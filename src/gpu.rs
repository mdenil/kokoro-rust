//! CUDA engine: the owner-accepted BF16x mixed-precision forward (owner #25), hydrated from the host
//! weight structs (weight norm, AdaIN affine defaults, ... resolved once), run on one stream.
//! - BF16 operands, f32 accumulation: every stride-1 convolution of the text encoder, predictor,
//!   decoder and generator (fused tensor-core kernel where the shape allows, per-tap cuBLAS
//!   otherwise) and every linear layer.
//! - f32: the small/strided convolutions, LSTM recurrences, attention products, normalization and
//!   statistics, source/STFT/iSTFT.
//! Kernels: kernels/kokoro.cu, built with -fmad=false. The only `unsafe` in this module: kernel
//! launches, uninitialized device allocations that are fully overwritten, and cuBLAS calls whose
//! operand extents are bounds-checked first.

use crate::albert::Albert;
use crate::model::{Kokoro, Output, HIDDEN, STYLE_DIM};
use crate::nn::{AdaIn1d, AdaInResBlock1, AdainResBlk1d, BiLstm, Conv1d, ConvTranspose1d, Linear};
use crate::vocoder::{HARMONICS, UPSAMPLE_SCALE};
use anyhow::{bail, ensure, Context, Result};
use cudarc::cublas::sys::cublasOperation_t as Op;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use std::sync::Arc;

type Buf = CudaSlice<f32>;

/// The embedded PTX (exposed for load-time probes).
pub const PTX_SRC: &str = PTX;
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kokoro.ptx"));
/// Kernel rounding mode of the accepted build (build.rs compiles with -fmad=false).
pub const KERNEL_ROUNDING: &str = "strict(-fmad=false)";

macro_rules! kernels {
    ($($name:ident),* $(,)?) => {
        #[allow(non_snake_case)]
        struct Kernels { $($name: CudaFunction,)* }
        impl Kernels {
            fn load(m: &Arc<CudaModule>) -> Result<Self> {
                Ok(Self { $($name: m.load_function(stringify!($name)).context(stringify!($name))?,)* })
            }
        }
    };
}

kernels!(
    fill_channels, fill_rows, leaky_relu, add_inplace, div_inplace, residual_scale,
    layer_norm_rows, gelu_new, softmax_rows, transpose,
    upsample_nearest2, dw_convT_k3s2, convT_gather, sine_phase_pre,
    sine_har_source, istft_ola, gen_noise, mask_gaps, adain_apply_seg, adaln_rows_seg,
    cat_style_rows_seg, gather_rows, gather_cols, lstm_seq_batched, reflect_pad_left1_seg, stft20_ld, istft_frames_ld,
    conv_direct_tiled, conv1d_strided, chan_stats_seg,
    bf16_transpose, bf16_convert, bf16_layout_a, conv1d_wmma_bf16, conv1d_wmma_bf16_snake, conv1d_wmma_bf16_snake_res,
);

macro_rules! launch {
    ($g:expr, $f:ident, $cfg:expr $(, $arg:expr)* $(,)?) => {{
        let cfg: LaunchConfig = $cfg;
        let mut b = $g.stream.launch_builder(&$g.k.$f);
        $( b.arg($arg); )*
        // SAFETY: argument list matches the kernel signature in kernels/kokoro.cu; every buffer
        // passed is sized by the caller for the index range the kernel's guard admits.
        unsafe { b.launch(cfg) }.map(|_| ()).context(stringify!($f))
    }};
}

fn cfg1(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (n.div_ceil(256).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    blas: CudaBlas,
    k: Kernels,
    tw: CudaSlice<f64>,
    win: Buf,
}

impl Gpu {
    /// Open CUDA device `ordinal` (an index into the devices visible under CUDA_VISIBLE_DEVICES).
    /// The kernels are compute-capability 8.9 PTX: older devices are refused; newer ones may JIT
    /// the PTX but are untested (a warning is printed).
    pub fn new(ordinal: usize) -> Result<Self> {
        let visible = std::env::var("CUDA_VISIBLE_DEVICES").map(|v| format!("CUDA_VISIBLE_DEVICES={v:?}")).unwrap_or_else(|_| "CUDA_VISIBLE_DEVICES unset".into());
        let ctx = CudaContext::new(ordinal).with_context(|| format!("opening CUDA device {ordinal} (--cuda-device, {visible}); an NVIDIA GPU with compute capability 8.9 and its driver are required"))?;
        let (major, minor) = ctx.compute_capability().context("querying the CUDA compute capability")?;
        let name = ctx.name().unwrap_or_else(|_| "unknown".into());
        ensure!((major, minor) >= (8, 9), "CUDA device {ordinal} ({name}) has compute capability {major}.{minor}; this build needs 8.9 (tested: RTX 4090)");
        if (major, minor) != (8, 9) {
            eprintln!("warning: CUDA device {ordinal} ({name}) has compute capability {major}.{minor}; only 8.9 (RTX 4090) is tested");
        }
        let stream = ctx.default_stream();
        let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(PTX)).context("loading the CUDA kernels (PTX JIT)")?;
        let blas = CudaBlas::new(stream.clone()).context("cuBLAS")?;
        let mut tw = vec![0.0f64; 440];
        for k in 0..11 {
            for n in 0..20 {
                let a = 2.0 * std::f64::consts::PI * ((k * n) % 20) as f64 / 20.0;
                tw[k * 20 + n] = a.cos();
                tw[220 + k * 20 + n] = a.sin();
            }
        }
        let win: Vec<f32> = (0..20).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / 20.0).cos()) as f32).collect();
        let tw = stream.clone_htod(&tw)?;
        let win = stream.clone_htod(&win)?;
        Ok(Self { k: Kernels::load(&module)?, ctx, stream, blas, tw, win })
    }

    pub fn device_name(&self) -> String {
        self.ctx.name().unwrap_or_else(|_| "unknown".into())
    }

    fn up(&self, x: &[f32]) -> Result<Buf> {
        Ok(self.stream.clone_htod(x)?)
    }

    pub fn down(&self, x: &Buf) -> Result<Vec<f32>> {
        Ok(self.stream.clone_dtoh(x)?)
    }

    /// f32 -> BF16 (round to nearest even), element-wise.
    fn bf16(&self, x: &Buf) -> Result<CudaSlice<u16>> {
        // SAFETY: fully written by the conversion kernel.
        let mut h = unsafe { self.stream.alloc::<u16>(x.len()) }?;
        let n = x.len() as i64;
        launch!(self, bf16_convert, cfg1(x.len()), x, &mut h, &n)?;
        Ok(h)
    }

    fn alloc(&self, n: usize) -> Result<Buf> {
        // SAFETY: every caller fully overwrites the buffer before it is read.
        Ok(unsafe { self.stream.alloc::<f32>(n.max(1)) }?)
    }

    /// Column-major C[m,n] = op(A)[m,k] op(B)[k,n] + beta C, with element offsets.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, ta: bool, tb: bool, m: usize, n: usize, k: usize, a: &Buf, a_off: usize, lda: usize, b: &Buf, b_off: usize, ldb: usize, beta: f32, c: &mut Buf, c_off: usize, ldc: usize) -> Result<()> {
        if m == 0 || n == 0 {
            return Ok(());
        }
        let ext = |off: usize, rows: usize, cols: usize, ld: usize| off + (cols - 1) * ld + rows;
        let (ar, ac) = if ta { (k, m) } else { (m, k) };
        let (br, bc) = if tb { (n, k) } else { (k, n) };
        ensure!(ar <= lda && br <= ldb && m <= ldc, "gemm: leading dimension too small");
        ensure!(ext(a_off, ar, ac, lda) <= a.len(), "gemm: A out of bounds");
        ensure!(ext(b_off, br, bc, ldb) <= b.len(), "gemm: B out of bounds");
        ensure!(ext(c_off, m, n, ldc) <= c.len(), "gemm: C out of bounds");
        let cfg = GemmConfig {
            transa: if ta { Op::CUBLAS_OP_T } else { Op::CUBLAS_OP_N },
            transb: if tb { Op::CUBLAS_OP_T } else { Op::CUBLAS_OP_N },
            m: m as i32,
            n: n as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: lda as i32,
            ldb: ldb as i32,
            beta,
            ldc: ldc as i32,
        };
        let (av, bv) = (a.slice(a_off..), b.slice(b_off..));
        let mut cv = c.slice_mut(c_off..);
        // SAFETY: operand extents checked above.
        unsafe { self.blas.gemm(cfg, &av, &bv, &mut cv) }.context("cublas sgemm")?;
        Ok(())
    }

    /// Raw cublasGemmEx (column-major) on device addresses: BF16 operands, f32 compute and output.
    /// Callers guarantee extents and alignment. alpha is 1; beta points to an f32.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_bf16(&self, ta: bool, tb: bool, m: usize, n: usize, k: usize, a: u64, lda: usize, b: u64, ldb: usize, beta: &f32, c: u64, ldc: usize) -> Result<()> {
        use cudarc::cublas::sys::{cublasComputeType_t as Ct, cublasGemmAlgo_t, cudaDataType as Dt};
        let one = 1.0f32;
        let alpha = &one as *const f32 as *const std::ffi::c_void;
        let beta = beta as *const f32 as *const std::ffi::c_void;
        let (at, bt, ct, compute) = (Dt::CUDA_R_16BF, Dt::CUDA_R_16BF, Dt::CUDA_R_32F, Ct::CUBLAS_COMPUTE_32F);
        cudarc::cublas::result::gemm_ex(
            *self.blas.handle(),
            if ta { Op::CUBLAS_OP_T } else { Op::CUBLAS_OP_N },
            if tb { Op::CUBLAS_OP_T } else { Op::CUBLAS_OP_N },
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a as *const std::ffi::c_void,
            at,
            lda as i32,
            b as *const std::ffi::c_void,
            bt,
            ldb as i32,
            beta,
            c as *mut std::ffi::c_void,
            ct,
            ldc as i32,
            compute,
            cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
        )
        .context("cublasGemmEx")
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_batched(&self, ta: bool, tb: bool, m: usize, n: usize, k: usize, a: &Buf, a_off: usize, lda: usize, sa: usize, b: &Buf, b_off: usize, ldb: usize, sb: usize, c: &mut Buf, c_off: usize, ldc: usize, sc: usize, batch: usize) -> Result<()> {
        let ext = |off: usize, rows: usize, cols: usize, ld: usize, s: usize| off + (batch - 1) * s + (cols - 1) * ld + rows;
        let (ar, ac) = if ta { (k, m) } else { (m, k) };
        let (br, bc) = if tb { (n, k) } else { (k, n) };
        ensure!(ext(a_off, ar, ac, lda, sa) <= a.len(), "gemm_batched: A out of bounds");
        ensure!(ext(b_off, br, bc, ldb, sb) <= b.len(), "gemm_batched: B out of bounds");
        ensure!(ext(c_off, m, n, ldc, sc) <= c.len(), "gemm_batched: C out of bounds");
        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: if ta { Op::CUBLAS_OP_T } else { Op::CUBLAS_OP_N },
                transb: if tb { Op::CUBLAS_OP_T } else { Op::CUBLAS_OP_N },
                m: m as i32,
                n: n as i32,
                k: k as i32,
                alpha: 1.0f32,
                lda: lda as i32,
                ldb: ldb as i32,
                beta: 0.0f32,
                ldc: ldc as i32,
            },
            batch_size: batch as i32,
            stride_a: sa as i64,
            stride_b: sb as i64,
            stride_c: sc as i64,
        };
        let av = a.slice(a_off..);
        let bv = b.slice(b_off..);
        let mut cv = c.slice_mut(c_off..);
        // SAFETY: operand extents (including the batch stride) checked above.
        unsafe { self.blas.gemm_strided_batched(cfg, &av, &bv, &mut cv) }.context("cublas sgemm batched")?;
        Ok(())
    }

    fn transpose(&self, x: &Buf, r: usize, c: usize) -> Result<Buf> {
        let mut y = self.alloc(r * c)?;
        let (ri, ci) = (r as i32, c as i32);
        let cfg = LaunchConfig { grid_dim: (c.div_ceil(32) as u32, r.div_ceil(32) as u32, 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 };
        launch!(self, transpose, cfg, x, &mut y, &ri, &ci)?;
        Ok(y)
    }

    fn leaky(&self, x: &mut Buf, slope: f32) -> Result<()> {
        let n = x.len() as i64;
        launch!(self, leaky_relu, cfg1(x.len()), x, &n, &slope)
    }

    fn add(&self, a: &mut Buf, b: &Buf) -> Result<()> {
        ensure!(a.len() == b.len(), "add: length mismatch");
        let n = a.len() as i64;
        launch!(self, add_inplace, cfg1(a.len()), a, b, &n)
    }

    /// Concatenate channel-major blocks along channels: parts are ([c_i, t]) buffers.
    fn cat_channels(&self, parts: &[&Buf]) -> Result<Buf> {
        let total: usize = parts.iter().map(|p| p.len()).sum();
        let mut y = self.alloc(total)?;
        let mut off = 0;
        for p in parts {
            let mut dst = y.slice_mut(off..off + p.len());
            self.stream.memcpy_dtod(*p, &mut dst)?;
            off += p.len();
        }
        Ok(y)
    }
}

// ------------------------------------------------------------------------------------ layers

pub struct GLinear {
    /// BF16 weights [dout][din]
    w: CudaSlice<u16>,
    b: Option<Buf>,
    din: usize,
    dout: usize,
}

impl GLinear {
    fn new(g: &Gpu, l: &Linear) -> Result<Self> {
        Ok(Self { w: g.bf16(&g.up(&l.w)?)?, b: l.b.as_ref().map(|b| g.up(b)).transpose()?, din: l.din, dout: l.dout })
    }

    /// x [t, din] -> [t, dout]: BF16 operands (x converted per call), f32 accumulate and output.
    fn fwd(&self, g: &Gpu, x: &Buf, t: usize) -> Result<Buf> {
        let mut y = g.alloc(t * self.dout)?;
        if t == 0 {
            return Ok(y);
        }
        if let Some(b) = &self.b {
            let (ri, di) = (t as i32, self.dout as i32);
            launch!(g, fill_rows, cfg1(t * self.dout), &mut y, b, &ri, &di)?;
        }
        let beta = if self.b.is_some() { 1.0f32 } else { 0.0 };
        let xh = g.bf16(x)?;
        {
            let (pw, _gw) = self.w.device_ptr(&g.stream);
            let (px, _gx) = xh.device_ptr(&g.stream);
            let (py, _gy) = y.device_ptr_mut(&g.stream);
            // SAFETY: A = W [dout][din] (read transposed), B = x [t][din], C = y [t][dout]; extents
            // din*dout, t*din and t*dout by construction.
            unsafe { g.gemm_bf16(true, false, self.dout, t, self.din, pw, self.din, px, self.din, &beta, py, self.dout) }?;
        }
        Ok(y)
    }
}

/// Input-channel chunk of conv1d_strided; MUST equal IG_BK in kernels/kokoro.cu.
const IG_BK: usize = 4;

/// Convolution weights by execution route (fixed per layer at load from its shape).
enum ConvW {
    /// f32 strided implicit-GEMM conv (conv1d_strided): weights [Cin][K][Cout]
    Strided(Buf),
    /// f32 small direct conv, stride 1 (conv_direct_tiled): weights [Cout][Cin][K]
    Direct(Buf),
    /// BF16 per-tap tensor-core GEMMs (shapes the fused kernel does not take): weights [K][Cout][Cin]
    Bf16PerTap(CudaSlice<u16>),
    /// BF16 fused tensor-core conv (conv1d_wmma_bf16*): weights in its A-fragment layout
    Bf16Fused(CudaSlice<u16>),
}

pub struct GConv {
    w: ConvW,
    b: Option<Buf>,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dil: usize,
}

impl GConv {
    fn new(g: &Gpu, c: &Conv1d) -> Result<Self> {
        let w = if c.stride > 1 {
            ensure!(c.dil == 1, "strided conv with dilation {} is not supported", c.dil);
            let xw = (128 - 1) * c.stride + (c.k - 1) * c.dil + 1;
            ensure!((IG_BK * xw + IG_BK * c.k * 64) * 4 <= 48 * 1024, "strided conv (stride {}, k {}) exceeds the kernel's shared memory", c.stride, c.k);
            let mut v = vec![0.0f32; c.w.len()];
            for co in 0..c.cout {
                for ci in 0..c.cin {
                    for kk in 0..c.k {
                        v[(ci * c.k + kk) * c.cout + co] = c.w[(co * c.cin + ci) * c.k + kk];
                    }
                }
            }
            ConvW::Strided(g.up(&v)?)
        } else if c.cin * c.k <= 32 {
            ensure!(c.dil == 1 && c.cin * (63 + c.k) <= 12288, "small conv (cin {}, k {}, dil {}) does not fit the tiled kernel", c.cin, c.k, c.dil);
            ConvW::Direct(g.up(&c.w)?)
        } else {
            let mut wt = vec![0.0f32; c.w.len()];
            for co in 0..c.cout {
                for ci in 0..c.cin {
                    for kk in 0..c.k {
                        wt[kk * c.cout * c.cin + co * c.cin + ci] = c.w[(co * c.cin + ci) * c.k + kk];
                    }
                }
            }
            let taps = g.bf16(&g.up(&wt)?)?;
            if c.cout % 64 == 0 && c.cin % 16 == 0 && 2 * c.pad == c.dil * (c.k - 1) {
                let na = c.cout.div_ceil(64) * c.cin.div_ceil(16) * c.k * 1024;
                // SAFETY: fully written by bf16_layout_a (every index < na).
                let mut wa = unsafe { g.stream.alloc::<u16>(na) }?;
                let (ci, co, kk, nn) = (c.cin as i32, c.cout as i32, c.k as i32, na as i64);
                launch!(g, bf16_layout_a, cfg1(na), &taps, &mut wa, &ci, &co, &kk, &nn)?;
                ConvW::Bf16Fused(wa)
            } else {
                ConvW::Bf16PerTap(taps)
            }
        };
        Ok(Self { w, b: c.b.as_ref().map(|b| g.up(b)).transpose()?, cin: c.cin, cout: c.cout, k: c.k, stride: c.stride, pad: c.pad, dil: c.dil })
    }

    /// Whether this layer runs on the fused tensor-core kernel (and so takes the AdaIN + Snake
    /// prologue in the Snake blocks).
    pub(crate) fn fused(&self) -> bool {
        matches!(self.w, ConvW::Bf16Fused(_))
    }

    fn fwd(&self, g: &Gpu, x: &Buf, t: usize) -> Result<(Buf, usize)> {
        ensure!(x.len() == self.cin * t, "conv input size");
        let tout = crate::ops::conv1d_out_len(t, self.k, self.stride, self.pad, self.dil);
        let mut y = g.alloc(self.cout * tout)?;
        let null = 0u64;
        match &self.w {
            ConvW::Strided(w) => {
                let xw = (128 - 1) * self.stride + (self.k - 1) * self.dil + 1;
                let smem = (IG_BK * xw + IG_BK * self.k * 64) * 4;
                let a = [self.cin as i32, t as i32, self.cout as i32, self.k as i32, self.dil as i32, self.pad as i32, self.stride as i32, tout as i32];
                let cfg = LaunchConfig { grid_dim: (tout.div_ceil(128) as u32, self.cout.div_ceil(64) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: smem as u32 };
                match &self.b {
                    Some(b) => launch!(g, conv1d_strided, cfg, x, w, b, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7])?,
                    None => launch!(g, conv1d_strided, cfg, x, w, &null, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7])?,
                }
            }
            ConvW::Direct(w) => {
                let win = (64 - 1) * self.stride + self.k;
                let a = [self.cin as i32, t as i32, self.cout as i32, self.k as i32, self.stride as i32, self.pad as i32, tout as i32];
                let cfg = LaunchConfig {
                    grid_dim: (tout.div_ceil(64) as u32, self.cout.div_ceil(16) as u32, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: (self.cin * win * 4) as u32,
                };
                match &self.b {
                    Some(b) => launch!(g, conv_direct_tiled, cfg, x, w, b, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6])?,
                    None => launch!(g, conv_direct_tiled, cfg, x, w, &null, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6])?,
                }
            }
            ConvW::Bf16Fused(_) => {
                ensure!(tout == t, "fused conv must preserve length");
                self.launch_fused(g, x, t, &mut y, None, false)?;
            }
            ConvW::Bf16PerTap(wh) => self.fwd_per_tap(g, wh, x, t, tout, &mut y)?,
        }
        Ok((y, tout))
    }

    /// BF16 per-tap conv on tensor cores. Activations are converted once per call into a transposed
    /// [T][Cin] BF16 operand, so every tap's A operand starts at a multiple of Cin; f32 accumulate.
    fn fwd_per_tap(&self, g: &Gpu, wh: &CudaSlice<u16>, x: &Buf, t: usize, tout: usize, y: &mut Buf) -> Result<()> {
        let (cin, cout) = (self.cin, self.cout);
        let null = 0u64;
        let tcfg = LaunchConfig { grid_dim: (t.div_ceil(32) as u32, cin.div_ceil(32) as u32, 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 };
        let (ci, ti) = (cin as i32, t as i32);
        // SAFETY: fully written by the transpose kernel (all t < T, c < Cin).
        let mut xt = unsafe { g.stream.alloc::<u16>(t * cin) }?;
        launch!(g, bf16_transpose, tcfg, x, &mut xt, &ci, &ti)?;
        let (co, to) = (cout as i32, tout as i32);
        match &self.b {
            Some(b) => launch!(g, fill_channels, cfg1(cout * tout), &mut *y, b, &co, &to)?,
            None => launch!(g, fill_channels, cfg1(cout * tout), &mut *y, &null, &co, &to)?,
        }
        let (px, _gx) = xt.device_ptr(&g.stream);
        let (pw, _gw) = wh.device_ptr(&g.stream);
        let (py, _gy) = y.device_ptr_mut(&g.stream);
        let beta = 1.0f32;
        for kk in 0..self.k {
            let shift = (kk * self.dil) as isize - self.pad as isize;
            let o_lo = if shift >= 0 { 0 } else { (-shift) as usize };
            let max_in = t as isize - 1 - shift;
            if max_in < 0 {
                continue;
            }
            let o_hi = (max_in as usize).min(tout - 1);
            if o_lo > o_hi {
                continue;
            }
            let (n, a_row) = (o_hi - o_lo + 1, (o_lo as isize + shift) as usize);
            // SAFETY: A = xt rows [a_row, a_row+n) x Cin (in bounds: a_row + n <= t);
            // B = tap kk weights Cin x Cout; C = y columns [o_lo, o_lo+n) of Cout rows.
            unsafe { g.gemm_bf16(true, false, n, cout, cin, px + 2 * (a_row * cin) as u64, cin, pw + 2 * (kk * cout * cin) as u64, cin, &beta, py + 4 * o_lo as u64, tout) }?;
        }
        Ok(())
    }

    /// Launch the fused tensor-core conv into `y` (same length `t`). With `pro`, the input is the raw
    /// AdaIN input and the kernel applies AdaIN + Snake while staging it; `res` accumulates into `y`
    /// (only with `pro`).
    fn launch_fused(&self, g: &Gpu, x: &Buf, t: usize, y: &mut Buf, pro: Option<SnakePro>, res: bool) -> Result<()> {
        let ConvW::Bf16Fused(wa) = &self.w else { bail!("not a fused conv") };
        let (cin, cout) = (self.cin, self.cout);
        let null = 0u64;
        let tw = 128 + (self.k - 1) * self.dil;
        let smem = (tw * 48 * 2).next_multiple_of(128) + 8 * 256 * 4;
        let cfg = LaunchConfig { grid_dim: (t.div_ceil(128) as u32, cout.div_ceil(128) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem as u32 };
        let k = &g.k;
        let f = match (&pro, res) {
            (None, false) => &k.conv1d_wmma_bf16,
            (Some(_), false) => &k.conv1d_wmma_bf16_snake,
            (Some(_), true) => &k.conv1d_wmma_bf16_snake_res,
            (None, true) => bail!("residual fused conv needs the Snake prologue"),
        };
        let a = [cin as i32, t as i32, cout as i32, self.k as i32, self.dil as i32, self.pad as i32];
        let mut lb = g.stream.launch_builder(f);
        lb.arg(x).arg(wa);
        match &self.b {
            Some(b) => lb.arg(b),
            None => lb.arg(&null),
        };
        lb.arg(y).arg(&a[0]).arg(&a[1]).arg(&a[2]).arg(&a[3]).arg(&a[4]).arg(&a[5]);
        if let Some(p) = &pro {
            lb.arg(p.col_item).arg(p.mean).arg(p.rstd).arg(p.nw).arg(p.nb).arg(p.gb).arg(p.alpha);
        }
        // SAFETY: matches conv1d_wmma_bf16(x, wa, b|null, y, Cin, T, Cout, K, dil, pad[, col_item, mean,
        // rstd, nw, nb, gb, alpha]); wa holds ceil(Cout/64) * ceil(Cin/16) * K * 1024 BF16 values (the
        // kernel's full index range; warps of a missing upper 64-co tile skip all loads); x Cin*T,
        // y Cout*T (callers check sizes); prologue shapes checked in fwd_snake; smem = window
        // [TW][48] BF16 + 8 KB epilogue tiles.
        unsafe { lb.launch(cfg) }.context("conv1d_wmma_bf16")?;
        Ok(())
    }

    /// y (= or += with `res`) conv(snake(adain(x))) with the AdaIN + Snake applied while staging
    /// the conv input (exactly adain_apply_seg's arithmetic; gap columns -> 0). `x` is the raw
    /// AdaIN input [Cin][t]; mean/rstd [B][Cin], gb [B][2 Cin], nw/nb/alpha [Cin].
    pub(crate) fn fwd_snake(&self, g: &Gpu, x: &Buf, t: usize, y: &mut Buf, res: bool, p: SnakePro) -> Result<()> {
        ensure!(self.fused() && x.len() == self.cin * t && y.len() == self.cout * t && p.col_item.len() >= t, "fused Snake conv not applicable");
        ensure!(p.nw.len() == self.cin && p.nb.len() == self.cin && p.alpha.len() == self.cin && p.mean.len() == p.rstd.len() && p.gb.len() == 2 * p.mean.len(), "prologue parameter shapes");
        self.launch_fused(g, x, t, y, Some(p), res)
    }
}

/// AdaIN + Snake prologue operands of the fused conv.
pub(crate) struct SnakePro<'a> {
    pub col_item: &'a CudaSlice<i32>,
    pub mean: &'a Buf,
    pub rstd: &'a Buf,
    pub nw: &'a Buf,
    pub nb: &'a Buf,
    pub gb: &'a Buf,
    pub alpha: &'a Buf,
}

pub struct GConvT {
    w: Buf, // native [Cin, Cout*K]
    b: Buf,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl GConvT {
    fn new(g: &Gpu, c: &ConvTranspose1d) -> Result<Self> {
        Ok(Self { w: g.up(&c.w)?, b: g.up(&c.b)?, cin: c.cin, cout: c.cout, k: c.k, stride: c.stride, pad: c.pad })
    }

    fn fwd(&self, g: &Gpu, x: &Buf, tin: usize) -> Result<(Buf, usize)> {
        let m = self.cout * self.k;
        let tout = (tin - 1) * self.stride + self.k - 2 * self.pad;
        let mut z = g.alloc(m * tin)?;
        // Z^T[i, m] = X^T[i, ci] * W[ci, m]
        g.gemm(false, true, tin, m, self.cin, x, 0, tin, &self.w, 0, m, 0.0, &mut z, 0, tin)?;
        let mut y = g.alloc(self.cout * tout)?;
        let a = [self.cout as i32, self.k as i32, tin as i32, self.stride as i32, self.pad as i32, tout as i32];
        launch!(g, convT_gather, cfg1(self.cout * tout), &z, &self.b, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5])?;
        Ok((y, tout))
    }
}

pub struct GAdaIn {
    nw: Buf,
    nb: Buf,
    fc: GLinear,
    c: usize,
}

impl GAdaIn {
    fn new(g: &Gpu, a: &AdaIn1d) -> Result<Self> {
        Ok(Self { nw: g.up(&a.norm_w)?, nb: g.up(&a.norm_b)?, fc: GLinear::new(g, &a.fc)?, c: a.c })
    }

}

pub struct GResBlk {
    conv1: GConv,
    conv2: GConv,
    norm1: GAdaIn,
    norm2: GAdaIn,
    conv1x1: Option<GConv>,
    pool: Option<(Buf, Buf)>,
    dim_in: usize,
    upsample: bool,
}

impl GResBlk {
    fn new(g: &Gpu, b: &AdainResBlk1d) -> Result<Self> {
        Ok(Self {
            conv1: GConv::new(g, &b.conv1)?,
            conv2: GConv::new(g, &b.conv2)?,
            norm1: GAdaIn::new(g, &b.norm1)?,
            norm2: GAdaIn::new(g, &b.norm2)?,
            conv1x1: b.conv1x1.as_ref().map(|c| GConv::new(g, c)).transpose()?,
            pool: b.pool.as_ref().map(|(w, bb)| Ok::<_, anyhow::Error>((g.up(w)?, g.up(bb)?))).transpose()?,
            dim_in: b.dim_in,
            upsample: b.upsample,
        })
    }

}

pub struct GSnakeBlk {
    convs1: Vec<GConv>,
    convs2: Vec<GConv>,
    adain1: Vec<GAdaIn>,
    adain2: Vec<GAdaIn>,
    alpha1: Vec<Buf>,
    alpha2: Vec<Buf>,
}

impl GSnakeBlk {
    fn new(g: &Gpu, b: &AdaInResBlock1) -> Result<Self> {
        let conv = |v: &Vec<Conv1d>| v.iter().map(|c| GConv::new(g, c)).collect::<Result<Vec<_>>>();
        let ada = |v: &Vec<AdaIn1d>| v.iter().map(|a| GAdaIn::new(g, a)).collect::<Result<Vec<_>>>();
        let al = |v: &Vec<Vec<f32>>| v.iter().map(|a| g.up(a)).collect::<Result<Vec<_>>>();
        Ok(Self { convs1: conv(&b.convs1)?, convs2: conv(&b.convs2)?, adain1: ada(&b.adain1)?, adain2: ada(&b.adain2)?, alpha1: al(&b.alpha1)?, alpha2: al(&b.alpha2)? })
    }

}

pub struct GLstm {
    wih: [GLinear; 2],
    whh: [Buf; 2],
    bhh: [Buf; 2],
    h: usize,
}

impl GLstm {
    fn new(g: &Gpu, l: &BiLstm) -> Result<Self> {
        let lin = |d: usize| -> Result<GLinear> { GLinear::new(g, &Linear { w: l.wih[d].clone(), b: Some(l.bih[d].clone()), din: l.din, dout: 4 * l.h }) };
        Ok(Self { wih: [lin(0)?, lin(1)?], whh: [g.up(&l.whh[0])?, g.up(&l.whh[1])?], bhh: [g.up(&l.bhh[0])?, g.up(&l.bhh[1])?], h: l.h })
    }

}

// ------------------------------------------------------------------------------------ model

pub struct GAlbert {
    map_in: GLinear,
    q: GLinear,
    k: GLinear,
    v: GLinear,
    dense: GLinear,
    attn_ln: (Buf, Buf),
    ffn: GLinear,
    ffn_out: GLinear,
    full_ln: (Buf, Buf),
}

impl GAlbert {
    fn new(g: &Gpu, a: &Albert) -> Result<Self> {
        Ok(Self {
            map_in: GLinear::new(g, &a.map_in)?,
            q: GLinear::new(g, &a.q)?,
            k: GLinear::new(g, &a.k)?,
            v: GLinear::new(g, &a.v)?,
            dense: GLinear::new(g, &a.dense)?,
            attn_ln: (g.up(&a.attn_ln.0)?, g.up(&a.attn_ln.1)?),
            ffn: GLinear::new(g, &a.ffn)?,
            ffn_out: GLinear::new(g, &a.ffn_out)?,
            full_ln: (g.up(&a.full_ln.0)?, g.up(&a.full_ln.1)?),
        })
    }

    fn ln(g: &Gpu, x: &mut Buf, rows: usize, d: usize, w: &Buf, b: &Buf, eps: f32) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (ri, di, null) = (rows as i32, d as i32, 0u64);
        launch!(g, layer_norm_rows, cfg, x, w, b, &null, &ri, &di, &eps)
    }

}

pub struct GpuKokoro {
    pub gpu: Gpu,
    albert: GAlbert,
    bert_encoder: GLinear,
    dur_lstms: Vec<GLstm>,
    dur_norms: Vec<GLinear>,
    pred_lstm: GLstm,
    dur_proj: GLinear,
    shared: GLstm,
    f0: Vec<GResBlk>,
    n: Vec<GResBlk>,
    f0_proj: GConv,
    n_proj: GConv,
    te_cnn: Vec<(GConv, Buf, Buf)>,
    te_lstm: GLstm,
    encode: GResBlk,
    decode: Vec<GResBlk>,
    f0_conv: GConv,
    n_conv: GConv,
    asr_res: GConv,
    l_w: Buf,
    l_b: f32,
    noise_convs: Vec<GConv>,
    noise_res: Vec<GSnakeBlk>,
    ups: Vec<GConvT>,
    resblocks: Vec<GSnakeBlk>,
    conv_post: GConv,
}

impl GpuKokoro {

    /// Build on an already-initialized device (lets the CUDA context be created concurrently with
    /// checkpoint parsing).
    pub fn with_gpu(g: Gpu, m: &Kokoro) -> Result<Self> {
        let p = &m.predictor;
        let d = &m.decoder;
        let gen = &d.generator;
        let rb = |v: &Vec<AdainResBlk1d>| v.iter().map(|b| GResBlk::new(&g, b)).collect::<Result<Vec<_>>>();
        let sb = |v: &Vec<AdaInResBlock1>| v.iter().map(|b| GSnakeBlk::new(&g, b)).collect::<Result<Vec<_>>>();
        let s = Self {
            albert: GAlbert::new(&g, &m.albert)?,
            bert_encoder: GLinear::new(&g, &m.bert_encoder)?,
            dur_lstms: p.dur_lstms.iter().map(|l| GLstm::new(&g, l)).collect::<Result<_>>()?,
            dur_norms: p.dur_norms.iter().map(|l| GLinear::new(&g, l)).collect::<Result<_>>()?,
            pred_lstm: GLstm::new(&g, &p.lstm)?,
            dur_proj: GLinear::new(&g, &p.duration_proj)?,
            shared: GLstm::new(&g, &p.shared)?,
            f0: rb(&p.f0)?,
            n: rb(&p.n)?,
            f0_proj: GConv::new(&g, &p.f0_proj)?,
            n_proj: GConv::new(&g, &p.n_proj)?,
            te_cnn: m.text_encoder.cnn.iter().map(|(c, ga, be)| Ok((GConv::new(&g, c)?, g.up(ga)?, g.up(be)?))).collect::<Result<_>>()?,
            te_lstm: GLstm::new(&g, &m.text_encoder.lstm)?,
            encode: GResBlk::new(&g, &d.encode)?,
            decode: rb(&d.decode)?,
            f0_conv: GConv::new(&g, &d.f0_conv)?,
            n_conv: GConv::new(&g, &d.n_conv)?,
            asr_res: GConv::new(&g, &d.asr_res)?,
            l_w: g.up(&gen.l_w)?,
            l_b: gen.l_b,
            noise_convs: gen.noise_convs.iter().map(|c| GConv::new(&g, c)).collect::<Result<_>>()?,
            noise_res: sb(&gen.noise_res)?,
            ups: gen.ups.iter().map(|c| GConvT::new(&g, c)).collect::<Result<_>>()?,
            resblocks: sb(&gen.resblocks)?,
            conv_post: GConv::new(&g, &gen.conv_post)?,
            gpu: g,
        };
        for b in s.noise_res.iter().chain(&s.resblocks) {
            ensure!(b.convs1.iter().chain(&b.convs2).all(|c| c.fused()), "Snake-block convs must run on the fused BF16 kernel");
        }
        s.gpu.stream.synchronize()?;
        Ok(s)
    }

}

#[path = "gpu_batch.rs"]
mod batch;
pub use batch::{BatchItem, ItemNoise};
