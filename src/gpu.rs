//! CUDA backend (feature `cuda`): the same forward as the CPU path, hydrated from the proven CPU
//! structs (weight norm, AdaIN affine defaults, ... resolved once), run on one stream.
//! Matrix products: cuBLAS SGEMM (default math mode = full f32, no TF32). Everything else:
//! kernels/kokoro.cu. The only `unsafe` in this module: kernel launches, uninitialized device
//! allocations that are fully overwritten, and cuBLAS calls whose operand extents are
//! bounds-checked in `gemm`/`gemm_batched` first.

use crate::albert::{self, Albert};
use crate::model::{self, Kokoro, Output, HIDDEN, STYLE_DIM};
use crate::nn::{AdaIn1d, AdaInResBlock1, AdainResBlk1d, BiLstm, Conv1d, ConvTranspose1d, Linear};
use crate::vocoder::{self, NoiseSource, HARMONICS, UPSAMPLE_SCALE};
use anyhow::{ensure, Context, Result};
use cudarc::cublas::sys::cublasOperation_t as Op;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

type Buf = CudaSlice<f32>;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kokoro.ptx"));
/// Kernel rounding mode baked in at build time ("fma" default, or "strict(-fmad=false)").
pub const KERNEL_ROUNDING: &str = env!("KOKORO_KERNEL_ROUNDING");

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
    fill_channels, fill_rows, leaky_relu, add_inplace, div_inplace, residual_scale, chan_stats, adain_apply,
    layer_norm_rows, gelu_new, softmax_rows, transpose, cat_style_rows, expand_rows, expand_cols, lstm_step,
    upsample_nearest2, dw_convT_k3s2, conv_direct, convT_gather, reflect_pad_left1, sine_phase_pre,
    sine_har_source, stft20, istft_frames, istft_ola, gen_noise, lstm_seq, mask_gaps, chan_stats_seg, adain_apply_seg, adaln_rows_seg,
    cat_style_rows_seg, gather_rows, gather_cols, lstm_seq_batched, reflect_pad_left1_seg, stft20_ld, istft_frames_ld, conv_direct_tiled, conv1d_igemm, conv1d_igemm_res,
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
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).context("CUDA context")?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(cudarc::nvrtc::Ptx::from_src(PTX)).context("loading PTX")?;
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

    /// With KOKORO_PROFILE=1, block until queued GPU work finishes so stage scopes measure
    /// device time instead of launch time. No-op otherwise.
    fn prof_sync(&self) {
        if crate::prof::enabled() {
            let _ = self.stream.synchronize();
        }
    }

    /// Public alias of the KOKORO_PROFILE device sync (no-op unless profiling).
    pub fn prof_sync_pub(&self) {
        self.prof_sync();
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
    w: Buf,
    b: Option<Buf>,
    din: usize,
    dout: usize,
}

impl GLinear {
    fn new(g: &Gpu, l: &Linear) -> Result<Self> {
        Ok(Self { w: g.up(&l.w)?, b: l.b.as_ref().map(|b| g.up(b)).transpose()?, din: l.din, dout: l.dout })
    }

    /// x [t, din] -> [t, dout]
    fn fwd(&self, g: &Gpu, x: &Buf, t: usize) -> Result<Buf> {
        let mut y = g.alloc(t * self.dout)?;
        if let Some(b) = &self.b {
            let (ri, di) = (t as i32, self.dout as i32);
            launch!(g, fill_rows, cfg1(t * self.dout), &mut y, b, &ri, &di)?;
        }
        let beta = if self.b.is_some() { 1.0 } else { 0.0 };
        g.gemm(true, false, self.dout, t, self.din, &self.w, 0, self.din, x, 0, self.din, beta, &mut y, 0, self.dout)?;
        Ok(y)
    }
}

/// Shape policy: the fused kernel wins where per-tap GEMMs are memory-bound (small Cin); wider
/// layers stay on cuBLAS (measured, PL-008). KOKORO_CONV_IGEMM_MAX_CIN overrides for A/B.
static IGEMM_MAX_CIN: std::sync::LazyLock<usize> =
    std::sync::LazyLock::new(|| std::env::var("KOKORO_CONV_IGEMM_MAX_CIN").ok().and_then(|v| v.parse().ok()).unwrap_or(128));
static IGEMM_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("KOKORO_CONV_IGEMM").map(|v| v != "0").unwrap_or(true));

pub struct GConv {
    wt: Buf, // [K][Cout][Cin] per-tap transposed (stride 1) or original [Cout][Cin][K] (direct)
    /// [Cin][K][Cout] for the fused implicit-GEMM kernel (stride-1, non-direct convs)
    w_ig: Option<Buf>,
    b: Option<Buf>,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dil: usize,
    direct: bool,
}

impl GConv {
    fn new(g: &Gpu, c: &Conv1d) -> Result<Self> {
        let direct = c.stride != 1 || c.cin * c.k <= 32;
        let wt = if direct {
            c.w.clone()
        } else {
            let mut wt = vec![0.0f32; c.w.len()];
            for co in 0..c.cout {
                for ci in 0..c.cin {
                    for kk in 0..c.k {
                        wt[kk * c.cout * c.cin + co * c.cin + ci] = c.w[(co * c.cin + ci) * c.k + kk];
                    }
                }
            }
            wt
        };
        let w_ig = if !direct && c.stride == 1 {
            let mut v = vec![0.0f32; c.w.len()];
            for co in 0..c.cout {
                for ci in 0..c.cin {
                    for kk in 0..c.k {
                        v[(ci * c.k + kk) * c.cout + co] = c.w[(co * c.cin + ci) * c.k + kk];
                    }
                }
            }
            Some(g.up(&v)?)
        } else {
            None
        };
        Ok(Self { wt: g.up(&wt)?, w_ig, b: c.b.as_ref().map(|b| g.up(b)).transpose()?, cin: c.cin, cout: c.cout, k: c.k, stride: c.stride, pad: c.pad, dil: c.dil, direct })
    }

    /// Whether the fused implicit-GEMM path applies to a same-length conv over length `t`.
    pub(crate) fn igemm_applicable(&self) -> bool {
        self.w_ig.is_some() && *IGEMM_ON && self.cin <= *IGEMM_MAX_CIN && (8 * (128 + (self.k - 1) * self.dil) + 8 * self.k * 64) * 4 <= 48 * 1024
    }

    /// res += conv(x) in place via the fused kernel's residual epilogue (x must already be masked).
    pub(crate) fn fwd_igemm_res(&self, g: &Gpu, x: &Buf, t: usize, res: &mut Buf) -> Result<()> {
        ensure!(self.igemm_applicable() && self.stride == 1 && x.len() == self.cin * t && res.len() == self.cout * t, "fused conv not applicable");
        let w_ig = self.w_ig.as_ref().unwrap();
        let smem = (8 * (128 + (self.k - 1) * self.dil) + 8 * self.k * 64) * 4;
        let a = [self.cin as i32, t as i32, self.cout as i32, self.k as i32, self.dil as i32, self.pad as i32];
        let cfg = LaunchConfig { grid_dim: (t.div_ceil(128) as u32, self.cout.div_ceil(64) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: smem as u32 };
        let null = 0u64;
        match &self.b {
            Some(b) => launch!(g, conv1d_igemm_res, cfg, x, w_ig, b, res, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5])?,
            None => launch!(g, conv1d_igemm_res, cfg, x, w_ig, &null, res, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5])?,
        }
        Ok(())
    }

    fn fwd(&self, g: &Gpu, x: &Buf, t: usize) -> Result<(Buf, usize)> {
        ensure!(x.len() == self.cin * t, "conv input size");
        let tout = crate::ops::conv1d_out_len(t, self.k, self.stride, self.pad, self.dil);
        let mut y = g.alloc(self.cout * tout)?;
        let null = 0u64;
        let win = (64 - 1) * self.stride + self.k;
        let tiled_ok = self.dil == 1 && self.cin * win <= 12288;
        if self.direct && tiled_ok && std::env::var("KOKORO_CONV_TILED").map(|v| v != "0").unwrap_or(true) {
            // LEVER PL-004 (kill switch KOKORO_CONV_TILED=0): same arithmetic order, shared-memory tile
            let a = [self.cin as i32, t as i32, self.cout as i32, self.k as i32, self.stride as i32, self.pad as i32, tout as i32];
            let cfg = LaunchConfig {
                grid_dim: (tout.div_ceil(64) as u32, self.cout.div_ceil(16) as u32, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: (self.cin * win * 4) as u32,
            };
            let null = 0u64;
            match &self.b {
                Some(b) => launch!(g, conv_direct_tiled, cfg, x, &self.wt, b, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6])?,
                None => launch!(g, conv_direct_tiled, cfg, x, &self.wt, &null, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6])?,
            }
            return Ok((y, tout));
        }
        if self.direct {
            let a = [self.cin as i32, t as i32, self.cout as i32, self.k as i32, self.stride as i32, self.pad as i32, tout as i32];
            match &self.b {
                Some(b) => launch!(g, conv_direct, cfg1(self.cout * tout), x, &self.wt, b, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6])?,
                None => launch!(g, conv_direct, cfg1(self.cout * tout), x, &self.wt, &null, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6])?,
            }
            return Ok((y, tout));
        }
        if let Some(w_ig) = &self.w_ig {
            if tout == t && *IGEMM_ON && self.cin <= *IGEMM_MAX_CIN {
                // LEVER PL-008 (kill switch KOKORO_CONV_IGEMM=0): fused implicit-GEMM conv
                let xw = 128 + (self.k - 1) * self.dil;
                let smem = (8 * xw + 8 * self.k * 64) * 4;
                if smem <= 48 * 1024 {
                    let a = [self.cin as i32, t as i32, self.cout as i32, self.k as i32, self.dil as i32, self.pad as i32];
                    let cfg = LaunchConfig { grid_dim: (t.div_ceil(128) as u32, self.cout.div_ceil(64) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: smem as u32 };
                    match &self.b {
                        Some(b) => launch!(g, conv1d_igemm, cfg, x, w_ig, b, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5])?,
                        None => launch!(g, conv1d_igemm, cfg, x, w_ig, &null, &mut y, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5])?,
                    }
                    return Ok((y, tout));
                }
            }
        }
        let (ci, ti) = (self.cout as i32, tout as i32);
        match &self.b {
            Some(b) => launch!(g, fill_channels, cfg1(self.cout * tout), &mut y, b, &ci, &ti)?,
            None => launch!(g, fill_channels, cfg1(self.cout * tout), &mut y, &null, &ci, &ti)?,
        }
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
            let n = o_hi - o_lo + 1;
            let b_off = (o_lo as isize + shift) as usize;
            // Y^T[o, co] += X^T[o+shift, ci] * Wt_k^T[ci, co]
            g.gemm(false, false, n, self.cout, self.cin, x, b_off, t, &self.wt, kk * self.cout * self.cin, self.cin, 1.0, &mut y, o_lo, tout)?;
        }
        Ok((y, tout))
    }
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

#[derive(Clone, Copy)]
enum Act<'a> {
    Leaky(f32),
    Snake(&'a Buf),
}

impl GAdaIn {
    fn new(g: &Gpu, a: &AdaIn1d) -> Result<Self> {
        Ok(Self { nw: g.up(&a.norm_w)?, nb: g.up(&a.norm_b)?, fc: GLinear::new(g, &a.fc)?, c: a.c })
    }

    fn fwd(&self, g: &Gpu, x: &Buf, t: usize, s: &Buf, act: Act) -> Result<Buf> {
        let gb = self.fc.fwd(g, s, 1)?;
        let mut mean = g.alloc(self.c)?;
        let mut rstd = g.alloc(self.c)?;
        let (ti, eps) = (t as i32, 1e-5f32);
        let cfg = LaunchConfig { grid_dim: (self.c as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        launch!(g, chan_stats, cfg, x, &ti, &eps, &mut mean, &mut rstd)?;
        let mut y = g.alloc(self.c * t)?;
        let ci = self.c as i32;
        let null = 0u64;
        match act {
            Act::Leaky(sl) => launch!(g, adain_apply, cfg1(self.c * t), x, &mut y, &mean, &rstd, &self.nw, &self.nb, &gb, &null, &ci, &ti, &1i32, &sl)?,
            Act::Snake(al) => launch!(g, adain_apply, cfg1(self.c * t), x, &mut y, &mean, &rstd, &self.nw, &self.nb, &gb, al, &ci, &ti, &2i32, &0.0f32)?,
        }
        Ok(y)
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

    fn fwd(&self, g: &Gpu, x: &Buf, t: usize, s: &Buf) -> Result<(Buf, usize)> {
        let r = self.norm1.fwd(g, x, t, s, Act::Leaky(0.2))?;
        let (r, tr) = match &self.pool {
            Some((w, b)) => {
                let mut y = g.alloc(self.dim_in * 2 * t)?;
                let (ci, ti) = (self.dim_in as i32, t as i32);
                launch!(g, dw_convT_k3s2, cfg1(self.dim_in * 2 * t), &r, w, b, &mut y, &ci, &ti)?;
                (y, 2 * t)
            }
            None => (r, t),
        };
        let (r, _) = self.conv1.fwd(g, &r, tr)?;
        let r = self.norm2.fwd(g, &r, tr, s, Act::Leaky(0.2))?;
        let (mut r, _) = self.conv2.fwd(g, &r, tr)?;
        let up;
        let sc_in = if self.upsample {
            let mut y = g.alloc(self.dim_in * 2 * t)?;
            let (ci, ti) = (self.dim_in as i32, t as i32);
            launch!(g, upsample_nearest2, cfg1(self.dim_in * 2 * t), x, &mut y, &ci, &ti)?;
            up = y;
            &up
        } else {
            x
        };
        let sc_conv;
        let sc = match &self.conv1x1 {
            Some(c) => {
                sc_conv = c.fwd(g, sc_in, tr)?.0;
                &sc_conv
            }
            None => sc_in,
        };
        let n = r.len() as i64;
        let s2 = 1.0f32 / 2.0f32.sqrt();
        launch!(g, residual_scale, cfg1(r.len()), &mut r, sc, &n, &s2)?;
        Ok((r, tr))
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

    fn fwd(&self, g: &Gpu, x: &Buf, t: usize, s: &Buf) -> Result<Buf> {
        let mut x = g.stream.clone_dtod(x)?;
        for i in 0..3 {
            let xt = self.adain1[i].fwd(g, &x, t, s, Act::Snake(&self.alpha1[i]))?;
            let (xt, _) = self.convs1[i].fwd(g, &xt, t)?;
            let xt = self.adain2[i].fwd(g, &xt, t, s, Act::Snake(&self.alpha2[i]))?;
            let (xt, _) = self.convs2[i].fwd(g, &xt, t)?;
            g.add(&mut x, &xt)?;
        }
        Ok(x)
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

    /// x [t, din] -> [t, 2h]
    fn fwd(&self, g: &Gpu, x: &Buf, t: usize) -> Result<Buf> {
        g.prof_sync();
        let _p = crate::prof::scope("gpu/lstm");
        let gf = self.wih[0].fwd(g, x, t)?;
        let gbk = self.wih[1].fwd(g, x, t)?;
        if std::env::var("KOKORO_LSTM_PERSISTENT").map(|v| v != "0").unwrap_or(true) {
            // LEVER PL-002 (kill switch KOKORO_LSTM_PERSISTENT=0): whole sequence, one launch.
            let mut hbuf = g.stream.alloc_zeros::<f32>(4 * self.h)?;
            let mut c = g.stream.alloc_zeros::<f32>(2 * self.h)?;
            let mut out = g.alloc(t * 2 * self.h)?;
            let cfg = LaunchConfig { grid_dim: (self.h as u32, 2, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            let (ti, hi) = (t as i32, self.h as i32);
            let mut b = g.stream.launch_builder(&g.k.lstm_seq);
            b.arg(&gf).arg(&gbk).arg(&self.whh[0]).arg(&self.whh[1]).arg(&self.bhh[0]).arg(&self.bhh[1]);
            b.arg(&mut hbuf).arg(&mut c).arg(&mut out).arg(&ti).arg(&hi);
            // SAFETY: arguments match lstm_seq; grid (H,2)x128 must be co-resident (cooperative
            // launch fails loudly otherwise); buffers sized 4H / 2H / T*2H as indexed.
            unsafe { b.launch_cooperative(cfg) }.context("lstm_seq cooperative launch")?;
            g.prof_sync();
            return Ok(out);
        }
        let mut ha = g.stream.alloc_zeros::<f32>(2 * self.h)?;
        let mut hb = g.stream.alloc_zeros::<f32>(2 * self.h)?;
        let mut c = g.stream.alloc_zeros::<f32>(2 * self.h)?;
        let mut out = g.alloc(t * 2 * self.h)?;
        let cfg = LaunchConfig { grid_dim: (self.h as u32, 2, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (ti, hi) = (t as i32, self.h as i32);
        for step in 0..t {
            let si = step as i32;
            if step % 2 == 0 {
                launch!(g, lstm_step, cfg, &gf, &gbk, &self.whh[0], &self.whh[1], &self.bhh[0], &self.bhh[1], &ha, &mut hb, &mut c, &mut out, &ti, &hi, &si)?;
            } else {
                launch!(g, lstm_step, cfg, &gf, &gbk, &self.whh[0], &self.whh[1], &self.bhh[0], &self.bhh[1], &hb, &mut ha, &mut c, &mut out, &ti, &hi, &si)?;
            }
        }
        g.prof_sync();
        Ok(out)
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

    fn layer(&self, g: &Gpu, h: &Buf, t: usize) -> Result<Buf> {
        use albert::{HEADS, HEAD_DIM, HID};
        let q = self.q.fwd(g, h, t)?;
        let k = self.k.fwd(g, h, t)?;
        let v = self.v.fwd(g, h, t)?;
        let mut scores = g.alloc(HEADS * t * t)?;
        // row-major S_h[i,j] = sum_d Q[i,hd] K[j,hd]  ==  col-major M = Kc^T Qc per head
        g.gemm_batched(true, false, t, t, HEAD_DIM, &k, 0, HID, HEAD_DIM, &q, 0, HID, HEAD_DIM, &mut scores, 0, t, t * t, HEADS)?;
        let cfg = LaunchConfig { grid_dim: ((HEADS * t) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (ni, scale) = (t as i32, (HEAD_DIM as f32).powf(-0.5));
        launch!(g, softmax_rows, cfg, &mut scores, &ni, &scale)?;
        let mut ctx = g.alloc(t * HID)?;
        // ctx_c[64, T] (ld 768) = Vc[64, T] * P_c   per head
        g.gemm_batched(false, false, HEAD_DIM, t, t, &v, 0, HID, HEAD_DIM, &scores, 0, t, t * t, &mut ctx, 0, HID, HEAD_DIM, HEADS)?;
        let mut a = self.dense.fwd(g, &ctx, t)?;
        g.add(&mut a, h)?;
        Self::ln(g, &mut a, t, HID, &self.attn_ln.0, &self.attn_ln.1, 1e-12)?;
        let mut f = self.ffn.fwd(g, &a, t)?;
        let n = f.len() as i64;
        launch!(g, gelu_new, cfg1(f.len()), &mut f, &n)?;
        let mut o = self.ffn_out.fwd(g, &f, t)?;
        g.add(&mut o, &a)?;
        Self::ln(g, &mut o, t, HID, &self.full_ln.0, &self.full_ln.1, 1e-12)?;
        Ok(o)
    }

    /// emb [t, 128] (host-computed embeddings, see Albert::embeddings) -> [t, 768]
    fn fwd(&self, g: &Gpu, emb: &Buf, t: usize) -> Result<Buf> {
        let mut h = self.map_in.fwd(g, emb, t)?;
        for _ in 0..albert::LAYERS {
            h = self.layer(g, &h, t)?;
        }
        Ok(h)
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
    pub fn new(m: &Kokoro, ordinal: usize) -> Result<Self> {
        let g = Gpu::new(ordinal)?;
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
        s.gpu.stream.synchronize()?;
        Ok(s)
    }

    fn cat_style(&self, x: &Buf, t: usize, d: usize, s: &Buf) -> Result<Buf> {
        let g = &self.gpu;
        let mut y = g.alloc(t * (d + STYLE_DIM))?;
        let (ti, di, si) = (t as i32, d as i32, STYLE_DIM as i32);
        launch!(g, cat_style_rows, cfg1(t * (d + STYLE_DIM)), x, s, &mut y, &ti, &di, &si)?;
        Ok(y)
    }

    fn duration_encoder(&self, d_en: &Buf, t: usize, s: &Buf) -> Result<Buf> {
        let g = &self.gpu;
        let mut x = self.cat_style(d_en, t, HIDDEN, s)?;
        for (lstm, fc) in self.dur_lstms.iter().zip(&self.dur_norms) {
            let mut h = lstm.fwd(g, &x, t)?;
            let gb = fc.fwd(g, s, 1)?;
            let cfg = LaunchConfig { grid_dim: (t as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let (ri, di, eps, null) = (t as i32, HIDDEN as i32, 1e-5f32, 0u64);
            launch!(g, layer_norm_rows, cfg, &mut h, &null, &null, &gb, &ri, &di, &eps)?;
            x = self.cat_style(&h, t, HIDDEN, s)?;
        }
        Ok(x)
    }

    fn f0n(&self, en: &Buf, nf: usize, s: &Buf) -> Result<(Buf, Buf)> {
        let g = &self.gpu;
        let x = self.shared.fwd(g, en, nf)?;
        let x = g.transpose(&x, nf, HIDDEN)?;
        let run = |blocks: &[GResBlk], proj: &GConv| -> Result<Buf> {
            let (mut h, mut t) = blocks[0].fwd(g, &x, nf, s)?;
            for b in &blocks[1..] {
                let (y, t2) = b.fwd(g, &h, t, s)?;
                h = y;
                t = t2;
            }
            Ok(proj.fwd(g, &h, t)?.0)
        };
        Ok((run(&self.f0, &self.f0_proj)?, run(&self.n, &self.n_proj)?))
    }

    fn text_encoder(&self, m: &Kokoro, ids: &[i64]) -> Result<Buf> {
        let g = &self.gpu;
        let t = ids.len();
        let te = &m.text_encoder;
        let mut e = vec![0.0f32; t * HIDDEN];
        for (i, &id) in ids.iter().enumerate() {
            ensure!(id >= 0 && (id as usize) < te.n_token, "token id {id} out of range");
            e[i * HIDDEN..(i + 1) * HIDDEN].copy_from_slice(&te.embedding[id as usize * HIDDEN..(id as usize + 1) * HIDDEN]);
        }
        let e = g.up(&e)?;
        let mut x = g.transpose(&e, t, HIDDEN)?;
        for (conv, ga, be) in &self.te_cnn {
            let (y, _) = conv.fwd(g, &x, t)?;
            let mut yt = g.transpose(&y, HIDDEN, t)?;
            GAlbert::ln(g, &mut yt, t, HIDDEN, ga, be, 1e-5)?;
            g.leaky(&mut yt, 0.2)?;
            x = g.transpose(&yt, t, HIDDEN)?;
        }
        let xt = g.transpose(&x, HIDDEN, t)?;
        let h = self.te_lstm.fwd(g, &xt, t)?;
        g.transpose(&h, t, HIDDEN)
    }

    fn pre_generator(&self, asr: &Buf, nf: usize, f0c: &Buf, nc: &Buf, s: &Buf) -> Result<(Buf, usize)> {
        let g = &self.gpu;
        let (f0, _) = self.f0_conv.fwd(g, f0c, 2 * nf)?;
        let (n, _) = self.n_conv.fwd(g, nc, 2 * nf)?;
        let x = g.cat_channels(&[asr, &f0, &n])?;
        let (mut x, _) = self.encode.fwd(g, &x, nf, s)?;
        let (asr_res, _) = self.asr_res.fwd(g, asr, nf)?;
        let mut t = nf;
        let mut res = true;
        for block in &self.decode {
            if res {
                x = g.cat_channels(&[&x, &asr_res, &f0, &n])?;
            }
            let (y, t2) = block.fwd(g, &x, t, s)?;
            x = y;
            t = t2;
            if block.upsample {
                res = false;
            }
        }
        Ok((x, t))
    }

    fn har(&self, f0c: &Buf, len2n: usize, noise: &mut dyn NoiseSource) -> Result<(Buf, usize)> {
        let g = &self.gpu;
        let s_len = len2n * UPSAMPLE_SCALE;
        let host_noise = std::env::var("KOKORO_GPU_HOST_NOISE").map(|v| v == "1").unwrap_or(false);
        let (ri, nz) = match noise.counter_seed() {
            Some(seed) if !host_noise => {
                let mut nz = g.alloc(s_len * HARMONICS)?;
                let n = (s_len * HARMONICS) as i64;
                launch!(g, gen_noise, cfg1(s_len * HARMONICS / 2 + 1), &seed, &mut nz, &n)?;
                (g.up(&vocoder::RngNoise::new(seed).rand_ini())?, nz)
            }
            _ => {
                let (rand_ini, sine_noise) = noise.draw(s_len)?;
                ensure!(sine_noise.len() == s_len * HARMONICS, "noise size");
                (g.up(&rand_ini)?, g.up(&sine_noise)?)
            }
        };
        let d = (s_len as f64 * (1.0 / UPSAMPLE_SCALE as f64)).floor() as usize;
        let mut pp = g.alloc(HARMONICS * d)?;
        let near = (1.0 / UPSAMPLE_SCALE as f64) as f32;
        let down = (1.0 / (1.0 / UPSAMPLE_SCALE as f64)) as f32;
        let up = (1.0 / UPSAMPLE_SCALE as f64) as f32;
        let (li, si, di) = (len2n as i32, s_len as i32, d as i32);
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        launch!(g, sine_phase_pre, one, f0c, &li, &si, &di, &near, &down, &ri, &mut pp)?;
        let mut har = g.alloc(s_len)?;
        launch!(g, sine_har_source, cfg1(s_len), f0c, &li, &si, &di, &near, &up, &pp, &nz, &self.l_w, &self.l_b, &mut har)?;
        let frames = 1 + s_len / vocoder::HOP;
        let mut spec = g.alloc(22 * frames)?;
        let fi = frames as i32;
        launch!(g, stft20, cfg1(frames), &har, &si, &mut spec, &fi, &g.tw, &g.win)?;
        Ok((spec, frames))
    }

    fn generator(&self, x: &Buf, t: usize, s: &Buf, har: &Buf, frames: usize) -> Result<Buf> {
        let g = &self.gpu;
        let mut x = g.stream.clone_dtod(x)?;
        let mut t = t;
        for i in 0..2 {
            let _p = crate::prof::scope(if i == 0 { "gpu.gen.stage0" } else { "gpu.gen.stage1" });
            g.leaky(&mut x, 0.1)?;
            let (xs, ts) = self.noise_convs[i].fwd(g, har, frames)?;
            let xs = self.noise_res[i].fwd(g, &xs, ts, s)?;
            let (mut y, mut ty) = self.ups[i].fwd(g, &x, t)?;
            if i == 1 {
                let c = self.ups[i].cout;
                let mut p = g.alloc(c * (ty + 1))?;
                let (ci, ti) = (c as i32, ty as i32);
                launch!(g, reflect_pad_left1, cfg1(c * (ty + 1)), &y, &mut p, &ci, &ti)?;
                y = p;
                ty += 1;
            }
            ensure!(ty == ts, "generator stage {i}: {ty} != {ts}");
            g.add(&mut y, &xs)?;
            let mut acc = self.resblocks[i * 3].fwd(g, &y, ty, s)?;
            for j in 1..3 {
                let r = self.resblocks[i * 3 + j].fwd(g, &y, ty, s)?;
                g.add(&mut acc, &r)?;
            }
            let (n, three) = (acc.len() as i64, 3.0f32);
            launch!(g, div_inplace, cfg1(acc.len()), &mut acc, &three, &n)?;
            x = acc;
            t = ty;
            g.prof_sync();
        }
        let _p = crate::prof::scope("gpu.gen.post+istft");
        g.leaky(&mut x, 0.01)?;
        let (post, tp) = self.conv_post.fwd(g, &x, t)?;
        // SAFETY: fully written by istft_frames before istft_ola reads it.
        let mut fr = unsafe { g.stream.alloc::<f64>(tp * 20) }?;
        let fi = tp as i32;
        launch!(g, istft_frames, cfg1(tp), &post, &fi, &mut fr, &g.tw, &g.win)?;
        let len = vocoder::HOP * (tp - 1);
        let mut audio = g.alloc(len)?;
        let li = len as i32;
        launch!(g, istft_ola, cfg1(len), &fr, &fi, &mut audio, &li, &g.win)?;
        Ok(audio)
    }

    /// Full forward. `m` supplies host-side embeddings and the vocabulary (same weights).
    pub fn forward_ids(&self, m: &Kokoro, ids: &[i64], ref_s: &[f32], speed: f32, noise: &mut dyn NoiseSource) -> Result<Output> {
        let g = &self.gpu;
        ensure!(ref_s.len() == 2 * STYLE_DIM, "ref_s must have 256 values");
        ensure!(ids.len() >= 2 && ids.len() <= model::CONTEXT_LEN, "input_ids length {} outside [2, {}]", ids.len(), model::CONTEXT_LEN);
        ensure!(speed.is_finite() && speed > 0.0, "speed must be positive and finite");
        let t = ids.len();
        let s_dec = g.up(&ref_s[..STYLE_DIM])?;
        let s = g.up(&ref_s[STYLE_DIM..])?;
        let bert = {
            let _p = crate::prof::scope("gpu.albert");
            let emb = g.up(&m.albert.embeddings(ids)?)?;
            let b = self.albert.fwd(g, &emb, t)?;
            g.prof_sync();
            b
        };
        let (d, logits) = {
            let _p = crate::prof::scope("gpu.duration");
            let d_en = self.bert_encoder.fwd(g, &bert, t)?;
            let d = self.duration_encoder(&d_en, t, &s)?;
            let x = self.pred_lstm.fwd(g, &d, t)?;
            let logits = g.down(&self.dur_proj.fwd(g, &x, t)?)?;
            (d, logits)
        };
        let _p = crate::prof::scope("gpu.f0n+text");
        let pred_dur = model::durations_from_logits(&logits, t, speed);
        let aln: Vec<i32> = model::alignment(&pred_dur).into_iter().map(|v| v as i32).collect();
        let nf = aln.len();
        let aln_d = g.stream.clone_htod(&aln)?;
        let dd = HIDDEN + STYLE_DIM;
        let mut en = g.alloc(nf * dd)?;
        let (nfi, ddi) = (nf as i32, dd as i32);
        launch!(g, expand_rows, cfg1(nf * dd), &d, &aln_d, &mut en, &nfi, &ddi)?;
        let (f0, n) = self.f0n(&en, nf, &s)?;
        let t_en = self.text_encoder(m, ids)?;
        let mut asr = g.alloc(HIDDEN * nf)?;
        let (ci, ti) = (HIDDEN as i32, t as i32);
        launch!(g, expand_cols, cfg1(HIDDEN * nf), &t_en, &aln_d, &mut asr, &ci, &ti, &nfi)?;
        g.prof_sync();
        drop(_p);
        if let Ok(dir) = std::env::var("KOKORO_DEBUG_F0_DIR") {
            // diagnostics only: dump the F0 curve keyed by the item's ids
            let key = crate::engine::sha256_bytes(&ids.iter().flat_map(|v| v.to_le_bytes()).chain(ref_s.iter().flat_map(|v| v.to_le_bytes())).chain(speed.to_le_bytes()).collect::<Vec<u8>>());
            let v: Vec<u8> = g.down(&f0)?.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(&dir).join(format!("single-{}.f32", &key[..16])), v)?;
        }
        let (x, t2) = {
            let _p = crate::prof::scope("gpu.decoder.pre");
            let r = self.pre_generator(&asr, nf, &f0, &n, &s_dec)?;
            g.prof_sync();
            r
        };
        let (har, frames) = {
            let _p = crate::prof::scope("gpu.gen.source");
            let r = self.har(&f0, 2 * nf, noise)?;
            g.prof_sync();
            r
        };
        let audio = self.generator(&x, t2, &s_dec, &har, frames)?;
        let _p = crate::prof::scope("gpu.download");
        let audio = g.down(&audio)?;
        Ok(Output { audio, pred_dur })
    }

    // ---- stage entry points for oracle seam tests (host in, host out)

    pub fn seam_gen_noise(&self, seed: u64, n: usize) -> Result<Vec<f32>> {
        let g = &self.gpu;
        let mut nz = g.alloc(n)?;
        let ni = n as i64;
        launch!(g, gen_noise, cfg1(n / 2 + 1), &seed, &mut nz, &ni)?;
        g.down(&nz)
    }

    pub fn seam_bert(&self, m: &Kokoro, ids: &[i64]) -> Result<Vec<f32>> {
        let g = &self.gpu;
        let emb = g.up(&m.albert.embeddings(ids)?)?;
        g.down(&self.albert.fwd(g, &emb, ids.len())?)
    }

    pub fn seam_dur_enc(&self, d_en: &[f32], t: usize, s: &[f32]) -> Result<Vec<f32>> {
        let g = &self.gpu;
        g.down(&self.duration_encoder(&g.up(d_en)?, t, &g.up(s)?)?)
    }

    pub fn seam_pred_lstm(&self, d: &[f32], t: usize) -> Result<Vec<f32>> {
        let g = &self.gpu;
        g.down(&self.pred_lstm.fwd(g, &g.up(d)?, t)?)
    }

    pub fn seam_f0n(&self, en: &[f32], nf: usize, s: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let g = &self.gpu;
        let (f, n) = self.f0n(&g.up(en)?, nf, &g.up(s)?)?;
        Ok((g.down(&f)?, g.down(&n)?))
    }

    pub fn seam_text_encoder(&self, m: &Kokoro, ids: &[i64]) -> Result<Vec<f32>> {
        self.gpu.down(&self.text_encoder(m, ids)?)
    }

    pub fn seam_pre_generator(&self, asr: &[f32], nf: usize, f0: &[f32], n: &[f32], s: &[f32]) -> Result<Vec<f32>> {
        let g = &self.gpu;
        let (x, _) = self.pre_generator(&g.up(asr)?, nf, &g.up(f0)?, &g.up(n)?, &g.up(s)?)?;
        g.down(&x)
    }

    /// (har_source-derived spectrum [22, F], frames) from the F0 curve and noise
    pub fn seam_har(&self, f0: &[f32], noise: &mut dyn NoiseSource) -> Result<(Vec<f32>, usize)> {
        let g = &self.gpu;
        let (h, f) = self.har(&g.up(f0)?, f0.len(), noise)?;
        Ok((g.down(&h)?, f))
    }

    pub fn seam_generator(&self, x: &[f32], t: usize, s: &[f32], har: &[f32], frames: usize) -> Result<Vec<f32>> {
        let g = &self.gpu;
        g.down(&self.generator(&g.up(x)?, t, &g.up(s)?, &g.up(har)?, frames)?)
    }
}

#[path = "gpu_batch.rs"]
mod batch;
pub use batch::{BatchItem, ItemNoise};
