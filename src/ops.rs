//! CPU f32 kernels. Layouts: sequence tensors are [T, D] row-major; channel tensors are
//! [C, T] row-major (batch dimension of 1 is implicit everywhere).
//!
//! All matrix products go through `gemm`, the single unsafe facade over matrixmultiply.

/// C[m,n] = alpha * A[m,k] B[k,n] + beta * C, with arbitrary element strides.
/// Every (row, col) index touched is bounds-checked against the slice lengths first.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    a_off: usize,
    rsa: usize,
    csa: usize,
    b: &[f32],
    b_off: usize,
    rsb: usize,
    csb: usize,
    beta: f32,
    c: &mut [f32],
    c_off: usize,
    rsc: usize,
    csc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for i in 0..m {
            for j in 0..n {
                let idx = c_off + i * rsc + j * csc;
                c[idx] *= beta;
            }
        }
        return;
    }
    let last = |off: usize, r: usize, rs: usize, cc: usize, cs: usize| off + (r - 1) * rs + (cc - 1) * cs;
    assert!(last(a_off, m, rsa, k, csa) < a.len(), "gemm: A out of bounds");
    assert!(last(b_off, k, rsb, n, csb) < b.len(), "gemm: B out of bounds");
    assert!(last(c_off, m, rsc, n, csc) < c.len(), "gemm: C out of bounds");
    // SAFETY: the maximal element offset of each operand was asserted in-bounds above and
    // all strides are non-negative, so every access matrixmultiply performs is in-bounds.
    // `c` is uniquely borrowed; A/B are shared borrows of distinct slices.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr().add(a_off),
            rsa as isize,
            csa as isize,
            b.as_ptr().add(b_off),
            rsb as isize,
            csb as isize,
            beta,
            c.as_mut_ptr().add(c_off),
            rsc as isize,
            csc as isize,
        );
    }
}

/// y[T, out] = x[T, in] W[out, in]^T + b
pub fn linear(x: &[f32], t: usize, w: &[f32], b: Option<&[f32]>, din: usize, dout: usize) -> Vec<f32> {
    assert_eq!(x.len(), t * din);
    assert_eq!(w.len(), dout * din);
    let mut y = vec![0.0f32; t * dout];
    if let Some(b) = b {
        for row in y.chunks_exact_mut(dout) {
            row.copy_from_slice(b);
        }
    }
    gemm(t, din, dout, x, 0, din, 1, w, 0, 1, din, if b.is_some() { 1.0 } else { 0.0 }, &mut y, 0, dout, 1);
    y
}

/// Single-row linear (style projections): y[out] = W[out,in] s[in] + b
pub fn linear_vec(s: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    linear(s, 1, w, Some(b), s.len(), b.len())
}

pub fn conv1d_out_len(t: usize, k: usize, stride: usize, pad: usize, dil: usize) -> usize {
    (t + 2 * pad - dil * (k - 1) - 1) / stride + 1
}

/// Conv1d, groups=1. x [cin, t], w [cout, cin, k] -> [cout, tout]. Implemented as K strided
/// GEMMs over the valid output range of each tap (zero padding = skipped contributions).
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    x: &[f32],
    cin: usize,
    t: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dil: usize,
) -> (Vec<f32>, usize) {
    assert_eq!(x.len(), cin * t);
    assert_eq!(w.len(), cout * cin * k);
    let tout = conv1d_out_len(t, k, stride, pad, dil);
    let mut y = vec![0.0f32; cout * tout];
    if let Some(b) = bias {
        for (co, row) in y.chunks_exact_mut(tout).enumerate() {
            row.fill(b[co]);
        }
    }
    for kk in 0..k {
        let shift = (kk * dil) as isize - pad as isize; // input index = o*stride + shift
        let o_lo = if shift >= 0 { 0 } else { ((-shift) as usize).div_ceil(stride) };
        let max_in = t as isize - 1 - shift;
        if max_in < 0 {
            continue;
        }
        let o_hi = ((max_in as usize) / stride).min(tout - 1);
        if o_lo > o_hi {
            continue;
        }
        let n = o_hi - o_lo + 1;
        let b_off = (o_lo as isize * stride as isize + shift) as usize;
        gemm(cout, cin, n, w, kk, cin * k, k, x, b_off, t, stride, 1.0, &mut y, o_lo, tout, 1);
    }
    (y, tout)
}

/// ConvTranspose1d, groups=1. x [cin, tin], w [cin, cout, k] -> [cout, tout].
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    x: &[f32],
    cin: usize,
    tin: usize,
    w: &[f32],
    bias: &[f32],
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    out_pad: usize,
) -> (Vec<f32>, usize) {
    assert_eq!(x.len(), cin * tin);
    assert_eq!(w.len(), cin * cout * k);
    let tout = (tin - 1) * stride + k + out_pad - 2 * pad;
    let mut y = vec![0.0f32; cout * tout];
    for (co, row) in y.chunks_exact_mut(tout).enumerate() {
        row.fill(bias[co]);
    }
    for kk in 0..k {
        let shift = kk as isize - pad as isize; // out index = i*stride + shift
        let i_lo = if shift >= 0 { 0 } else { ((-shift) as usize).div_ceil(stride) };
        let max_o = tout as isize - 1 - shift;
        if max_o < 0 {
            continue;
        }
        let i_hi = ((max_o as usize) / stride).min(tin - 1);
        if i_lo > i_hi {
            continue;
        }
        let n = i_hi - i_lo + 1;
        let c_off = (i_lo as isize * stride as isize + shift) as usize;
        gemm(cout, cin, n, w, kk, k, cout * k, x, i_lo, tin, 1, 1.0, &mut y, c_off, tout, stride);
    }
    (y, tout)
}

/// Depthwise ConvTranspose1d (groups = channels), w [c, 1, k].
pub fn conv_transpose1d_depthwise(
    x: &[f32],
    c: usize,
    tin: usize,
    w: &[f32],
    bias: &[f32],
    k: usize,
    stride: usize,
    pad: usize,
    out_pad: usize,
) -> (Vec<f32>, usize) {
    let tout = (tin - 1) * stride + k + out_pad - 2 * pad;
    let mut y = vec![0.0f32; c * tout];
    for ch in 0..c {
        let yr = &mut y[ch * tout..(ch + 1) * tout];
        yr.fill(bias[ch]);
        let xr = &x[ch * tin..(ch + 1) * tin];
        for (i, &xv) in xr.iter().enumerate() {
            for kk in 0..k {
                let o = (i * stride + kk) as isize - pad as isize;
                if o >= 0 && (o as usize) < tout {
                    yr[o as usize] += w[ch * k + kk] * xv;
                }
            }
        }
    }
    (y, tout)
}

/// torch._weight_norm(v, g, dim=0): w = v * (g / ||v||) with the norm over all dims but 0.
pub fn weight_norm(v: &[f32], g: &[f32]) -> Vec<f32> {
    let rows = g.len();
    assert_eq!(v.len() % rows, 0);
    let per = v.len() / rows;
    let mut w = vec![0.0f32; v.len()];
    for r in 0..rows {
        let vr = &v[r * per..(r + 1) * per];
        let norm = vr.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>().sqrt() as f32;
        let scale = g[r] / norm;
        for (o, &a) in w[r * per..(r + 1) * per].iter_mut().zip(vr) {
            *o = a * scale;
        }
    }
    w
}

/// LayerNorm over the last dim of [rows, d]; optional affine.
pub fn layer_norm_rows(x: &mut [f32], d: usize, gamma: Option<&[f32]>, beta: Option<&[f32]>, eps: f32) {
    for row in x.chunks_exact_mut(d) {
        let mean = (row.iter().map(|&a| a as f64).sum::<f64>() / d as f64) as f32;
        let var = (row.iter().map(|&a| {
            let c = (a - mean) as f64;
            c * c
        })
        .sum::<f64>()
            / d as f64) as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (j, v) in row.iter_mut().enumerate() {
            let mut o = (*v - mean) * inv;
            if let Some(g) = gamma {
                o *= g[j];
            }
            if let Some(b) = beta {
                o += b[j];
            }
            *v = o;
        }
    }
}

/// Per-channel normalization over time of [c, t] (InstanceNorm1d statistics, biased var).
pub fn instance_norm(x: &mut [f32], t: usize, eps: f32) {
    for row in x.chunks_exact_mut(t) {
        let mean = (row.iter().map(|&a| a as f64).sum::<f64>() / t as f64) as f32;
        let var = (row.iter().map(|&a| {
            let c = (a - mean) as f64;
            c * c
        })
        .sum::<f64>()
            / t as f64) as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) * inv;
        }
    }
}

pub fn leaky_relu(x: &mut [f32], slope: f32) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// transformers "gelu_new": 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
pub fn gelu_new(x: &mut [f32]) {
    let c = (2.0f64 / std::f64::consts::PI).sqrt() as f32;
    for v in x.iter_mut() {
        let a = *v;
        *v = 0.5 * a * (1.0 + (c * (a + 0.044715 * a * a * a)).tanh());
    }
}

/// Transpose [r, c] -> [c, r]
pub fn transpose(x: &[f32], r: usize, c: usize) -> Vec<f32> {
    assert_eq!(x.len(), r * c);
    let mut y = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            y[j * r + i] = x[i * c + j];
        }
    }
    y
}

/// Nearest ×2 upsample along time of [c, t] (torch nearest with out == 2*in: src = dst >> 1).
pub fn upsample_nearest2(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; c * 2 * t];
    for ch in 0..c {
        for i in 0..t {
            let v = x[ch * t + i];
            y[ch * 2 * t + 2 * i] = v;
            y[ch * 2 * t + 2 * i + 1] = v;
        }
    }
    y
}
