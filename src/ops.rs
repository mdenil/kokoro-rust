//! Host-side numeric helpers of the model definition (the model itself runs on the GPU).

/// Output length of a 1-D convolution.
pub fn conv1d_out_len(t: usize, k: usize, stride: usize, pad: usize, dil: usize) -> usize {
    (t + 2 * pad - dil * (k - 1) - 1) / stride + 1
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

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
