//! Layer structs hydrated once from `Weights`. Channel tensors are [C, T].

use crate::ops;
use crate::weights::Weights;
use anyhow::Result;

pub struct Linear {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    pub din: usize,
    pub dout: usize,
}

impl Linear {
    pub fn load(w: &Weights, prefix: &str, din: usize, dout: usize, bias: bool) -> Result<Self> {
        Ok(Self {
            w: w.get(&format!("{prefix}.weight"), &[dout, din])?,
            b: if bias { Some(w.get(&format!("{prefix}.bias"), &[dout])?) } else { None },
            din,
            dout,
        })
    }

    /// x [t, din] -> [t, dout]
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        ops::linear(x, t, &self.w, self.b.as_deref(), self.din, self.dout)
    }
}

pub struct Conv1d {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub dil: usize,
}

impl Conv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        w: &Weights,
        prefix: &str,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dil: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            w: w.conv_weight(prefix, &[cout, cin, k])?,
            b: if bias { Some(w.get(&format!("{prefix}.bias"), &[cout])?) } else { None },
            cin,
            cout,
            k,
            stride,
            pad,
            dil,
        })
    }

    pub fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        ops::conv1d(x, self.cin, t, &self.w, self.b.as_deref(), self.cout, self.k, self.stride, self.pad, self.dil)
    }
}

pub struct ConvTranspose1d {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
}

impl ConvTranspose1d {
    pub fn load(w: &Weights, prefix: &str, cin: usize, cout: usize, k: usize, stride: usize, pad: usize) -> Result<Self> {
        Ok(Self {
            w: w.conv_weight(prefix, &[cin, cout, k])?,
            b: w.get(&format!("{prefix}.bias"), &[cout])?,
            cin,
            cout,
            k,
            stride,
            pad,
        })
    }

    pub fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        ops::conv_transpose1d(x, self.cin, t, &self.w, &self.b, self.cout, self.k, self.stride, self.pad, 0)
    }
}

/// AdaIN1d: (1 + gamma(s)) * InstanceNorm1d_affine(x) + beta(s)
pub struct AdaIn1d {
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub fc: Linear,
    pub c: usize,
}

impl AdaIn1d {
    pub fn load(w: &Weights, prefix: &str, style_dim: usize, c: usize) -> Result<Self> {
        Ok(Self {
            norm_w: w.get_instance_norm_affine(&format!("{prefix}.norm.weight"), c, 1.0)?,
            norm_b: w.get_instance_norm_affine(&format!("{prefix}.norm.bias"), c, 0.0)?,
            fc: Linear::load(w, &format!("{prefix}.fc"), style_dim, 2 * c, true)?,
            c,
        })
    }

    pub fn forward(&self, x: &[f32], t: usize, s: &[f32]) -> Vec<f32> {
        let h = self.fc.forward(s, 1);
        let (gamma, beta) = h.split_at(self.c);
        let mut y = x.to_vec();
        ops::instance_norm(&mut y, t, 1e-5);
        for ch in 0..self.c {
            let (nw, nb) = (self.norm_w[ch], self.norm_b[ch]);
            let (g1, b) = (1.0 + gamma[ch], beta[ch]);
            for v in &mut y[ch * t..(ch + 1) * t] {
                *v = g1 * (*v * nw + nb) + b;
            }
        }
        y
    }
}

/// StyleTTS2 AdainResBlk1d (predictor F0/N stacks and decoder encode/decode).
pub struct AdainResBlk1d {
    conv1: Conv1d,
    conv2: Conv1d,
    norm1: AdaIn1d,
    norm2: AdaIn1d,
    conv1x1: Option<Conv1d>,
    pool: Option<(Vec<f32>, Vec<f32>)>,
    pub dim_in: usize,
    pub dim_out: usize,
    pub upsample: bool,
}

impl AdainResBlk1d {
    pub fn load(w: &Weights, prefix: &str, dim_in: usize, dim_out: usize, style_dim: usize, upsample: bool) -> Result<Self> {
        let p = |n: &str| format!("{prefix}.{n}");
        Ok(Self {
            conv1: Conv1d::load(w, &p("conv1"), dim_in, dim_out, 3, 1, 1, 1, true)?,
            conv2: Conv1d::load(w, &p("conv2"), dim_out, dim_out, 3, 1, 1, 1, true)?,
            norm1: AdaIn1d::load(w, &p("norm1"), style_dim, dim_in)?,
            norm2: AdaIn1d::load(w, &p("norm2"), style_dim, dim_out)?,
            conv1x1: if dim_in != dim_out {
                Some(Conv1d::load(w, &p("conv1x1"), dim_in, dim_out, 1, 1, 0, 1, false)?)
            } else {
                None
            },
            pool: if upsample {
                Some((w.conv_weight(&p("pool"), &[dim_in, 1, 3])?, w.get(&p("pool.bias"), &[dim_in])?))
            } else {
                None
            },
            dim_in,
            dim_out,
            upsample,
        })
    }

    /// x [dim_in, t] -> ([dim_out, t or 2t], t_out)
    pub fn forward(&self, x: &[f32], t: usize, s: &[f32]) -> (Vec<f32>, usize) {
        // residual branch
        let mut r = self.norm1.forward(x, t, s);
        ops::leaky_relu(&mut r, 0.2);
        let (r, tr) = match &self.pool {
            Some((pw, pb)) => ops::conv_transpose1d_depthwise(&r, self.dim_in, t, pw, pb, 3, 2, 1, 1),
            None => (r, t),
        };
        let (r, _) = self.conv1.forward(&r, tr);
        let mut r = self.norm2.forward(&r, tr, s);
        ops::leaky_relu(&mut r, 0.2);
        let (mut r, _) = self.conv2.forward(&r, tr);
        // shortcut branch
        let sc = if self.upsample { ops::upsample_nearest2(x, self.dim_in, t) } else { x.to_vec() };
        let sc = match &self.conv1x1 {
            Some(c) => c.forward(&sc, tr).0,
            None => sc,
        };
        let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();
        for (a, b) in r.iter_mut().zip(&sc) {
            *a = (*a + *b) * inv_sqrt2;
        }
        (r, tr)
    }
}

/// iSTFTNet AdaINResBlock1 with Snake activations.
pub struct AdaInResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIn1d>,
    adain2: Vec<AdaIn1d>,
    alpha1: Vec<Vec<f32>>,
    alpha2: Vec<Vec<f32>>,
    c: usize,
}

fn snake(x: &mut [f32], t: usize, alpha: &[f32]) {
    for (ch, row) in x.chunks_exact_mut(t).enumerate() {
        let a = alpha[ch];
        let inv = 1.0 / a;
        for v in row.iter_mut() {
            let sn = (a * *v).sin();
            *v += inv * (sn * sn);
        }
    }
}

impl AdaInResBlock1 {
    pub fn load(w: &Weights, prefix: &str, c: usize, k: usize, dil: [usize; 3], style_dim: usize) -> Result<Self> {
        let mut s = Self { convs1: vec![], convs2: vec![], adain1: vec![], adain2: vec![], alpha1: vec![], alpha2: vec![], c };
        for (i, &d) in dil.iter().enumerate() {
            s.convs1.push(Conv1d::load(w, &format!("{prefix}.convs1.{i}"), c, c, k, 1, (k * d - d) / 2, d, true)?);
            s.convs2.push(Conv1d::load(w, &format!("{prefix}.convs2.{i}"), c, c, k, 1, (k - 1) / 2, 1, true)?);
            s.adain1.push(AdaIn1d::load(w, &format!("{prefix}.adain1.{i}"), style_dim, c)?);
            s.adain2.push(AdaIn1d::load(w, &format!("{prefix}.adain2.{i}"), style_dim, c)?);
            s.alpha1.push(w.get(&format!("{prefix}.alpha1.{i}"), &[1, c, 1])?);
            s.alpha2.push(w.get(&format!("{prefix}.alpha2.{i}"), &[1, c, 1])?);
        }
        Ok(s)
    }

    pub fn forward(&self, x: &[f32], t: usize, s: &[f32]) -> Vec<f32> {
        let mut x = x.to_vec();
        for i in 0..3 {
            let mut xt = self.adain1[i].forward(&x, t, s);
            snake(&mut xt, t, &self.alpha1[i]);
            let (xt, _) = self.convs1[i].forward(&xt, t);
            let mut xt = self.adain2[i].forward(&xt, t, s);
            snake(&mut xt, t, &self.alpha2[i]);
            let (xt, _) = self.convs2[i].forward(&xt, t);
            for (a, b) in x.iter_mut().zip(&xt) {
                *a += *b;
            }
        }
        debug_assert_eq!(x.len(), self.c * t);
        x
    }
}

/// Single-layer bidirectional LSTM (torch gate order i, f, g, o). x [t, din] -> [t, 2h].
pub struct BiLstm {
    wih: [Vec<f32>; 2],
    whh: [Vec<f32>; 2],
    bih: [Vec<f32>; 2],
    bhh: [Vec<f32>; 2],
    pub din: usize,
    pub h: usize,
}

impl BiLstm {
    pub fn load(w: &Weights, prefix: &str, din: usize, h: usize) -> Result<Self> {
        let g = |n: &str, sfx: &str, shape: &[usize]| w.get(&format!("{prefix}.{n}_l0{sfx}"), shape);
        Ok(Self {
            wih: [g("weight_ih", "", &[4 * h, din])?, g("weight_ih", "_reverse", &[4 * h, din])?],
            whh: [g("weight_hh", "", &[4 * h, h])?, g("weight_hh", "_reverse", &[4 * h, h])?],
            bih: [g("bias_ih", "", &[4 * h])?, g("bias_ih", "_reverse", &[4 * h])?],
            bhh: [g("bias_hh", "", &[4 * h])?, g("bias_hh", "_reverse", &[4 * h])?],
            din,
            h,
        })
    }

    fn direction(&self, x: &[f32], t: usize, dir: usize, out: &mut [f32]) {
        let h = self.h;
        let gx = ops::linear(x, t, &self.wih[dir], Some(&self.bih[dir]), self.din, 4 * h);
        let whh = &self.whh[dir];
        let bhh = &self.bhh[dir];
        let mut hs = vec![0.0f32; h];
        let mut cs = vec![0.0f32; h];
        let mut gates = vec![0.0f32; 4 * h];
        for step in 0..t {
            let ti = if dir == 0 { step } else { t - 1 - step };
            let gxr = &gx[ti * 4 * h..(ti + 1) * 4 * h];
            for (r, g) in gates.iter_mut().enumerate() {
                let wr = &whh[r * h..(r + 1) * h];
                let dot: f32 = wr.iter().zip(&hs).map(|(a, b)| a * b).sum();
                *g = gxr[r] + (dot + bhh[r]);
            }
            for j in 0..h {
                let i = ops::sigmoid(gates[j]);
                let f = ops::sigmoid(gates[h + j]);
                let gg = gates[2 * h + j].tanh();
                let o = ops::sigmoid(gates[3 * h + j]);
                cs[j] = f * cs[j] + i * gg;
                hs[j] = o * cs[j].tanh();
            }
            let orow = &mut out[ti * 2 * h + dir * h..ti * 2 * h + (dir + 1) * h];
            orow.copy_from_slice(&hs);
        }
    }

    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        assert_eq!(x.len(), t * self.din);
        let mut out = vec![0.0f32; t * 2 * self.h];
        self.direction(x, t, 0, &mut out);
        self.direction(x, t, 1, &mut out);
        out
    }
}
