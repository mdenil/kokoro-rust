//! Layer weight structs hydrated once from `Weights` (the GPU engine uploads from these).
//! Channel tensors are [C, T].

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
}

/// StyleTTS2 AdainResBlk1d (predictor F0/N stacks and decoder encode/decode).
pub struct AdainResBlk1d {
    pub(crate) conv1: Conv1d,
    pub(crate) conv2: Conv1d,
    pub(crate) norm1: AdaIn1d,
    pub(crate) norm2: AdaIn1d,
    pub(crate) conv1x1: Option<Conv1d>,
    pub(crate) pool: Option<(Vec<f32>, Vec<f32>)>,
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
}

/// iSTFTNet AdaINResBlock1 with Snake activations.
pub struct AdaInResBlock1 {
    pub(crate) convs1: Vec<Conv1d>,
    pub(crate) convs2: Vec<Conv1d>,
    pub(crate) adain1: Vec<AdaIn1d>,
    pub(crate) adain2: Vec<AdaIn1d>,
    pub(crate) alpha1: Vec<Vec<f32>>,
    pub(crate) alpha2: Vec<Vec<f32>>,
}

impl AdaInResBlock1 {
    pub fn load(w: &Weights, prefix: &str, c: usize, k: usize, dil: [usize; 3], style_dim: usize) -> Result<Self> {
        let mut s = Self { convs1: vec![], convs2: vec![], adain1: vec![], adain2: vec![], alpha1: vec![], alpha2: vec![] };
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
}

/// Single-layer bidirectional LSTM (torch gate order i, f, g, o). x [t, din] -> [t, 2h].
pub struct BiLstm {
    pub(crate) wih: [Vec<f32>; 2],
    pub(crate) whh: [Vec<f32>; 2],
    pub(crate) bih: [Vec<f32>; 2],
    pub(crate) bhh: [Vec<f32>; 2],
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
}
