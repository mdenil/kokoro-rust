//! iSTFTNet decoder + generator weights (the stages run on the GPU), their constants, and the
//! counter-based excitation noise seeding (the GPU generates the same stream in place).

use crate::nn::{AdaInResBlock1, AdainResBlk1d, Conv1d, ConvTranspose1d};
use crate::weights::Weights;
use anyhow::Result;

pub const N_FFT: usize = 20;
pub const HOP: usize = 5;
pub const HARMONICS: usize = 9;
pub const UPSAMPLE_SCALE: usize = 300;

/// Counter-based native noise: value i is a pure function of (seed, i), so a GPU backend can
/// generate the identical integer stream in parallel (kernels/kokoro.cu `gen_noise`).
/// Uniforms: splitmix64 outputs; the GPU's normals: f32 Box-Muller. Distributionally equivalent
/// to torch's draws, not bit-identical (the reference itself is unseeded in production).
pub struct RngNoise {
    pub seed: u64,
}

/// splitmix64 output number `ctr` of the stream starting at `seed`.
pub fn splitmix_at(seed: u64, ctr: u64) -> u64 {
    let mut x = seed.wrapping_add(ctr.wrapping_add(1).wrapping_mul(0x9E3779B97F4A7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

impl RngNoise {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn rand_ini(&self) -> [f32; HARMONICS] {
        let mut ri = [0.0f32; HARMONICS];
        for (h, v) in ri.iter_mut().enumerate() {
            *v = (splitmix_at(self.seed, h as u64) >> 40) as f32 * (1.0 / 16_777_216.0);
        }
        ri
    }
}

pub struct Generator {
    pub(crate) l_w: Vec<f32>,
    pub(crate) l_b: f32,
    pub(crate) noise_convs: Vec<Conv1d>,
    pub(crate) noise_res: Vec<AdaInResBlock1>,
    pub(crate) ups: Vec<ConvTranspose1d>,
    pub(crate) resblocks: Vec<AdaInResBlock1>,
    pub(crate) conv_post: Conv1d,
}

impl Generator {
    fn load(w: &Weights, style_dim: usize) -> Result<Self> {
        let p = "decoder.generator";
        let lb = w.get(&format!("{p}.m_source.l_linear.bias"), &[1])?;
        let kernels = [3usize, 7, 11];
        let mut resblocks = vec![];
        for i in 0..2 {
            let ch = 512 >> (i + 1);
            for (j, &k) in kernels.iter().enumerate() {
                resblocks.push(AdaInResBlock1::load(w, &format!("{p}.resblocks.{}", i * 3 + j), ch, k, [1, 3, 5], style_dim)?);
            }
        }
        Ok(Self {
            l_w: w.get(&format!("{p}.m_source.l_linear.weight"), &[1, HARMONICS])?,
            l_b: lb[0],
            noise_convs: vec![
                Conv1d::load(w, &format!("{p}.noise_convs.0"), N_FFT + 2, 256, 12, 6, 3, 1, true)?,
                Conv1d::load(w, &format!("{p}.noise_convs.1"), N_FFT + 2, 128, 1, 1, 0, 1, true)?,
            ],
            noise_res: vec![
                AdaInResBlock1::load(w, &format!("{p}.noise_res.0"), 256, 7, [1, 3, 5], style_dim)?,
                AdaInResBlock1::load(w, &format!("{p}.noise_res.1"), 128, 11, [1, 3, 5], style_dim)?,
            ],
            ups: vec![
                ConvTranspose1d::load(w, &format!("{p}.ups.0"), 512, 256, 20, 10, 5)?,
                ConvTranspose1d::load(w, &format!("{p}.ups.1"), 256, 128, 12, 6, 3)?,
            ],
            resblocks,
            conv_post: Conv1d::load(w, &format!("{p}.conv_post"), 128, N_FFT + 2, 7, 1, 3, 1, true)?,
        })
    }
}

pub struct Decoder {
    pub(crate) encode: AdainResBlk1d,
    pub(crate) decode: Vec<AdainResBlk1d>,
    pub(crate) f0_conv: Conv1d,
    pub(crate) n_conv: Conv1d,
    pub(crate) asr_res: Conv1d,
    pub generator: Generator,
}

impl Decoder {
    pub fn load(w: &Weights) -> Result<Self> {
        let sd = crate::model::STYLE_DIM;
        let mut decode = vec![];
        for i in 0..3 {
            decode.push(AdainResBlk1d::load(w, &format!("decoder.decode.{i}"), 1090, 1024, sd, false)?);
        }
        decode.push(AdainResBlk1d::load(w, "decoder.decode.3", 1090, 512, sd, true)?);
        Ok(Self {
            encode: AdainResBlk1d::load(w, "decoder.encode", 514, 1024, sd, false)?,
            decode,
            f0_conv: Conv1d::load(w, "decoder.F0_conv", 1, 1, 3, 2, 1, 1, true)?,
            n_conv: Conv1d::load(w, "decoder.N_conv", 1, 1, 3, 2, 1, 1, true)?,
            asr_res: Conv1d::load(w, "decoder.asr_res.0", 512, 64, 1, 1, 0, 1, true)?,
            generator: Generator::load(w, sd)?,
        })
    }
}
