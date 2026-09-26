//! iSTFTNet decoder + generator: harmonic source (SineGen), STFT/iSTFT, upsampling stacks.
//! Interpolation/remainder/cumsum semantics follow torch CPU kernels (see OQ_INDEX OQ-9/10).

use crate::nn::{AdaInResBlock1, AdainResBlk1d, Conv1d, ConvTranspose1d};
use crate::ops;
use crate::weights::Weights;
use anyhow::{ensure, Result};

pub const N_FFT: usize = 20;
pub const HOP: usize = 5;
pub const N_BINS: usize = N_FFT / 2 + 1;
pub const HARMONICS: usize = 9;
pub const UPSAMPLE_SCALE: usize = 300;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
const SR: f32 = 24000.0;

/// Supplies SineGen's stochastic inputs. `rand_ini` is the raw torch.rand(1, 9) draw (index 0
/// is zeroed by the model), `sine_noise` is randn [samples, 9] (time-major).
pub trait NoiseSource {
    fn draw(&mut self, samples: usize) -> Result<([f32; HARMONICS], Vec<f32>)>;
}

/// Replays recorded draws (parity harness).
pub struct FixedNoise {
    pub rand_ini: [f32; HARMONICS],
    pub sine_noise: Vec<f32>,
}

impl NoiseSource for FixedNoise {
    fn draw(&mut self, samples: usize) -> Result<([f32; HARMONICS], Vec<f32>)> {
        ensure!(
            self.sine_noise.len() == samples * HARMONICS,
            "fixed noise has {} values, forward needs {}",
            self.sine_noise.len(),
            samples * HARMONICS
        );
        Ok((self.rand_ini, self.sine_noise.clone()))
    }
}

/// Native generator: xoshiro256** uniform + Box-Muller normal. Distributionally equivalent to
/// torch's draws, not bit-identical (the reference itself is unseeded in production).
pub struct RngNoise {
    pub(crate) s: [u64; 4],
}

impl RngNoise {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Self { s: [next(), next(), next(), next()] }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1) with 24 bits of mantissa.
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }
}

impl NoiseSource for RngNoise {
    fn draw(&mut self, samples: usize) -> Result<([f32; HARMONICS], Vec<f32>)> {
        let mut ri = [0.0f32; HARMONICS];
        for v in ri.iter_mut() {
            *v = self.uniform();
        }
        let n = samples * HARMONICS;
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let u1 = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let u2 = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let r = (-2.0 * (1.0 - u1).ln()).sqrt();
            let th = 2.0 * std::f64::consts::PI * u2;
            out.push((r * th.cos()) as f32);
            out.push((r * th.sin()) as f32);
        }
        out.truncate(n);
        Ok((ri, out))
    }
}

/// torch.remainder(a, b) for float: fmod then shift into the divisor's sign.
fn torch_remainder(a: f32, b: f32) -> f32 {
    let m = a % b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        m + b
    } else {
        m
    }
}

/// F.interpolate(mode="linear", align_corners=False, scale_factor=sf) along time of [c, tin].
pub fn interp_linear(x: &[f32], c: usize, tin: usize, sf: f64) -> (Vec<f32>, usize) {
    let tout = (tin as f64 * sf).floor() as usize;
    let mut y = vec![0.0f32; c * tout];
    if tout == tin {
        y.copy_from_slice(x);
        return (y, tout);
    }
    let scale = (1.0 / sf) as f32;
    let mut idx = Vec::with_capacity(tout);
    for d in 0..tout {
        let mut real = scale.mul_add(d as f32 + 0.5, -0.5);
        if real < 0.0 {
            real = 0.0;
        }
        let i0 = (real.floor() as usize).min(tin - 1);
        let l1 = (real - i0 as f32).clamp(0.0, 1.0);
        let i1 = if i0 < tin - 1 { i0 + 1 } else { i0 };
        idx.push((i0, i1, 1.0 - l1, l1));
    }
    for ch in 0..c {
        let xr = &x[ch * tin..(ch + 1) * tin];
        let yr = &mut y[ch * tout..(ch + 1) * tout];
        for (o, &(i0, i1, l0, l1)) in yr.iter_mut().zip(&idx) {
            *o = xr[i0].mul_add(l0, xr[i1] * l1);
        }
    }
    (y, tout)
}

/// nn.Upsample(scale_factor, mode="nearest") on a single channel.
pub fn upsample_nearest(x: &[f32], sf: usize) -> Vec<f32> {
    let tin = x.len();
    let tout = tin * sf;
    let scale = (1.0 / sf as f64) as f32;
    (0..tout)
        .map(|d| {
            let src = ((d as f32 * scale).floor() as usize).min(tin - 1);
            x[src]
        })
        .collect()
}

/// Intermediates of SineGen's phase path, all channel-major [HARMONICS, len].
pub struct SourcePhase {
    pub rad_down: Vec<f32>,
    pub phase_pre_up: Vec<f32>,
    pub phase: Vec<f32>,
}

/// SourceModuleHnNSF harmonic branch: f0 at sample rate [S] -> har_source [S].
pub fn harmonic_source(f0_up: &[f32], rand_ini: &[f32; HARMONICS], sine_noise: &[f32], l_w: &[f32], l_b: f32) -> Vec<f32> {
    let s = f0_up.len();
    assert_eq!(sine_noise.len(), s * HARMONICS);
    let up = source_phase(f0_up, rand_ini).phase;
    let mut har = vec![0.0f32; s];
    for t in 0..s {
        let uv = if f0_up[t] > VOICED_THRESHOLD { 1.0f32 } else { 0.0 };
        let noise_amp = uv * NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0;
        let mut acc = 0.0f32;
        for h in 0..HARMONICS {
            let sine = up[h * s + t].sin() * SINE_AMP;
            let v = sine * uv + noise_amp * sine_noise[t * HARMONICS + h];
            acc += l_w[h] * v;
        }
        har[t] = (acc + l_b).tanh();
    }
    har
}

/// SineGen instantaneous phase (pre-sin) at sample rate.
pub fn source_phase(f0_up: &[f32], rand_ini: &[f32; HARMONICS]) -> SourcePhase {
    let s = f0_up.len();
    // rad [HARMONICS, S] (channel-major for interpolation)
    let mut rad = vec![0.0f32; HARMONICS * s];
    for h in 0..HARMONICS {
        let mult = (h + 1) as f32;
        for t in 0..s {
            rad[h * s + t] = torch_remainder(f0_up[t] * mult / SR, 1.0);
        }
        if h > 0 {
            rad[h * s] += rand_ini[h];
        }
    }
    let (down, td) = interp_linear(&rad, HARMONICS, s, 1.0 / UPSAMPLE_SCALE as f64);
    let pi = std::f32::consts::PI;
    let mut phase = vec![0.0f32; HARMONICS * td];
    for h in 0..HARMONICS {
        let mut acc = 0.0f64;
        for t in 0..td {
            acc += down[h * td + t] as f64;
            phase[h * td + t] = ((acc as f32) * 2.0 * pi) * UPSAMPLE_SCALE as f32;
        }
    }
    let (up, tu) = interp_linear(&phase, HARMONICS, td, UPSAMPLE_SCALE as f64);
    assert_eq!(tu, s);
    SourcePhase { rad_down: down, phase_pre_up: phase, phase: up }
}

fn hann_periodic() -> [f32; N_FFT] {
    let mut w = [0.0f32; N_FFT];
    for (n, v) in w.iter_mut().enumerate() {
        *v = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32;
    }
    w
}

fn twiddles() -> ([[f64; N_FFT]; N_BINS], [[f64; N_FFT]; N_BINS]) {
    let mut c = [[0.0f64; N_FFT]; N_BINS];
    let mut s = [[0.0f64; N_FFT]; N_BINS];
    for k in 0..N_BINS {
        for n in 0..N_FFT {
            let a = 2.0 * std::f64::consts::PI * ((k * n) % N_FFT) as f64 / N_FFT as f64;
            c[k][n] = a.cos();
            s[k][n] = a.sin();
        }
    }
    (c, s)
}

/// torch.stft(n_fft=20, hop=5, hann periodic, center=True, reflect) -> (mag, phase) [11, F].
pub fn stft(x: &[f32]) -> (Vec<f32>, Vec<f32>, usize) {
    let l = x.len();
    let pad = N_FFT / 2;
    assert!(l > pad, "signal too short for reflect padding");
    let mut p = Vec::with_capacity(l + 2 * pad);
    for i in 0..pad {
        p.push(x[pad - i]);
    }
    p.extend_from_slice(x);
    for j in 0..pad {
        p.push(x[l - 2 - j]);
    }
    let frames = 1 + l / HOP;
    let w = hann_periodic();
    let (cs, sn) = twiddles();
    let mut mag = vec![0.0f32; N_BINS * frames];
    let mut ph = vec![0.0f32; N_BINS * frames];
    let mut buf = [0.0f64; N_FFT];
    for f in 0..frames {
        for n in 0..N_FFT {
            buf[n] = (p[f * HOP + n] * w[n]) as f64;
        }
        for k in 0..N_BINS {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for n in 0..N_FFT {
                re += buf[n] * cs[k][n];
                im -= buf[n] * sn[k][n];
            }
            if k == 0 || k == N_BINS - 1 {
                im = 0.0; // real-to-complex FFTs emit exactly +0 imaginary at DC and Nyquist
            }
            let (re, im) = (re as f32, im as f32);
            mag[k * frames + f] = re.hypot(im);
            ph[k * frames + f] = im.atan2(re);
        }
    }
    (mag, ph, frames)
}

/// torch.istft(mag * exp(i*phase), n_fft=20, hop=5, hann periodic, center=True) -> [(F-1)*hop].
pub fn istft(mag: &[f32], phase: &[f32], frames: usize) -> Vec<f32> {
    let w = hann_periodic();
    let (cs, sn) = twiddles();
    let full = N_FFT + HOP * (frames - 1);
    let mut acc = vec![0.0f64; full];
    let mut env = vec![0.0f64; full];
    let mut re = [0.0f64; N_BINS];
    let mut im = [0.0f64; N_BINS];
    for f in 0..frames {
        for k in 0..N_BINS {
            let m = mag[k * frames + f];
            let p = phase[k * frames + f];
            re[k] = (m * p.cos()) as f64;
            im[k] = (m * p.sin()) as f64;
        }
        for n in 0..N_FFT {
            let mut v = re[0] + re[N_BINS - 1] * if n % 2 == 0 { 1.0 } else { -1.0 };
            for k in 1..N_BINS - 1 {
                v += 2.0 * (re[k] * cs[k][n] - im[k] * sn[k][n]);
            }
            let v = v / N_FFT as f64;
            acc[f * HOP + n] += v * w[n] as f64;
            env[f * HOP + n] += (w[n] as f64) * (w[n] as f64);
        }
    }
    let start = N_FFT / 2;
    let len = HOP * (frames - 1);
    (start..start + len).map(|i| (acc[i] / env[i]) as f32).collect()
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

    /// har [22, F] from har_source [S]
    pub fn source_spec(&self, har_source: &[f32]) -> (Vec<f32>, usize) {
        let (mag, ph, frames) = stft(har_source);
        let mut har = mag;
        har.extend_from_slice(&ph);
        (har, frames)
    }

    pub fn har_source(&self, f0_curve: &[f32], noise: &mut dyn NoiseSource) -> Result<Vec<f32>> {
        let f0_up = upsample_nearest(f0_curve, UPSAMPLE_SCALE);
        let (rand_ini, sine_noise) = noise.draw(f0_up.len())?;
        Ok(harmonic_source(&f0_up, &rand_ini, &sine_noise, &self.l_w, self.l_b))
    }

    /// x [512, t] (t = 2N), s [128], f0_curve [2N] -> waveform [600 N]
    pub fn forward(&self, x: &[f32], t: usize, s: &[f32], f0_curve: &[f32], noise: &mut dyn NoiseSource) -> Result<Vec<f32>> {
        let (har, frames) = {
            let _p = crate::prof::scope("gen.source+stft");
            let har_source = self.har_source(f0_curve, noise)?;
            self.source_spec(&har_source)
        };
        self.forward_from_har(x, t, s, &har, frames)
    }

    pub fn forward_from_har(&self, x: &[f32], t: usize, s: &[f32], har: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut x = x.to_vec();
        let mut t = t;
        for i in 0..2 {
            let _p = crate::prof::scope(if i == 0 { "gen.stage0" } else { "gen.stage1" });
            ops::leaky_relu(&mut x, 0.1);
            let (xs, ts) = {
                let _q = crate::prof::scope("gen/noise_branch");
                let (xs, ts) = self.noise_convs[i].forward(har, frames);
                (self.noise_res[i].forward(&xs, ts, s), ts)
            };
            let (mut y, mut ty) = {
                let _q = crate::prof::scope("gen/ups");
                self.ups[i].forward(&x, t)
            };
            let c = self.ups[i].cout;
            if i == 1 {
                let mut padded = vec![0.0f32; c * (ty + 1)];
                for ch in 0..c {
                    let src = &y[ch * ty..(ch + 1) * ty];
                    let dst = &mut padded[ch * (ty + 1)..(ch + 1) * (ty + 1)];
                    dst[0] = src[1];
                    dst[1..].copy_from_slice(src);
                }
                y = padded;
                ty += 1;
            }
            ensure!(ty == ts, "generator stage {i}: upsampled length {ty} != source length {ts}");
            for (a, b) in y.iter_mut().zip(&xs) {
                *a += *b;
            }
            let _q = crate::prof::scope("gen/resblocks");
            let mut acc = self.resblocks[i * 3].forward(&y, ty, s);
            for j in 1..3 {
                let r = self.resblocks[i * 3 + j].forward(&y, ty, s);
                for (a, b) in acc.iter_mut().zip(&r) {
                    *a += *b;
                }
            }
            for a in acc.iter_mut() {
                *a /= 3.0;
            }
            x = acc;
            t = ty;
        }
        let _p = crate::prof::scope("gen.post+istft");
        ops::leaky_relu(&mut x, 0.01);
        let (post, tp) = self.conv_post.forward(&x, t);
        let mut mag = post[..N_BINS * tp].to_vec();
        for v in mag.iter_mut() {
            *v = v.exp();
        }
        let mut ph = post[N_BINS * tp..].to_vec();
        for v in ph.iter_mut() {
            *v = v.sin();
        }
        Ok(istft(&mag, &ph, tp))
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

fn cat_channels(parts: &[(&[f32], usize)], t: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.1).sum::<usize>() * t);
    for (x, c) in parts {
        assert_eq!(x.len(), c * t);
        out.extend_from_slice(x);
    }
    out
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

    /// Everything before the generator: -> ([512, 2N], 2N)
    pub fn pre_generator(&self, asr: &[f32], nf: usize, f0_curve: &[f32], n_curve: &[f32], s: &[f32]) -> Result<(Vec<f32>, usize)> {
        ensure!(f0_curve.len() == 2 * nf && n_curve.len() == 2 * nf, "F0/N curves must have 2N values");
        let (f0, tf) = self.f0_conv.forward(f0_curve, 2 * nf);
        let (n, _) = self.n_conv.forward(n_curve, 2 * nf);
        ensure!(tf == nf, "F0_conv length {tf} != {nf}");
        let x = cat_channels(&[(asr, 512), (&f0, 1), (&n, 1)], nf);
        let (mut x, _) = self.encode.forward(&x, nf, s);
        let (asr_res, _) = self.asr_res.forward(asr, nf);
        let mut t = nf;
        let mut res = true;
        for block in &self.decode {
            if res {
                x = cat_channels(&[(&x, 1024), (&asr_res, 64), (&f0, 1), (&n, 1)], t);
            }
            let (y, t2) = block.forward(&x, t, s);
            x = y;
            t = t2;
            if block.upsample {
                res = false;
            }
        }
        Ok((x, t))
    }

    pub fn forward(&self, asr: &[f32], nf: usize, f0_curve: &[f32], n_curve: &[f32], s: &[f32], noise: &mut dyn NoiseSource) -> Result<Vec<f32>> {
        let (x, t) = {
            let _p = crate::prof::scope("decoder.pre");
            self.pre_generator(asr, nf, f0_curve, n_curve, s)?
        };
        self.generator.forward(&x, t, s, f0_curve, noise)
    }
}
