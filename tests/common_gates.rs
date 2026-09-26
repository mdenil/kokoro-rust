// Shared binding gates (included by the parity ladders via `#[path]`).
//
// G-SPEC (NONDET_FLOOR.md original text): "log-magnitude STFT (n_fft 1024, hop 256) mean |Δ dB|
// ≤ 2 × floor-pair value (computed by the same comparator on the t1/t8 pair)". The original text
// leaves the dB floor and framing unspecified; the literal reading implemented here was FIXED
// BEFORE its first evaluation (2026-09-26) and is flagged for owner confirmation:
//   periodic Hann(1024), hop 256, frames fully inside the signal (no padding),
//   dB = 20·log10(max(|X|, 1e-5)), metric = mean over all bins×frames of |dB_a − dB_b|,
//   floor pair = pinned reference cpu-t1 vs cpu-t8 on the floor case s02_fox__af_heart__s1.0.

pub const SPEC_NFFT: usize = 1024;
pub const SPEC_HOP: usize = 256;
pub const FLOOR_CASE: &str = "s02_fox__af_heart__s1.0";

fn log_mag_frames(x: &[f32]) -> Vec<f64> {
    let n = SPEC_NFFT;
    let w: Vec<f64> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).collect();
    let frames = if x.len() >= n { 1 + (x.len() - n) / SPEC_HOP } else { 0 };
    let bins = n / 2 + 1;
    let mut out = Vec::with_capacity(frames * bins);
    let (cs, sn): (Vec<f64>, Vec<f64>) = (0..n).map(|i| {
        let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        (a.cos(), a.sin())
    }).unzip();
    let mut buf = vec![0.0f64; n];
    for f in 0..frames {
        for i in 0..n {
            buf[i] = x[f * SPEC_HOP + i] as f64 * w[i];
        }
        for k in 0..bins {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for i in 0..n {
                let idx = (k * i) % n;
                re += buf[i] * cs[idx];
                im -= buf[i] * sn[idx];
            }
            out.push(20.0 * (re.hypot(im)).max(1e-5).log10());
        }
    }
    out
}

/// Mean |Δ dB| between log-magnitude spectrograms of equal-length signals.
pub fn spec_mean_abs_db(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (la, lb) = (log_mag_frames(a), log_mag_frames(b));
    assert!(!la.is_empty(), "signal shorter than one spectral frame");
    la.iter().zip(&lb).map(|(x, y)| (x - y).abs()).sum::<f64>() / la.len() as f64
}
