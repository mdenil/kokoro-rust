//! Parity-forensics for the SineGen / STFT seams against oracle/diag_source.py dumps.
//! Run: KOKORO_CASE=<case> cargo test --release --test diag_source -- --nocapture

use kokoro::st;
use kokoro::vocoder::{self, HARMONICS, N_BINS};
use std::path::PathBuf;

fn stats(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name} length");
    let (mut d2, mut w2, mut mx, mut arg, mut exact) = (0.0f64, 0.0f64, 0.0f64, 0usize, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g as f64 - w as f64).abs();
        if g.to_bits() == w.to_bits() {
            exact += 1;
        }
        d2 += d * d;
        w2 += (w as f64).powi(2);
        if d > mx {
            mx = d;
            arg = i;
        }
    }
    println!(
        "{name:<14} n={:>7} bit-exact={:>6.2}% rel_l2={:.3e} max={:.3e} @{} (got {} want {})",
        want.len(),
        100.0 * exact as f64 / want.len() as f64,
        (d2 / w2.max(1e-300)).sqrt(),
        mx,
        arg,
        got[arg],
        want[arg]
    );
}

#[test]
fn diag_source_path() {
    let root = PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()));
    let case = std::env::var("KOKORO_CASE").unwrap_or_else(|_| "s01_hello__af_heart__s1.0".into());
    let fx = st::load(&root.join(format!("fixtures/cpu-t1/{case}/fixture.safetensors"))).unwrap();
    let dg = st::load(&root.join(format!("fixtures/diag/cpu-t1/{case}/source_diag.safetensors"))).expect("run oracle/diag_source.py first");
    let f = |m: &st::TensorMap, k: &str| m[k].f32().unwrap().to_vec();

    let f0_up = vocoder::upsample_nearest(&f(&fx, "seam.gen.arg2"), vocoder::UPSAMPLE_SCALE);
    stats("f0_up", &f0_up, &f(&dg, "f0_up"));
    let rand_ini: [f32; HARMONICS] = f(&fx, "noise.rand_ini").try_into().unwrap();
    let sp = vocoder::source_phase(&f(&dg, "f0_up"), &rand_ini);
    stats("rad_down", &sp.rad_down, &f(&dg, "rad_down"));
    stats("phase_pre_up", &sp.phase_pre_up, &f(&dg, "phase_pre_up"));
    stats("phase", &sp.phase, &f(&dg, "phase"));

    // sines from the ORACLE phase: isolates sin() from the phase path
    let oph = f(&dg, "phase");
    let s = f0_up.len();
    let mut sines = vec![0.0f32; s * HARMONICS];
    for t in 0..s {
        for h in 0..HARMONICS {
            sines[t * HARMONICS + h] = oph[h * s + t].sin() * 0.1;
        }
    }
    stats("sin(oracle ph)", &sines, &f(&dg, "sines"));

    // phase error growth with |phase|
    let dph: Vec<f64> = sp.phase.iter().zip(&oph).map(|(a, b)| (*a as f64 - *b as f64).abs()).collect();
    let maxph = oph.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!("max |phase| = {maxph:.1} rad; f32 ulp there = {:.3e}; max |dphase| = {:.3e}",
        maxph * f32::EPSILON, dph.iter().cloned().fold(0.0, f64::max));

    // STFT: where does the phase differ by ~2pi, and what are the oracle's re/im there?
    let (_, ph, frames) = vocoder::stft(&f(&dg, "har_source"));
    let (ore, oim) = (f(&dg, "stft_re"), f(&dg, "stft_im"));
    let oph_stft: Vec<f32> = ore.iter().zip(&oim).map(|(r, i)| i.atan2(*r)).collect();
    let mut flips = vec![0usize; N_BINS];
    let mut shown = 0;
    for k in 0..N_BINS {
        for fr in 0..frames {
            let i = k * frames + fr;
            if (ph[i] - oph_stft[i]).abs() > 1.0 {
                flips[k] += 1;
                if shown < 6 {
                    println!("  flip bin {k} frame {fr}: ours {:.6} oracle {:.6} oracle re {:e} im {:e} (im sign bit {})",
                        ph[i], oph_stft[i], ore[i], oim[i], oim[i].is_sign_negative());
                    shown += 1;
                }
            }
        }
    }
    println!("2pi flips per bin: {flips:?}");
}
