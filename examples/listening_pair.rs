//! Owner listening export: frozen cpu-t1 reference audio vs native CUDA forward on the SAME
//! frozen inputs (ids, style vector, speed, recorded noise). Raw 24 kHz mono float32 WAVs, no
//! normalization/clipping/trimming/alignment. Also recomputes the ladder metrics and compares
//! them to a recorded ladder receipt row.
//!
//! cargo run --release --features cuda --example listening_pair -- <case> <recorded-receipt.json> <out-dir>

use anyhow::{ensure, Context, Result};
use kokoro::engine::{sha256_bytes, sha256_file};
use kokoro::gpu::GpuKokoro;
use kokoro::model::Kokoro;
use kokoro::vocoder::{FixedNoise, HARMONICS};
use kokoro::wav::{self, Format};
use kokoro::{st, weights::Weights};
use std::path::PathBuf;

fn metrics(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    // identical formula to tests/gpu_parity.rs::rel
    let (mut d2, mut w2, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sg, mut sw, mut sgg, mut sww, mut sgw) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (g as f64, w as f64);
        d2 += (g - w).powi(2);
        w2 += w * w;
        mx = mx.max((g - w).abs());
        sg += g;
        sw += w;
        sgg += g * g;
        sww += w * w;
        sgw += g * w;
    }
    let n = got.len() as f64;
    let corr = (sgw / n - sg / n * sw / n) / ((sgg / n - (sg / n).powi(2)).sqrt() * (sww / n - (sw / n).powi(2)).sqrt());
    ((d2 / w2.max(1e-300)).sqrt(), mx, corr)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 4, "usage: listening_pair <case> <recorded-receipt.json> <out-dir>");
    let (case, receipt, out) = (&args[1], PathBuf::from(&args[2]), PathBuf::from(&args[3]));
    let data = PathBuf::from(std::env::var("KOKORO_DATA").context("source scripts/env.sh")?);
    let snap = data.join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987");
    let fxdir = data.join("fixtures/cpu-t1").join(case);
    let fx = st::load(&fxdir.join("fixture.safetensors"))?;
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fxdir.join("meta.json"))?)?;
    let f = |k: &str| fx[k].f32().map(|v| v.to_vec());

    let w = Weights::load_pth(&snap.join("kokoro-v1_0.pth"))?;
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(snap.join("config.json"))?)?;
    let m = Kokoro::from_weights(&w, &cfg)?;
    let gm = GpuKokoro::new(&m, 0)?;

    let ps = meta["phonemes"].as_str().context("phonemes")?;
    let (ids, dropped) = m.phonemes_to_ids(ps);
    ensure!(dropped.is_empty(), "dropped phoneme chars");
    ensure!(ids == fx["input_ids"].i64()?.to_vec(), "input ids differ from fixture");
    let ref_s = f("ref_s")?;
    let speed = meta["speed"].as_f64().context("speed")? as f32;
    let rand_ini: [f32; HARMONICS] = f("noise.rand_ini")?.try_into().map_err(|_| anyhow::anyhow!("rand_ini"))?;
    let reference = f("audio")?;

    // two independent CUDA runs: determinism of the subject itself
    let run = || gm.forward_ids(&m, &ids, &ref_s, speed, &mut FixedNoise { rand_ini, sine_noise: f("noise.sine").unwrap() });
    let a = run()?;
    let b = run()?;
    let deterministic = a.audio.iter().zip(&b.audio).all(|(x, y)| x.to_bits() == y.to_bits()) && a.pred_dur == b.pred_dur;
    ensure!(a.pred_dur == fx["pred_dur"].i64()?.to_vec(), "durations differ from fixture");
    ensure!(a.audio.len() == reference.len(), "sample count differs");
    let (rel, mx, corr) = metrics(&a.audio, &reference);

    let rec: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&receipt)?)?;
    let row = rec.as_array().context("receipt array")?.iter().find(|r| r["case"] == case.as_str() && r["seam"].as_str().unwrap_or("").starts_with("E2E")).context("recorded E2E row")?;
    let (rrel, rmax, rcorr) = (row["rel_l2"].as_f64().unwrap(), row["max_abs"].as_f64().unwrap(), row["corr"].as_f64().unwrap());
    let reproduces = rel == rrel && mx == rmax && corr == rcorr;

    std::fs::create_dir_all(&out)?;
    let ref_wav = wav::encode(&reference, 24000, Format::Float32);
    let rust_wav = wav::encode(&a.audio, 24000, Format::Float32);
    wav::write_atomic(&out.join("worst-peak-reference.wav"), &ref_wav.bytes)?;
    wav::write_atomic(&out.join("worst-peak-rust-cuda.wav"), &rust_wav.bytes)?;
    let peak = |x: &[f32]| x.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let argmax = a.audio.iter().zip(&reference).enumerate().fold((0usize, 0.0f64), |acc, (i, (g, w))| {
        let d = (*g as f64 - *w as f64).abs();
        if d > acc.1 { (i, d) } else { acc }
    });

    let manifest = serde_json::json!({
        "purpose": "owner listening review of the worst peak-error case among revised-gate GPU rows; NOT an acceptance artifact",
        "case": case, "text": meta["text"], "phonemes": ps, "voice": meta["voice"], "speed": meta["speed"],
        "sample_rate": 24000, "format": "RIFF WAVE, IEEE float32, mono, no normalization/clipping/trimming/alignment",
        "samples": reference.len(), "duration_s": reference.len() as f64 / 24000.0,
        "reference": {
            "file": "worst-peak-reference.wav",
            "identity": "pinned PyTorch reference kokoro==0.9.4 / torch==2.12.1 on CPU, 1 thread (fixture set cpu-t1) — NOT the production CUDA reference",
            "source_fixture": fxdir.join("fixture.safetensors"), "source_fixture_sha256": sha256_file(&fxdir.join("fixture.safetensors"))?,
            "source_meta": meta, "wav_sha256": sha256_bytes(&ref_wav.bytes), "peak_abs": peak(&reference),
        },
        "rust_cuda": {
            "file": "worst-peak-rust-cuda.wav",
            "identity": "kokoro-rust native CUDA forward (cuBLAS SGEMM default math = full f32, no TF32; kernels/kokoro.cu, -fmad=false)",
            "device": gm.gpu.device_name(),
            "git_head": std::env::var("KOKORO_GIT_HEAD").unwrap_or_default(),
            "git_dirty_paths": std::env::var("KOKORO_GIT_DIRTY").unwrap_or_default(),
            "note_vs_recorded_commit": "recorded receipt produced at ceef72f; fixed-noise CUDA forward path unchanged since (diff is additive: gen_noise kernel + counter-seed branch not taken for FixedNoise)",
            "model_sha256": sha256_file(&snap.join("kokoro-v1_0.pth"))?, "wav_sha256": sha256_bytes(&rust_wav.bytes),
            "peak_abs": peak(&a.audio), "clipped_samples_if_pcm16": wav::encode(&a.audio, 24000, Format::Pcm16).clipped,
            "deterministic_two_runs_bitwise": deterministic,
        },
        "inputs_identical": {"input_ids": true, "ref_s_from_fixture": true, "speed": meta["speed"], "noise": "fixture noise.sine + noise.rand_ini (recorded reference draws)", "pred_dur_equal": true},
        "metrics_recomputed": {"rel_l2": rel, "max_abs": mx, "corr": corr, "max_abs_at_sample": argmax.0, "max_abs_at_s": argmax.0 as f64 / 24000.0},
        "metrics_recorded": {"receipt": receipt, "receipt_sha256": sha256_file(&receipt)?, "rel_l2": rrel, "max_abs": rmax, "corr": rcorr},
        "reproduces_recorded_bitwise": reproduces,
        "binding_original_gates": {"rel_l2_le": 0.019, "max_abs_le": 0.033, "corr_ge": 0.9995,
            "verdict": if rel <= 0.019 && mx <= 0.033 && corr >= 0.9995 { "PASS" } else { "FAIL" }},
    });
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest["metrics_recomputed"])?);
    println!("reproduces recorded: {reproduces}; deterministic: {deterministic}; verdict under binding gates: {}", manifest["binding_original_gates"]["verdict"]);
    Ok(())
}
