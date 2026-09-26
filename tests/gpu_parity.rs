//! GPU (CUDA, full f32, no TF32) parity ladder against the frozen pinned-reference fixtures.
//! Same seams and gates as tests/parity.rs: stage-isolated seams fed the ORACLE's inputs
//! (gate 1e-4 rel L2), discrete outputs exact, E2E waveform under G-E2E-v2.
//! Run: source scripts/env.sh && cargo test --release --features cuda --test gpu_parity -- --nocapture
#![cfg(feature = "cuda")]

use kokoro::gpu::GpuKokoro;
use kokoro::model::{self, Kokoro};
use kokoro::st::{self, TensorMap};
use kokoro::vocoder::{FixedNoise, HARMONICS};
use kokoro::weights::Weights;
use std::path::PathBuf;

#[path = "common_gates.rs"]
mod common_gates;

fn data_root() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

struct Fx {
    name: String,
    t: TensorMap,
    meta: serde_json::Value,
}
impl Fx {
    fn f(&self, k: &str) -> Vec<f32> {
        self.t.get(k).unwrap_or_else(|| panic!("{}: missing {k}", self.name)).f32().unwrap().to_vec()
    }
    fn i(&self, k: &str) -> Vec<i64> {
        self.t[k].i64().unwrap().to_vec()
    }
}

fn fixtures() -> Vec<Fx> {
    let dir = data_root().join("fixtures/cpu-t1");
    let mut names: Vec<_> = std::fs::read_dir(&dir).expect("fixtures missing — NOT a pass").map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    assert!(!names.is_empty(), "no fixtures — NOT a pass");
    let filter = std::env::var("KOKORO_CASE").ok();
    let names: Vec<String> = names.into_iter().filter(|n| filter.as_ref().map(|c| n.contains(c.as_str())).unwrap_or(true)).collect();
    assert!(!names.is_empty(), "case filter {filter:?} selected ZERO fixtures — NOT a pass");
    if filter.is_none() {
        assert_eq!(names.len(), 15, "fixture set incomplete");
    }
    names
        .into_iter()
        .map(|n| {
            let p = dir.join(&n);
            Fx { t: st::load(&p.join("fixture.safetensors")).unwrap(), meta: serde_json::from_str(&std::fs::read_to_string(p.join("meta.json")).unwrap()).unwrap(), name: n }
        })
        .collect()
}

fn rel(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    assert_eq!(got.len(), want.len());
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

fn check(rows: &mut Vec<serde_json::Value>, fails: &mut Vec<String>, case: &str, seam: &str, got: &[f32], want: &[f32], pass_fn: &dyn Fn(f64, f64, f64) -> bool) {
    let (r, mx, c) = rel(got, want);
    let ok = pass_fn(r, mx, c) && got.iter().all(|v| v.is_finite());
    println!("{case:<32} {seam:<22} rel {r:>10.3e} max {mx:>10.3e} corr {c:>9.6} {}", if ok { "PASS" } else { "FAIL" });
    rows.push(serde_json::json!({"case": case, "seam": seam, "rel_l2": r, "max_abs": mx, "corr": c, "pass": ok}));
    if !ok {
        fails.push(format!("{case}/{seam}"));
    }
}

const SEAM_GATE: f64 = 1e-4;

#[test]
fn gpu_ladder() {
    let snap = data_root().join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987");
    let w = Weights::load_pth(&snap.join("kokoro-v1_0.pth")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(snap.join("config.json")).unwrap()).unwrap();
    let m = Kokoro::from_weights(&w, &cfg).unwrap();
    let ordinal: usize = std::env::var("KOKORO_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let gm = GpuKokoro::new(&m, ordinal).unwrap();
    println!("device: {}", gm.gpu.device_name());
    let floors: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(data_root().join("fixtures/floor_per_case.json")).unwrap()).unwrap();
    let mut fails: Vec<String> = vec![];
    let mut rows: Vec<serde_json::Value> = vec![];
    let mut cases_seen = 0usize;
    let spec_gate = {
        let c = common_gates::FLOOR_CASE;
        let t1 = st::load(&data_root().join(format!("fixtures/cpu-t1/{c}/fixture.safetensors"))).unwrap()["audio"].f32().unwrap().to_vec();
        let t8 = st::load(&data_root().join(format!("fixtures/f64/{c}/f64.safetensors"))).unwrap()["audio_f32_t8"].f32().unwrap().to_vec();
        2.0 * common_gates::spec_mean_abs_db(&t1, &t8)
    };
    println!("G-SPEC gate (2 x reference cpu-t1 vs cpu-t8 on {}): {spec_gate:.6} dB", common_gates::FLOOR_CASE);
    let seam = |r: f64, _m: f64, _c: f64| r <= SEAM_GATE;
    for fx in fixtures() {
        let c = fx.name.as_str();
        let (ids, _) = m.phonemes_to_ids(fx.meta["phonemes"].as_str().unwrap());
        assert_eq!(ids, fx.i("input_ids"));
        let t = ids.len();
        let ref_s = fx.f("ref_s");
        let (s_dec, s_pro) = ref_s.split_at(model::STYLE_DIM);
        let speed = fx.meta["speed"].as_f64().unwrap() as f32;

        check(&mut rows, &mut fails, c, "bert", &gm.seam_bert(&m, &ids).unwrap(), &fx.f("seam.bert"), &seam);
        check(&mut rows, &mut fails, c, "dur_enc", &gm.seam_dur_enc(&fx.f("seam.bert_encoder"), t, s_pro).unwrap(), &fx.f("seam.dur_enc"), &seam);
        check(&mut rows, &mut fails, c, "pred_lstm", &gm.seam_pred_lstm(&fx.f("seam.dur_enc"), t).unwrap(), &fx.f("seam.pred_lstm"), &seam);
        let nf: usize = fx.i("pred_dur").iter().sum::<i64>() as usize;
        let (f0, n) = gm.seam_f0n(&fx.f("seam.shared.arg0"), nf, s_pro).unwrap();
        check(&mut rows, &mut fails, c, "F0_pred", &f0, &fx.f("seam.decoder.arg1"), &seam);
        check(&mut rows, &mut fails, c, "N_pred", &n, &fx.f("seam.decoder.arg2"), &seam);
        check(&mut rows, &mut fails, c, "text_encoder", &gm.seam_text_encoder(&m, &ids).unwrap(), &fx.f("seam.text_encoder"), &seam);
        let pre = gm.seam_pre_generator(&fx.f("seam.decoder.arg0"), nf, &fx.f("seam.decoder.arg1"), &fx.f("seam.decoder.arg2"), s_dec).unwrap();
        check(&mut rows, &mut fails, c, "decoder.pre_generator", &pre, &fx.f("seam.gen.arg0"), &seam);
        let rand_ini: [f32; HARMONICS] = fx.f("noise.rand_ini").try_into().unwrap();
        let (spec, frames) = gm.seam_har(&fx.f("seam.gen.arg2"), &mut FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") }).unwrap();
        let nb = 11 * frames;
        check(&mut rows, &mut fails, c, "stft.mag(gpu source)", &spec[..nb], &fx.f("seam.har_spec"), &seam);
        let mut har = fx.f("seam.har_spec");
        har.extend(fx.f("seam.har_phase"));
        let audio_g = gm.seam_generator(&fx.f("seam.gen.arg0"), 2 * nf, s_dec, &har, frames).unwrap();
        check(&mut rows, &mut fails, c, "generator(oracle in)", &audio_g, &fx.f("audio"), &seam);

        let out = gm.forward_ids(&m, &ids, &ref_s, speed, &mut FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") }).unwrap();
        assert_eq!(out.pred_dur, fx.i("pred_dur"), "{c}: GPU durations differ");
        assert_eq!(out.audio.len(), fx.f("audio").len(), "{c}: sample count differs");
        // ENFORCED: original precommitted gates (binding owner ruling)
        check(&mut rows, &mut fails, c, "E2E [G-v1 original, enforced]", &out.audio, &fx.f("audio"), &|r, mx, co| r <= 0.019 && mx <= 0.033 && co >= 0.9995);
        let sv = common_gates::spec_mean_abs_db(&out.audio, &fx.f("audio"));
        let sok = sv <= spec_gate;
        println!("{c:<32} {:<22} mean|dB| {sv:>9.5} gate {spec_gate:.5} {}", "E2E spectral [G-SPEC]", if sok { "PASS" } else { "FAIL" });
        rows.push(serde_json::json!({"case": c, "seam": "E2E spectral mean|dB| [G-SPEC original, enforced]", "value": sv, "gate": spec_gate, "pass": sok}));
        if !sok {
            fails.push(format!("{c}/E2E spectral"));
        }
        // DIAGNOSTIC ONLY: revised per-case gate, NOT approved, never asserted
        let fl = &floors["cases"][c];
        let (fr, fm) = (fl["floor_rel"].as_f64().unwrap(), fl["floor_max"].as_f64().unwrap());
        let (r, mx, co) = rel(&out.audio, &fx.f("audio"));
        let v2 = r <= 2.0 * fr && mx <= 2.0 * fm && co >= 0.9995;
        println!("{c:<32} {:<22} (UNAPPROVED diagnostic) {}", "E2E [G-v2]", if v2 { "diag-ok" } else { "diag-FAIL" });
        rows.push(serde_json::json!({"case": c, "seam": "E2E [G-v2 UNAPPROVED diagnostic]", "rel_l2": r, "max_abs": mx, "corr": co, "pass": v2, "asserted": false}));
        cases_seen += 1;
    }
    assert!(cases_seen > 0, "zero cases evaluated — NOT a pass");
    for req in ["bert", "dur_enc", "pred_lstm", "F0_pred", "N_pred", "text_encoder", "decoder.pre_generator", "stft.mag", "generator(oracle in)", "E2E [G-v1", "E2E spectral"] {
        let n = rows.iter().filter(|r| r["seam"].as_str().unwrap_or("").starts_with(req)).count();
        assert_eq!(n, cases_seen, "required seam {req}: {n} rows for {cases_seen} cases");
    }
    println!("coverage: {cases_seen} cases x 11 required seams present");
    let dir = data_root().join("evidence/ladder");
    std::fs::create_dir_all(&dir).unwrap();
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let p = dir.join(format!("ladder-gpu-{ts}.json"));
    std::fs::write(&p, serde_json::to_string_pretty(&rows).unwrap()).unwrap();
    println!("receipt: {}", p.display());
    assert!(fails.is_empty(), "{} GPU seam(s) failed: {fails:?}", fails.len());
}

/// Lever NOISE-ON-DEVICE parity: the device counter-based generator reproduces the CPU stream,
/// and E2E audio with device noise matches the kill-switch (host-drawn) path.
#[test]
fn gpu_noise_matches_cpu_stream() {
    use kokoro::vocoder::{NoiseSource, RngNoise};
    let snap = data_root().join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987");
    let w = Weights::load_pth(&snap.join("kokoro-v1_0.pth")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(snap.join("config.json")).unwrap()).unwrap();
    let m = Kokoro::from_weights(&w, &cfg).unwrap();
    let gm = GpuKokoro::new(&m, 0).unwrap();
    let n = 9 * 200_001;
    let (_, cpu) = RngNoise::new(42).draw(n / 9).unwrap();
    let gpu = gm.seam_gen_noise(42, n).unwrap();
    let exact = cpu.iter().zip(&gpu).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
    let mx = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let mean = cpu.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var = cpu.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    println!("noise: {:.2}% bit-exact, max |cpu-gpu| {mx:.3e}; mean {mean:.4} var {var:.4}", 100.0 * exact as f64 / n as f64);
    assert!(mx <= 1e-5, "device noise diverges from CPU stream");
    assert!(mean.abs() < 0.01 && (var - 1.0).abs() < 0.01, "noise is not standard normal");

    let (ids, _) = m.phonemes_to_ids("ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ.");
    let pack = kokoro::torchpt::load_voice_pack(&snap.join("voices/af_heart.pt")).unwrap();
    let r = 47usize; // len(phonemes) - 1
    let ref_s = pack.data[r * 256..(r + 1) * 256].to_vec();
    let dev = gm.forward_ids(&m, &ids, &ref_s, 1.0, &mut RngNoise::new(7)).unwrap();
    std::env::set_var("KOKORO_GPU_HOST_NOISE", "1");
    let host = gm.forward_ids(&m, &ids, &ref_s, 1.0, &mut RngNoise::new(7)).unwrap();
    std::env::remove_var("KOKORO_GPU_HOST_NOISE");
    assert_eq!(dev.pred_dur, host.pred_dur);
    let (r, mx, c) = rel(&dev.audio, &host.audio);
    println!("E2E device-noise vs host-noise: rel {r:.3e} max {mx:.3e} corr {c:.8}");
    assert!(r <= 1e-4, "device noise changes audio beyond ulp level");
}

/// GPU negative controls: every enforced comparator must FAIL under a deliberate perturbation
/// of the CUDA path (brief gate 4), and the unperturbed control must behave as recorded.
#[test]
fn gpu_negative_controls() {
    let snap = data_root().join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987");
    let w = Weights::load_pth(&snap.join("kokoro-v1_0.pth")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(snap.join("config.json")).unwrap()).unwrap();
    let m = Kokoro::from_weights(&w, &cfg).unwrap();
    let gm = GpuKokoro::new(&m, 0).unwrap();
    let dir = data_root().join("fixtures/cpu-t1/s02_fox__af_heart__s1.0");
    let fx = Fx { t: st::load(&dir.join("fixture.safetensors")).unwrap(), meta: serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap(), name: "s02_fox__af_heart__s1.0".into() };
    let (ids, _) = m.phonemes_to_ids(fx.meta["phonemes"].as_str().unwrap());
    let ref_s = fx.f("ref_s");
    let rand_ini: [f32; HARMONICS] = fx.f("noise.rand_ini").try_into().unwrap();
    let want = fx.f("audio");
    let v1 = |a: &[f32]| {
        let (r, mx, c) = rel(a, &want);
        (r <= 0.019 && mx <= 0.033 && c >= 0.9995, r, mx)
    };
    let t1 = want.clone();
    let t8 = st::load(&data_root().join("fixtures/f64/s02_fox__af_heart__s1.0/f64.safetensors")).unwrap()["audio_f32_t8"].f32().unwrap().to_vec();
    let spec_gate = 2.0 * common_gates::spec_mean_abs_db(&t1, &t8);
    let noise = || FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") };

    let base = gm.forward_ids(&m, &ids, &ref_s, 1.0, &mut noise()).unwrap();
    let (ok, r, mx) = v1(&base.audio);
    println!("control (unperturbed): v1 {ok} rel {r:.3e} max {mx:.3e}; spec {:.4} dB (gate {spec_gate:.4})", common_gates::spec_mean_abs_db(&base.audio, &want));
    assert!(ok, "unperturbed control must pass the original gate on this case (recorded PASS)");

    // F0 curve x1.001 into the decoder -> v1 must fail
    let nf: usize = base.pred_dur.iter().sum::<i64>() as usize;
    let f0: Vec<f32> = fx.f("seam.decoder.arg1").iter().map(|v| v * 1.001).collect();
    let pre = gm.seam_pre_generator(&fx.f("seam.decoder.arg0"), nf, &f0, &fx.f("seam.decoder.arg2"), &ref_s[..128]).unwrap();
    let (spec, frames) = gm.seam_har(&f0, &mut noise()).unwrap();
    let a = gm.seam_generator(&pre, 2 * nf, &ref_s[..128], &spec, frames).unwrap();
    let (ok, r, mx) = v1(&a);
    println!("F0 x1.001: v1 {ok} rel {r:.3e} max {mx:.3e}");
    assert!(!ok, "F0 perturbation not detected by v1");

    // pitch shift x1.05: spectral gate must fail
    let f0: Vec<f32> = fx.f("seam.decoder.arg1").iter().map(|v| v * 1.05).collect();
    let pre = gm.seam_pre_generator(&fx.f("seam.decoder.arg0"), nf, &f0, &fx.f("seam.decoder.arg2"), &ref_s[..128]).unwrap();
    let (spec, frames) = gm.seam_har(&f0, &mut noise()).unwrap();
    let a = gm.seam_generator(&pre, 2 * nf, &ref_s[..128], &spec, frames).unwrap();
    let sv = common_gates::spec_mean_abs_db(&a, &want);
    println!("F0 x1.05: spectral {sv:.4} dB (gate {spec_gate:.4})");
    assert!(sv > spec_gate, "pitch perturbation not detected by G-SPEC");

    // excitation noise perturbed -> GPU source spectrum seam must fail
    let mut sn = fx.f("noise.sine");
    for v in sn.iter_mut().step_by(HARMONICS) {
        *v *= 1.5;
    }
    let (spec, _) = gm.seam_har(&fx.f("seam.gen.arg2"), &mut FixedNoise { rand_ini, sine_noise: sn }).unwrap();
    let nb = spec.len() / 2;
    let (r, _, _) = rel(&spec[..nb], &fx.f("seam.har_spec"));
    println!("noise h0 x1.5: stft.mag seam rel {r:.3e}");
    assert!(r > SEAM_GATE, "noise perturbation not detected by the GPU source seam");

    // wrong voice -> exact duration gate must fail
    let other = st::load(&data_root().join("fixtures/cpu-t1/s02_fox__am_adam__s1.0/fixture.safetensors")).unwrap();
    let o = gm.forward_ids(&m, &ids, other["ref_s"].f32().unwrap(), 1.0, &mut FixedNoise { rand_ini, sine_noise: other["noise.sine"].f32().unwrap().to_vec() }).unwrap();
    assert_ne!(o.pred_dur, fx.i("pred_dur"), "voice swap not detected by the exact duration gate");

    // one-sample shift of the unperturbed output -> v1 must fail
    let mut sh = vec![0.0f32];
    sh.extend_from_slice(&base.audio[..base.audio.len() - 1]);
    let (ok, r, _) = v1(&sh);
    println!("1-sample shift: v1 {ok} rel {r:.3e}");
    assert!(!ok, "time shift not detected by v1");

    // seam gate: 1e-3 relative perturbation must fail
    let x: Vec<f32> = fx.f("seam.gen.arg0").iter().enumerate().map(|(i, v)| v * (1.0 + if i % 2 == 0 { 1e-3 } else { -1e-3 })).collect();
    let (r, _, _) = rel(&x, &fx.f("seam.gen.arg0"));
    assert!(r > SEAM_GATE, "seam perturbation not detected");
    println!("all GPU negative controls detected");
}

fn read_f32_wav(p: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap();
    assert_eq!(&b[0..4], b"RIFF");
    assert_eq!(u16::from_le_bytes([b[20], b[21]]), 3, "not IEEE float");
    b[44..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Bounded-variation regression policy RB-1 (owner clarification 1553392534095527999; bounds
/// fixed 2026-09-26 BEFORE judging any further optimization; docs/conformance/REGRESSION_POLICY.md).
/// Per case, a candidate CUDA output may drift from the approved baseline output by at most
///   min(reference arithmetic sensitivity of that case, owner-accepted listened divergence)
/// on rel-L2, max|Δ| and spectral mean|ΔdB|; discrete outputs exact; the set of cases failing the
/// binding original gates may not gain any (case, gate) pair. Bit identity is reported, never required.
/// One-time setup: KOKORO_PIN_BOUNDS=1 (refuses to overwrite).
#[test]
fn gpu_regression_bounded() {
    use sha2::{Digest, Sha256};
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/pinned");
    // AUTHORITATIVE: always the owner-accepted strict baseline, whatever the build's rounding mode. An
    // FMA build may FAIL this truthfully (owner #11 authorizes FMA exploration, not approval of its
    // deltas). KOKORO_DIAG_FMA_SNAPSHOT=1 instead compares against the PROVISIONAL FMA snapshot
    // (diagnostic only; never parity, never approval).
    let diag = std::env::var("KOKORO_DIAG_FMA_SNAPSHOT").map(|v| v == "1").unwrap_or(false);
    let (pin_path, bounds_path) = if diag {
        (root.join("gpu_envelope_fma_PROVISIONAL_diagnostic.json"), root.join("regression_bounds.json"))
    } else {
        (root.join("gpu_envelope.json"), root.join("regression_bounds.json"))
    };
    let base_dir = data_root().join(if diag { "evidence/pinned-snapshot-fma-PROVISIONAL-diagnostic" } else { "evidence/pinned-baseline" });
    println!("RB-1 reference: {} (build kernels={})", if diag { "PROVISIONAL FMA snapshot (diagnostic)" } else { "owner-accepted strict baseline (authoritative)" }, kokoro::gpu::KERNEL_ROUNDING);
    let pin_baseline = false; // baselines are pinned once; no re-pinning from this test
    let snap = data_root().join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987");
    let w = Weights::load_pth(&snap.join("kokoro-v1_0.pth")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(snap.join("config.json")).unwrap()).unwrap();
    let m = Kokoro::from_weights(&w, &cfg).unwrap();
    let gm = GpuKokoro::new(&m, 0).unwrap();
    if pin_baseline {
        // One-time pin of the approved baseline for this rounding build (refuses to overwrite).
        assert!(!pin_path.exists() && !base_dir.exists(), "refusing to overwrite an existing baseline pin");
        std::fs::create_dir_all(&base_dir).unwrap();
        let spec_gate = {
            let c = common_gates::FLOOR_CASE;
            let t1 = st::load(&data_root().join(format!("fixtures/cpu-t1/{c}/fixture.safetensors"))).unwrap()["audio"].f32().unwrap().to_vec();
            let t8 = st::load(&data_root().join(format!("fixtures/f64/{c}/f64.safetensors"))).unwrap()["audio_f32_t8"].f32().unwrap().to_vec();
            2.0 * common_gates::spec_mean_abs_db(&t1, &t8)
        };
        let _ = spec_gate;
        let mut cases = serde_json::Map::new();
        for fx in fixtures() {
            let (ids, _) = m.phonemes_to_ids(fx.meta["phonemes"].as_str().unwrap());
            let rand_ini: [f32; HARMONICS] = fx.f("noise.rand_ini").try_into().unwrap();
            let out = gm.forward_ids(&m, &ids, &fx.f("ref_s"), fx.meta["speed"].as_f64().unwrap() as f32, &mut FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") }).unwrap();
            let want = fx.f("audio");
            let (r, mx, co) = rel(&out.audio, &want);
            let sp = common_gates::spec_mean_abs_db(&out.audio, &want);
            let bytes: Vec<u8> = out.audio.iter().flat_map(|v| v.to_le_bytes()).collect();
            let sha: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
            std::fs::write(base_dir.join(format!("{}.f32le", fx.name)), &bytes).unwrap();
            cases.insert(fx.name.clone(), serde_json::json!({"rel_l2": r, "max_abs": mx, "corr": co, "spec_db": sp, "audio_f32le_sha256": sha}));
        }
        std::fs::write(&pin_path, serde_json::to_string_pretty(&serde_json::json!({
            "pinned": format!("2026-09-26 baseline for kernel rounding '{}' (owner #11); fixtures cpu-t1; FixedNoise", kokoro::gpu::KERNEL_ROUNDING),
            "cases": cases})).unwrap()).unwrap();
        println!("pinned {} + {}", pin_path.display(), base_dir.display());
        return;
    }
    let pin: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&pin_path).unwrap()).unwrap();
    let setup = std::env::var("KOKORO_PIN_BOUNDS").map(|v| v == "1").unwrap_or(false);
    let listened_ref = read_f32_wav(&data_root().join("evidence/listening/worst-peak-reference.wav"));
    let listened_rust = read_f32_wav(&data_root().join("evidence/listening/worst-peak-rust-cuda.wav"));
    let (l_rel, l_max, _) = rel(&listened_rust, &listened_ref);
    let l_spec = common_gates::spec_mean_abs_db(&listened_rust, &listened_ref);
    let spec_gate = {
        let c = common_gates::FLOOR_CASE;
        let t1 = st::load(&data_root().join(format!("fixtures/cpu-t1/{c}/fixture.safetensors"))).unwrap()["audio"].f32().unwrap().to_vec();
        let t8 = st::load(&data_root().join(format!("fixtures/f64/{c}/f64.safetensors"))).unwrap()["audio_f32_t8"].f32().unwrap().to_vec();
        2.0 * common_gates::spec_mean_abs_db(&t1, &t8)
    };
    // per-gate verdicts: (v1 waveform fails, G-SPEC fails)
    let binding_fail = |r: f64, mx: f64, co: f64, sp: f64| (!(r <= 0.019 && mx <= 0.033 && co >= 0.9995), sp > spec_gate);
    if setup {
        assert!(!bounds_path.exists(), "refusing to overwrite {}", bounds_path.display());
        std::fs::create_dir_all(&base_dir).unwrap();
    }
    let bounds: serde_json::Value = if setup { serde_json::json!({}) } else { serde_json::from_str(&std::fs::read_to_string(&bounds_path).expect("no bounds; run setup")).unwrap() };
    let floors: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(data_root().join("fixtures/floor_per_case.json")).unwrap()).unwrap();
    let mut new_bounds = serde_json::Map::new();
    let (mut violations, mut new_binding_fails, mut bitwise) = (vec![], vec![], 0usize);
    let mut n = 0usize;
    for fx in fixtures() {
        let c = fx.name.clone();
        let (ids, _) = m.phonemes_to_ids(fx.meta["phonemes"].as_str().unwrap());
        let rand_ini: [f32; HARMONICS] = fx.f("noise.rand_ini").try_into().unwrap();
        let speed = fx.meta["speed"].as_f64().unwrap() as f32;
        let out = gm.forward_ids(&m, &ids, &fx.f("ref_s"), speed, &mut FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") }).unwrap();
        assert_eq!(out.pred_dur, fx.i("pred_dur"), "{c}: durations (exact invariant)");
        let want = fx.f("audio");
        assert_eq!(out.audio.len(), want.len(), "{c}: sample count (exact invariant)");
        let bytes: Vec<u8> = out.audio.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sha: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        let base_file = base_dir.join(format!("{c}.f32le"));
        if setup {
            assert_eq!(sha, pin["cases"][&c]["audio_f32le_sha256"].as_str().unwrap(), "{c}: current output is not the approved baseline; cannot dump");
            std::fs::write(&base_file, &bytes).unwrap();
            // reference arithmetic sensitivity: pinned torch f32 at 2/4/8 threads vs 1 thread
            let f64fx = st::load(&data_root().join(format!("fixtures/f64/{c}/f64.safetensors"))).unwrap();
            let sens_spec = [2, 4, 8].iter().map(|t| common_gates::spec_mean_abs_db(f64fx[&format!("audio_f32_t{t}")].f32().unwrap(), &want)).fold(0.0f64, f64::max);
            let fl = &floors["cases"][&c];
            let (sr, sm) = (fl["floor_rel"].as_f64().unwrap(), fl["floor_max"].as_f64().unwrap());
            new_bounds.insert(c.clone(), serde_json::json!({
                "rel_l2": sr.min(l_rel), "max_abs": sm.min(l_max), "spec_db": sens_spec.min(l_spec),
                "ref_sensitivity": {"rel_l2": sr, "max_abs": sm, "spec_db": sens_spec}}));
            n += 1;
            continue;
        }
        let base: Vec<f32> = std::fs::read(&base_file).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        if base == out.audio {
            bitwise += 1;
        }
        let (dr, dm, _) = rel(&out.audio, &base);
        let ds = common_gates::spec_mean_abs_db(&out.audio, &base);
        let b = &bounds["cases"][&c];
        for (k, v) in [("rel_l2", dr), ("max_abs", dm), ("spec_db", ds)] {
            if v > b[k].as_f64().unwrap() {
                violations.push(format!("{c}.{k}: drift {v:.5} > bound {:.5}", b[k].as_f64().unwrap()));
            }
        }
        let (r, mx, co) = rel(&out.audio, &want);
        let sp = common_gates::spec_mean_abs_db(&out.audio, &want);
        let p = &pin["cases"][&c];
        let was_fail = binding_fail(p["rel_l2"].as_f64().unwrap(), p["max_abs"].as_f64().unwrap(), p["corr"].as_f64().unwrap(), p["spec_db"].as_f64().unwrap());
        let now = binding_fail(r, mx, co, sp);
        if now.0 && !was_fail.0 {
            new_binding_fails.push(format!("{c}/v1"));
        }
        if now.1 && !was_fail.1 {
            new_binding_fails.push(format!("{c}/G-SPEC"));
        }
        println!("{c:<32} drift rel {dr:.2e} max {dm:.2e} spec {ds:.4} dB | vs ref max {mx:.4} (base {:.4})", p["max_abs"].as_f64().unwrap());
        n += 1;
    }
    assert_eq!(n, 15, "all 15 cases required");
    if setup {
        std::fs::write(&bounds_path, serde_json::to_string_pretty(&serde_json::json!({
            "policy": "RB-1: per case min(reference arithmetic sensitivity [pinned torch CPU f32 t2/t4/t8 vs t1], owner-accepted listened divergence); fixed 2026-09-26 before judging further levers",
            "listened_divergence": {"rel_l2": l_rel, "max_abs": l_max, "spec_db": l_spec},
            "baseline_audio_dir": base_dir, "cases": new_bounds})).unwrap()).unwrap();
        println!("wrote {} and baseline audio to {}", bounds_path.display(), base_dir.display());
        return;
    }
    println!("RB-1: {bitwise}/15 bitwise identical to baseline (optional evidence); {} drift violations; new binding-gate failures: {new_binding_fails:?}", violations.len());
    assert!(violations.is_empty(), "drift beyond RB-1 bounds — escalate with paired audio: {violations:#?}");
    assert!(new_binding_fails.is_empty(), "cases newly failing the binding original gates — escalate: {new_binding_fails:?}");
}
