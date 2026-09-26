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
    names
        .into_iter()
        .filter(|n| std::env::var("KOKORO_CASE").map(|c| n.contains(&c)).unwrap_or(true))
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
    let mut fails = vec![];
    let mut rows = vec![];
    let mut check = |case: &str, seam: &str, got: &[f32], want: &[f32], pass_fn: &dyn Fn(f64, f64, f64) -> bool| {
        let (r, mx, c) = rel(got, want);
        let ok = pass_fn(r, mx, c) && got.iter().all(|v| v.is_finite());
        println!("{case:<32} {seam:<22} rel {r:>10.3e} max {mx:>10.3e} corr {c:>9.6} {}", if ok { "PASS" } else { "FAIL" });
        rows.push(serde_json::json!({"case": case, "seam": seam, "rel_l2": r, "max_abs": mx, "corr": c, "pass": ok}));
        if !ok {
            fails.push(format!("{case}/{seam}"));
        }
    };
    let seam = |r: f64, _m: f64, _c: f64| r <= SEAM_GATE;
    for fx in fixtures() {
        let c = fx.name.as_str();
        let (ids, _) = m.phonemes_to_ids(fx.meta["phonemes"].as_str().unwrap());
        assert_eq!(ids, fx.i("input_ids"));
        let t = ids.len();
        let ref_s = fx.f("ref_s");
        let (s_dec, s_pro) = ref_s.split_at(model::STYLE_DIM);
        let speed = fx.meta["speed"].as_f64().unwrap() as f32;

        check(c, "bert", &gm.seam_bert(&m, &ids).unwrap(), &fx.f("seam.bert"), &seam);
        check(c, "dur_enc", &gm.seam_dur_enc(&fx.f("seam.bert_encoder"), t, s_pro).unwrap(), &fx.f("seam.dur_enc"), &seam);
        check(c, "pred_lstm", &gm.seam_pred_lstm(&fx.f("seam.dur_enc"), t).unwrap(), &fx.f("seam.pred_lstm"), &seam);
        let nf: usize = fx.i("pred_dur").iter().sum::<i64>() as usize;
        let (f0, n) = gm.seam_f0n(&fx.f("seam.shared.arg0"), nf, s_pro).unwrap();
        check(c, "F0_pred", &f0, &fx.f("seam.decoder.arg1"), &seam);
        check(c, "N_pred", &n, &fx.f("seam.decoder.arg2"), &seam);
        check(c, "text_encoder", &gm.seam_text_encoder(&m, &ids).unwrap(), &fx.f("seam.text_encoder"), &seam);
        let pre = gm.seam_pre_generator(&fx.f("seam.decoder.arg0"), nf, &fx.f("seam.decoder.arg1"), &fx.f("seam.decoder.arg2"), s_dec).unwrap();
        check(c, "decoder.pre_generator", &pre, &fx.f("seam.gen.arg0"), &seam);
        let rand_ini: [f32; HARMONICS] = fx.f("noise.rand_ini").try_into().unwrap();
        let (spec, frames) = gm.seam_har(&fx.f("seam.gen.arg2"), &mut FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") }).unwrap();
        let nb = 11 * frames;
        check(c, "stft.mag(gpu source)", &spec[..nb], &fx.f("seam.har_spec"), &seam);
        let mut har = fx.f("seam.har_spec");
        har.extend(fx.f("seam.har_phase"));
        let audio_g = gm.seam_generator(&fx.f("seam.gen.arg0"), 2 * nf, s_dec, &har, frames).unwrap();
        check(c, "generator(oracle in)", &audio_g, &fx.f("audio"), &seam);

        let out = gm.forward_ids(&m, &ids, &ref_s, speed, &mut FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") }).unwrap();
        assert_eq!(out.pred_dur, fx.i("pred_dur"), "{c}: GPU durations differ");
        let fl = &floors["cases"][c];
        let (fr, fm) = (fl["floor_rel"].as_f64().unwrap(), fl["floor_max"].as_f64().unwrap());
        check(c, "E2E [G-v2]", &out.audio, &fx.f("audio"), &|r, mx, co| r <= 2.0 * fr && mx <= 2.0 * fm && co >= 0.9995);
    }
    let dir = data_root().join("evidence/ladder");
    std::fs::create_dir_all(&dir).unwrap();
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let p = dir.join(format!("ladder-gpu-{ts}.json"));
    std::fs::write(&p, serde_json::to_string_pretty(&rows).unwrap()).unwrap();
    println!("receipt: {}", p.display());
    assert!(fails.is_empty(), "{} GPU seam(s) failed: {fails:?}", fails.len());
}
