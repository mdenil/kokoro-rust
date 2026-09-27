//! BF16x product vs the frozen Python reference fixtures ($KOKORO_DATA/fixtures/cpu-t1: kokoro 0.9.4 /
//! torch 2.12.1 on CPU, 15 cases x {af_heart, am_adam} x speeds).
//! - Hard check: the phoneme -> input_ids mapping and the voice style vector used by the engine equal
//!   the reference exactly for every case (data pins).
//! - Diagnostic (ignored; no thresholds): distances of the product forward, fed the reference's own
//!   excitation noise, to the reference audio. The owner accepted BF16x by listening (#25), not by a
//!   numerical gate; the phase-1 f32 conformance gates do not apply to it and are not re-imposed
//!   here. Where BF16x predicts different durations than the reference, the replayed noise no longer
//!   fits and the case is reported as such.

use kokoro::engine::Engine;
use kokoro::gpu::{BatchItem, ItemNoise};
use kokoro::st::{self, TensorMap};
use kokoro::vocoder::HARMONICS;
use std::path::PathBuf;

#[path = "common_gates.rs"]
mod common_gates;

fn data() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

fn model_dir() -> PathBuf {
    data().join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987")
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
}

fn fixtures() -> Vec<Fx> {
    let dir = data().join("fixtures/cpu-t1");
    let mut names: Vec<String> = std::fs::read_dir(&dir).expect("fixtures missing - NOT a pass").map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    assert_eq!(names.len(), 15, "fixture set incomplete - NOT a pass");
    names
        .into_iter()
        .map(|n| {
            let p = dir.join(&n);
            Fx { t: st::load(&p.join("fixture.safetensors")).unwrap(), meta: serde_json::from_str(&std::fs::read_to_string(p.join("meta.json")).unwrap()).unwrap(), name: n }
        })
        .collect()
}

#[test]
fn ids_and_style_vectors_match_reference() {
    let mut e = Engine::load(&model_dir(), 0).unwrap();
    for fx in fixtures() {
        let ph = fx.meta["phonemes"].as_str().unwrap();
        let (ids, dropped) = e.model.phonemes_to_ids(ph);
        assert!(dropped.is_empty(), "{}: dropped {dropped:?}", fx.name);
        assert_eq!(ids, fx.t["input_ids"].i64().unwrap().to_vec(), "{}: input_ids differ from the reference", fx.name);
        let voice = fx.meta["voice"].as_str().unwrap();
        let ref_s = e.voice(voice).unwrap().ref_s(ph.chars().count()).unwrap().to_vec();
        assert_eq!(ref_s, fx.f("ref_s"), "{}: style vector differs from the reference", fx.name);
    }
}

#[test]
#[ignore = "diagnostic; run explicitly (prints distances, no thresholds)"]
fn bf16x_vs_reference_diagnostic() {
    let e = Engine::load(&model_dir(), 0).unwrap();
    println!("{:<30} {:>7} {:>10} {:>10} {:>9}", "case (vs reference cpu-t1)", "frames", "rel L2", "max abs", "spec dB");
    for fx in fixtures() {
        let (ids, _) = e.model.phonemes_to_ids(fx.meta["phonemes"].as_str().unwrap());
        let ref_s = fx.f("ref_s");
        let rand_ini: [f32; HARMONICS] = fx.f("noise.rand_ini").try_into().unwrap();
        let item = BatchItem { ids: &ids, ref_s: &ref_s, speed: fx.meta["speed"].as_f64().unwrap() as f32, noise: ItemNoise::Fixed { rand_ini, sine: fx.f("noise.sine") } };
        let want = fx.f("audio");
        match e.gpu.forward_batch(&e.model, &[item]) {
            Ok(out) => {
                let a = &out[0].audio;
                assert!(!a.is_empty() && a.iter().all(|v| v.is_finite()), "{}: empty or non-finite audio", fx.name);
                assert_eq!(a.len(), want.len(), "{}: equal noise size implies equal durations", fx.name);
                let (mut d2, mut w2, mut mx) = (0.0f64, 0.0f64, 0.0f64);
                for (&x, &y) in a.iter().zip(&want) {
                    d2 += (x as f64 - y as f64).powi(2);
                    w2 += (y as f64).powi(2);
                    mx = mx.max((x as f64 - y as f64).abs());
                }
                let spec = common_gates::spec_mean_abs_db(a, &want);
                println!("{:<30} {:>7} {:>10.4} {:>10.4} {spec:>9.4}", fx.name, out[0].pred_dur.iter().sum::<i64>(), (d2 / w2).sqrt(), mx);
            }
            Err(err) => println!("{:<30} durations differ from the reference ({err:#})", fx.name),
        }
    }
}
