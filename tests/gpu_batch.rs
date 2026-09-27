//! B1 batched-forward correctness (docs/design/BATCHING.md). Bounds are the RB-1 per-case drift
//! bounds already fixed in tests/pinned/regression_bounds.json (not chosen after seeing results):
//! each item's batched output vs its single-item output (identical inputs and noise) must stay
//! within its case bound; ids/durations/sample counts EXACT. Negative controls must be DETECTED.
#![cfg(feature = "cuda")]

use kokoro::gpu::{BatchItem, GpuKokoro, ItemNoise};
use kokoro::model::{Kokoro, Output};
use kokoro::st;
use kokoro::vocoder::{FixedNoise, HARMONICS};
use kokoro::weights::Weights;
use std::path::PathBuf;

#[path = "common_gates.rs"]
mod common_gates;

fn data() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

struct Case {
    name: String,
    ids: Vec<i64>,
    ref_s: Vec<f32>,
    speed: f32,
    rand_ini: [f32; HARMONICS],
    sine: Vec<f32>,
}

fn setup() -> (Kokoro, GpuKokoro, Vec<Case>, serde_json::Value) {
    let snap = data().join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987");
    let w = Weights::load_pth(&snap.join("kokoro-v1_0.pth")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(snap.join("config.json")).unwrap()).unwrap();
    let m = Kokoro::from_weights(&w, &cfg).unwrap();
    let gm = GpuKokoro::new(&m, 0).unwrap();
    let dir = data().join("fixtures/cpu-t1");
    let mut names: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    assert_eq!(names.len(), 15, "fixture set incomplete");
    let cases = names
        .into_iter()
        .map(|n| {
            let t = st::load(&dir.join(&n).join("fixture.safetensors")).unwrap();
            let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(&n).join("meta.json")).unwrap()).unwrap();
            let f = |k: &str| t[k].f32().unwrap().to_vec();
            Case {
                ids: t["input_ids"].i64().unwrap().to_vec(),
                ref_s: f("ref_s"),
                speed: meta["speed"].as_f64().unwrap() as f32,
                rand_ini: f("noise.rand_ini").try_into().unwrap(),
                sine: f("noise.sine"),
                name: n,
            }
        })
        .collect();
    let bounds = serde_json::from_str(&std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/pinned/regression_bounds.json")).unwrap()).unwrap();
    (m, gm, cases, bounds)
}

fn single(m: &Kokoro, gm: &GpuKokoro, c: &Case) -> Output {
    gm.forward_ids(m, &c.ids, &c.ref_s, c.speed, &mut FixedNoise { rand_ini: c.rand_ini, sine_noise: c.sine.clone() }).unwrap()
}

fn batch(m: &Kokoro, gm: &GpuKokoro, cs: &[&Case]) -> Vec<Output> {
    let items: Vec<BatchItem> = cs
        .iter()
        .map(|c| BatchItem { ids: &c.ids, ref_s: &c.ref_s, speed: c.speed, noise: ItemNoise::Fixed { rand_ini: c.rand_ini, sine: c.sine.clone() } })
        .collect();
    gm.forward_batch(m, &items).unwrap()
}

fn drift(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut d2, mut w2, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = x as f64 - y as f64;
        d2 += d * d;
        w2 += (y as f64).powi(2);
        mx = mx.max(d.abs());
    }
    ((d2 / w2).sqrt(), mx, common_gates::spec_mean_abs_db(a, b))
}

/// Returns violations (empty = within bounds) for a batch evaluated against single-item outputs.
fn check(cs: &[&Case], outs: &[Output], singles: &std::collections::HashMap<String, Output>, bounds: &serde_json::Value, label: &str) -> Vec<String> {
    let mut v = vec![];
    // cardinality first: never let zip() silently drop a missing or extra output
    if outs.len() != cs.len() {
        v.push(format!("{label}: {} outputs for {} inputs", outs.len(), cs.len()));
        return v;
    }
    for (c, o) in cs.iter().zip(outs) {
        let s = &singles[&c.name];
        if o.pred_dur != s.pred_dur {
            v.push(format!("{label}/{}: durations differ", c.name));
            continue;
        }
        if o.audio.len() != s.audio.len() {
            v.push(format!("{label}/{}: sample count {} vs {}", c.name, o.audio.len(), s.audio.len()));
            continue;
        }
        let (r, mx, sp) = drift(&o.audio, &s.audio);
        let b = &bounds["cases"][&c.name];
        let ok = r <= b["rel_l2"].as_f64().unwrap() && mx <= b["max_abs"].as_f64().unwrap() && sp <= b["spec_db"].as_f64().unwrap();
        println!("{label:<22} {:<32} rel {r:.2e} max {mx:.2e} spec {sp:.4} {}", c.name, if ok { "ok" } else { "VIOLATION" });
        if !ok {
            v.push(format!("{label}/{}: rel {r:.3e} max {mx:.3e} spec {sp:.4}", c.name));
        }
    }
    v
}

#[test]
fn batched_matches_single_item_within_rb1() {
    let (m, gm, cases, bounds) = setup();
    let singles: std::collections::HashMap<String, Output> = cases.iter().map(|c| (c.name.clone(), single(&m, &gm, c))).collect();
    let all: Vec<&Case> = cases.iter().collect();
    let mut viol = check(&all, &batch(&m, &gm, &all), &singles, &bounds, "all15");
    let rev: Vec<&Case> = all.iter().rev().copied().collect();
    viol.extend(check(&rev, &batch(&m, &gm, &rev), &singles, &bounds, "reversed"));
    // deterministic shuffle + sub-batches of sizes 1, 2, 4, 7 and a tail
    let mut sh = all.clone();
    let mut x = 12345u64;
    for i in (1..sh.len()).rev() {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        sh.swap(i, (x >> 33) as usize % (i + 1));
    }
    let mut i = 0;
    for size in [1usize, 2, 4, 7, 1] {
        let part = &sh[i..(i + size).min(sh.len())];
        viol.extend(check(part, &batch(&m, &gm, part), &singles, &bounds, &format!("shuffled/b{size}")));
        i += size;
    }
    assert!(viol.is_empty(), "batched output outside RB-1 bounds: {viol:#?}");
}

#[test]
fn batching_negative_controls_are_detected() {
    let (m, gm, cases, bounds) = setup();
    let singles: std::collections::HashMap<String, Output> = cases.iter().map(|c| (c.name.clone(), single(&m, &gm, c))).collect();
    // a short item next to the longest, loudest one: gap masking must isolate them
    let short = cases.iter().find(|c| c.name.starts_with("s05_word__af_heart")).unwrap();
    let long = cases.iter().find(|c| c.name.starts_with("s06_long")).unwrap();
    let pair = [short, long];
    assert!(check(&pair, &batch(&m, &gm, &pair), &singles, &bounds, "control").is_empty(), "control pair must pass");

    std::env::set_var("KOKORO_BATCH_NEGCTL_NOMASK", "1");
    let v = check(&pair, &batch(&m, &gm, &pair), &singles, &bounds, "negctl/nomask");
    std::env::remove_var("KOKORO_BATCH_NEGCTL_NOMASK");
    assert!(!v.is_empty(), "disabled gap masking (padding contamination) NOT detected");

    // style swapped between the two items -> must be detected
    // (durations change under a style swap, so counter noise is used; the exact duration gate judges it)
    let items = vec![
        BatchItem { ids: &short.ids, ref_s: &long.ref_s, speed: short.speed, noise: ItemNoise::Counter(1) },
        BatchItem { ids: &long.ids, ref_s: &short.ref_s, speed: long.speed, noise: ItemNoise::Counter(2) },
    ];
    let outs = gm.forward_batch(&m, &items).unwrap();
    assert!(!check(&pair, &outs, &singles, &bounds, "negctl/style").is_empty(), "style swap NOT detected");

    // noise stream of another item (same length requirement: use a same-case duplicate with a
    // shifted stream) -> must be detected
    let mut shifted = short.sine.clone();
    shifted.rotate_left(HARMONICS * 997);
    let items = vec![BatchItem { ids: &short.ids, ref_s: &short.ref_s, speed: short.speed, noise: ItemNoise::Fixed { rand_ini: short.rand_ini, sine: shifted } }];
    let outs = gm.forward_batch(&m, &items).unwrap();
    assert!(!check(&[short], &outs, &singles, &bounds, "negctl/noise").is_empty(), "foreign noise stream NOT detected");
}

/// Diagnostic: attribute batch-vs-single waveform drift to the F0 curve (the known amplifier).
#[test]
#[ignore = "diagnostic; run explicitly"]
fn diag_f0_batch_vs_single() {
    let (m, gm, cases, _) = setup();
    let dir = data().join("tmp/diag_f0");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_var("KOKORO_DEBUG_F0_DIR", &dir);
    for c in &cases {
        single(&m, &gm, c);
    }
    let all: Vec<&Case> = cases.iter().collect();
    batch(&m, &gm, &all);
    std::env::remove_var("KOKORO_DEBUG_F0_DIR");
    let read = |p: PathBuf| -> Vec<f32> { std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect() };
    for c in &cases {
        use sha2::Digest;
        let key: String = sha2::Sha256::digest(c.ids.iter().flat_map(|v| v.to_le_bytes()).chain(c.ref_s.iter().flat_map(|v| v.to_le_bytes())).chain(c.speed.to_le_bytes()).collect::<Vec<u8>>()).iter().map(|b| format!("{b:02x}")).collect();
        let s = read(dir.join(format!("single-{}.f32", &key[..16])));
        let b = read(dir.join(format!("batch-{}.f32", &key[..16])));
        assert_eq!(s.len(), b.len());
        let (mut d2, mut w2, mut mx) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in b.iter().zip(&s) {
            d2 += (x as f64 - y as f64).powi(2);
            w2 += (y as f64).powi(2);
            mx = mx.max((x as f64 - y as f64).abs());
        }
        println!("{:<32} F0 batch-vs-single rel {:.3e} max {mx:.3e} Hz", c.name, (d2 / w2).sqrt());
    }
}

/// Diagnostic (owner #11 evaluation): is batched output any farther from the pinned reference
/// than single-item output? Same frozen inputs/noise; reference = cpu-t1 fixture audio.
#[test]
#[ignore = "diagnostic; run explicitly"]
fn diag_batch_vs_reference() {
    let (m, gm, cases, _) = setup();
    let dir = data().join("fixtures/cpu-t1");
    let all: Vec<&Case> = cases.iter().collect();
    let outs = batch(&m, &gm, &all);
    assert_eq!(outs.len(), cases.len(), "batched output count");
    let spec_gate = {
        let t1 = st::load(&dir.join("s02_fox__af_heart__s1.0/fixture.safetensors")).unwrap()["audio"].f32().unwrap().to_vec();
        let t8 = st::load(&data().join("fixtures/f64/s02_fox__af_heart__s1.0/f64.safetensors")).unwrap()["audio_f32_t8"].f32().unwrap().to_vec();
        2.0 * common_gates::spec_mean_abs_db(&t1, &t8)
    };
    let corr = |a: &[f32], b: &[f32]| {
        let n = a.len() as f64;
        let (sa, sb) = (a.iter().map(|&x| x as f64).sum::<f64>() / n, b.iter().map(|&x| x as f64).sum::<f64>() / n);
        let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
        for (&x, &y) in a.iter().zip(b) {
            c += (x as f64 - sa) * (y as f64 - sb);
            va += (x as f64 - sa).powi(2);
            vb += (y as f64 - sb).powi(2);
        }
        c / (va * vb).sqrt()
    };
    let v1 = |r: f64, mx: f64, co: f64, sp: f64| r <= 0.019 && mx <= 0.033 && co >= 0.9995 && sp <= spec_gate;
    let (mut fs, mut fb) = (0, 0);
    let (mut sum_s, mut sum_b) = ([0.0f64; 3], [0.0f64; 3]);
    println!("{:<32} {:>28} | {:>28}", "case (vs reference cpu-t1)", "single: rel / max / spec", "batched(15): rel / max / spec");
    for (c, o) in cases.iter().zip(&outs) {
        let want = st::load(&dir.join(&c.name).join("fixture.safetensors")).unwrap()["audio"].f32().unwrap().to_vec();
        let s = single(&m, &gm, c);
        assert_eq!(o.audio.len(), want.len(), "{}: batched sample count", c.name);
        assert_eq!(s.audio.len(), want.len(), "{}: single sample count", c.name);
        let (rs, ms, ss) = drift(&s.audio, &want);
        let (rb, mb, sb) = drift(&o.audio, &want);
        let (cs, cb) = (corr(&s.audio, &want), corr(&o.audio, &want));
        fs += !v1(rs, ms, cs, ss) as usize;
        fb += !v1(rb, mb, cb, sb) as usize;
        for (acc, v) in [(&mut sum_s, [rs, ms, ss]), (&mut sum_b, [rb, mb, sb])] {
            for k in 0..3 {
                acc[k] += v[k];
            }
        }
        println!("{:<32} {rs:>8.4} {ms:>8.4} {ss:>8.4}     | {rb:>8.4} {mb:>8.4} {sb:>8.4}", c.name);
    }
    let n = cases.len() as f64;
    println!("mean single  rel {:.4} max {:.4} spec {:.4}; binding-gate failures {fs}/15", sum_s[0] / n, sum_s[1] / n, sum_s[2] / n);
    println!("mean batched rel {:.4} max {:.4} spec {:.4}; binding-gate failures {fb}/15", sum_b[0] / n, sum_b[1] / n, sum_b[2] / n);
}

/// Owner listening escalation (owner #11): worst batched peak divergence vs the reference.
/// Writes raw float32 WAVs (no normalization/trim): reference cpu-t1, Rust single, Rust batched(15).
#[test]
#[ignore = "export for owner listening; run explicitly"]
fn export_batch_worst_listening() {
    let (m, gm, cases, _) = setup();
    let all: Vec<&Case> = cases.iter().collect();
    let outs = batch(&m, &gm, &all);
    let k = cases.iter().position(|c| c.name == "s04_alice__am_adam__s1.0").unwrap();
    let c = &cases[k];
    let want = st::load(&data().join("fixtures/cpu-t1").join(&c.name).join("fixture.safetensors")).unwrap()["audio"].f32().unwrap().to_vec();
    let s = single(&m, &gm, c);
    let dir = data().join("evidence/listening/batch-worst-s04_alice_am_adam");
    std::fs::create_dir_all(&dir).unwrap();
    let mut manifest = serde_json::json!({"case": c.name, "note": "owner escalation: batched(15) peak vs reference 0.0706 > accepted 0.0464; not an acceptance artifact",
        "format": "RIFF WAVE IEEE float32 mono 24 kHz, raw"});
    for (name, audio) in [("reference-cpu-t1", &want), ("rust-cuda-single", &s.audio), ("rust-cuda-batched15", &outs[k].audio)] {
        let enc = kokoro::wav::encode(audio, 24000, kokoro::wav::Format::Float32);
        std::fs::write(dir.join(format!("{name}.wav")), &enc.bytes).unwrap();
        let (r, mx, sp) = drift(audio, &want);
        manifest[name] = serde_json::json!({"sha256": kokoro::engine::sha256_bytes(&enc.bytes), "vs_reference": {"rel": r, "max": mx, "spec_db": sp}, "samples": audio.len()});
    }
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    println!("{}", serde_json::to_string_pretty(&manifest).unwrap());
}

/// Owner listening escalation (owner #11): worst FMA-build peak vs the reference (s03_moon/am_adam).
/// reference cpu-t1, Rust strict (approved pinned baseline audio), Rust current build (FMA).
#[test]
#[ignore = "export for owner listening; run explicitly"]
fn export_fma_worst_listening() {
    let (m, gm, cases, _) = setup();
    let c = cases.iter().find(|c| c.name == "s03_moon__am_adam__s1.0").unwrap();
    let want = st::load(&data().join("fixtures/cpu-t1").join(&c.name).join("fixture.safetensors")).unwrap()["audio"].f32().unwrap().to_vec();
    let strict: Vec<f32> = std::fs::read(data().join(format!("evidence/pinned-baseline/{}.f32le", c.name))).unwrap()
        .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let cur = single(&m, &gm, c);
    let dir = data().join("evidence/listening/fma-worst-s03_moon_am_adam");
    std::fs::create_dir_all(&dir).unwrap();
    let mut manifest = serde_json::json!({"case": c.name, "kernel_rounding": kokoro::gpu::KERNEL_ROUNDING,
        "note": "owner escalation: FMA build peak vs reference 0.054 > accepted 0.046; not an acceptance artifact",
        "format": "RIFF WAVE IEEE float32 mono 24 kHz, raw"});
    for (name, audio) in [("reference-cpu-t1", &want), ("rust-cuda-strict-approved", &strict), ("rust-cuda-fma", &cur.audio)] {
        let enc = kokoro::wav::encode(audio, 24000, kokoro::wav::Format::Float32);
        std::fs::write(dir.join(format!("{name}.wav")), &enc.bytes).unwrap();
        let (r, mx, sp) = drift(audio, &want);
        manifest[name] = serde_json::json!({"sha256": kokoro::engine::sha256_bytes(&enc.bytes), "vs_reference": {"rel": r, "max": mx, "spec_db": sp}});
    }
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    println!("{}", serde_json::to_string_pretty(&manifest).unwrap());
}


/// PL-014 audit (supervisor): the two batched (case, metric) failure-set additions, as raw WAVs.
/// Run once with KOKORO_STATS_1PASS=0 (two-pass statistics) and once with =1 (single-pass, the
/// default); the switch is read once per process. Writes reference cpu-t1, Rust single (unchanged
/// by PL-014) and Rust batched(15) under the current statistics mode, plus a per-mode manifest.
#[test]
#[ignore = "export for owner listening; run explicitly"]
fn export_pl014_pairs_listening() {
    let mode = if std::env::var("KOKORO_STATS_1PASS").map(|v| v != "0").unwrap_or(true) { "stats1pass" } else { "stats2pass" };
    let (m, gm, cases, _) = setup();
    let all: Vec<&Case> = cases.iter().collect();
    let outs = batch(&m, &gm, &all);
    for name in ["s03_moon__am_adam__s1.0", "s06_long__af_heart__s1.0"] {
        let k = cases.iter().position(|c| c.name == name).unwrap();
        let c = &cases[k];
        let want = st::load(&data().join("fixtures/cpu-t1").join(&c.name).join("fixture.safetensors")).unwrap()["audio"].f32().unwrap().to_vec();
        let s = single(&m, &gm, c);
        let dir = data().join("evidence/listening/phase2-pack/pl014-pairs").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = serde_json::json!({"case": c.name, "mode": mode, "kernel_rounding": kokoro::gpu::KERNEL_ROUNDING,
            "note": "PL-014 batched failure-set addition (supervisor audit); disclosed, not an acceptance artifact",
            "format": "RIFF WAVE IEEE float32 mono 24 kHz, raw"});
        let bname = format!("rust-cuda-batched15-{mode}");
        for (label, audio) in [("reference-cpu-t1", &want), ("rust-cuda-single", &s.audio), (bname.as_str(), &outs[k].audio)] {
            let enc = kokoro::wav::encode(audio, 24000, kokoro::wav::Format::Float32);
            std::fs::write(dir.join(format!("{label}.wav")), &enc.bytes).unwrap();
            let (r, mx, sp) = drift(audio, &want);
            manifest[label] = serde_json::json!({"sha256": kokoro::engine::sha256_bytes(&enc.bytes), "vs_reference": {"rel": r, "max": mx, "spec_db": sp}, "samples": audio.len()});
        }
        std::fs::write(dir.join(format!("manifest-{mode}.json")), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
        println!("{}", serde_json::to_string_pretty(&manifest).unwrap());
    }
}

/// Mapping negative controls: the checker must catch a missing output, a duplicated output and a
/// swapped (misattributed) output — i.e. batch result -> item mapping errors.
#[test]
fn batch_mapping_negative_controls_are_detected() {
    let (m, gm, cases, bounds) = setup();
    let singles: std::collections::HashMap<String, Output> = cases.iter().map(|c| (c.name.clone(), single(&m, &gm, c))).collect();
    let three: Vec<&Case> = cases.iter().filter(|c| c.name.ends_with("af_heart__s1.0")).take(3).collect();
    assert_eq!(three.len(), 3);
    let _ = &bounds;
    let outs = batch(&m, &gm, &three);
    // Mapping is a DISCRETE property: count, per-item durations and sample counts must match the
    // single-item run exactly (waveform bounds are a separate, escalated question).
    let mapping = |outs: &[Output]| -> Vec<String> {
        let mut v = vec![];
        if outs.len() != three.len() {
            return vec![format!("{} outputs for {} inputs", outs.len(), three.len())];
        }
        for (c, o) in three.iter().zip(outs) {
            let s = &singles[&c.name];
            if o.pred_dur != s.pred_dur || o.audio.len() != s.audio.len() {
                v.push(format!("{}: output does not belong to this item", c.name));
            }
        }
        v
    };
    assert!(mapping(&outs).is_empty(), "control mapping must be exact: {:?}", mapping(&outs));
    let clone = |o: &Output| Output { audio: o.audio.clone(), pred_dur: o.pred_dur.clone() };
    let missing: Vec<Output> = outs.iter().take(2).map(clone).collect();
    assert!(!mapping(&missing).is_empty(), "missing output NOT detected");
    let dup: Vec<Output> = vec![clone(&outs[0]), clone(&outs[0]), clone(&outs[2])];
    assert!(!mapping(&dup).is_empty(), "duplicate/misassigned output NOT detected");
    let swapped: Vec<Output> = vec![clone(&outs[1]), clone(&outs[0]), clone(&outs[2])];
    assert!(!mapping(&swapped).is_empty(), "swapped outputs NOT detected");
}
