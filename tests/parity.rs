//! Oracle parity ladder: every stage is fed the ORACLE's exact input tensor (from frozen
//! fixtures produced by the pinned PyTorch reference) and compared to the oracle's output.
//! Missing fixtures FAIL (skip-honest); expected values never come from this crate.
//!
//! Run: source scripts/env.sh && cargo test --release --test parity -- --nocapture

use kokoro::model::{self, Kokoro};
use kokoro::st::{self, TensorMap};
use kokoro::vocoder::{self, FixedNoise, HARMONICS};
use kokoro::weights::Weights;
use std::path::PathBuf;
use std::sync::OnceLock;

#[path = "common_gates.rs"]
mod common_gates;

fn data_root() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

fn model() -> &'static Kokoro {
    static M: OnceLock<Kokoro> = OnceLock::new();
    M.get_or_init(|| {
        // Product loading path: the upstream checkpoint read natively (bitwise-equal to the
        // reference's loaded state; tests/native_load.rs).
        let snap = data_root().join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987");
        let w = Weights::load_pth(&snap.join("kokoro-v1_0.pth")).expect("kokoro-v1_0.pth");
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(snap.join("config.json")).unwrap()).unwrap();
        Kokoro::from_weights(&w, &cfg).expect("model hydrate")
    })
}

struct Fixture {
    name: String,
    t: TensorMap,
    meta: serde_json::Value,
}

impl Fixture {
    fn f(&self, k: &str) -> Vec<f32> {
        self.t.get(k).unwrap_or_else(|| panic!("{}: missing {k}", self.name)).f32().unwrap().to_vec()
    }
    fn i(&self, k: &str) -> Vec<i64> {
        self.t.get(k).unwrap_or_else(|| panic!("{}: missing {k}", self.name)).i64().unwrap().to_vec()
    }
    fn has(&self, k: &str) -> bool {
        self.t.contains_key(k)
    }
}

fn fixtures(device: &str) -> Vec<Fixture> {
    let dir = data_root().join("fixtures").join(device);
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("fixture dir {} unreadable: {e} — ladder cannot run (NOT a pass)", dir.display()))
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no fixtures in {} — ladder cannot run (NOT a pass)", dir.display());
    let filter = std::env::var("KOKORO_CASE").ok();
    let names: Vec<String> = names.into_iter().filter(|n| filter.as_ref().map(|c| n.contains(c.as_str())).unwrap_or(true)).collect();
    assert!(!names.is_empty(), "case filter {filter:?} selected ZERO fixtures — NOT a pass");
    if filter.is_none() {
        assert_eq!(names.len(), EXPECTED_CASES, "fixture set {device} is incomplete");
    }
    names
        .into_iter()
        .map(|n| {
            let p = dir.join(&n);
            Fixture {
                t: st::load(&p.join("fixture.safetensors")).unwrap(),
                meta: serde_json::from_str(&std::fs::read_to_string(p.join("meta.json")).unwrap()).unwrap(),
                name: n,
            }
        })
        .collect()
}

#[derive(Debug, Clone, serde::Serialize)]
struct Cmp {
    case: String,
    seam: String,
    n: usize,
    rel_l2: f64,
    max_abs: f64,
    ref_rms: f64,
    corr: f64,
    gate: f64,
    pass: bool,
}

fn compare(case: &str, seam: &str, got: &[f32], want: &[f32], gate_rel: f64) -> Cmp {
    assert_eq!(got.len(), want.len(), "{case}/{seam}: length {} vs oracle {}", got.len(), want.len());
    let n = want.len();
    let (mut d2, mut w2, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sg, mut sw, mut sgg, mut sww, mut sgw) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (g as f64, w as f64);
        let d = g - w;
        d2 += d * d;
        w2 += w * w;
        mx = mx.max(d.abs());
        sg += g;
        sw += w;
        sgg += g * g;
        sww += w * w;
        sgw += g * w;
    }
    let nf = n as f64;
    let cov = sgw / nf - (sg / nf) * (sw / nf);
    let corr = cov / ((sgg / nf - (sg / nf).powi(2)).sqrt() * (sww / nf - (sw / nf).powi(2)).sqrt());
    let rel = if w2 > 0.0 { (d2 / w2).sqrt() } else { d2.sqrt() };
    let ok = rel <= gate_rel && got.iter().all(|v| v.is_finite());
    Cmp { case: case.into(), seam: seam.into(), n, rel_l2: rel, max_abs: mx, ref_rms: (w2 / nf).sqrt(), corr, gate: gate_rel, pass: ok }
}

/// Angle seam: element error is the wrapped angular distance (±π are the same angle).
/// Branch flips (|raw Δ| > π, i.e. the reference's rounding-noise sign of a zero imaginary
/// part) are counted into the seam label; see DISCREPANCIES.md DISC-001.
fn compare_angle(case: &str, seam: &str, got: &[f32], want: &[f32], gate_rel: f64) -> Cmp {
    let pi = std::f32::consts::PI;
    let mut flips = 0usize;
    let wrapped: Vec<f32> = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| {
            let mut d = g - w;
            if d.abs() > pi {
                flips += 1;
                d -= (2.0 * pi).copysign(d);
            }
            w + d
        })
        .collect();
    compare(case, &format!("{seam} [flips={flips}]"), &wrapped, want, gate_rel)
}

fn floors() -> &'static serde_json::Value {
    static F: OnceLock<serde_json::Value> = OnceLock::new();
    F.get_or_init(|| {
        let p = data_root().join("fixtures/floor_per_case.json");
        serde_json::from_str(&std::fs::read_to_string(&p).expect("run oracle/floor_all.py (per-case floor)")).unwrap()
    })
}

/// G-E2E-v2 (NONDET_FLOOR.md Revision 1): rel ≤ 2·floor_rel(case), max ≤ 2·floor_max(case), corr ≥ 0.9995.
fn gate_e2e_v2(case: &str, got: &[f32], want: &[f32]) -> Cmp {
    let f = &floors()["cases"][case];
    let (fr, fm) = (f["floor_rel"].as_f64().expect("floor_rel"), f["floor_max"].as_f64().expect("floor_max"));
    let mut r = compare(case, "E2E waveform [G-v2 UNAPPROVED diagnostic]", got, want, 2.0 * fr);
    r.pass = r.pass && r.max_abs <= 2.0 * fm && r.corr >= 0.9995;
    r
}

/// Stage-isolated linear-algebra seam gate (docs/conformance/NONDET_FLOOR.md).
const SEAM_GATE: f64 = 1e-4;

/// Corpus size: 1+1 (s01) + 6 (s02) + 2 (s03) + 2 (s04) + 2 (s05) + 1 (s06).
const EXPECTED_CASES: usize = 15;

/// Every case must produce each of these seams (prefix match); missing coverage fails.
const REQUIRED_SEAMS: &[&str] = &[
    "bert", "bert_encoder", "dur_enc", "pred_lstm", "duration_proj", "F0_pred", "N_pred", "text_encoder",
    "decoder.pre_generator", "har_source", "stft.mag", "stft.phase", "generator(oracle in)", "istft(oracle in)",
    "E2E waveform [G-v1", "E2E spectral",
];

/// ORIGINAL precommitted E2E gate (NONDET_FLOOR.md, before any subject result): rel ≤ 1.9%,
/// max ≤ 3.3e-2, corr ≥ 0.9995. This is the ENFORCED E2E gate; no revision is approved.
fn gate_e2e_v1(case: &str, got: &[f32], want: &[f32]) -> Cmp {
    let mut r = compare(case, "E2E waveform [G-v1 original, enforced]", got, want, 0.019);
    r.pass = r.pass && r.max_abs <= 3.3e-2 && r.corr >= 0.9995;
    r
}

/// G-SPEC floor: 2 × mean|ΔdB|(reference cpu-t1, reference cpu-t8) on the floor case.
fn spec_gate() -> f64 {
    static G: OnceLock<f64> = OnceLock::new();
    *G.get_or_init(|| {
        let c = common_gates::FLOOR_CASE;
        let t1 = st::load(&data_root().join(format!("fixtures/cpu-t1/{c}/fixture.safetensors"))).unwrap()["audio"].f32().unwrap().to_vec();
        let t8 = st::load(&data_root().join(format!("fixtures/f64/{c}/f64.safetensors"))).expect("run oracle/gen_f64.py")["audio_f32_t8"].f32().unwrap().to_vec();
        let floor = common_gates::spec_mean_abs_db(&t1, &t8);
        println!("G-SPEC floor pair (reference cpu-t1 vs cpu-t8, {c}): {floor:.6} dB -> gate {:.6} dB", 2.0 * floor);
        2.0 * floor
    })
}

fn gate_spec(case: &str, got: &[f32], want: &[f32]) -> Cmp {
    let v = common_gates::spec_mean_abs_db(got, want);
    let g = spec_gate();
    Cmp { case: case.into(), seam: "E2E spectral mean|dB| [G-SPEC original, enforced]".into(), n: got.len(), rel_l2: v, max_abs: 0.0, ref_rms: 0.0, corr: f64::NAN, gate: g, pass: v <= g }
}

fn unapproved(r: &Cmp) -> bool {
    r.seam.contains("UNAPPROVED")
}

fn run_ladder(device: &str) -> Vec<Cmp> {
    let m = model();
    let mut rows = vec![];
    for fx in fixtures(device) {
        let c = fx.name.as_str();
        let ps = fx.meta["phonemes"].as_str().unwrap();
        let speed = fx.meta["speed"].as_f64().unwrap() as f32;

        // L0: phonemes -> ids EXACT
        let (ids, dropped) = m.phonemes_to_ids(ps);
        let want_ids = fx.i("input_ids");
        assert_eq!(ids, want_ids, "{c}: input_ids mismatch");
        assert!(dropped.is_empty(), "{c}: dropped phoneme chars {dropped:?}");
        let t = ids.len();
        let ref_s = fx.f("ref_s");
        let (s_dec, s_pro) = ref_s.split_at(model::STYLE_DIM);

        // L1 seams in pipeline order, each fed the oracle's own input
        if fx.has("seam.albert_embeddings") {
            rows.push(compare(c, "albert.embeddings", &m.albert.embeddings(&ids).unwrap(), &fx.f("seam.albert_embeddings"), SEAM_GATE));
            let l0in = fx.f("seam.albert_embed_proj");
            rows.push(compare(c, "albert.layer0(oracle in)", &m.albert.layer(&l0in, t), &fx.f("seam.albert_layer0"), SEAM_GATE));
        }
        rows.push(compare(c, "bert", &m.albert.forward(&ids).unwrap(), &fx.f("seam.bert"), SEAM_GATE));
        rows.push(compare(c, "bert_encoder", &m.bert_encoder.forward(&fx.f("seam.bert"), t), &fx.f("seam.bert_encoder"), SEAM_GATE));
        rows.push(compare(c, "dur_enc", &m.predictor.duration_encoder(&fx.f("seam.bert_encoder"), t, s_pro), &fx.f("seam.dur_enc"), SEAM_GATE));
        rows.push(compare(c, "pred_lstm", &m.predictor.dur_lstm(&fx.f("seam.dur_enc"), t), &fx.f("seam.pred_lstm"), SEAM_GATE));
        rows.push(compare(c, "duration_proj", &m.predictor.duration_logits(&fx.f("seam.pred_lstm"), t), &fx.f("seam.duration_proj"), SEAM_GATE));

        // durations from the ORACLE's logits must be EXACT
        let pd = model::durations_from_logits(&fx.f("seam.duration_proj"), t, speed);
        assert_eq!(pd, fx.i("pred_dur"), "{c}: durations from oracle logits differ");
        let nf: usize = pd.iter().sum::<i64>() as usize;

        let shared_in = fx.f("seam.shared.arg0");
        let (f0, n) = m.predictor.f0n(&shared_in, nf, s_pro);
        rows.push(compare(c, "F0_pred", &f0, &fx.f("seam.decoder.arg1"), SEAM_GATE));
        rows.push(compare(c, "N_pred", &n, &fx.f("seam.decoder.arg2"), SEAM_GATE));
        rows.push(compare(c, "text_encoder", &m.text_encoder.forward(&ids).unwrap(), &fx.f("seam.text_encoder"), SEAM_GATE));

        let (pre, t2) = m
            .decoder
            .pre_generator(&fx.f("seam.decoder.arg0"), nf, &fx.f("seam.decoder.arg1"), &fx.f("seam.decoder.arg2"), s_dec)
            .unwrap();
        assert_eq!(t2, 2 * nf);
        rows.push(compare(c, "decoder.pre_generator", &pre, &fx.f("seam.gen.arg0"), SEAM_GATE));

        // vocoder source path with the oracle's recorded noise
        let rand_ini: [f32; HARMONICS] = fx.f("noise.rand_ini").try_into().unwrap();
        let mut noise = FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") };
        let gen = &m.decoder.generator;
        let har_src = gen.har_source(&fx.f("seam.gen.arg2"), &mut noise).unwrap();
        rows.push(compare(c, "har_source", &har_src, &fx.f("seam.har_source"), SEAM_GATE));
        let (mag, ph, frames) = vocoder::stft(&fx.f("seam.har_source"));
        rows.push(compare(c, "stft.mag", &mag, &fx.f("seam.har_spec"), SEAM_GATE));
        rows.push(compare_angle(c, "stft.phase", &ph, &fx.f("seam.har_phase"), SEAM_GATE));

        let mut har = fx.f("seam.har_spec");
        har.extend(fx.f("seam.har_phase"));
        let audio_g = gen.forward_from_har(&fx.f("seam.gen.arg0"), 2 * nf, s_dec, &har, frames).unwrap();
        rows.push(compare(c, "generator(oracle in)", &audio_g, &fx.f("audio"), SEAM_GATE));

        let istft_only = vocoder::istft(&fx.f("seam.gen_spec"), &fx.f("seam.gen_phase"), frames);
        rows.push(compare(c, "istft(oracle in)", &istft_only, &fx.f("audio"), SEAM_GATE));

        // L3 end-to-end with frozen noise: discrete EXACT + waveform floor-derived gates
        let mut noise = FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") };
        let out = m.forward_ids(&ids, &ref_s, speed, &mut noise).unwrap();
        assert_eq!(out.pred_dur, fx.i("pred_dur"), "{c}: e2e durations differ");
        let want = fx.f("audio");
        assert_eq!(out.audio.len(), want.len(), "{c}: e2e sample count differs");
        rows.push(gate_e2e_v1(c, &out.audio, &want));
        rows.push(gate_spec(c, &out.audio, &want));
        rows.push(gate_e2e_v2(c, &out.audio, &want));
    }
    rows
}

fn report(rows: &[Cmp], tag: &str) {
    println!("\n{:<34} {:<26} {:>9} {:>10} {:>10} {:>9} {:>8}", "case", "seam", "n", "rel_l2", "max_abs", "corr", "verdict");
    for r in rows {
        println!(
            "{:<34} {:<26} {:>9} {:>10.3e} {:>10.3e} {:>9.6} {:>8}",
            r.case, r.seam, r.n, r.rel_l2, r.max_abs, r.corr,
            match (unapproved(r), r.pass) {
                (true, true) => "diag-ok",
                (true, false) => "diag-FAIL",
                (false, true) => "PASS",
                (false, false) => "FAIL",
            }
        );
    }
    let dir = data_root().join("evidence/ladder");
    std::fs::create_dir_all(&dir).unwrap();
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let path = dir.join(format!("{tag}-{ts}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(rows).unwrap()).unwrap();
    println!("receipt: {}", path.display());
}

#[test]
fn ladder_cpu_f32() {
    let rows = run_ladder("cpu-t1");
    report(&rows, "ladder-cpu-t1");
    let mut cases: Vec<&str> = rows.iter().map(|r| r.case.as_str()).collect();
    cases.dedup();
    for c in &cases {
        for req in REQUIRED_SEAMS {
            assert!(rows.iter().any(|r| r.case == *c && r.seam.starts_with(req)), "{c}: required seam {req} missing");
        }
    }
    println!("coverage: {} cases x {} required seams present", cases.len(), REQUIRED_SEAMS.len());
    let fails: Vec<_> = rows.iter().filter(|r| !r.pass && !unapproved(r)).map(|r| format!("{}/{}", r.case, r.seam)).collect();
    assert!(fails.is_empty(), "{} seam(s) failed: {fails:?}", fails.len());
}

/// Attribution: native pipeline with ONLY the oracle F0/N curves substituted. If E2E error
/// collapses, the E2E divergence is attributable to F0-integration amplification.
#[test]
fn e2e_attribution_f0() {
    let m = model();
    println!("\n{:<34} {:>12} {:>12} {:>12} {:>12}", "case", "native rel", "native max", "oF0 rel", "oF0 max");
    for fx in fixtures("cpu-t1") {
        let ps = fx.meta["phonemes"].as_str().unwrap();
        let (ids, _) = m.phonemes_to_ids(ps);
        let t = ids.len();
        let ref_s = fx.f("ref_s");
        let (s_dec, _) = ref_s.split_at(model::STYLE_DIM);
        let rand_ini: [f32; HARMONICS] = fx.f("noise.rand_ini").try_into().unwrap();
        let want = fx.f("audio");
        let speed = fx.meta["speed"].as_f64().unwrap() as f32;
        let mut noise = FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") };
        let native = m.forward_ids(&ids, &ref_s, speed, &mut noise).unwrap();
        let aln = model::alignment(&native.pred_dur);
        let nf = aln.len();
        let t_en = m.text_encoder.forward(&ids).unwrap();
        let mut asr = vec![0.0f32; 512 * nf];
        for c in 0..512 {
            for (f, &tok) in aln.iter().enumerate() {
                asr[c * nf + f] = t_en[c * t + tok];
            }
        }
        let mut noise = FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") };
        let a = m.decoder.forward(&asr, nf, &fx.f("seam.decoder.arg1"), &fx.f("seam.decoder.arg2"), s_dec, &mut noise).unwrap();
        let n = compare(&fx.name, "native", &native.audio, &want, 1.0);
        let o = compare(&fx.name, "oracleF0", &a, &want, 1.0);
        println!("{:<34} {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e}", fx.name, n.rel_l2, n.max_abs, o.rel_l2, o.max_abs);
    }
}

/// Distance to the float64 "true-math" oracle (oracle/gen_f64.py): the Rust f32 subject vs the
/// pinned torch f32 reference at thread counts {1,2,4,8}, all measured against exact arithmetic.
#[test]
fn truth_distance() {
    let m = model();
    let root = data_root().join("fixtures/f64");
    println!("\n{:<32} {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>9} {:>9}",
        "case", "rust rel", "torch min", "torch max", "rust max", "tmax min", "tmax max", "F0 rust", "F0 torch");
    let mut worse = 0;
    for fx in fixtures("cpu-t1") {
        let tp = st::load(&root.join(&fx.name).join("f64.safetensors")).expect("run oracle/gen_f64.py");
        let truth = tp["audio_f64"].f64().unwrap().to_vec();
        let f0_truth = tp["F0_f64"].f64().unwrap().to_vec();
        let ps = fx.meta["phonemes"].as_str().unwrap();
        let (ids, _) = m.phonemes_to_ids(ps);
        let rand_ini: [f32; HARMONICS] = fx.f("noise.rand_ini").try_into().unwrap();
        let mut noise = FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") };
        let speed = fx.meta["speed"].as_f64().unwrap() as f32;
        let out = m.forward_ids(&ids, &fx.f("ref_s"), speed, &mut noise).unwrap();
        assert_eq!(out.pred_dur, tp["pred_dur_f64"].i64().unwrap().to_vec(), "{}: durations vs f64 truth", fx.name);
        let dist = |x: &[f32]| {
            let (mut d2, mut t2, mut mx) = (0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in x.iter().zip(&truth) {
                let d = a as f64 - b;
                d2 += d * d;
                t2 += b * b;
                mx = mx.max(d.abs());
            }
            ((d2 / t2).sqrt(), mx)
        };
        let (rr, rm) = dist(&out.audio);
        let torch: Vec<(f64, f64)> = [1, 2, 4, 8].iter().map(|t| dist(tp[&format!("audio_f32_t{t}")].f32().unwrap())).collect();
        let (tmin, tmax) = torch.iter().fold((f64::MAX, 0.0f64), |a, b| (a.0.min(b.0), a.1.max(b.0)));
        let (mmin, mmax) = torch.iter().fold((f64::MAX, 0.0f64), |a, b| (a.0.min(b.1), a.1.max(b.1)));
        // F0 relative error vs truth: subject (native F0 from the full pipeline) and torch t1 (fixture seam)
        let nf: usize = out.pred_dur.iter().sum::<i64>() as usize;
        let t = ids.len();
        let d = {
            let bert = m.albert.forward(&ids).unwrap();
            let d_en = m.bert_encoder.forward(&bert, t);
            m.predictor.duration_encoder(&d_en, t, &fx.f("ref_s")[128..])
        };
        let aln = model::alignment(&out.pred_dur);
        let mut en = vec![0.0f32; nf * 640];
        for (f, &tok) in aln.iter().enumerate() {
            en[f * 640..(f + 1) * 640].copy_from_slice(&d[tok * 640..(tok + 1) * 640]);
        }
        let (f0, _) = m.predictor.f0n(&en, nf, &fx.f("ref_s")[128..]);
        let frel = |x: &[f32]| {
            let (mut d2, mut t2) = (0.0f64, 0.0f64);
            for (&a, &b) in x.iter().zip(&f0_truth) {
                d2 += (a as f64 - b).powi(2);
                t2 += b * b;
            }
            (d2 / t2).sqrt()
        };
        let f0r = frel(&f0);
        let f0t = frel(&fx.f("seam.decoder.arg1"));
        if rr > tmax {
            worse += 1;
        }
        println!("{:<32} {:>10.3e} {:>10.3e} {:>10.3e} | {:>10.3e} {:>10.3e} {:>10.3e} | {:>9.2e} {:>9.2e}",
            fx.name, rr, tmin, tmax, rm, mmin, mmax, f0r, f0t);
    }
    println!("cases where Rust is farther from truth (rel) than every torch reorder: {worse}");
}

/// The comparators must FAIL under deliberate perturbation (brief gate 4).
#[test]
fn perturbation_detected() {
    let m = model();
    let fx = fixtures("cpu-t1").into_iter().find(|f| f.name == "s02_fox__af_heart__s1.0").expect("case");
    let c = fx.name.as_str();
    let (ids, _) = m.phonemes_to_ids(fx.meta["phonemes"].as_str().unwrap());
    let ref_s = fx.f("ref_s");
    let rand_ini: [f32; HARMONICS] = fx.f("noise.rand_ini").try_into().unwrap();
    let want = fx.f("audio");
    let noise = || FixedNoise { rand_ini, sine_noise: fx.f("noise.sine") };

    // baseline passes
    let base = m.forward_ids(&ids, &ref_s, 1.0, &mut noise()).unwrap();
    assert!(gate_e2e_v2(c, &base.audio, &want).pass, "unperturbed baseline must pass");

    // 1. F0 curve scaled by 1.001 at the decoder input
    let nf: usize = base.pred_dur.iter().sum::<i64>() as usize;
    let f0: Vec<f32> = fx.f("seam.decoder.arg1").iter().map(|v| v * 1.001).collect();
    let a = m.decoder.forward(&fx.f("seam.decoder.arg0"), nf, &f0, &fx.f("seam.decoder.arg2"), &ref_s[..128], &mut noise()).unwrap();
    let r = gate_e2e_v2(c, &a, &want);
    println!("F0 x1.001: rel {:.3e} max {:.3e} -> {}", r.rel_l2, r.max_abs, r.pass);
    assert!(!r.pass, "F0 perturbation not detected");

    // 2a. rand_ini is provably DEAD: it is added only at sample 0, which the x1/300 linear
    //     downsample never reads (source index 300d+149.5). Output must be bit-identical.
    let mut ri = rand_ini;
    ri[3] = (ri[3] + 0.25) % 1.0;
    let a = m.forward_ids(&ids, &ref_s, 1.0, &mut FixedNoise { rand_ini: ri, sine_noise: fx.f("noise.sine") }).unwrap();
    assert_eq!(a.audio, base.audio, "rand_ini unexpectedly affects output");
    // 2b. excitation noise perturbed: harmonic-0 Gaussian noise scaled x1.5
    let mut sn = fx.f("noise.sine");
    for v in sn.iter_mut().step_by(HARMONICS) {
        *v *= 1.5;
    }
    //     The E2E waveform gate has NO power here (effect ~2.3% rel is inside the model's own
    //     f32 reorder envelope); the stage-isolated har_source seam must catch it.
    let har = m.decoder.generator.har_source(&fx.f("seam.gen.arg2"), &mut FixedNoise { rand_ini, sine_noise: sn.clone() }).unwrap();
    let rs = compare(c, "har_source(perturbed)", &har, &fx.f("seam.har_source"), SEAM_GATE);
    let a = m.forward_ids(&ids, &ref_s, 1.0, &mut FixedNoise { rand_ini, sine_noise: sn }).unwrap();
    let r = gate_e2e_v2(c, &a.audio, &want);
    println!("sine noise h0 x1.5: har_source seam rel {:.3e} -> {} | E2E v2 rel {:.3e} -> {} (E2E gate lacks power; documented)",
        rs.rel_l2, rs.pass, r.rel_l2, r.pass);
    assert!(!rs.pass, "excitation-noise perturbation not detected by the har_source seam");

    // 3. wrong voice (am_adam style vector for an af_heart fixture)
    let other = fixtures("cpu-t1").into_iter().find(|f| f.name == "s02_fox__am_adam__s1.0").unwrap();
    let a = m.forward_ids(&ids, &other.f("ref_s"), 1.0, &mut FixedNoise { rand_ini, sine_noise: other.f("noise.sine") });
    let durations_differ = a.as_ref().map(|o| o.pred_dur != fx.i("pred_dur")).unwrap_or(true);
    println!("voice swap: durations differ = {durations_differ}");
    assert!(durations_differ, "voice swap not detected by the exact duration gate");

    // 4. speed 1.0 -> 1.05 changes discrete durations
    let pd = model::durations_from_logits(&fx.f("seam.duration_proj"), ids.len(), 1.05);
    assert_ne!(pd, fx.i("pred_dur"), "speed perturbation not detected");

    // 5. seam gate: 1e-3 relative perturbation of a stage output
    let mut x = fx.f("seam.gen.arg0");
    for (i, v) in x.iter_mut().enumerate() {
        *v *= 1.0 + 1e-3 * if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let r = compare(c, "perturbed", &x, &fx.f("seam.gen.arg0"), SEAM_GATE);
    println!("seam 1e-3 perturbation: rel {:.3e} -> {}", r.rel_l2, r.pass);
    assert!(!r.pass, "seam perturbation not detected");

    // 6. one-sample time shift of the correct waveform
    let mut shifted = vec![0.0f32];
    shifted.extend_from_slice(&base.audio[..base.audio.len() - 1]);
    let r = gate_e2e_v2(c, &shifted, &want);
    println!("1-sample shift: rel {:.3e} -> {}", r.rel_l2, r.pass);
    assert!(!r.pass, "time shift not detected");
}
