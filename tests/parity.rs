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

fn data_root() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

fn model() -> &'static Kokoro {
    static M: OnceLock<Kokoro> = OnceLock::new();
    M.get_or_init(|| {
        let root = data_root().join("models");
        let w = Weights::load(&root.join("raw_state.safetensors")).expect("raw_state.safetensors (run oracle/dump_weights.py)");
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("config.json")).unwrap()).unwrap();
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
    names
        .into_iter()
        .filter(|n| std::env::var("KOKORO_CASE").map(|c| n.contains(&c)).unwrap_or(true))
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

/// Stage-isolated linear-algebra seam gate (docs/conformance/NONDET_FLOOR.md).
const SEAM_GATE: f64 = 1e-4;

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
        let mut e2e = compare(c, "E2E waveform", &out.audio, &want, 0.019);
        e2e.pass = e2e.pass && e2e.max_abs <= 3.3e-2 && e2e.corr >= 0.9995;
        rows.push(e2e);
    }
    rows
}

fn report(rows: &[Cmp], tag: &str) {
    println!("\n{:<34} {:<26} {:>9} {:>10} {:>10} {:>9} {:>8}", "case", "seam", "n", "rel_l2", "max_abs", "corr", "verdict");
    for r in rows {
        println!(
            "{:<34} {:<26} {:>9} {:>10.3e} {:>10.3e} {:>9.6} {:>8}",
            r.case, r.seam, r.n, r.rel_l2, r.max_abs, r.corr, if r.pass { "PASS" } else { "FAIL" }
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
    let fails: Vec<_> = rows.iter().filter(|r| !r.pass).map(|r| format!("{}/{}", r.case, r.seam)).collect();
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
