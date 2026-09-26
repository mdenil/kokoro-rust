//! Milestone I1 (owner #15/#17): the ACTUAL `kokoro` binary, text line file -> per-line WAVs on the
//! RTX 4090 with the NATIVE frontend, Python unavailable (scrubbed env, PATH without Python) and an
//! execve audit (strace -f: the only exec is the binary itself; no helper processes).
//! Pronunciation fidelity: each line's chunk graphemes/phonemes must equal the pinned reference
//! pipeline's (oracle/frontend_oracle.py fixtures). Binary under test: $KOKORO_BIN (e.g. the FMA
//! build) or this crate's binary.

use std::path::{Path, PathBuf};
use std::process::Command;

fn data() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

fn model_dir() -> PathBuf {
    data().join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987")
}

fn bin() -> PathBuf {
    std::env::var("KOKORO_BIN").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_BIN_EXE_kokoro")))
}

fn scratch(name: &str) -> PathBuf {
    let tag = bin().file_name().unwrap().to_string_lossy().into_owned();
    let d = data().join("tmp/cli_text_native").join(tag).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Run {
    code: i32,
    stderr: String,
    execs: Vec<String>,
}

/// `kokoro synth` (text, native frontend, CUDA GPU 0) under `strace -f -e trace=execve` with a
/// cleared environment: no PATH entry with Python, no PYTHON* variables, no HOME.
fn synth(input: &Path, out: &Path, extra: &[&str]) -> Run {
    let log = out.with_extension("strace");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    let o = Command::new("/usr/bin/strace")
        .args(["-f", "-qq", "-e", "trace=execve,execveat", "-o"])
        .arg(&log)
        .arg(bin())
        .env_clear()
        .env("PATH", "/nonexistent")
        .env("CUDA_VISIBLE_DEVICES", "0")
        .env("KOKORO_FRONTEND_DIR", data().join("frontend"))
        .args(["synth", "--device", "cuda", "--model-dir"])
        .arg(model_dir())
        .arg("--input")
        .arg(input)
        .arg("--out-dir")
        .arg(out)
        .args(extra)
        .output()
        .expect("run strace + kokoro");
    let execs = std::fs::read_to_string(&log).unwrap_or_default().lines().filter(|l| l.contains("execve")).map(String::from).collect();
    Run { code: o.status.code().unwrap_or(-1), stderr: String::from_utf8_lossy(&o.stderr).into_owned(), execs }
}

fn assert_no_helpers(r: &Run) {
    assert_eq!(r.execs.len(), 1, "expected exactly one execve (the binary itself), got:\n{}", r.execs.join("\n"));
    assert!(r.execs[0].contains(&*bin().to_string_lossy()), "{}", r.execs[0]);
    assert!(!r.execs.iter().any(|e| e.contains("python")));
}

fn json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|_| panic!("missing {}", p.display()))).unwrap()
}

/// (text, expected chunks [(graphemes, phonemes)]) from a reference oracle fixture.
fn oracle_lines(file: &str, pick: impl Fn(u64) -> bool) -> Vec<(String, Vec<(String, String)>)> {
    let t = std::fs::read_to_string(data().join(file)).unwrap_or_else(|_| panic!("{file} missing — NOT a pass"));
    t.lines()
        .skip(1)
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .filter(|r| r["blank"].as_bool() != Some(true) && r.get("error").is_none() && pick(r["line"].as_u64().unwrap()))
        .map(|r| {
            let ch = r["chunks"].as_array().unwrap().iter().map(|c| (c["graphemes"].as_str().unwrap().to_string(), c["phonemes"].as_str().unwrap().to_string())).collect();
            (r["text"].as_str().unwrap().to_string(), ch)
        })
        .collect()
}

/// Public corpus for I1: all edge cases + link features + every multi-chunk Alice line + a few
/// ordinary Alice lines.
fn corpus() -> Vec<(String, Vec<(String, String)>)> {
    let mut v = oracle_lines("fixtures/frontend/frontend_edge_cases.oracle.jsonl", |_| true);
    v.extend(oracle_lines("fixtures/frontend/link_features.oracle.jsonl", |_| true));
    v.extend(oracle_lines("fixtures/frontend/alice_full.oracle.jsonl", |l| [5, 18, 49, 121, 332, 361, 866, 1399, 1400, 1401, 2, 3, 10].contains(&l)));
    v
}

fn wav_samples(p: &Path) -> Vec<i16> {
    let b = std::fs::read(p).unwrap();
    // pcm16 mono: find the "data" chunk
    let pos = b.windows(4).position(|w| w == b"data").expect("data chunk");
    let n = u32::from_le_bytes(b[pos + 4..pos + 8].try_into().unwrap()) as usize;
    b[pos + 8..pos + 8 + n].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

fn sha256_hex(b: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

/// Full check of one completed run against the reference pronunciations.
fn verify_run(out: &Path, stem: &str, corpus: &[(String, Vec<(String, String)>)], voice: &str) -> Vec<usize> {
    let m = json(&out.join(format!("{stem}.manifest.json")));
    assert_eq!(m["complete"], true);
    assert_eq!(m["input_lines"], corpus.len());
    assert!(m["config"]["frontend"].as_str().unwrap().starts_with("native misaki-0.9.4"), "{}", m["config"]["frontend"]);
    assert!(m["config"]["engine"].as_str().unwrap().contains("cuda"), "{}", m["config"]["engine"]);
    assert_eq!(m["config"]["voice"], voice);
    let mut samples = vec![];
    for (i, (text, want)) in corpus.iter().enumerate() {
        let line = i + 1;
        let sc = json(&out.join(format!("{stem}_{line:05}.json")));
        assert_eq!(sc["line"], line);
        assert_eq!(sc["status"], "ok", "line {line}: {}", sc["error"]);
        assert_eq!(sc["text"].as_str().unwrap(), text);
        let ps: Vec<&str> = sc["phonemes"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let gs: Vec<&str> = sc["graphemes"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let wps: Vec<&str> = want.iter().map(|c| c.1.as_str()).collect();
        let wgs: Vec<&str> = want.iter().map(|c| c.0.as_str()).collect();
        assert_eq!(ps, wps, "line {line}: phonemes differ from the pinned reference");
        assert_eq!(gs, wgs, "line {line}: chunk graphemes differ from the pinned reference");
        // production KModel silently drops phoneme chars outside the vocab; we must drop exactly
        // those (and record them)
        let vocab = vocab();
        let want_dropped: Vec<String> = wps.iter().flat_map(|p| p.chars()).filter(|c| !vocab.contains(c)).map(|c| c.to_string()).collect();
        let dropped: Vec<String> = sc["dropped_phoneme_chars"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        assert_eq!(dropped, want_dropped, "line {line}: dropped phoneme chars");
        let wav = out.join(format!("{stem}_{line:05}.wav"));
        let bytes = std::fs::read(&wav).unwrap();
        assert_eq!(sc["audio_sha256"].as_str().unwrap(), sha256_hex(&bytes), "line {line}: audio hash");
        let s = wav_samples(&wav);
        assert_eq!(s.len(), sc["samples"].as_u64().unwrap() as usize);
        assert!(s.len() > 2400, "line {line}: implausibly short audio");
        let peak = s.iter().map(|x| x.unsigned_abs()).max().unwrap();
        assert!(peak > 1000, "line {line}: silent audio (peak {peak})");
        samples.push(s.len());
    }
    samples
}

fn vocab() -> std::collections::HashSet<char> {
    let cfg = json(&model_dir().join("config.json"));
    cfg["vocab"].as_object().unwrap().keys().map(|k| k.chars().next().unwrap()).collect()
}

fn write_corpus(path: &Path, corpus: &[(String, Vec<(String, String)>)]) {
    std::fs::write(path, corpus.iter().map(|c| format!("{}\n", c.0)).collect::<String>()).unwrap();
}

#[test]
fn native_text_file_to_wavs_both_voices_python_free() {
    let corpus = corpus();
    for voice in ["af_heart", "am_adam"] {
        let d = scratch(&format!("voices-{voice}"));
        let input = d.join("book.txt");
        write_corpus(&input, &corpus);
        let r = synth(&input, &d.join("out"), &["--voice", voice]);
        assert_eq!(r.code, 0, "{}", r.stderr);
        assert_no_helpers(&r);
        let n = verify_run(&d.join("out"), "book", &corpus, voice);
        println!("{voice}: {} lines, {} samples total, execve audit: 1 exec; {}", n.len(), n.iter().sum::<usize>(), r.stderr.lines().last().unwrap_or(""));
    }
}

/// Batched (default) vs batch-1: identical line structure and sample counts (durations exact);
/// waveforms close (sanity only; quality variation of batching is owner-accepted, #14).
#[test]
fn batched_and_batch1_agree_on_structure() {
    let corpus = corpus();
    let d = scratch("batching");
    let input = d.join("book.txt");
    write_corpus(&input, &corpus);
    let rb = synth(&input, &d.join("batched"), &[]);
    assert_eq!(rb.code, 0, "{}", rb.stderr);
    let r1 = synth(&input, &d.join("single"), &["--batch-phonemes", "0"]);
    assert_eq!(r1.code, 0, "{}", r1.stderr);
    assert_no_helpers(&rb);
    assert_no_helpers(&r1);
    let nb = verify_run(&d.join("batched"), "book", &corpus, "af_heart");
    let n1 = verify_run(&d.join("single"), "book", &corpus, "af_heart");
    assert_eq!(nb, n1, "per-line sample counts differ between batched and batch-1");
    let mut worst = 1.0f64;
    for line in 1..=corpus.len() {
        let a = wav_samples(&d.join(format!("batched/book_{line:05}.wav")));
        let b = wav_samples(&d.join(format!("single/book_{line:05}.wav")));
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            ab += *x as f64 * *y as f64;
            aa += (*x as f64).powi(2);
            bb += (*y as f64).powi(2);
        }
        worst = worst.min(ab / (aa * bb).sqrt());
    }
    println!("batched vs batch-1: sample counts identical on {} lines; worst waveform corr {worst:.5}", corpus.len());
    assert!(worst > 0.95, "batched output diverges grossly from batch-1 (corr {worst})");
}

/// Long lines: every chunk is synthesized and joined in order — the line's audio length equals the
/// sum of its chunks synthesized one per line in phoneme mode (no dropped/truncated suffix).
#[test]
fn long_lines_are_complete() {
    let corpus: Vec<_> = corpus().into_iter().filter(|c| c.1.len() > 1).collect();
    assert!(corpus.len() >= 10 && corpus.iter().any(|c| c.1.len() >= 3));
    let d = scratch("long");
    let input = d.join("long.txt");
    write_corpus(&input, &corpus);
    let r = synth(&input, &d.join("out"), &["--batch-phonemes", "0"]);
    assert_eq!(r.code, 0, "{}", r.stderr);
    let n = verify_run(&d.join("out"), "long", &corpus, "af_heart");
    let chunks: Vec<&str> = corpus.iter().flat_map(|c| c.1.iter().map(|x| x.1.as_str())).collect();
    let pin = d.join("chunks.txt");
    std::fs::write(&pin, chunks.iter().map(|c| format!("{c}\n")).collect::<String>()).unwrap();
    let rp = synth(&pin, &d.join("per_chunk"), &["--batch-phonemes", "0", "--input-format", "phonemes"]);
    assert_eq!(rp.code, 0, "{}", rp.stderr);
    let mut k = 0;
    for (i, c) in corpus.iter().enumerate() {
        let mut sum = 0;
        for _ in &c.1 {
            k += 1;
            sum += json(&d.join(format!("per_chunk/chunks_{k:05}.json")))["samples"].as_u64().unwrap() as usize;
        }
        assert_eq!(n[i], sum, "long line {}: {} chunks, line audio {} != sum of chunk audio {sum}", i + 1, c.1.len(), n[i]);
    }
    println!("long lines: {} lines / {} chunks complete", corpus.len(), chunks.len());
}

/// Failures are explicit and never shift numbering; restart resumes only verified outputs;
/// edits, config changes and damaged outputs invalidate exactly what they should.
#[test]
fn failures_restart_and_invalidation() {
    let d = scratch("resume");
    let input = d.join("sec.txt");
    let oversize = vec!["alpha"; 150].join("-");
    let lines = ["The quick brown fox jumps over the lazy dog.", "", "Bad \u{7} bell.", oversize.as_str(), "It was 3:45 p.m. on the 21st of May, 1999.", "[Kokoro](/kˈOkəɹO/) reads this."];
    std::fs::write(&input, lines.iter().map(|l| format!("{l}\n")).collect::<String>()).unwrap();
    let out = d.join("out");
    let r = synth(&input, &out, &[]);
    assert_eq!(r.code, 1, "incomplete run must exit 1: {}", r.stderr);
    assert_no_helpers(&r);
    let m = json(&out.join("sec.manifest.json"));
    let st: Vec<&str> = m["lines"].as_array().unwrap().iter().map(|l| l["status"].as_str().unwrap()).collect();
    assert_eq!(st, ["ok", "blank", "invalid", "oversize", "ok", "ok"]);
    assert_eq!(m["complete"], false);
    for (line, ok) in [(1, true), (2, false), (3, false), (4, false), (5, true), (6, true)] {
        assert_eq!(out.join(format!("sec_{line:05}.wav")).exists(), ok, "line {line}");
    }
    let sc4 = json(&out.join("sec_00004.json"));
    assert!(sc4["error"].as_str().unwrap().contains("refusing to truncate"));
    // restart: verified outputs resumed, failures re-evaluated (still failing)
    let r2 = synth(&input, &out, &[]);
    assert_eq!(r2.code, 1);
    let m2 = json(&out.join("sec.manifest.json"));
    assert_eq!(m2["counts"]["resumed"], 3);
    // blank allowed by policy -> everything else still the same, blank is not a failure
    let r3 = synth(&input, &out, &["--blank-lines", "skip"]);
    assert_eq!(r3.code, 1, "invalid + oversize still fail");
    // fix the failing lines: only they are synthesized; edit line 5: only it is redone
    let fixed = ["The quick brown fox jumps over the lazy dog.", "Now a real line.", "Bell removed.", "Alpha alpha alpha.", "It was 4:45 p.m. on the 22nd of May, 1999.", "[Kokoro](/kˈOkəɹO/) reads this."];
    std::fs::write(&input, fixed.iter().map(|l| format!("{l}\n")).collect::<String>()).unwrap();
    let r4 = synth(&input, &out, &[]);
    assert_eq!(r4.code, 0, "{}", r4.stderr);
    let m4 = json(&out.join("sec.manifest.json"));
    let resumed: Vec<bool> = m4["lines"].as_array().unwrap().iter().map(|l| l["resumed"].as_bool().unwrap()).collect();
    assert_eq!(resumed, [true, false, false, false, false, true]);
    // damaged output (corrupt WAV) and a missing WAV are redone; the rest resumes
    std::fs::write(out.join("sec_00001.wav"), b"RIFFjunk").unwrap();
    std::fs::remove_file(out.join("sec_00006.wav")).unwrap();
    let r5 = synth(&input, &out, &[]);
    assert_eq!(r5.code, 0);
    let m5 = json(&out.join("sec.manifest.json"));
    let resumed: Vec<bool> = m5["lines"].as_array().unwrap().iter().map(|l| l["resumed"].as_bool().unwrap()).collect();
    assert_eq!(resumed, [false, true, true, true, true, false]);
    // config change (voice / speed) invalidates everything
    for extra in [&["--voice", "am_adam"][..], &["--voice", "am_adam", "--speed", "1.1"][..]] {
        let r6 = synth(&input, &out, extra);
        assert_eq!(r6.code, 0);
        assert_eq!(json(&out.join("sec.manifest.json"))["counts"]["resumed"], 0, "{extra:?}");
    }
    // missing frontend data is a job-level error (exit 2), never a silent fallback
    let o = Command::new(bin())
        .env_clear()
        .env("PATH", "/nonexistent")
        .env("KOKORO_FRONTEND_DIR", d.join("no-such-dir"))
        .args(["synth", "--device", "cuda", "--model-dir"])
        .arg(model_dir())
        .arg("--input")
        .arg(&input)
        .arg("--out-dir")
        .arg(d.join("out2"))
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2), "{}", String::from_utf8_lossy(&o.stderr));
}
