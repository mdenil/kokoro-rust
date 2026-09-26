//! Milestone I1 (owner #15/#17): the ACTUAL `kokoro` binary, text line file -> per-line WAVs on the
//! RTX 4090 with the NATIVE frontend, Python unavailable (scrubbed env, PATH without Python) and an
//! execve audit (strace -f: the only exec is the binary itself; no helper processes).
//! Pronunciation fidelity: each line's chunk graphemes/phonemes must equal the pinned reference
//! pipeline's (oracle/frontend_oracle.py fixtures). Binary under test: $KOKORO_BIN (e.g. the FMA
//! build) or this crate's binary.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/mod.rs"]
mod support;
use support::{Corpus, ALICE, CHAPTER, EDGE, FUZZ, LINKS};

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
    synth_path(input, out, extra, "/nonexistent")
}

fn synth_path(input: &Path, out: &Path, extra: &[&str], path: &str) -> Run {
    let log = out.with_extension("strace");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    let o = Command::new("/usr/bin/strace")
        .args(["-f", "-qq", "-e", "trace=execve,execveat", "-o"])
        .arg(&log)
        .arg(bin())
        .env_clear()
        .env("PATH", path)
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

fn try_json(p: &Path) -> Result<serde_json::Value, String> {
    let name = p.file_name().unwrap().to_string_lossy().into_owned();
    let s = std::fs::read_to_string(p).map_err(|_| format!("{name}: missing"))?;
    serde_json::from_str(&s).map_err(|_| format!("{name}: unparsable (truncated/corrupt)"))
}

type Expected = Vec<(String, Vec<(String, String)>)>;

/// (text, expected chunks [(graphemes, phonemes)]) from a pinned reference oracle corpus.
fn oracle_lines(c: &Corpus, pick: impl Fn(u64) -> bool) -> Expected {
    let recs = support::load_corpus(c).unwrap_or_else(|e| panic!("{e}"));
    recs.iter()
        .filter(|r| r["blank"].as_bool() != Some(true) && r.get("error").is_none() && pick(r["line"].as_u64().unwrap()))
        .map(|r| {
            let ch = r["chunks"].as_array().unwrap().iter().map(|c| (c["graphemes"].as_str().unwrap().to_string(), c["phonemes"].as_str().unwrap().to_string())).collect();
            (r["text"].as_str().unwrap().to_string(), ch)
        })
        .collect()
}

const ALICE_PICK: [u64; 13] = [5, 18, 49, 121, 332, 361, 866, 1399, 1400, 1401, 2, 3, 10];

/// Public corpus for I1: all edge cases + link features + every multi-chunk Alice line + a few
/// ordinary Alice lines. Cardinality pinned: 98 lines, 111 chunks, 11 multi-chunk lines.
fn corpus() -> Expected {
    let mut v = oracle_lines(&EDGE, |_| true);
    v.extend(oracle_lines(&LINKS, |_| true));
    v.extend(oracle_lines(&ALICE, |l| ALICE_PICK.contains(&l)));
    assert_eq!(v.len(), 65 + 20 + ALICE_PICK.len());
    assert_eq!(v.iter().map(|c| c.1.len()).sum::<usize>(), 111, "pinned chunk total");
    assert_eq!(v.iter().filter(|c| c.1.len() > 1).count(), 11, "pinned multi-chunk lines");
    v
}

/// pcm16 mono WAV -> samples; malformed/truncated files are an error.
fn wav_samples(p: &Path) -> Result<Vec<i16>, String> {
    let name = p.file_name().unwrap().to_string_lossy().into_owned();
    let b = std::fs::read(p).map_err(|_| format!("{name}: missing"))?;
    if b.len() < 44 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(format!("{name}: not a WAV"));
    }
    let pos = b.windows(4).position(|w| w == b"data").ok_or(format!("{name}: no data chunk"))?;
    let n = u32::from_le_bytes(b[pos + 4..pos + 8].try_into().unwrap()) as usize;
    if b.len() < pos + 8 + n {
        return Err(format!("{name}: truncated data chunk"));
    }
    Ok(b[pos + 8..pos + 8 + n].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
}

fn sha256_hex(b: &[u8]) -> String {
    support::sha256_hex(b)
}

macro_rules! ensure {
    ($c:expr, $($m:tt)*) => { if !$c { return Err(format!($($m)*)); } };
}

/// Full check of one completed run against the expected (reference) pronunciations. Returns the
/// per-line sample counts. Private-safe: messages carry line numbers / file names only.
/// Cardinality first (manifest entries, sidecars, WAVs, chunk arrays), then content.
fn verify_outputs(out: &Path, stem: &str, corpus: &Expected, voice: &str) -> Result<Vec<usize>, String> {
    let n = corpus.len();
    let m = try_json(&out.join(format!("{stem}.manifest.json")))?;
    ensure!(m["complete"] == true, "manifest: not complete");
    ensure!(m["input_lines"].as_u64() == Some(n as u64), "manifest: input_lines {} != expected {n}", m["input_lines"]);
    let entries = m["lines"].as_array().ok_or("manifest: no lines")?;
    ensure!(entries.len() == n, "manifest: {} line entries != {n} (missing/duplicate)", entries.len());
    for (i, e) in entries.iter().enumerate() {
        ensure!(e["line"].as_u64() == Some(i as u64 + 1), "manifest entry {}: line {} (duplicate/missing/reordered)", i + 1, e["line"]);
        ensure!(e["status"] == "ok", "manifest entry {}: status {}", i + 1, e["status"]);
        ensure!(e["sidecar"].as_str() == Some(format!("{stem}_{:05}.json", i + 1).as_str()), "manifest entry {}: sidecar name", i + 1);
    }
    let c = &m["counts"];
    ensure!(c["failed"] == 0 && c["done"].as_u64().unwrap_or(0) + c["resumed"].as_u64().unwrap_or(0) == n as u64, "manifest counts {c}");
    ensure!(m["config"]["frontend"].as_str().unwrap_or("").starts_with("native misaki-0.9.4"), "config.frontend is not the native frontend");
    ensure!(m["config"]["engine"].as_str().unwrap_or("").contains("cuda"), "config.engine is not CUDA");
    ensure!(m["config"]["voice"] == voice, "config.voice");
    // exactly one WAV + one sidecar per line, nothing extra
    let listing: Vec<String> = std::fs::read_dir(out).map_err(|_| "out dir missing")?.map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    let wavs = listing.iter().filter(|f| f.ends_with(".wav")).count();
    let jsons = listing.iter().filter(|f| f.ends_with(".json") && !f.ends_with(".manifest.json")).count();
    ensure!(wavs == n && jsons == n, "{wavs} WAVs / {jsons} sidecars != {n} lines (missing or extra outputs)");
    let vocab = vocab();
    let mut samples = vec![];
    for (i, (text, want)) in corpus.iter().enumerate() {
        let line = i + 1;
        let sc = try_json(&out.join(format!("{stem}_{line:05}.json")))?;
        ensure!(sc["line"].as_u64() == Some(line as u64), "sidecar {line}: line field {} (duplicated/misplaced sidecar)", sc["line"]);
        ensure!(sc["status"] == "ok", "line {line}: status {}", sc["status"]);
        ensure!(sc["text"].as_str() == Some(text.as_str()), "line {line}: text differs from input");
        ensure!(sc["text_sha256"].as_str() == Some(sha256_hex(text.as_bytes()).as_str()), "line {line}: text hash");
        let ps: Vec<&str> = sc["phonemes"].as_array().ok_or("phonemes")?.iter().map(|v| v.as_str().unwrap_or("")).collect();
        let gs: Vec<&str> = sc["graphemes"].as_array().ok_or("graphemes")?.iter().map(|v| v.as_str().unwrap_or("")).collect();
        ensure!(ps.len() == want.len() && gs.len() == want.len(), "line {line}: {} phoneme / {} grapheme chunks != expected {} (dropped/extra chunk)", ps.len(), gs.len(), want.len());
        for (k, ((p, g), (wg, wp))) in ps.iter().zip(&gs).zip(want).enumerate() {
            ensure!(p == wp, "line {line} chunk {}: phonemes differ from the pinned reference", k + 1);
            ensure!(g == wg, "line {line} chunk {}: graphemes differ from the pinned reference", k + 1);
        }
        // production KModel silently drops phoneme chars outside the vocab; we must drop exactly those
        let want_dropped: Vec<String> = want.iter().flat_map(|c| c.1.chars()).filter(|c| !vocab.contains(c)).map(|c| c.to_string()).collect();
        let dropped: Vec<String> = sc["dropped_phoneme_chars"].as_array().ok_or("dropped")?.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        ensure!(dropped == want_dropped, "line {line}: dropped phoneme chars differ");
        let wav = out.join(format!("{stem}_{line:05}.wav"));
        ensure!(sc["wav"].as_str() == Some(format!("{stem}_{line:05}.wav").as_str()), "line {line}: sidecar wav name");
        let bytes = std::fs::read(&wav).map_err(|_| format!("line {line}: WAV missing"))?;
        ensure!(sc["audio_sha256"].as_str() == Some(sha256_hex(&bytes).as_str()), "line {line}: audio hash mismatch (damaged/replaced WAV)");
        let s = wav_samples(&wav)?;
        ensure!(s.len() == sc["samples"].as_u64().unwrap_or(0) as usize, "line {line}: sample count vs sidecar");
        ensure!(s.len() > 2400, "line {line}: implausibly short audio");
        let peak = s.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
        ensure!(peak > 1000, "line {line}: silent audio");
        samples.push(s.len());
    }
    ensure!(samples.len() == n, "verified {} lines != {n}", samples.len());
    Ok(samples)
}

fn verify_run(out: &Path, stem: &str, corpus: &Expected, voice: &str) -> Vec<usize> {
    verify_outputs(out, stem, corpus, voice).unwrap_or_else(|e| panic!("{e}"))
}

fn vocab() -> std::collections::HashSet<char> {
    let cfg = json(&model_dir().join("config.json"));
    cfg["vocab"].as_object().unwrap().keys().map(|k| k.chars().next().unwrap()).collect()
}

/// Copy a verified output dir, damage it one way, and require verify_outputs to reject it.
fn output_negative_controls(good: &Path, stem: &str, corpus: &Expected, voice: &str) -> usize {
    assert!(verify_outputs(good, stem, corpus, voice).is_ok());
    let multi = corpus.iter().position(|c| c.1.len() > 1).expect("a multi-chunk line") + 1;
    type Damage<'a> = Box<dyn Fn(&Path) + 'a>;
    let wav = |l: usize| format!("{stem}_{l:05}.wav");
    let sc = |l: usize| format!("{stem}_{l:05}.json");
    let edit_json = |p: &Path, f: &dyn Fn(&mut serde_json::Value)| {
        let mut v = json(p);
        f(&mut v);
        std::fs::write(p, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    };
    let man = format!("{stem}.manifest.json");
    let cases: Vec<(&str, Damage)> = vec![
        ("missing WAV", Box::new(move |d: &Path| std::fs::remove_file(d.join(wav(3))).unwrap())),
        ("truncated WAV", Box::new(move |d: &Path| {
            let b = std::fs::read(d.join(wav(4))).unwrap();
            std::fs::write(d.join(wav(4)), &b[..b.len() / 2]).unwrap();
        })),
        ("missing sidecar", Box::new(move |d: &Path| std::fs::remove_file(d.join(sc(5))).unwrap())),
        ("truncated sidecar", Box::new(move |d: &Path| {
            let b = std::fs::read(d.join(sc(6))).unwrap();
            std::fs::write(d.join(sc(6)), &b[..b.len() / 2]).unwrap();
        })),
        ("duplicated sidecar", Box::new(move |d: &Path| { std::fs::copy(d.join(sc(2)), d.join(sc(7))).unwrap(); })),
        ("duplicated WAV", Box::new(move |d: &Path| { std::fs::copy(d.join(wav(2)), d.join(wav(8))).unwrap(); })),
        ("extra stray WAV", Box::new(move |d: &Path| { std::fs::copy(d.join(wav(1)), d.join(format!("{stem}_99999.wav"))).unwrap(); })),
        ("manifest entry dropped", Box::new({ let man = man.clone(); move |d: &Path| edit_json(&d.join(&man), &|v| { v["lines"].as_array_mut().unwrap().pop(); }) })),
        ("manifest entry duplicated", Box::new({ let man = man.clone(); move |d: &Path| edit_json(&d.join(&man), &|v| {
            let a = v["lines"].as_array_mut().unwrap();
            let x = a[1].clone();
            a[2] = x;
        }) })),
        ("manifest truncated", Box::new({ let man = man.clone(); move |d: &Path| {
            let b = std::fs::read(d.join(&man)).unwrap();
            std::fs::write(d.join(&man), &b[..b.len() - 20]).unwrap();
        } })),
        ("chunk dropped from a multi-chunk line", Box::new(move |d: &Path| edit_json(&d.join(sc(multi)), &|v| {
            v["phonemes"].as_array_mut().unwrap().pop();
            v["graphemes"].as_array_mut().unwrap().pop();
        }))),
        ("phoneme altered", Box::new(move |d: &Path| edit_json(&d.join(sc(9)), &|v| {
            let p = v["phonemes"][0].as_str().unwrap().replacen('ə', "ɪ", 1) + "ə";
            v["phonemes"][0] = p.into();
        }))),
    ];
    let mut n = 0;
    for (name, damage) in &cases {
        let d = good.with_file_name(format!("negctl-{}", name.replace(' ', "_")));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for e in std::fs::read_dir(good).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), d.join(e.file_name())).unwrap();
        }
        damage(&d);
        let r = verify_outputs(&d, stem, corpus, voice);
        assert!(r.is_err(), "output negative control '{name}' was NOT detected");
        println!("  output negative control '{name}': rejected ({})", r.unwrap_err());
        let _ = std::fs::remove_dir_all(&d);
        n += 1;
    }
    // expected-corpus side: truncated or duplicated expectations must not verify either
    let mut shorter = corpus.clone();
    shorter.pop();
    assert!(verify_outputs(good, stem, &shorter, voice).is_err(), "truncated expectation accepted");
    let mut dup = corpus.clone();
    dup[1] = dup[0].clone();
    assert!(verify_outputs(good, stem, &dup, voice).is_err(), "duplicated expectation accepted");
    assert!(verify_outputs(good, stem, corpus, if voice == "af_heart" { "am_adam" } else { "af_heart" }).is_err(), "wrong voice accepted");
    n + 3
}

fn write_corpus(path: &Path, corpus: &[(String, Vec<(String, String)>)]) {
    std::fs::write(path, corpus.iter().map(|c| format!("{}\n", c.0)).collect::<String>()).unwrap();
    assert_eq!(std::fs::read_to_string(path).unwrap().matches('\n').count(), corpus.len(), "input line count");
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
        if voice == "af_heart" {
            let k = output_negative_controls(&d.join("out"), "book", &corpus, voice);
            println!("{k} output/expectation negative controls rejected");
        }
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
        let a = wav_samples(&d.join(format!("batched/book_{line:05}.wav"))).unwrap();
        let b = wav_samples(&d.join(format!("single/book_{line:05}.wav"))).unwrap();
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

/// Complete private chapter acceptance (owner #15): the prepared chapter line file -> all per-line
/// WAVs through the binary, both voices, Python unavailable + execve audit; pronunciations vs the
/// pinned reference. Private-safe: outputs under evidence/private (outside Git), only aggregates
/// printed, no text/phonemes in messages.
#[test]
#[ignore = "private chapter (local only)"]
fn private_chapter_acceptance() {
    // pinned input (sha256 + 316 lines) and pinned reference oracle (sha256, 316 records 1..=316,
    // 317 chunks, 9137 tokens) — a missing/truncated/duplicated/altered fixture fails here
    support::read_pinned_bytes(&support::CHAPTER_INPUT).unwrap_or_else(|e| panic!("{e}"));
    let input = support::data().join(support::CHAPTER_INPUT.path);
    let expected = oracle_lines(&CHAPTER, |_| true);
    assert_eq!(expected.len(), CHAPTER.lines);
    assert_eq!(expected.iter().map(|c| c.1.len()).sum::<usize>(), CHAPTER.chunks);
    let stem = "002_hidden_curriculum_of_youth_whaddaya_want_from_me";
    let tag = bin().file_name().unwrap().to_string_lossy().into_owned();
    for voice in ["af_heart", "am_adam"] {
        let out = data().join("evidence/private/acceptance").join(&tag).join(voice);
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        let t = std::time::Instant::now();
        let r = synth(&input, &out, &["--voice", voice]);
        let wall = t.elapsed().as_secs_f64();
        assert_eq!(r.code, 0, "chapter run failed (exit {}); see private sidecars", r.code);
        assert_no_helpers(&r);
        let samples = verify_outputs(&out, stem, &expected, voice).unwrap_or_else(|e| panic!("private chapter: {e}"));
        // expectation-side negative controls on the real private outputs
        let mut shorter = expected.clone();
        shorter.pop();
        assert!(verify_outputs(&out, stem, &shorter, voice).is_err(), "truncated expectation accepted");
        let mut dup = expected.clone();
        dup[1] = dup[0].clone();
        assert!(verify_outputs(&out, stem, &dup, voice).is_err(), "duplicated expectation accepted");
        println!(
            "PRIVATE chapter [{tag}] {voice}: {} lines verified, {} chunks, {:.1} s audio, 0 pronunciation/dropped-char mismatches, execs {}, process wall {wall:.2} s (under strace; not a benchmark)",
            samples.len(),
            CHAPTER.chunks,
            samples.iter().sum::<usize>() as f64 / 24000.0,
            r.execs.len()
        );
    }
}

/// Optional `--encode` (owner-approved external ffmpeg, #18): encoded files exist, hashes recorded,
/// and the only extra processes are ffmpeg (one per line) — still no Python.
#[test]
fn encode_with_ffmpeg_when_enabled() {
    let d = scratch("encode");
    let input = d.join("enc.txt");
    std::fs::write(&input, "First line here.\nSecond line, a bit longer.\nThird.\n").unwrap();
    let r = synth_path(&input, &d.join("out"), &["--encode", "flac"], "/usr/bin");
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert!(!r.execs.iter().any(|e| e.contains("python")));
    let ff = r.execs.iter().filter(|e| e.contains("ffmpeg")).count();
    assert_eq!(r.execs.len(), 1 + ff);
    assert_eq!(ff, 3, "{}", r.execs.join("\n"));
    for line in 1..=3 {
        let sc = json(&d.join(format!("out/enc_{line:05}.json")));
        let flac = std::fs::read(d.join(format!("out/enc_{line:05}.flac"))).unwrap();
        assert_eq!(&flac[..4], b"fLaC");
        assert_eq!(sc["encoded_sha256"].as_str().unwrap(), sha256_hex(&flac));
        assert!(d.join(format!("out/enc_{line:05}.wav")).exists());
    }
}

/// Whole synthetic fuzz corpus (4000 lines) through the binary: every line gets exactly the status
/// the pinned reference implies — "error" where the reference frontend raises, "oversize" where a
/// reference chunk exceeds 510 phonemes (production would truncate; we refuse explicitly), "ok"
/// with reference-identical chunks otherwise; exit code 1 (incomplete) and no helper processes.
#[test]
fn fuzz_corpus_through_binary() {
    let recs = support::load_corpus(&FUZZ).unwrap_or_else(|e| panic!("{e}"));
    let d = scratch("fuzz");
    let input = d.join("fuzz.txt");
    std::fs::write(&input, recs.iter().map(|r| format!("{}\n", r["text"].as_str().unwrap())).collect::<String>()).unwrap();
    let r = synth(&input, &d.join("out"), &[]);
    assert_eq!(r.code, 1, "reference-error and oversize lines must make the run incomplete: {}", r.stderr.lines().last().unwrap_or(""));
    assert_no_helpers(&r);
    let m = json(&d.join("out/fuzz.manifest.json"));
    let entries = m["lines"].as_array().unwrap();
    assert_eq!(entries.len(), FUZZ.lines);
    let (mut ok, mut err, mut over, mut chunks) = (0, 0, 0, 0);
    for (i, rec) in recs.iter().enumerate() {
        let line = i + 1;
        assert_eq!(entries[i]["line"].as_u64(), Some(line as u64));
        let sc = json(&d.join(format!("out/fuzz_{line:05}.json")));
        let want = if rec.get("error").is_some() {
            "error"
        } else if rec["chunks"].as_array().unwrap().iter().any(|c| c["oversize"] == true) {
            "oversize"
        } else {
            "ok"
        };
        assert_eq!(sc["status"], want, "line {line}: {}", sc["error"]);
        match want {
            "ok" => {
                ok += 1;
                let exp: Vec<(&str, &str)> = rec["chunks"].as_array().unwrap().iter().map(|c| (c["graphemes"].as_str().unwrap(), c["phonemes"].as_str().unwrap())).collect();
                let got: Vec<(&str, &str)> = sc["graphemes"].as_array().unwrap().iter().zip(sc["phonemes"].as_array().unwrap()).map(|(g, p)| (g.as_str().unwrap(), p.as_str().unwrap())).collect();
                assert_eq!(sc["graphemes"].as_array().unwrap().len(), exp.len(), "line {line}: chunk count");
                assert_eq!(got, exp, "line {line}: chunks differ from the reference");
                chunks += exp.len();
                let wav = std::fs::read(d.join(format!("out/fuzz_{line:05}.wav"))).unwrap();
                assert_eq!(sc["audio_sha256"].as_str().unwrap(), sha256_hex(&wav));
            }
            "error" => {
                err += 1;
                assert!(sc["error"].as_str().unwrap().contains("TypeError"), "line {line}: {}", sc["error"]);
                assert!(!d.join(format!("out/fuzz_{line:05}.wav")).exists());
            }
            _ => {
                over += 1;
                assert!(sc["error"].as_str().unwrap().contains("refusing to truncate"));
                assert!(!d.join(format!("out/fuzz_{line:05}.wav")).exists());
            }
        }
    }
    assert_eq!((ok + err + over, err, over), (FUZZ.lines, 1, 2), "pinned status counts");
    assert_eq!(m["counts"]["failed"], err + over);
    println!("fuzz through binary: {ok} ok ({chunks} chunks, reference-identical), {err} error (reference TypeError), {over} oversize (refused), exit 1, 1 exec");
}
