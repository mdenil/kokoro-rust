//! Shared helpers for the regression and diagnostic tests: render a pinned case with the binary
//! under test and compare per-line WAV hashes with a pinned table.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "paths.rs"]
pub mod paths;

pub fn sha256_file(p: &Path) -> String {
    kokoro::engine::sha256_bytes(&std::fs::read(p).unwrap())
}

/// Render `input` with the binary under test (cleared environment, no Python on PATH, the reference
/// eSpeak NG copy, the caller's CUDA_VISIBLE_DEVICES passed through, extra `env`); returns line -> WAV
/// sha256 of the lines that succeeded.
/// GPU renders inside one test binary run one at a time: concurrent processes can exhaust GPU memory,
/// and a batch that fails is split into smaller batches. The BF16 predictor's results depend on the
/// batch shape, so a split can change the output. Serializing keeps the comparisons meaningful.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn render_env(input: &Path, voice: &str, args: &[String], out: &Path, env: &[(&str, &str)]) -> BTreeMap<usize, String> {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let _ = std::fs::remove_dir_all(out);
    let mut cmd = Command::new(paths::bin());
    // the pinned tables were rendered with the reference eSpeak NG copy
    cmd.env_clear();
    paths::pass_reference_espeak(&mut cmd);
    cmd.envs(env.iter().copied()).env("PATH", "/nonexistent").env("KOKORO_FRONTEND_DIR", paths::frontend_dir());
    let o = paths::pass_cuda_env(&mut cmd)
        .args(["synth", "--per-line", "--diagnostics", "--model-dir"])
        .arg(paths::model_dir())
        .arg("--input")
        .arg(input)
        .arg("--out-dir")
        .arg(out)
        .args(["--voice", voice])
        .args(args)
        .output()
        .unwrap();
    // exit 1 is accepted only for lines refused because a word has no pronunciation (checked below)
    assert!(o.status.success() || o.status.code() == Some(1), "synth failed ({}): {}", o.status, String::from_utf8_lossy(&o.stderr));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        !stderr.contains("splitting"),
        "a batch was split under GPU memory pressure (other GPU work running?); byte comparisons are not meaningful for this run:\n{stderr}"
    );
    let stem = input.file_stem().unwrap().to_string_lossy().into_owned();
    let mut h = BTreeMap::new();
    for e in std::fs::read_dir(out).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if let Some(rest) = name.strip_prefix(&format!("{stem}_")).and_then(|r| r.strip_suffix(".wav")) {
            h.insert(rest.parse::<usize>().unwrap(), sha256_file(&p));
        }
        if let Some(rest) = name.strip_prefix(&format!("{stem}_")).and_then(|r| r.strip_suffix(".json")) {
            let sc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            if sc["status"] != "ok" {
                let err = sc["error"].as_str().unwrap_or("");
                assert!(err.contains("unresolved word"), "{name}: line failed: {err}");
                h.insert(rest.parse::<usize>().unwrap(), UNRESOLVED.to_string());
            }
        }
    }
    h
}

/// Hash-table value for a line the binary refused because a word has no pronunciation (the
/// no-dropped-words rule), so such lines stay visible in every comparison.
pub const UNRESOLVED: &str = "refused:unresolved-word";

pub fn render(input: &Path, voice: &str, args: &[String], out: &Path) -> BTreeMap<usize, String> {
    render_env(input, voice, args, out, &[])
}

/// Per-line differences between rendered hashes and a pinned table (empty = identical, including
/// line coverage).
pub fn diff(got: &BTreeMap<usize, String>, want: &serde_json::Map<String, serde_json::Value>) -> Vec<String> {
    let mut d = vec![];
    for (k, v) in want {
        let line: usize = k.parse().unwrap();
        match got.get(&line) {
            None => d.push(format!("line {line}: missing")),
            Some(h) if h != v.as_str().unwrap() => d.push(format!("line {line}: sha256 {} != pinned {}", if h == UNRESOLVED { h } else { &h[..12] }, &v.as_str().unwrap()[..12])),
            _ => {}
        }
    }
    for line in got.keys() {
        if !want.contains_key(&line.to_string()) {
            d.push(format!("line {line}: unexpected extra output"));
        }
    }
    d
}

/// Input file and arguments of a pinned case (repo-relative input unless `input_override`).
pub fn case_input(c: &serde_json::Value, input_override: Option<&Path>) -> (PathBuf, Vec<String>) {
    let input = match input_override {
        Some(p) => p.to_path_buf(),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(c["input"].as_str().unwrap()),
    };
    assert_eq!(sha256_file(&input), c["input_sha256"].as_str().unwrap(), "input file of a pinned case changed");
    (input, c["args"].as_array().unwrap().iter().map(|a| a.as_str().unwrap().to_string()).collect())
}

/// Render every case of `table`; returns (case, lines rendered, per-line differences).
pub fn run_table(table: &serde_json::Value, input_override: Option<&Path>, scratch: &Path) -> Vec<(String, usize, Vec<String>)> {
    let cases = table["cases"].as_object().unwrap();
    assert!(!cases.is_empty());
    cases
        .iter()
        .map(|(case, c)| {
            let (input, args) = case_input(c, input_override);
            let got = render(&input, c["voice"].as_str().unwrap(), &args, &scratch.join(case));
            let d = diff(&got, c["wav_sha256"].as_object().unwrap());
            (case.clone(), got.len(), d)
        })
        .collect()
}
