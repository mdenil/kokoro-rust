//! BF16x product regression: the binary under test must reproduce, byte for byte,
//! the per-line WAVs of the accepted reference artifact (phase2-9b39d48, sha256 6fde9d88…12ea) on the
//! pinned public cases (tests/pinned/bf16x_golden.json, rendered by bench/make_bf16x_golden.py:
//! deterministic across two runs of the accepted binary). Covers both voices, the float32 and default
//! pcm16 outputs, speed, seed and one-item-per-batch execution. The optional private-text case reads its
//! hashes from $KOKORO_DATA/evidence/private (never in Git).
//! Binary under test: $KOKORO_BIN or this crate's binary.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/paths.rs"]
mod paths;
use paths::{bin, data, model_dir};

fn sha256_file(p: &Path) -> String {
    kokoro::engine::sha256_bytes(&std::fs::read(p).unwrap())
}

/// Render `input` with the binary under test; returns line -> WAV sha256.
fn render(input: &Path, voice: &str, args: &[String], out: &Path) -> BTreeMap<usize, String> {
    render_env(input, voice, args, out, &[])
}

fn render_env(input: &Path, voice: &str, args: &[String], out: &Path, env: &[(&str, &str)]) -> BTreeMap<usize, String> {
    let _ = std::fs::remove_dir_all(out);
    let mut cmd = Command::new(bin());
    cmd.env_clear().envs(env.iter().copied()).env("PATH", "/nonexistent").env("KOKORO_FRONTEND_DIR", paths::frontend_dir());
    let o = paths::pass_cuda_env(&mut cmd)
        .args(["synth", "--model-dir"])
        .arg(model_dir())
        .arg("--input")
        .arg(input)
        .arg("--out-dir")
        .arg(out)
        .args(["--voice", voice])
        .args(args)
        .output()
        .unwrap();
    assert!(o.status.success(), "synth failed ({}): {}", o.status, String::from_utf8_lossy(&o.stderr));
    let stem = input.file_stem().unwrap().to_string_lossy().into_owned();
    let mut h = BTreeMap::new();
    for e in std::fs::read_dir(out).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if let Some(rest) = name.strip_prefix(&format!("{stem}_")).and_then(|r| r.strip_suffix(".wav")) {
            h.insert(rest.parse::<usize>().unwrap(), sha256_file(&p));
        }
    }
    h
}

/// Differences between rendered and pinned hashes (empty = identical, incl. line coverage).
fn diff(got: &BTreeMap<usize, String>, want: &serde_json::Map<String, serde_json::Value>) -> Vec<String> {
    let mut d = vec![];
    for (k, v) in want {
        let line: usize = k.parse().unwrap();
        match got.get(&line) {
            None => d.push(format!("line {line}: missing")),
            Some(h) if h != v.as_str().unwrap() => d.push(format!("line {line}: sha256 {} != accepted {}", &h[..12], &v.as_str().unwrap()[..12])),
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

fn run_table(table: &serde_json::Value, input_root: Option<&Path>, scratch: &Path) {
    let cases = table["cases"].as_object().unwrap();
    assert!(!cases.is_empty());
    let mut failures = vec![];
    for (case, c) in cases {
        let input = match input_root {
            Some(p) => p.to_path_buf(),
            None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(c["input"].as_str().unwrap()),
        };
        assert_eq!(sha256_file(&input), c["input_sha256"].as_str().unwrap(), "{case}: input file changed");
        let args: Vec<String> = c["args"].as_array().unwrap().iter().map(|a| a.as_str().unwrap().to_string()).collect();
        let got = render(&input, c["voice"].as_str().unwrap(), &args, &scratch.join(case));
        let d = diff(&got, c["wav_sha256"].as_object().unwrap());
        println!("{case}: {} lines, {} differences from the accepted artifact", got.len(), d.len());
        if !d.is_empty() {
            failures.push(format!("{case}: {} of {} lines differ; first: {:?}", d.len(), c["lines"], &d[..d.len().min(3)]));
        }
    }
    assert!(failures.is_empty(), "outputs differ from the accepted BF16x artifact:\n{}", failures.join("\n"));
}

#[test]
fn public_cases_match_accepted_artifact() {
    let table: serde_json::Value = serde_json::from_str(include_str!("pinned/bf16x_golden.json")).unwrap();
    assert_eq!(table["accepted_sha256"], "6fde9d88990a8dc518ec1f366fb17db0869f7e7df5fc1fdea973ea0a371612ea");
    run_table(&table, None, &paths::scratch(&["bf16x_golden"]));
}

/// The checker must catch a changed line, a missing line and an extra line.
#[test]
fn golden_checker_negative_controls() {
    let mut want = serde_json::Map::new();
    want.insert("1".into(), serde_json::json!("a".repeat(64)));
    want.insert("2".into(), serde_json::json!("b".repeat(64)));
    let ok: BTreeMap<usize, String> = [(1, "a".repeat(64)), (2, "b".repeat(64))].into();
    assert!(diff(&ok, &want).is_empty());
    let changed: BTreeMap<usize, String> = [(1, "a".repeat(64)), (2, "c".repeat(64))].into();
    assert_eq!(diff(&changed, &want).len(), 1);
    let missing: BTreeMap<usize, String> = [(1, "a".repeat(64))].into();
    assert_eq!(diff(&missing, &want).len(), 1);
    let extra: BTreeMap<usize, String> = [(1, "a".repeat(64)), (2, "b".repeat(64)), (3, "d".repeat(64))].into();
    assert_eq!(diff(&extra, &want).len(), 1);
}

/// Negative control for the product comparison: with the batch gap masking disabled (fault hook
/// KOKORO_BATCH_NEGCTL_NOMASK), activations leak across batch items; the golden check must see it.
#[test]
fn golden_detects_cross_item_leakage() {
    let table: serde_json::Value = serde_json::from_str(include_str!("pinned/bf16x_golden.json")).unwrap();
    let c = &table["cases"]["edge_af"];
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(c["input"].as_str().unwrap());
    let args: Vec<String> = c["args"].as_array().unwrap().iter().map(|a| a.as_str().unwrap().to_string()).collect();
    let got = render_env(&input, c["voice"].as_str().unwrap(), &args, &paths::scratch(&["bf16x_golden_negctl"]), &[("KOKORO_BATCH_NEGCTL_NOMASK", "1")]);
    let d = diff(&got, c["wav_sha256"].as_object().unwrap());
    println!("edge_af with gap masking disabled: {} of {} lines differ", d.len(), c["lines"]);
    assert!(!d.is_empty(), "cross-item leakage was not detected by the golden comparison");
}

/// Optional private long-form text (316 lines, both voices; KOKORO_PRIVATE_CHAPTER) vs the accepted artifact; hashes and text stay under
/// $KOKORO_DATA/evidence/private.
#[test]
#[ignore = "private corpus; run explicitly"]
fn private_chapter_matches_accepted_artifact() {
    let dir = data().join("evidence/private/release-cleanup/golden");
    let table: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("private_golden.json")).unwrap()).unwrap();
    let input = paths::private_chapter();
    run_table(&table, Some(&input), &paths::scratch(&["bf16x_golden_private"]));
}
