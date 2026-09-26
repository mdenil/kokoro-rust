//! Audiobook line-file interface (owner #7) exercised through the ACTUAL `kokoro` binary.
//! Phoneme-mode input (Python-free today); the text path joins at milestone I1 (native frontend).
//! Uses the CPU engine with short lines so the suite stays portable.

use std::path::{Path, PathBuf};
use std::process::Command;

fn data() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

fn model_dir() -> PathBuf {
    data().join("hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987")
}

fn scratch(name: &str) -> PathBuf {
    let d = data().join("tmp/cli_linefile").join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Runs `kokoro synth` with a PATH that contains no Python and a scrubbed environment.
fn synth(input: &Path, out: &Path, extra: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_kokoro"))
        .env_clear()
        .env("PATH", "/nonexistent")
        .args(["synth", "--device", "cpu", "--threads", "4", "--input-format", "phonemes", "--model-dir"])
        .arg(model_dir())
        .arg("--input")
        .arg(input)
        .arg("--out-dir")
        .arg(out)
        .args(extra)
        .output()
        .expect("run kokoro");
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn manifest(out: &Path, stem: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(out.join(format!("{stem}.manifest.json"))).unwrap()).unwrap()
}

const L1: &str = "həlˈO.";
const L2: &str = "ðə kwˈɪk bɹˈWn fˈɑks.";

#[test]
fn one_output_per_line_in_order_with_duplicates() {
    let d = scratch("order");
    let input = d.join("sec.txt");
    std::fs::write(&input, format!("{L1}\n{L2}\n{L1}\n{L2}\n")).unwrap();
    let (code, err) = synth(&input, &d.join("out"), &[]);
    assert_eq!(code, 0, "{err}");
    let m = manifest(&d.join("out"), "sec");
    let lines = m["lines"].as_array().unwrap();
    assert_eq!(lines.len(), 4);
    for (i, l) in lines.iter().enumerate() {
        assert_eq!(l["line"], i + 1);
        assert_eq!(l["status"], "ok");
        let wav = d.join("out").join(format!("sec_{:05}.wav", i + 1));
        assert!(wav.is_file() && std::fs::metadata(&wav).unwrap().len() > 44, "line {} wav missing", i + 1);
        assert!(l["duration_s"].as_f64().unwrap() > 0.2);
        let sc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("out").join(format!("sec_{:05}.json", i + 1))).unwrap()).unwrap();
        assert_eq!(sc["line"], i + 1);
        assert_eq!(sc["input_sha256"], m["input_sha256"]);
    }
    // duplicates are separate identities (same text hash, separate files)
    assert_eq!(lines[0]["text_sha256"], lines[2]["text_sha256"]);
    assert_eq!(m["complete"], true);
}

#[test]
fn blank_lines_fail_by_default_and_never_shift_numbering() {
    let d = scratch("blank");
    let input = d.join("sec.txt");
    std::fs::write(&input, format!("{L1}\n   \n{L2}\n")).unwrap();
    let (code, err) = synth(&input, &d.join("out"), &[]);
    assert_eq!(code, 1, "blank line must make the job incomplete: {err}");
    assert!(err.contains("line 2: blank"), "{err}");
    let m = manifest(&d.join("out"), "sec");
    assert_eq!(m["lines"][1]["status"], "blank");
    assert!(d.join("out/sec_00001.wav").is_file() && d.join("out/sec_00003.wav").is_file());
    assert!(!d.join("out/sec_00002.wav").exists());
    let (code, err) = synth(&input, &d.join("out"), &["--blank-lines", "skip"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(manifest(&d.join("out"), "sec")["complete"], true);
}

#[test]
fn malformed_inputs_are_explicit() {
    let d = scratch("malformed");
    let bad = d.join("bad.txt");
    std::fs::write(&bad, [L1.as_bytes(), b"\nabc\xff\xfedef\n"].concat()).unwrap();
    let (code, err) = synth(&bad, &d.join("out"), &[]);
    assert_eq!(code, 2);
    assert!(err.contains("not valid UTF-8: line 2, byte 4"), "{err}");
    assert!(!d.join("out/bad.manifest.json").exists(), "nothing may be synthesized for invalid UTF-8");

    let ctl = d.join("ctl.txt");
    std::fs::write(&ctl, format!("{L1}\nhə\u{7}lˈO.\n")).unwrap();
    let (code, err) = synth(&ctl, &d.join("out2"), &[]);
    assert_eq!(code, 1);
    assert!(err.contains("line 2: invalid") && err.contains("U+0007"), "{err}");

    let crlf = d.join("crlf.txt");
    std::fs::write(&crlf, format!("\u{feff}{L1}\r\n{L2}\r\n")).unwrap();
    let (code, err) = synth(&crlf, &d.join("out3"), &[]);
    assert_eq!(code, 0, "{err}");
    let m = manifest(&d.join("out3"), "crlf");
    assert_eq!(m["bom_stripped"], true);
    assert_eq!(m["crlf_lines"], 2);
    assert_eq!(m["lines"].as_array().unwrap().len(), 2);
}

#[test]
fn oversize_line_is_refused_not_truncated() {
    let d = scratch("oversize");
    let input = d.join("sec.txt");
    std::fs::write(&input, format!("{L1}\n{}\n{L2}\n", "a".repeat(511))).unwrap();
    let (code, err) = synth(&input, &d.join("out"), &[]);
    assert_eq!(code, 1);
    assert!(err.contains("line 2: oversize"), "{err}");
    assert!(!d.join("out/sec_00002.wav").exists());
    assert!(d.join("out/sec_00003.wav").is_file(), "later lines keep their identity and are still produced");
}

#[test]
fn resume_skips_only_verified_outputs_and_invalidates_on_change() {
    let d = scratch("resume");
    let input = d.join("sec.txt");
    let out = d.join("out");
    std::fs::write(&input, format!("{L1}\n{L2}\n{L1}\n")).unwrap();
    assert_eq!(synth(&input, &out, &[]).0, 0);
    let resumed = |m: &serde_json::Value| m["lines"].as_array().unwrap().iter().map(|l| l["resumed"].as_bool().unwrap()).collect::<Vec<_>>();

    assert_eq!(synth(&input, &out, &[]).0, 0);
    assert_eq!(resumed(&manifest(&out, "sec")), vec![true, true, true], "unchanged rerun must resume everything");

    // interrupted-write leftovers and a corrupted WAV: only the corrupted line is redone
    std::fs::write(out.join("sec_00002.partial"), b"junk").unwrap();
    let mut w = std::fs::read(out.join("sec_00001.wav")).unwrap();
    w.push(0);
    std::fs::write(out.join("sec_00001.wav"), w).unwrap();
    assert_eq!(synth(&input, &out, &[]).0, 0);
    assert_eq!(resumed(&manifest(&out, "sec")), vec![false, true, true]);

    // changed text on line 3 only
    std::fs::write(&input, format!("{L1}\n{L2}\n{L2}\n")).unwrap();
    assert_eq!(synth(&input, &out, &[]).0, 0);
    assert_eq!(resumed(&manifest(&out, "sec")), vec![true, true, false]);

    // changed settings invalidate every line
    assert_eq!(synth(&input, &out, &["--speed", "1.1"]).0, 0);
    assert_eq!(resumed(&manifest(&out, "sec")), vec![false, false, false]);

    // --force regenerates everything
    assert_eq!(synth(&input, &out, &["--speed", "1.1", "--force"]).0, 0);
    assert_eq!(resumed(&manifest(&out, "sec")), vec![false, false, false]);
}
