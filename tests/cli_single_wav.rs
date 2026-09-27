//! Output contract of `kokoro synth` through the actual binary (GPU; runs serialized):
//! - default: one WAV for the whole input, `<stem>.wav`, in the current directory;
//! - --diagnostics adds `<stem>.manifest.json`; --per-line writes `<stem>_<line>.wav` instead;
//!   --per-line --diagnostics adds the manifest and one JSON file per WAV;
//! - resume bookkeeping stays in `.kokoro/<stem>/`.
//! Most runs use phoneme input (no text frontend) to stay fast; one uses text input.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

#[path = "support/paths.rs"]
mod paths;

static GPU: Mutex<()> = Mutex::new(());

const LINES: &str = "həlˈO.\nwˈɜɹld.\nðə ˈɛnd ɪz nˈɪɹ.\n";

struct Run {
    code: i32,
    stderr: String,
}

/// `kokoro synth <extra...>` in `cwd`, empty environment (no ffmpeg on PATH unless `path` says so).
fn kokoro(cwd: &Path, extra: &[&str], path: &str, text: bool) -> Run {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut cmd = Command::new(paths::bin());
    cmd.current_dir(cwd).env_clear().env("PATH", path).env("KOKORO_MODEL_DIR", paths::model_dir());
    if text {
        cmd.env("KOKORO_FRONTEND_DIR", paths::frontend_dir());
    }
    paths::pass_cuda_env(&mut cmd).arg("synth");
    if !text {
        cmd.args(["--input-format", "phonemes"]);
    }
    let o = cmd.args(extra).output().expect("run kokoro");
    Run { code: o.status.code().unwrap_or(-1), stderr: String::from_utf8_lossy(&o.stderr).into_owned() }
}

fn ph(cwd: &Path, extra: &[&str]) -> Run {
    kokoro(cwd, extra, "/nonexistent", false)
}

/// A fresh working directory plus an input file in a DIFFERENT directory.
fn setup(name: &str, lines: &str) -> (PathBuf, PathBuf) {
    let d = paths::scratch(&["cli_single_wav", name]);
    let (cwd, inputs) = (d.join("cwd"), d.join("inputs"));
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&inputs).unwrap();
    let input = inputs.join("book.txt");
    std::fs::write(&input, lines).unwrap();
    (cwd, input)
}

/// Files in `dir`, excluding the internal `.kokoro/` bookkeeping.
fn deliverables(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).filter(|n| n != ".kokoro").collect()
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn payload(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap()[44..].to_vec()
}

fn sha(p: &Path) -> String {
    kokoro::engine::sha256_bytes(&std::fs::read(p).unwrap())
}

fn json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn output_sets_for_all_four_flag_combinations() {
    let s = |a: &str| a.to_string();
    let cases: [(&str, Vec<&str>, BTreeSet<String>); 4] = [
        ("default", vec![], set(&["book.wav"])),
        ("diagnostics", vec!["--diagnostics"], set(&["book.wav", "book.manifest.json"])),
        ("per_line", vec!["--per-line"], set(&["book_00001.wav", "book_00002.wav", "book_00003.wav"])),
        (
            "per_line_diagnostics",
            vec!["--per-line", "--diagnostics"],
            set(&["book_00001.wav", "book_00002.wav", "book_00003.wav", "book.manifest.json", "book_00001.json", "book_00002.json", "book_00003.json"]),
        ),
    ];
    for (name, flags, want) in cases {
        let (cwd, input) = setup(name, LINES);
        let mut args = flags.clone();
        let inp = input.to_string_lossy().into_owned();
        args.push(&inp);
        let r = ph(&cwd, &args);
        assert_eq!(r.code, 0, "{name}: {}", r.stderr);
        assert_eq!(deliverables(&cwd), want, "{name}: deliverables in the current directory");
        assert_eq!(deliverables(input.parent().unwrap()), set(&["book.txt"]), "{name}: nothing written next to the input");
        // enabling --diagnostics on a finished run: everything resumes, the JSON appears
        if !flags.contains(&"--diagnostics") {
            let mut args2 = flags.clone();
            args2.push("--diagnostics");
            args2.push(&inp);
            let r = ph(&cwd, &args2);
            assert_eq!(r.code, 0, "{name} + --diagnostics: {}", r.stderr);
            let m = json(&cwd.join("book.manifest.json"));
            assert_eq!(m["counts"]["resumed"], 3, "{name} + --diagnostics: all lines resumed");
            let mut want2 = want.clone();
            want2.insert(s("book.manifest.json"));
            if flags.contains(&"--per-line") {
                for l in 1..=3 {
                    want2.insert(format!("book_{l:05}.json"));
                }
            } else {
                assert_eq!(m["output"]["resumed"], true);
                assert_eq!(m["output"]["sha256"], sha(&cwd.join("book.wav")));
            }
            assert_eq!(deliverables(&cwd), want2, "{name} + --diagnostics on a resumed run");
        }
    }
}

#[test]
fn exact_command_with_text_input_writes_book_wav_in_the_current_directory() {
    // `kokoro synth book.txt`, input in the current directory; then an input elsewhere
    let (cwd, other) = setup("text", "Hello there.\nThis is the second line.\n");
    std::fs::copy(&other, cwd.join("book.txt")).unwrap();
    let r = kokoro(&cwd, &["book.txt"], "/nonexistent", true);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(deliverables(&cwd), set(&["book.txt", "book.wav"]));
    let elsewhere = paths::scratch(&["cli_single_wav", "text-elsewhere"]);
    let r = kokoro(&elsewhere, &[other.to_str().unwrap()], "/nonexistent", true);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(deliverables(&elsewhere), set(&["book.wav"]), "output goes to the process's current directory");
    assert_eq!(sha(&elsewhere.join("book.wav")), sha(&cwd.join("book.wav")));
    // explicit --out-dir still works, and the hidden --input spelling too
    let out = elsewhere.join("chosen");
    let r = kokoro(&elsewhere, &["--input", other.to_str().unwrap(), "--out-dir", out.to_str().unwrap()], "/nonexistent", true);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(deliverables(&out), set(&["book.wav"]));
}

#[test]
fn single_wav_is_the_exact_concatenation_of_the_per_line_audio() {
    for fmt in ["pcm16", "float32"] {
        let (cwd, input) = setup(&format!("exact-{fmt}"), LINES);
        let inp = input.to_str().unwrap();
        let r = ph(&cwd, &["--format", fmt, inp]);
        assert_eq!(r.code, 0, "{}", r.stderr);
        let per = cwd.join("per");
        let r = ph(&cwd, &["--format", fmt, "--per-line", "--out-dir", per.to_str().unwrap(), inp]);
        assert_eq!(r.code, 0, "{}", r.stderr);
        let concat: Vec<u8> = (1..=3).flat_map(|l| payload(&per.join(format!("book_{l:05}.wav")))).collect();
        let whole = std::fs::read(cwd.join("book.wav")).unwrap();
        assert_eq!(&whole[44..], &concat[..], "{fmt}: samples = per-line samples, in order, nothing added");
        let head = &whole[..44];
        assert_eq!(u32::from_le_bytes(head[40..44].try_into().unwrap()) as usize, concat.len(), "{fmt}: data size");
        assert_eq!(u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize, concat.len() + 36, "{fmt}: RIFF size");
        let first = std::fs::read(per.join("book_00001.wav")).unwrap();
        assert_eq!(&head[8..40], &first[8..40], "{fmt}: same format fields as the per-line files");
    }
}

#[test]
fn resume_reuses_line_audio_and_the_finished_file() {
    let (cwd, input) = setup("resume", LINES);
    let inp = input.to_str().unwrap();
    assert_eq!(ph(&cwd, &[inp]).code, 0);
    let before = sha(&cwd.join("book.wav"));
    let mtime = std::fs::metadata(cwd.join("book.wav")).unwrap().modified().unwrap();
    let r = ph(&cwd, &["--diagnostics", inp]);
    assert_eq!(r.code, 0);
    assert!(r.stderr.contains("is up to date"), "{}", r.stderr);
    assert_eq!(std::fs::metadata(cwd.join("book.wav")).unwrap().modified().unwrap(), mtime, "finished file not rewritten");
    let m = json(&cwd.join("book.manifest.json"));
    assert_eq!(m["counts"], serde_json::json!({"done": 0, "resumed": 3, "failed": 0}));
    // a damaged and a missing cached line are synthesized again; the result is the same file
    let cache = cwd.join(".kokoro/book");
    let mut b = std::fs::read(cache.join("book_00002.wav")).unwrap();
    b[100] ^= 0xff;
    std::fs::write(cache.join("book_00002.wav"), &b).unwrap();
    std::fs::remove_file(cache.join("book_00003.wav")).unwrap();
    let r = ph(&cwd, &["--diagnostics", inp]);
    assert_eq!(r.code, 0, "{}", r.stderr);
    let m = json(&cwd.join("book.manifest.json"));
    assert_eq!(m["counts"], serde_json::json!({"done": 2, "resumed": 1, "failed": 0}));
    assert_eq!(sha(&cwd.join("book.wav")), before, "same audio after re-synthesis");
    // editing one line changes only that line and rebuilds the file
    std::fs::write(&input, LINES.replace("wˈɜɹld.", "wˈɜɹldz.")).unwrap();
    let r = ph(&cwd, &["--diagnostics", inp]);
    assert_eq!(r.code, 0);
    let m = json(&cwd.join("book.manifest.json"));
    assert_eq!(m["counts"], serde_json::json!({"done": 1, "resumed": 2, "failed": 0}));
    assert_ne!(sha(&cwd.join("book.wav")), before);
}

#[test]
fn failed_lines_never_produce_a_shortened_file() {
    let (cwd, input) = setup("failed", "həlˈO.\nbad\u{7}line\nðə ˈɛnd.\n");
    let inp = input.to_str().unwrap();
    let r = ph(&cwd, &[inp]);
    assert_eq!(r.code, 1, "{}", r.stderr);
    assert!(r.stderr.contains("line 2: invalid"), "failure is on stderr without --diagnostics: {}", r.stderr);
    assert!(r.stderr.contains("was not written"), "{}", r.stderr);
    assert_eq!(deliverables(&cwd), set(&[]), "no WAV, no JSON");
    // a good earlier output is kept but flagged, never presented as current
    std::fs::write(&input, LINES).unwrap();
    assert_eq!(ph(&cwd, &[inp]).code, 0);
    let good = sha(&cwd.join("book.wav"));
    std::fs::write(&input, "həlˈO.\nbad\u{7}line\nðə ˈɛnd.\n").unwrap();
    let r = ph(&cwd, &["--diagnostics", inp]);
    assert_eq!(r.code, 1);
    assert!(r.stderr.contains("was NOT updated"), "{}", r.stderr);
    assert_eq!(sha(&cwd.join("book.wav")), good, "earlier output left in place");
    let m = json(&cwd.join("book.manifest.json"));
    assert_eq!(m["complete"], false);
    assert_eq!(m["output"]["wav"], serde_json::Value::Null);
    assert_eq!(m["output"]["stale_wav"]["sha256"], good);
}

#[test]
fn blank_and_empty_inputs() {
    // --blank-lines skip: blank lines add nothing (no silence)
    let (cwd, input) = setup("blank", "həlˈO.\n\nðə ˈɛnd.\n");
    let inp = input.to_str().unwrap();
    assert_eq!(ph(&cwd, &["--blank-lines", "error", inp]).code, 1);
    assert!(!cwd.join("book.wav").exists());
    assert_eq!(ph(&cwd, &["--blank-lines", "skip", inp]).code, 0);
    let per = cwd.join("per");
    assert_eq!(ph(&cwd, &["--blank-lines", "skip", "--per-line", "--out-dir", per.to_str().unwrap(), inp]).code, 0);
    let concat: Vec<u8> = [1, 3].iter().flat_map(|l| payload(&per.join(format!("book_{l:05}.wav")))).collect();
    assert_eq!(payload(&cwd.join("book.wav")), concat);
    // empty or all-blank input: explicit failure, no WAV
    for (name, text) in [("empty", ""), ("all-blank", "\n  \n")] {
        let (cwd, input) = setup(name, text);
        let r = ph(&cwd, &["--blank-lines", "skip", input.to_str().unwrap()]);
        assert_eq!(r.code, 1, "{name}: {}", r.stderr);
        assert!(r.stderr.contains("nothing to synthesize"), "{name}: {}", r.stderr);
        assert_eq!(deliverables(&cwd), set(&[]), "{name}");
    }
}

#[test]
fn encode_converts_the_whole_file_once() {
    let ffmpeg = paths::which("ffmpeg").expect("ffmpeg on PATH is needed for this test - NOT a pass");
    let path = format!("{}:/nonexistent", ffmpeg.parent().unwrap().display());
    let (cwd, input) = setup("encode", LINES);
    let inp = input.to_str().unwrap();
    let r = kokoro(&cwd, &["--encode", "flac", inp], &path, false);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(deliverables(&cwd), set(&["book.wav", "book.flac"]));
    let mtime = std::fs::metadata(cwd.join("book.flac")).unwrap().modified().unwrap();
    let r = kokoro(&cwd, &["--encode", "flac", inp], &path, false);
    assert_eq!(r.code, 0);
    assert_eq!(std::fs::metadata(cwd.join("book.flac")).unwrap().modified().unwrap(), mtime, "not converted again");
    // without ffmpeg: the WAV is written, the failed conversion is reported
    let (cwd, input) = setup("encode-missing", LINES);
    let r = ph(&cwd, &["--encode", "flac", input.to_str().unwrap()]);
    assert_eq!(r.code, 1, "{}", r.stderr);
    assert!(r.stderr.contains("converting it to flac failed"), "{}", r.stderr);
    assert_eq!(deliverables(&cwd), set(&["book.wav"]));
}

#[test]
fn changing_flags_leaves_other_files_alone() {
    let (cwd, input) = setup("flags", LINES);
    let inp = input.to_str().unwrap();
    std::fs::write(cwd.join("notes.txt"), "mine").unwrap();
    assert_eq!(ph(&cwd, &["--per-line", inp]).code, 0);
    assert_eq!(ph(&cwd, &[inp]).code, 0);
    assert_eq!(
        deliverables(&cwd),
        set(&["notes.txt", "book.wav", "book_00001.wav", "book_00002.wav", "book_00003.wav"]),
        "earlier per-line WAVs and unrelated files are kept"
    );
}
