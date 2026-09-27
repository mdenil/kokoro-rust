//! The product's eSpeak NG backend: the separately installed system library and data (default
//! discovery, no explicit paths). Needs eSpeak NG installed (Debian/Ubuntu: libespeak-ng1); a missing
//! installation fails these tests (NOT a pass). Pronunciations are whatever the installed version
//! gives; these tests check loading, identity, coverage and the no-dropped-words rule, not equality
//! with the reference's eSpeak NG 1.52.0 (that is tests/frontend_espeak.rs).
use kokoro::frontend::espeak::{Espeak, EspeakSource, MIN_VERSION};
use kokoro::frontend::pipeline::{EnglishFrontend, FrontendPaths};
use sha2::Digest;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

#[path = "support/mod.rs"]
mod support;
use support::paths;

fn fe() -> &'static EnglishFrontend {
    static FE: OnceLock<EnglishFrontend> = OnceLock::new();
    FE.get_or_init(|| EnglishFrontend::load(&FrontendPaths::under(&paths::frontend_dir())).expect("frontend with the system eSpeak NG — missing is NOT a pass"))
}

fn sha256_hex(b: &[u8]) -> String {
    sha2::Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn system_backend_loads_and_reports_identity() {
    let e = fe().espeak();
    let v: Vec<u32> = e.version.split(|c: char| !c.is_ascii_digit()).take(2).map(|x| x.parse().unwrap()).collect();
    assert!((v[0], v[1]) >= MIN_VERSION, "version {}", e.version);
    assert!(e.library.is_file(), "loaded library {} is not a file", e.library.display());
    assert!(e.data_dir.join("phontab").is_file(), "selected data dir {}", e.data_dir.display());
    assert_eq!(e.library_sha256, sha256_hex(&std::fs::read(&e.library).unwrap()), "library hash is of the mapped file");
    assert_eq!(e.source, EspeakSource::default());
    assert!(fe().ident().contains(&e.describe()), "frontend identity carries the backend: {}", fe().ident());
    assert!(Espeak::get(&EspeakSource { library: Some(e.library.clone()), data: None }).is_err(), "a second, different source must be refused in-process");
    println!("system eSpeak NG {}: library {} (sha256 {}), data {} (en-us sha256 {})", e.version, e.library.display(), &e.library_sha256[..16], e.data_dir.display(), &e.data_sha256[..16]);
}

#[test]
fn known_and_oov_words() {
    let e = fe().espeak();
    for w in ["hello", "Zyxquorbl", "Kaczmarczyk"] {
        let p = e.fallback(w).unwrap().unwrap_or_default();
        assert!(p.chars().any(char::is_alphabetic), "{w}: {p:?}");
    }
    // lexicon words, and an OOV word that goes through the fallback, inside ordinary sentences
    for line in ["The cat sat on the mat.", "Mr. Zyxquorbl visited Kaczmarczyk yesterday!"] {
        let ch = fe().line_chunks(line).unwrap();
        assert!(!ch.is_empty() && ch.iter().all(|c| c.phonemes.chars().any(char::is_alphabetic)), "{line}: {ch:?}");
    }
}

/// Negative controls: words the reference silently drops make the line fail; punctuation alone does not.
#[test]
fn no_dropped_words() {
    for (line, word) in [("The temperature dropped to -12 degrees overnight.", "\"-12\""), ("\u{201C}I won\u{2019}t!\u{201D}", "n\u{2019}t")] {
        let err = format!("{:#}", fe().line_chunks(line).expect_err(line));
        assert!(err.contains("unresolved word") && err.contains(word), "{line}: {err}");
    }
    for line in ["...", "\u{2014} !", "(\u{201C}\u{201D})"] {
        fe().line_chunks(line).unwrap_or_else(|e| panic!("{line:?}: punctuation-only line failed: {e:#}"));
    }
}

/// Every public edge case either synthesizes or is refused for an unresolved word, and exactly the
/// lines where the reference dropped a word are refused. Differences from the reference's phonemes
/// (eSpeak NG version) are counted, not asserted.
#[test]
fn edge_corpus_coverage() {
    let recs = support::load_corpus(&support::EDGE).unwrap_or_else(|e| panic!("{e}"));
    let (mut ok, mut refused, mut differ) = (0, vec![], 0);
    for r in recs.iter().filter(|r| r["blank"].as_bool() != Some(true) && r.get("error").is_none()) {
        let line = r["line"].as_u64().unwrap();
        match fe().g2p(r["text"].as_str().unwrap()) {
            Ok((ps, _)) => {
                ok += 1;
                differ += (ps != r["phonemes"].as_str().unwrap()) as usize;
            }
            Err(e) => {
                assert!(format!("{e:#}").contains("unresolved word"), "line {line}: {e:#}");
                refused.push(line);
            }
        }
        assert_eq!(support::reference_dropped_words(r).is_empty(), !refused.contains(&line), "line {line}: refusal vs reference drop");
    }
    println!("edge cases with eSpeak NG {}: {ok} ok ({differ} with phonemes differing from the reference), refused {refused:?}", fe().espeak().version);
    assert_eq!(refused, vec![7]);
}

fn run(input: &Path, out: &Path, extra: &[&str], env: &[(&str, &Path)]) -> (i32, String) {
    let mut cmd = Command::new(paths::bin());
    cmd.env_clear().env("PATH", "/nonexistent").env("KOKORO_FRONTEND_DIR", paths::frontend_dir());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let o = paths::pass_cuda_env(&mut cmd).args(["synth", "--model-dir"]).arg(paths::model_dir()).arg("--input").arg(input).arg("--out-dir").arg(out).args(extra).output().unwrap();
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// A shared library without the eSpeak NG API (for the symbol probe).
fn non_espeak_library() -> PathBuf {
    ["/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/lib64", "/usr/lib64", "/lib", "/usr/lib"]
        .iter()
        .map(|d| Path::new(d).join("libm.so.6"))
        .find(|p| p.is_file())
        .expect("no libm.so.6 found for the symbol probe - NOT a pass")
}

/// Unusable installations are job-level errors (exit 2) with an actionable message.
#[test]
fn unusable_backend_is_an_actionable_error() {
    let d = paths::scratch(&["system_espeak", "probes"]);
    let input = d.join("p.txt");
    std::fs::write(&input, "Hello there.\n").unwrap();
    let missing = d.join("no-such-libespeak-ng.so.1");
    let nodata = d.join("empty");
    std::fs::create_dir_all(&nodata).unwrap();
    for (what, args, want) in [
        ("missing library", vec!["--espeak-lib".to_string(), missing.display().to_string()], "apt install libespeak-ng1"),
        ("library without the API", vec!["--espeak-lib".to_string(), non_espeak_library().display().to_string()], "no symbol espeak_Initialize"),
        ("missing data", vec!["--espeak-data".to_string(), nodata.display().to_string()], "espeak-ng-data (with phontab) not found"),
    ] {
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        let (code, err) = run(&input, &d.join("out"), &a, &[]);
        assert_eq!(code, 2, "{what}: {err}");
        assert!(err.contains(want), "{what}: {err}");
    }
}

/// Native text file -> WAVs with the system backend; the manifest records the backend; switching
/// to another eSpeak NG installation invalidates resume; the same one resumes everything.
#[test]
fn text_to_wav_and_resume_follow_the_backend() {
    let d = paths::scratch(&["system_espeak", "resume"]);
    let input = d.join("s.txt");
    std::fs::write(&input, "Hello there.\nMr. Zyxquorbl visited Kaczmarczyk yesterday!\nThe temperature dropped to -12 degrees overnight.\n").unwrap();
    let out = d.join("out");
    let (code, err) = run(&input, &out, &[], &[]);
    assert_eq!(code, 1, "line 3 must be refused: {err}");
    let m = json(&out.join("s.manifest.json"));
    let e = fe().espeak();
    assert_eq!(m["espeak"]["version"], e.version.as_str());
    assert_eq!(m["espeak"]["library_sha256"], e.library_sha256.as_str());
    assert_eq!(m["counts"]["done"], 2);
    for l in 1..=2 {
        let sc = json(&out.join(format!("s_0000{l}.json")));
        assert_eq!(sc["status"], "ok");
        assert!(sc["config"]["frontend"].as_str().unwrap().contains(&e.describe()));
        assert!(out.join(format!("s_0000{l}.wav")).is_file());
    }
    let sc3 = json(&out.join("s_00003.json"));
    assert!(sc3["error"].as_str().unwrap().contains("unresolved word"), "{}", sc3["error"]);
    assert!(!out.join("s_00003.wav").exists());
    // same backend: everything done resumes
    run(&input, &out, &[], &[]);
    assert_eq!(json(&out.join("s.manifest.json"))["counts"]["resumed"], 2);
    // another installation (the reference copy): nothing is reused
    let (lib, data) = paths::espeak_reference();
    run(&input, &out, &[], &[("KOKORO_ESPEAK_LIB", &lib), ("KOKORO_ESPEAK_DATA", &data)]);
    let m = json(&out.join("s.manifest.json"));
    assert_eq!(m["counts"]["resumed"], 0, "a different eSpeak NG must not reuse outputs");
    assert_ne!(m["espeak"]["library_sha256"], e.library_sha256.as_str());
}
