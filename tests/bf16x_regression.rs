//! Regression gates for the BF16x product build (FMA contraction, -fmad=true):
//! - identical input, options and seed give byte-identical WAVs on every run (reproducibility);
//! - disabling the batch gap masking (fault hook KOKORO_BATCH_NEGCTL_NOMASK) changes outputs, so
//!   cross-item leakage would be visible (negative control against the same binary);
//! - the per-line hash comparison catches changed, missing and extra lines (checker controls).
//! The earlier strict-rounding artifact is compared separately (tests/strict_reference_diff.rs,
//! diagnostic only).

#[path = "support/golden.rs"]
mod golden;
use golden::paths;
use std::collections::BTreeMap;

fn edge_case() -> (std::path::PathBuf, Vec<String>) {
    let input = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/frontend_edge_cases.txt");
    (input, vec!["--format".into(), "float32".into()])
}

fn as_table(h: &BTreeMap<usize, String>) -> serde_json::Map<String, serde_json::Value> {
    h.iter().map(|(k, v)| (k.to_string(), serde_json::json!(v))).collect()
}

#[test]
fn reproducible_across_runs() {
    let (input, args) = edge_case();
    for voice in ["af_heart", "am_adam"] {
        let a = golden::render(&input, voice, &args, &paths::scratch(&["regression", voice, "run1"]));
        let b = golden::render(&input, voice, &args, &paths::scratch(&["regression", voice, "run2"]));
        assert_eq!(a.len(), 65, "{voice}: every line rendered");
        let d = golden::diff(&b, &as_table(&a));
        assert!(d.is_empty(), "{voice}: two runs differ: {d:?}");
        println!("{voice}: {} lines byte-identical across two runs", a.len());
    }
}

#[test]
fn detects_cross_item_leakage() {
    let (input, args) = edge_case();
    let good = golden::render(&input, "af_heart", &args, &paths::scratch(&["regression", "negctl", "masked"]));
    let bad = golden::render_env(&input, "af_heart", &args, &paths::scratch(&["regression", "negctl", "unmasked"]), &[("KOKORO_BATCH_NEGCTL_NOMASK", "1")]);
    let d = golden::diff(&bad, &as_table(&good));
    println!("gap masking disabled: {} of {} lines differ", d.len(), good.len());
    assert!(!d.is_empty(), "cross-item leakage was not detected");
}

#[test]
fn checker_negative_controls() {
    let mut want = serde_json::Map::new();
    want.insert("1".into(), serde_json::json!("a".repeat(64)));
    want.insert("2".into(), serde_json::json!("b".repeat(64)));
    let ok: BTreeMap<usize, String> = [(1, "a".repeat(64)), (2, "b".repeat(64))].into();
    assert!(golden::diff(&ok, &want).is_empty());
    let changed: BTreeMap<usize, String> = [(1, "a".repeat(64)), (2, "c".repeat(64))].into();
    assert_eq!(golden::diff(&changed, &want).len(), 1);
    let missing: BTreeMap<usize, String> = [(1, "a".repeat(64))].into();
    assert_eq!(golden::diff(&missing, &want).len(), 1);
    let extra: BTreeMap<usize, String> = [(1, "a".repeat(64)), (2, "b".repeat(64)), (3, "d".repeat(64))].into();
    assert_eq!(golden::diff(&extra, &want).len(), 1);
}
