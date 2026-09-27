//! DIAGNOSTIC (not a gate): differences between the binary under test and the earlier strict-rounding
//! BF16x artifact (built with -fmad=false; phase2-9b39d48, sha256 6fde9d88…12ea), whose per-line WAV
//! hashes are pinned in tests/pinned/bf16x_golden.json. The current build compiles its kernels with
//! FMA contraction (-fmad=true), so byte differences are expected. These tests report them and fail
//! only when a run breaks or line coverage is wrong. No listening acceptance is implied either way.
//! Run explicitly: cargo test --release --test strict_reference_diff -- --ignored --nocapture

#[path = "support/golden.rs"]
mod golden;
use golden::paths;

fn report(rows: &[(String, usize, Vec<String>)]) {
    let (mut lines, mut differ) = (0, 0);
    for (case, n, d) in rows {
        let coverage = d.iter().filter(|x| x.contains("missing") || x.contains("extra")).count();
        assert_eq!(coverage, 0, "{case}: line coverage differs from the pinned table: {d:?}");
        println!("{case}: {n} lines, {} byte-identical to the strict artifact, {} differ", n - d.len(), d.len());
        lines += n;
        differ += d.len();
    }
    println!("TOTAL: {lines} lines, {} byte-identical, {differ} differ (diagnostic only)", lines - differ);
}

#[test]
#[ignore = "diagnostic: strict (-fmad=false) artifact vs this build; run explicitly"]
fn public_cases_vs_strict_artifact() {
    let table: serde_json::Value = serde_json::from_str(include_str!("pinned/bf16x_golden.json")).unwrap();
    assert_eq!(table["accepted_sha256"], "6fde9d88990a8dc518ec1f366fb17db0869f7e7df5fc1fdea973ea0a371612ea");
    report(&golden::run_table(&table, None, &paths::scratch(&["strict_diff_public"])));
}

/// Optional private long-form text (KOKORO_PRIVATE_CHAPTER; hashes under the data root, never in Git).
#[test]
#[ignore = "diagnostic, private text (local only; KOKORO_PRIVATE_CHAPTER)"]
fn private_text_vs_strict_artifact() {
    let dir = paths::data().join("evidence/private/release-cleanup/golden");
    let table: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("private_golden.json")).unwrap()).unwrap();
    report(&golden::run_table(&table, Some(&paths::private_chapter()), &paths::scratch(&["strict_diff_private"])));
}
