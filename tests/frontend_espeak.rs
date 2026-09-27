//! F3 differential test: native misaki EspeakFallback (phonemizer EspeakBackend semantics over the
//! runtime-loaded pinned libespeak-ng 1.52.0) must reproduce every fallback call the pinned
//! reference made on the corpora EXACTLY (phonemes + rating).
use kokoro::frontend::espeak::Espeak;
use std::collections::BTreeSet;

#[path = "support/mod.rs"]
mod support;
use support::{Corpus, ALICE, CHAPTER, EDGE, LINKS};

fn espeak() -> &'static Espeak {
    let d = support::paths::frontend_dir().join("espeak-ng-1.52.0");
    Espeak::get(&d.join("libespeak-ng.so.1.52.0"), &d).expect("pinned espeak-ng (scripts/stage_espeak.sh) — missing is NOT a pass")
}

fn recorded(c: &Corpus) -> BTreeSet<(String, Option<String>, Option<i64>)> {
    let recs = support::load_corpus(c).unwrap_or_else(|e| panic!("{e}"));
    let mut s = BTreeSet::new();
    for r in &recs {
        for c in r["fallback"].as_array().into_iter().flatten() {
            s.insert((c["text"].as_str().unwrap().to_string(), c["phonemes"].as_str().map(String::from), c["rating"].as_i64()));
        }
    }
    assert_eq!(s.len(), c.distinct_fallback, "{}: distinct fallback calls != pinned", c.oracle.path);
    s
}

fn check(c: &Corpus) {
    let (file, private) = (c.oracle.path, c.private);
    let e = espeak();
    let calls = recorded(c);
    let mut bad = vec![];
    for (text, want, rating) in &calls {
        let got = e.fallback(text).unwrap();
        let got_rating = got.as_ref().map(|_| 2);
        if &got != want || got_rating != *rating {
            bad.push(if private { "(private token)".to_string() } else { format!("{text:?}: got {got:?} want {want:?} (rating {rating:?})") });
        }
    }
    println!("{file}: {}/{} distinct fallback calls exact (espeak {})", calls.len() - bad.len(), calls.len(), e.version);
    for b in bad.iter().take(15) {
        println!("  {b}");
    }
    assert!(!calls.is_empty());
    assert!(bad.is_empty(), "{} fallback outputs differ", bad.len());
}

#[test]
fn fallback_matches_reference_public() {
    check(&EDGE);
    check(&ALICE);
    check(&LINKS);
}

#[test]
#[ignore = "private chapter (local only)"]
fn fallback_matches_reference_private() {
    check(&CHAPTER);
}

/// Synthetic unit set (oracle/espeak_oracle.py): phonemizer punctuation preserve/restore, clauses,
/// digits, non-ASCII, empty results. Both the raw phonemize() list and the misaki mapping must match.
#[test]
fn fallback_matches_reference_synthetic() {
    let e = espeak();
    let rows = support::load_jsonl(&support::ESPEAK_SYNTHETIC).unwrap_or_else(|e| panic!("{e}"));
    let (mut n, mut bad) = (0, vec![]);
    for r in &rows {
        let t = r["text"].as_str().unwrap();
        let raw: Vec<String> = r["phonemize"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        let got_raw = e.phonemize(t).unwrap();
        let got = e.fallback(t).unwrap();
        n += 1;
        if got_raw != raw || got.as_deref() != r["phonemes"].as_str() {
            bad.push(format!("{t:?}: raw {got_raw:?} want {raw:?}; got {got:?} want {}", r["phonemes"]));
        }
    }
    assert_eq!(n, support::ESPEAK_SYNTHETIC.records);
    println!("synthetic: {}/{n} exact", n - bad.len());
    for b in &bad {
        println!("  {b}");
    }
    assert!(bad.is_empty());
}
