//! Exhaustive differential test of the Python str-semantics helpers (src/frontend/pystr.rs and the
//! spaCy tokenizer's py_isspace) against the pinned reference interpreter (Python 3.12.3, Unicode
//! 15.0.0; oracle/pystr_oracle.py): every Unicode scalar value, isalpha / isdigit (+ numeric value) /
//! isspace / isupper and lower / upper / capitalize-first-char mappings.
use kokoro::frontend::pystr;
use kokoro::frontend::spacy_tok::py_isspace;
use std::collections::HashMap;

#[path = "support/mod.rs"]
mod support;

const TABLE: support::Pin = support::Pin {
    path: "fixtures/frontend/pystr_unicode.json",
    sha256: "499e980845cc88d73f2a0a7d72a4a6c5189117dd18edf249d5f4a7a8b2e9f514",
    records: 0,
    meta: false,
    numbered: false,
};

fn in_ranges(r: &[serde_json::Value], cp: u32) -> bool {
    r.iter().any(|x| x[0].as_u64().unwrap() as u32 <= cp && cp <= x[1].as_u64().unwrap() as u32)
}

#[test]
fn pystr_matches_python_exhaustively() {
    let t: serde_json::Value = serde_json::from_slice(&support::read_pinned_bytes(&TABLE).unwrap_or_else(|e| panic!("{e}"))).unwrap();
    assert_eq!(t["python"], "3.12.3");
    assert_eq!(pystr::table_basis(), ("3.12.3", "15.0.0"), "compiled tables vs pinned interpreter");
    let map = |k: &str| -> HashMap<u32, String> { t[k].as_object().unwrap().iter().map(|(a, b)| (a.parse().unwrap(), b.as_str().unwrap().to_string())).collect() };
    let (lower, upper, title) = (map("lower"), map("upper"), map("title"));
    let digits: HashMap<u32, u32> = t["isdigit"].as_object().unwrap().iter().map(|(a, b)| (a.parse().unwrap(), b.as_u64().unwrap() as u32)).collect();
    let (alpha, space, isup) = (t["isalpha"].as_array().unwrap().clone(), t["isspace"].as_array().unwrap().clone(), t["isupper"].as_array().unwrap().clone());
    // precompute range membership as bitsets
    let bits = |r: &[serde_json::Value]| {
        let mut v = vec![false; 0x110000];
        for x in r {
            for cp in x[0].as_u64().unwrap()..=x[1].as_u64().unwrap() {
                v[cp as usize] = true;
            }
        }
        v
    };
    let (alpha, space, isup) = (bits(&alpha), bits(&space), bits(&isup));
    let _ = in_ranges;
    let mut diff: HashMap<&str, Vec<u32>> = HashMap::new();
    let mut n = 0usize;
    for cp in 0..0x110000u32 {
        let Some(c) = char::from_u32(cp) else { continue };
        n += 1;
        let s = c.to_string();
        let mut d = |k: &'static str, bad: bool| {
            if bad {
                diff.entry(k).or_default().push(cp);
            }
        };
        d("isalpha", pystr::char_isalpha(c) != alpha[cp as usize]);
        d("isspace", py_isspace(c) != space[cp as usize]);
        d("isupper", pystr::char_isupper(c) != isup[cp as usize]);
        d("isspace(pystr)", pystr::char_isspace(c) != space[cp as usize]);
        d("isdigit", pystr::char_isdigit(c) != digits.contains_key(&cp));
        if let Some(v) = digits.get(&cp) {
            d("digit_value", pystr::digit_value(c) != Some(*v));
        }
        d("lower", pystr::lower(&s) != *lower.get(&cp).unwrap_or(&s));
        d("upper", pystr::upper(&s) != *upper.get(&cp).unwrap_or(&s));
        d("capitalize", pystr::capitalize(&s) != *title.get(&cp).unwrap_or(&s));
    }
    assert_eq!(n, 0x110000 - 0x800);
    // Final_Sigma context + multi-char strings vs Python results computed in the oracle probes
    assert_eq!(pystr::lower("ΟΔΟΣ"), "οδος");
    assert_eq!(pystr::lower("ΣΑΣ"), "σας");
    assert_eq!(pystr::lower("Σ"), "σ");
    assert_eq!(pystr::lower("A.Σ."), "a.ς.");
    assert_eq!(pystr::capitalize("ΣΑΣ ΣΑΣ"), "Σας σας");
    assert_eq!(pystr::capitalize("ǆemal"), "ǅemal");
    assert_eq!(pystr::capitalize("ßa"), "Ssa");
    assert_eq!(pystr::lower("İstanbul"), "i\u{307}stanbul");
    assert_eq!(pystr::upper("straße"), "STRASSE");
    assert_eq!(pystr::strip("\u{1c} a \u{85}"), "a");
    let mut keys: Vec<_> = diff.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        let v = &diff[k];
        println!("{k}: {} code points differ; first: {:?}", v.len(), v.iter().take(12).map(|c| format!("U+{c:04X}")).collect::<Vec<_>>());
    }
    println!("checked {n} scalar values");
    assert!(diff.is_empty(), "pystr differs from Python 3.12.3 on: {keys:?}");
}
