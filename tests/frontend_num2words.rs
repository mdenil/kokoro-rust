//! Differential test: native number words vs num2words 0.5.14 (all 37k oracle rows, exact strings).
use kokoro::frontend::num2words;

#[path = "support/mod.rs"]
mod support;

#[test]
fn num2words_matches_oracle_exactly() {
    let rows = support::load_jsonl(&support::NUM2WORDS).unwrap_or_else(|e| panic!("{e}"));
    let (mut n, mut bad) = (0usize, vec![]);
    let mut kinds = std::collections::BTreeMap::new();
    for row in &rows {
        let r: Vec<String> = serde_json::from_value(row.clone()).unwrap();
        assert_eq!(r.len(), 3, "row shape");
        *kinds.entry(r[0].clone()).or_insert(0usize) += 1;
        let got = match r[0].as_str() {
            "cardinal" => num2words::cardinal(r[1].parse().unwrap()),
            "ordinal" => num2words::ordinal(r[1].parse().unwrap()),
            "year" => num2words::year(r[1].parse().unwrap()),
            "float" => num2words::float_str(&r[1]).unwrap_or_default(),
            k => panic!("unknown kind {k}"),
        };
        n += 1;
        if got != r[2] {
            bad.push(format!("{} {}: got {got:?} want {:?}", r[0], r[1], r[2]));
        }
    }
    assert_eq!(n, support::NUM2WORDS.records, "oracle rows checked != pinned");
    assert_eq!(kinds.len(), 4, "every kind (cardinal/ordinal/year/float) present: {kinds:?}");
    println!("{n} rows {kinds:?}, {} mismatches", bad.len());
    assert!(bad.is_empty(), "first mismatches: {:#?}", &bad[..bad.len().min(20)]);
}
