//! Differential test: native number words vs num2words 0.5.14 (all 37k oracle rows, exact strings).
use kokoro::frontend::num2words;
use std::path::PathBuf;

#[test]
fn num2words_matches_oracle_exactly() {
    let data = PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()));
    let text = std::fs::read_to_string(data.join("fixtures/frontend/num2words_en.oracle.jsonl")).expect("oracle table (NOT a pass if missing)");
    let (mut n, mut bad) = (0usize, vec![]);
    for line in text.lines() {
        let r: Vec<String> = serde_json::from_str(line).unwrap();
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
    assert!(n > 37000, "oracle table incomplete: {n}");
    println!("{n} rows, {} mismatches", bad.len());
    assert!(bad.is_empty(), "first mismatches: {:#?}", &bad[..bad.len().min(20)]);
}
