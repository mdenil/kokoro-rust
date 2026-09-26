//! Integrated native frontend differential test: raw line text -> (misaki phoneme string, KPipeline
//! chunks) with the NATIVE spaCy tokenizer/tagger and the NATIVE espeak fallback, vs the pinned
//! reference (oracle/frontend_oracle.py). No oracle tokens/tags/fallback outputs are replayed.
use kokoro::frontend::pipeline::{Chunk, EnglishFrontend, FrontendPaths};
use std::path::PathBuf;
use std::sync::OnceLock;

fn data() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

fn fe() -> &'static EnglishFrontend {
    static FE: OnceLock<EnglishFrontend> = OnceLock::new();
    FE.get_or_init(|| EnglishFrontend::load(&FrontendPaths::under(&data())).expect("native frontend data — missing is NOT a pass"))
}

fn check(file: &str, private: bool) {
    let text = std::fs::read_to_string(data().join(file)).unwrap_or_else(|_| panic!("{file} missing — NOT a pass"));
    let (mut n, mut bad, mut nchunks) = (0usize, vec![], 0usize);
    for l in text.lines().skip(1) {
        let r: serde_json::Value = serde_json::from_str(l).unwrap();
        if r["blank"].as_bool() == Some(true) {
            continue;
        }
        n += 1;
        let line = r["line"].as_u64().unwrap();
        let t = r["text"].as_str().unwrap();
        let mut why = vec![];
        if r.get("error").is_some() {
            if fe().line_chunks(t).is_ok() {
                why.push("reference fails this line; native did not".to_string());
            }
        } else {
            let want: Vec<Chunk> = r["chunks"].as_array().unwrap().iter().map(|c| Chunk { graphemes: c["graphemes"].as_str().unwrap().into(), phonemes: c["phonemes"].as_str().unwrap().into() }).collect();
            nchunks += want.len();
            let (ps, _) = fe().g2p(t).unwrap();
            if ps != r["phonemes"].as_str().unwrap() {
                why.push(if private { "phonemes".into() } else { format!("phonemes\n   got  {ps}\n   want {}", r["phonemes"].as_str().unwrap()) });
            }
            let got = fe().line_chunks(t).unwrap();
            if got != want {
                why.push(if private { "chunks".into() } else { format!("chunks\n   got  {got:?}\n   want {want:?}") });
            }
        }
        if !why.is_empty() {
            bad.push(format!("line {line}: {}", why.join("; ")));
        }
    }
    println!("{file}: {}/{n} lines exact (phonemes + chunks), {nchunks} chunks", n - bad.len());
    for b in bad.iter().take(10) {
        println!("  {b}");
    }
    assert!(n > 0);
    assert!(bad.is_empty(), "{} lines differ", bad.len());
}

#[test]
fn native_frontend_matches_reference_public() {
    println!("{}", fe().ident());
    check("fixtures/frontend/frontend_edge_cases.oracle.jsonl", false);
    check("fixtures/frontend/link_features.oracle.jsonl", false);
    check("fixtures/frontend/alice_full.oracle.jsonl", false);
}

#[test]
#[ignore = "private chapter (local only)"]
fn native_frontend_matches_reference_private() {
    check("evidence/private/frontend/chapter.oracle.jsonl", true);
}
