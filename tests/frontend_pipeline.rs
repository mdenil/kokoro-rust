//! Integrated native frontend differential test: raw line text -> (misaki phoneme string, KPipeline
//! chunks) with the NATIVE spaCy tokenizer/tagger and the NATIVE espeak fallback, vs the pinned
//! reference (oracle/frontend_oracle.py). No oracle tokens/tags/fallback outputs are replayed.
use kokoro::frontend::pipeline::{Chunk, EnglishFrontend, FrontendPaths};
use std::path::PathBuf;
use std::sync::OnceLock;

#[path = "support/mod.rs"]
mod support;
use support::{Corpus, ALICE, CHAPTER, EDGE, FUZZ, LINKS};

fn data() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

fn fe() -> &'static EnglishFrontend {
    static FE: OnceLock<EnglishFrontend> = OnceLock::new();
    FE.get_or_init(|| EnglishFrontend::load(&FrontendPaths::under(&data().join("frontend"))).expect("native frontend data — missing is NOT a pass"))
}

fn check(c: &Corpus) {
    let (file, private) = (c.oracle.path, c.private);
    let recs = support::load_corpus(c).unwrap_or_else(|e| panic!("{e}"));
    let (mut n, mut bad, mut nchunks, mut got_chunks) = (0usize, vec![], 0usize, 0usize);
    for r in &recs {
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
            got_chunks += got.len();
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
    assert_eq!(n, c.lines, "lines checked != pinned");
    assert_eq!(nchunks, c.chunks, "reference chunks != pinned");
    assert_eq!(got_chunks, c.chunks, "native chunks != pinned");
    assert!(bad.is_empty(), "{} lines differ", bad.len());
}

#[test]
fn native_frontend_matches_reference_public() {
    println!("{}", fe().ident());
    check(&EDGE);
    check(&LINKS);
    check(&ALICE);
}

/// Synthetic fuzz corpus: 4000 generated lines of hard constructs (numbers/currency/abbreviations/
/// quotes/links/unicode/long lines), incl. one line the reference itself fails on.
#[test]
fn native_frontend_matches_reference_fuzz() {
    check(&FUZZ);
}

#[test]
#[ignore = "private chapter (local only)"]
fn native_frontend_matches_reference_private() {
    check(&CHAPTER);
}

/// The fixture validators must reject missing, truncated (dropped record / cut line), duplicated
/// and altered fixtures — so no differential test above can pass on a damaged or partial fixture.
#[test]
fn fixture_validators_reject_damaged_fixtures() {
    for c in [EDGE, ALICE, LINKS] {
        support::validator_negative_controls(&c.oracle);
        if let Some(p) = c.spacy_tokens {
            support::validator_negative_controls(&p);
        }
    }
    support::validator_negative_controls(&support::ESPEAK_SYNTHETIC);
    support::validator_negative_controls(&support::NUM2WORDS);
    // a corpus whose pinned cardinalities disagree with its fixture is rejected
    let wrong = Corpus { chunks: EDGE.chunks + 1, ..EDGE };
    assert!(support::load_corpus(&wrong).is_err());
    let wrong = Corpus { tokens: EDGE.tokens - 1, ..EDGE };
    assert!(support::load_corpus(&wrong).is_err());
    println!("validator negative controls: missing / dropped / cut / duplicate / altered rejected for 7 public fixtures");
}

/// Same validator negative controls on the private fixtures (local only; hashes pinned, no content).
#[test]
#[ignore = "private chapter (local only)"]
fn fixture_validators_reject_damaged_fixtures_private() {
    support::validator_negative_controls(&CHAPTER.oracle);
    support::validator_negative_controls(&CHAPTER.spacy_tokens.unwrap());
    support::read_pinned_bytes(&CHAPTER.spacy_seams.unwrap()).unwrap();
    let input = support::read_pinned_bytes(&support::CHAPTER_INPUT).unwrap();
    assert_eq!(input.iter().filter(|&&b| b == b'\n').count(), support::CHAPTER_INPUT.records);
    println!("private fixtures: pins verified; validator negative controls pass");
}

/// Regression (found by the fuzz corpus): misaki's `ord(c.lower())` raises TypeError in the reference
/// when a punctuation-tagged token contains a char whose lowercase is 2 code points (e.g. 'İ') and no
/// earlier char short-circuited; the reference fails the line, so the native frontend must too.
#[test]
fn reference_typeerror_is_reproduced_as_an_error() {
    let recs = support::load_corpus(&FUZZ).unwrap();
    let bad: Vec<_> = recs.iter().filter(|r| r.get("error").is_some()).collect();
    assert_eq!(bad.len(), 1, "pinned: exactly one reference error line in the fuzz corpus");
    let e = fe().line_chunks(bad[0]["text"].as_str().unwrap()).unwrap_err();
    assert!(format!("{e:#}").contains("TypeError"), "{e:#}");
    // same line with İ replaced by I synthesizes fine (the error is specific, not a blanket refusal)
    let fixed = bad[0]["text"].as_str().unwrap().replace('İ', "I");
    assert!(fe().line_chunks(&fixed).is_ok());
}
