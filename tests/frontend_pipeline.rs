//! Integrated native frontend differential test: raw line text -> (misaki phoneme string, KPipeline
//! chunks) with the NATIVE spaCy tokenizer/tagger and the NATIVE espeak fallback, vs the pinned
//! reference (oracle/frontend_oracle.py). No oracle tokens/tags/fallback outputs are replayed.
//! Runs with the reference eSpeak NG copy (the fixtures' backend). Lines where the reference silently
//! dropped a word (a token left without phonemes) must instead FAIL natively (no dropped words).
use kokoro::frontend::espeak::EspeakSource;
use kokoro::frontend::pipeline::{Chunk, EnglishFrontend, FrontendPaths};
use std::sync::OnceLock;

#[path = "support/mod.rs"]
mod support;
use support::{Corpus, ALICE, CHAPTER, EDGE, FUZZ, LINKS, SOUP};


fn fe() -> &'static EnglishFrontend {
    static FE: OnceLock<EnglishFrontend> = OnceLock::new();
    FE.get_or_init(|| {
        let mut p = FrontendPaths::under(&support::paths::frontend_dir());
        let (library, data) = support::paths::espeak_reference();
        p.espeak = EspeakSource { library: Some(library), data: Some(data) };
        EnglishFrontend::load(&p).expect("native frontend data — missing is NOT a pass")
    })
}

fn check(c: &Corpus) {
    let (file, private) = (c.oracle.path, c.private);
    let recs = support::load_corpus(c).unwrap_or_else(|e| panic!("{e}"));
    let (mut n, mut bad, mut nchunks, mut got_chunks, mut refused, mut refused_chunks) = (0usize, vec![], 0usize, 0usize, 0usize, 0usize);
    for r in &recs {
        if r["blank"].as_bool() == Some(true) {
            continue;
        }
        n += 1;
        let line = r["line"].as_u64().unwrap();
        let t = r["text"].as_str().unwrap();
        let mut why = vec![];
        let dropped = support::reference_dropped_words(r);
        if r.get("error").is_none() && !dropped.is_empty() {
            refused += 1;
            nchunks += r["chunks"].as_array().unwrap().len();
            refused_chunks += r["chunks"].as_array().unwrap().len();
            match fe().line_chunks(t) {
                Ok(_) => why.push(format!("reference dropped {} word(s); native did not refuse the line", dropped.len())),
                Err(e) if !format!("{e:#}").contains("unresolved word") => why.push("reference dropped a word; native failed differently".into()),
                Err(_) => {}
            }
        } else if let Some(cls) = r.get("error").and_then(|e| e.as_str()) {
            match fe().line_chunks(t) {
                Ok(_) => why.push("reference fails this line; native did not".to_string()),
                Err(e) if !format!("{e:#}").contains(cls) => why.push(format!("reference raises {cls}; native fails differently")),
                Err(_) => {}
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
    println!("{file}: {}/{n} lines as expected (exact phonemes + chunks; {refused} refused where the reference dropped a word), {nchunks} chunks", n - bad.len());
    for b in bad.iter().take(10) {
        println!("  {b}");
    }
    assert_eq!(n, c.lines, "lines checked != pinned");
    assert_eq!(nchunks, c.chunks, "reference chunks != pinned");
    assert_eq!(got_chunks + refused_chunks, c.chunks, "native chunks + reference chunks of refused lines != pinned");
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

/// Character-level soup (regex / Unicode / reference-crash edges): 3000 lines, 22 reference errors.
#[test]
fn native_frontend_matches_reference_soup() {
    let recs = support::load_corpus(&SOUP).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(recs.iter().filter(|r| r.get("error").is_some()).count(), 22, "pinned reference error lines");
    check(&SOUP);
}

#[test]
#[ignore = "private text (local only; KOKORO_PRIVATE_CHAPTER)"]
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
#[ignore = "private text (local only; KOKORO_PRIVATE_CHAPTER)"]
fn fixture_validators_reject_damaged_fixtures_private() {
    support::validator_negative_controls(&CHAPTER.oracle);
    support::validator_negative_controls(&CHAPTER.spacy_tokens.unwrap());
    support::read_pinned_bytes(&CHAPTER.spacy_seams.unwrap()).unwrap();
    let pin = support::chapter_input();
    let input = support::read_pinned_bytes(&pin).unwrap();
    assert_eq!(input.iter().filter(|&&b| b == b'\n').count(), pin.records);
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
