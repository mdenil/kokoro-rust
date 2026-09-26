//! F1 differential test (seam): native misaki logic fed the ORACLE's spaCy tokens/tags and the
//! recorded espeak fallback outputs must reproduce the oracle's phonemes, tokens and KPipeline chunks
//! EXACTLY. Tokenizer/tagger (F2) and fallback (F3) are tested separately.
use kokoro::frontend::g2p::{en_tokenize, g2p, Fallback};
use kokoro::frontend::lexicon::Lexicon;
use kokoro::frontend::{MToken, SpacyToken};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

#[path = "support/mod.rs"]
mod support;
use support::{Corpus, ALICE, CHAPTER, EDGE};

fn data() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

struct Recorded {
    map: HashMap<String, (Option<String>, Option<i32>)>,
    missing: RefCell<Vec<String>>,
}

impl Fallback for Recorded {
    fn g2p(&self, tk: &MToken) -> (Option<String>, Option<i32>) {
        match self.map.get(&tk.text) {
            Some(v) => v.clone(),
            None => {
                self.missing.borrow_mut().push(tk.text.clone());
                (None, None)
            }
        }
    }
}

fn run(c: &Corpus) -> (usize, usize, Vec<String>) {
    let private = c.private;
    let lex = Lexicon::load(&data().join("frontend/misaki-0.9.4"), false).unwrap();
    let recs = support::load_corpus(c).unwrap_or_else(|e| panic!("{e}"));
    let (mut n, mut ok, mut bad) = (0, 0, vec![]);
    for r in &recs {
        if r["blank"].as_bool() == Some(true) {
            continue;
        }
        n += 1;
        let spacy: Vec<SpacyToken> = r["spacy"].as_array().unwrap().iter().map(|t| SpacyToken {
            text: t["text"].as_str().unwrap().into(), ws: t["ws"].as_str().unwrap().into(), tag: t["tag"].as_str().unwrap().into() }).collect();
        let map = r["fallback"].as_array().unwrap().iter().map(|c| (c["text"].as_str().unwrap().to_string(),
            (c["phonemes"].as_str().map(String::from), c["rating"].as_i64().map(|x| x as i32)))).collect();
        let fb = Recorded { map, missing: RefCell::new(vec![]) };
        let (ps, mut toks) = g2p(&lex, &spacy, Some(&fb), "").unwrap();
        let chunks = en_tokenize(&mut toks);
        let want_chunks: Vec<(String, String)> = r["chunks"].as_array().unwrap().iter().map(|c| (c["graphemes"].as_str().unwrap().into(), c["phonemes"].as_str().unwrap().into())).collect();
        let line = r["line"].as_u64().unwrap();
        let mut why = vec![];
        if ps != r["phonemes"].as_str().unwrap() {
            why.push(if private { "phonemes differ".to_string() } else { format!("phonemes\n   got  {ps}\n   want {}", r["phonemes"].as_str().unwrap()) });
        }
        if chunks != want_chunks {
            why.push("chunks differ".into());
        }
        if !fb.missing.borrow().is_empty() {
            why.push(if private { "unexpected fallback call".into() } else { format!("unexpected fallback on {:?}", fb.missing.borrow()) });
        }
        if why.is_empty() { ok += 1 } else { bad.push(format!("line {line}: {}", why.join("; "))) }
    }
    assert_eq!(n, c.lines, "{}: checked {n} lines != pinned {}", c.oracle.path, c.lines);
    (n, ok, bad)
}

#[test]
fn g2p_matches_oracle_on_public_corpora() {
    for c in [EDGE, ALICE] {
        let (n, ok, bad) = run(&c);
        println!("{}: {ok}/{n} lines exact", c.oracle.path);
        for b in bad.iter().take(12) {
            println!("  {b}");
        }
        assert!(n > 0);
        assert!(bad.is_empty(), "{}: {} lines differ", c.oracle.path, bad.len());
    }
}

/// Private chapter: aggregate counts only (no text printed).
#[test]
#[ignore = "private fixture; run explicitly"]
fn g2p_matches_oracle_on_private_chapter() {
    let (n, ok, bad) = run(&CHAPTER);
    println!("private chapter: {ok}/{n} lines exact; mismatching line numbers: {:?}", bad.iter().map(|b| b.split(':').next().unwrap().to_string()).collect::<Vec<_>>());
    assert!(bad.is_empty());
}

/// The comparator must fail under deliberate perturbation: flattened POS tags, no fallback, and the
/// British lexicon each have to produce mismatches on the public Alice corpus.
#[test]
fn g2p_negative_controls_are_detected() {
    let recs = support::load_corpus(&ALICE).unwrap_or_else(|e| panic!("{e}"));
    let us = Lexicon::load(&data().join("frontend/misaki-0.9.4"), false).unwrap();
    let gb = Lexicon::load(&data().join("frontend/misaki-0.9.4"), true).unwrap();
    let (mut tag_diff, mut nofb_diff, mut gb_diff) = (0, 0, 0);
    let mut n = 0;
    for r in recs.iter().take(400) {
        n += 1;
        if r["blank"].as_bool() == Some(true) {
            continue;
        }
        let want = r["phonemes"].as_str().unwrap();
        let spacy: Vec<SpacyToken> = r["spacy"].as_array().unwrap().iter().map(|t| SpacyToken {
            text: t["text"].as_str().unwrap().into(), ws: t["ws"].as_str().unwrap().into(), tag: t["tag"].as_str().unwrap().into() }).collect();
        let map: HashMap<String, (Option<String>, Option<i32>)> = r["fallback"].as_array().unwrap().iter().map(|c| (c["text"].as_str().unwrap().to_string(),
            (c["phonemes"].as_str().map(String::from), c["rating"].as_i64().map(|x| x as i32)))).collect();
        let fb = Recorded { map, missing: RefCell::new(vec![]) };
        let flat: Vec<SpacyToken> = spacy.iter().map(|t| SpacyToken { tag: "NN".into(), ..t.clone() }).collect();
        tag_diff += (g2p(&us, &flat, Some(&fb), "").unwrap().0 != want) as usize;
        nofb_diff += (g2p(&us, &spacy, None, "").unwrap().0 != want) as usize;
        gb_diff += (g2p(&gb, &spacy, Some(&fb), "").unwrap().0 != want) as usize;
    }
    assert_eq!(n, 400);
    println!("mismatching lines of 400: flattened tags {tag_diff}, no fallback {nofb_diff}, british lexicon {gb_diff}");
    assert!(tag_diff > 0 && nofb_diff > 0 && gb_diff > 0, "a perturbation went undetected");
}
