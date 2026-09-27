//! Shared fixture pins + validators so differential tests cannot pass vacuously: every fixture is
//! checked for existence, exact sha256, exact record count, complete final line, parseable JSON and
//! (where numbered) line numbers exactly 1..=N in order (no duplicate / missing / reordered rows).
//! Private fixtures are pinned by hash only (their content never enters the repo).
#![allow(dead_code)]

use sha2::Digest;
use std::path::PathBuf;

#[path = "paths.rs"]
pub mod paths;

pub fn data() -> PathBuf {
    paths::data()
}

#[derive(Clone, Copy, Debug)]
pub struct Pin {
    pub path: &'static str,
    pub sha256: &'static str,
    /// data records (excluding a leading meta record)
    pub records: usize,
    /// first JSONL line is a {"meta": ...} record
    pub meta: bool,
    /// records carry "line" = 1..=records in order
    pub numbered: bool,
}

/// Expected aggregate cardinalities of one frontend oracle corpus (pinned from the fixtures).
#[derive(Clone, Copy, Debug)]
pub struct Corpus {
    pub oracle: Pin,
    pub spacy_tokens: Option<Pin>,
    pub spacy_seams: Option<Pin>,
    pub lines: usize,
    pub chunks: usize,
    pub tokens: usize,
    pub distinct_fallback: usize,
    pub private: bool,
}

const fn pin(path: &'static str, sha256: &'static str, records: usize, meta: bool, numbered: bool) -> Pin {
    Pin { path, sha256, records, meta, numbered }
}

pub const EDGE: Corpus = Corpus {
    oracle: pin("fixtures/frontend/frontend_edge_cases.oracle.jsonl", "3c17187485fdd5634e70d9e7b9ae391a9337cb6e8d50854633d9074ae8d59e18", 65, true, true),
    spacy_tokens: Some(pin("fixtures/frontend/edge.spacy.tokens.jsonl", "dd930cafccfc60c24cc564122d2449b5c04957d8917379ac73fe0b0b5aeea7a7", 65, false, true)),
    spacy_seams: Some(pin("fixtures/frontend/edge.spacy.seams.safetensors", "c39cb406726aaa11f1408d439f036514e515a6589cc67f57e5f4f521371bc883", 130, false, false)),
    lines: 65,
    chunks: 67,
    tokens: 822,
    distinct_fallback: 31,
    private: false,
};

pub const ALICE: Corpus = Corpus {
    oracle: pin("fixtures/frontend/alice_full.oracle.jsonl", "4e603df28b636f4d359ff2c82fdcf4ccd55dfc307377a9725af4e18f8a5c8223", 1402, true, true),
    spacy_tokens: Some(pin("fixtures/frontend/alice.spacy.tokens.jsonl", "7009568510b38d46e834a9e586f3ec8326ec47dcd14e434349d71f4b912c78a7", 1402, false, true)),
    spacy_seams: Some(pin("fixtures/frontend/alice.spacy.seams.safetensors", "aa0d720834abcb747dd322813072f496068851623c4d36c5eec666457137b26f", 2804, false, false)),
    lines: 1402,
    chunks: 1413,
    tokens: 34473,
    distinct_fallback: 71,
    private: false,
};

pub const LINKS: Corpus = Corpus {
    oracle: pin("fixtures/frontend/link_features.oracle.jsonl", "b4af830a675c685188eb9e60074befa347ff663464efeb6e61e7d28c982dc62f", 20, true, true),
    spacy_tokens: None,
    spacy_seams: None,
    lines: 20,
    chunks: 20,
    tokens: 137,
    distinct_fallback: 2,
    private: false,
};

pub const CHAPTER: Corpus = Corpus {
    oracle: pin("evidence/private/frontend/chapter.oracle.jsonl", "27c1cd691eae7eb15e4734f2579f0155207e45f1c0ed3ac19ceac9b7f75b3442", 316, true, true),
    spacy_tokens: Some(pin("evidence/private/frontend/chapter.spacy.tokens.jsonl", "047eda875d2a0d15c2fd34251576c8bd2b94ea3e2f06c5d69a36cbe1c110d886", 316, false, true)),
    spacy_seams: Some(pin("evidence/private/frontend/chapter.spacy.seams.safetensors", "55b40645defa2ede0d94b9ddcca82bed811ef23f16a7cc495f5a90b5dcadeebc", 632, false, false)),
    lines: 316,
    chunks: 317,
    tokens: 9137,
    distinct_fallback: 37,
    private: true,
};

/// Synthetic fuzz corpus (bench/make_frontend_fuzz.py, seed 20260926; 4000 lines, 1 reference error).
pub const FUZZ: Corpus = Corpus {
    oracle: pin("fixtures/frontend/fuzz.oracle.jsonl", "6096bf1220b15d4d807a017f36f10c24795fb3bfd7dac801a0efd98294b34f25", 4000, true, true),
    spacy_tokens: None,
    spacy_seams: None,
    lines: 4000,
    chunks: 4388,
    tokens: 127514,
    distinct_fallback: 1843,
    private: false,
};

/// Character-soup corpus (bench/make_frontend_soup.py, seed 11; 3000 lines, 22 reference errors).
pub const SOUP: Corpus = Corpus {
    oracle: pin("fixtures/frontend/soup.oracle.jsonl", "9718e8c16945e77d95537d87f9d1ef2ee4202b53f81c238ad83e6e08c2093674", 3000, true, true),
    spacy_tokens: None,
    spacy_seams: None,
    lines: 3000,
    chunks: 3457,
    tokens: 31843,
    distinct_fallback: 13086,
    private: false,
};

/// The optional private long-form text (316 lines; location from KOKORO_PRIVATE_CHAPTER, content
/// pinned by sha256; never in Git).
pub fn chapter_input() -> Pin {
    let path: &'static str = Box::leak(paths::private_chapter().to_string_lossy().into_owned().into_boxed_str());
    pin(path, "8129112a801ffb67d8974dddaa1de492232ea896f84bd121d45f8cf115b7aaad", 316, false, false)
}

pub const ESPEAK_SYNTHETIC: Pin = pin("fixtures/frontend/espeak_synthetic.oracle.jsonl", "e46f396becce138f193017d46239917aaaa58054d8a038774ada089dd11d4daf", 83, false, false);
pub const NUM2WORDS: Pin = pin("fixtures/frontend/num2words_en.oracle.jsonl", "b1864ab2c33126df92d902d954e091cb4625a219d49d27f8f3940a83c452915e", 37048, false, false);

pub fn sha256_hex(b: &[u8]) -> String {
    sha2::Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

/// Read a pinned file: must exist and match its sha256 exactly.
pub fn read_pinned_bytes(p: &Pin) -> Result<Vec<u8>, String> {
    let path = data().join(p.path);
    let b = std::fs::read(&path).map_err(|e| format!("{}: missing/unreadable ({e}) — NOT a pass", p.path))?;
    check_hash(&b, p)?;
    Ok(b)
}

pub fn check_hash(b: &[u8], p: &Pin) -> Result<(), String> {
    let h = sha256_hex(b);
    if h != p.sha256 {
        return Err(format!("{}: sha256 {h} != pinned {} (fixture changed, truncated or replaced)", p.path, p.sha256));
    }
    Ok(())
}

/// Structural validation of JSONL content, independent of the hash (so it also guards regenerated
/// fixtures): final newline, every line parses, optional meta first, exact record count, and
/// numbered records exactly 1..=N in order.
pub fn parse_jsonl(text: &str, p: &Pin) -> Result<Vec<serde_json::Value>, String> {
    if !text.ends_with('\n') {
        return Err(format!("{}: no final newline (truncated)", p.path));
    }
    let mut recs = vec![];
    for (i, l) in text.split_terminator('\n').enumerate() {
        let v: serde_json::Value = serde_json::from_str(l).map_err(|e| format!("{}: JSONL line {} does not parse ({e}) — truncated/corrupt", p.path, i + 1))?;
        recs.push(v);
    }
    if p.meta {
        if recs.first().map(|r| r.get("meta").is_some()) != Some(true) {
            return Err(format!("{}: missing meta record", p.path));
        }
        recs.remove(0);
    }
    if recs.len() != p.records {
        return Err(format!("{}: {} records != pinned {}", p.path, recs.len(), p.records));
    }
    if p.numbered {
        for (i, r) in recs.iter().enumerate() {
            if r["line"].as_u64() != Some(i as u64 + 1) {
                return Err(format!("{}: record {} has line {} (duplicate/missing/reordered)", p.path, i + 1, r["line"]));
            }
        }
    }
    Ok(recs)
}

/// Pinned JSONL fixture -> validated records.
pub fn load_jsonl(p: &Pin) -> Result<Vec<serde_json::Value>, String> {
    let b = read_pinned_bytes(p)?;
    let text = String::from_utf8(b).map_err(|_| format!("{}: not UTF-8", p.path))?;
    parse_jsonl(&text, p)
}

/// Aggregate cardinalities of a frontend oracle record set (for pinning against `Corpus`).
pub fn oracle_totals(recs: &[serde_json::Value]) -> (usize, usize, usize) {
    let chunks = recs.iter().map(|r| r["chunks"].as_array().map(|a| a.len()).unwrap_or(0)).sum();
    let tokens = recs.iter().map(|r| r["spacy"].as_array().map(|a| a.len()).unwrap_or(0)).sum();
    let mut fb = std::collections::BTreeSet::new();
    for r in recs {
        for c in r["fallback"].as_array().into_iter().flatten() {
            fb.insert((c["text"].to_string(), c["phonemes"].to_string(), c["rating"].to_string()));
        }
    }
    (chunks, tokens, fb.len())
}

/// Load a corpus's oracle and check its aggregate cardinalities against the pins.
pub fn load_corpus(c: &Corpus) -> Result<Vec<serde_json::Value>, String> {
    let recs = load_jsonl(&c.oracle)?;
    let (chunks, tokens, fb) = oracle_totals(&recs);
    if (recs.len(), chunks, tokens, fb) != (c.lines, c.chunks, c.tokens, c.distinct_fallback) {
        return Err(format!(
            "{}: (lines, chunks, tokens, distinct fallback) = {:?} != pinned {:?}",
            c.oracle.path,
            (recs.len(), chunks, tokens, fb),
            (c.lines, c.chunks, c.tokens, c.distinct_fallback)
        ));
    }
    Ok(recs)
}

/// Negative controls for the validators themselves: missing file, truncated content (last record
/// dropped, last line cut mid-JSON), duplicated record, and a single changed byte must all be
/// rejected — by the hash AND, independently, by the structure checks.
pub fn validator_negative_controls(p: &Pin) {
    let good = String::from_utf8(read_pinned_bytes(p).unwrap()).unwrap();
    assert!(parse_jsonl(&good, p).is_ok());
    // missing file
    let missing = Pin { path: "fixtures/frontend/__no_such_fixture__.jsonl", ..*p };
    assert!(read_pinned_bytes(&missing).is_err(), "missing fixture accepted");
    let lines: Vec<&str> = good.split_terminator('\n').collect();
    let join = |v: &[&str]| v.iter().map(|l| format!("{l}\n")).collect::<String>();
    // dropped last record
    let dropped = join(&lines[..lines.len() - 1]);
    // cut mid-line (no final newline, unparsable tail)
    let cut = String::from_utf8_lossy(&good.as_bytes()[..good.len() - 7]).into_owned();
    // duplicated record (and, separately, duplicated with the last one removed to keep the count)
    let mut dup = lines.clone();
    dup.insert(lines.len() / 2, lines[lines.len() / 2]);
    let mut dup_same_count = dup.clone();
    dup_same_count.pop();
    // one changed byte inside a string value
    let flipped = {
        let mut b = good.clone().into_bytes();
        let i = b.iter().rposition(|&c| c.is_ascii_lowercase()).unwrap();
        b[i] = if b[i] == b'a' { b'b' } else { b'a' };
        String::from_utf8(b).unwrap()
    };
    for (name, bad) in [("dropped", dropped), ("cut", cut), ("duplicate", join(&dup)), ("duplicate-same-count", join(&dup_same_count)), ("flipped", flipped)] {
        assert!(check_hash(bad.as_bytes(), p).is_err(), "{name}: hash check accepted a mutated fixture");
        if name != "flipped" && !(name == "duplicate-same-count" && !p.numbered) {
            assert!(parse_jsonl(&bad, p).is_err(), "{name}: structure check accepted a mutated fixture ({})", p.path);
        }
    }
}

/// Words the reference left without a pronunciation in one oracle record: it silently drops them
/// (e.g. "-12", or the "n’t" of "won’t"), whereas the native frontend fails such a line. Same rule as
/// the native check: a token with a letter or digit, not an explicit [text](/phonemes/) feature
/// (rating 5), whose phonemes contain no letter.
pub fn reference_dropped_words(rec: &serde_json::Value) -> Vec<String> {
    let Some(toks) = rec.get("tokens").and_then(|t| t.as_array()) else { return vec![] };
    toks.iter()
        .filter(|t| {
            let text = t["text"].as_str().unwrap_or("");
            let ph = t["phonemes"].as_str().unwrap_or("");
            text.chars().any(char::is_alphanumeric) && t["rating"].as_i64() != Some(5) && !ph.chars().any(char::is_alphabetic)
        })
        .map(|t| t["text"].as_str().unwrap_or("").to_string())
        .collect()
}
