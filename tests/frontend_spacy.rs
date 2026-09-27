//! F2 differential tests: native spaCy tokenizer / lexeme attributes / tok2vec+tagger vs the pinned
//! en_core_web_sm 3.8.0 oracle (oracle/spacy_oracle.py) on misaki-preprocessed lines.
use kokoro::frontend::spacy_tag::{word_shape, Tagger};
use kokoro::frontend::spacy_tok::Tokenizer;
use std::path::PathBuf;

#[path = "support/mod.rs"]
mod support;
use support::{Corpus, ALICE, CHAPTER, EDGE};

fn spacy_dir() -> PathBuf {
    support::paths::frontend_dir().join("spacy-en_core_web_sm-3.8.0")
}

struct Line {
    line: u64,
    text: String,
    toks: Vec<serde_json::Value>,
}

/// Pinned spaCy token fixture of a corpus: hash, record count, lines 1..=N, total tokens pinned.
fn load(c: &Corpus) -> Vec<Line> {
    let pin = c.spacy_tokens.expect("corpus has spaCy token fixtures");
    let recs = support::load_jsonl(&pin).unwrap_or_else(|e| panic!("{e}"));
    let lines: Vec<Line> = recs
        .iter()
        .map(|r| Line { line: r["line"].as_u64().unwrap(), text: r["text"].as_str().unwrap().into(), toks: r["tokens"].as_array().unwrap().clone() })
        .collect();
    assert_eq!(lines.len(), c.lines, "{}: lines", pin.path);
    assert_eq!(lines.iter().map(|l| l.toks.len()).sum::<usize>(), c.tokens, "{}: total tokens != pinned", pin.path);
    lines
}

fn check_tokens(c: &Corpus) {
    let private = c.private;
    let file = c.spacy_tokens.unwrap().path;
    let tk = Tokenizer::load(&spacy_dir()).unwrap();
    let lines = load(c);
    let mut bad = vec![];
    for l in &lines {
        let got: Vec<(String, String)> = tk.tokenize(&l.text).into_iter().map(|t| (t.text, if t.space { " ".into() } else { String::new() })).collect();
        let want: Vec<(String, String)> =
            l.toks.iter().map(|t| (t["text"].as_str().unwrap().to_string(), t["ws"].as_str().unwrap().to_string())).collect();
        if got != want {
            bad.push(if private { format!("line {}", l.line) } else { format!("line {}:\n  got  {:?}\n  want {:?}", l.line, got, want) });
        }
    }
    println!("{file}: {}/{} lines token-exact", lines.len() - bad.len(), lines.len());
    for b in bad.iter().take(10) {
        println!("{b}");
    }
    assert!(!lines.is_empty());
    assert!(bad.is_empty(), "{} lines differ", bad.len());
}

#[test]
fn tokenizer_matches_oracle_public() {
    check_tokens(&EDGE);
    check_tokens(&ALICE);
}

#[test]
#[ignore = "private chapter (local only)"]
fn tokenizer_matches_oracle_private() {
    check_tokens(&CHAPTER);
}

/// Feature ids / strings exact; tok2vec tensor within max|d| <= 1e-4 (fixed before judging; f32 BLAS
/// vs f64-accumulated dots); tags EXACT. Fed the NATIVE tokenizer output (already proven exact).
fn check_tagger(c: &Corpus) {
    let private = c.private;
    let prefix = c.spacy_tokens.unwrap().path;
    let tk = Tokenizer::load(&spacy_dir()).unwrap();
    let tg = Tagger::load(&spacy_dir()).unwrap();
    let lines = load(c);
    let seam_pin = c.spacy_seams.unwrap();
    let seams = kokoro::st::parse(&support::read_pinned_bytes(&seam_pin).unwrap_or_else(|e| panic!("{e}"))).unwrap();
    assert_eq!(seams.len(), seam_pin.records, "{}: tensor count (2 per line)", seam_pin.path);
    let (mut ntok, mut tag_bad, mut feat_bad, mut max_d, mut max_ds, mut bad_lines) = (0usize, 0usize, 0usize, 0f32, 0f32, vec![]);
    for l in &lines {
        let toks = tk.tokenize(&l.text);
        assert_eq!(toks.len(), l.toks.len(), "line {}: token count", l.line);
        let mut why = vec![];
        for (t, w) in toks.iter().zip(&l.toks) {
            let ids = tg.features(t);
            let want: Vec<u64> = w["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            assert_eq!(want.len(), 6, "line {}: oracle feature row width", l.line);
            let strings_ok = tg.norm(t) == w["norm"].as_str().unwrap() && word_shape(&t.text) == w["shape"].as_str().unwrap();
            if ids[..] != want[..] || !strings_ok {
                feat_bad += 1;
                why.push(if private { "features".to_string() } else { format!("features {:?}: got {:?} norm {:?} want {:?} {}", t.text, ids, tg.norm(t), want, w) });
            }
        }
        let out = tg.tag(&toks);
        assert_eq!(out.tags.len(), toks.len(), "line {}: tag count", l.line);
        assert_eq!(out.scores.len(), toks.len() * tg.labels.len(), "line {}: score count", l.line);
        let tensor = seams.get(&format!("l{}.tensor", l.line)).unwrap_or_else(|| panic!("seam tensor for line {} missing", l.line)).f32().unwrap();
        assert_eq!(tensor.len(), out.tensor.len(), "line {}: tensor size", l.line);
        assert_eq!(tensor.len(), toks.len() * 96, "line {}: tensor size vs tokens", l.line);
        let scores = seams.get(&format!("l{}.scores", l.line)).unwrap_or_else(|| panic!("seam scores for line {} missing", l.line)).f32().unwrap();
        assert_eq!(scores.len(), out.scores.len(), "line {}: scores size", l.line);
        let ds = scores.iter().zip(&out.scores).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        max_ds = max_ds.max(ds);
        if ds > 1e-4 {
            why.push(format!("scores max|d| {ds:.3e}"));
        }
        let d = tensor.iter().zip(&out.tensor).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        max_d = max_d.max(d);
        if d > 1e-4 {
            why.push(format!("tensor max|d| {d:.3e}"));
        }
        for (i, (got, w)) in out.tags.iter().zip(&l.toks).enumerate() {
            ntok += 1;
            if got != w["tag"].as_str().unwrap() {
                tag_bad += 1;
                why.push(if private { "tag".into() } else { format!("tag[{i}] {:?}: got {got} want {}", l.toks[i]["text"], w["tag"]) });
            }
        }
        if !why.is_empty() {
            bad_lines.push(format!("line {}: {}", l.line, why.join("; ")));
        }
    }
    println!("{prefix}: {} lines, {ntok} tokens; feature mismatches {feat_bad}; tag mismatches {tag_bad}; tensor max|d| {max_d:.3e}; score max|d| {max_ds:.3e}; lines with any diff {}", lines.len(), bad_lines.len());
    assert_eq!(ntok, c.tokens, "tokens checked != pinned");
    assert_eq!(lines.len(), c.lines);
    for b in bad_lines.iter().take(10) {
        println!("  {b}");
    }
    assert!(bad_lines.is_empty());
}

#[test]
fn tagger_matches_oracle_public() {
    check_tagger(&EDGE);
    check_tagger(&ALICE);
}

#[test]
#[ignore = "private chapter (local only)"]
fn tagger_matches_oracle_private() {
    check_tagger(&CHAPTER);
}

/// Negative controls: every subtle rule the port depends on must be load-bearing on the public corpora.
#[test]
fn tagger_negative_controls() {
    let tk = Tokenizer::load(&spacy_dir()).unwrap();
    let lines = load(&ALICE);
    let docs: Vec<_> = lines.iter().map(|l| tk.tokenize(&l.text)).collect();
    let base = Tagger::load(&spacy_dir()).unwrap();
    let want: Vec<Vec<String>> = docs.iter().map(|d| base.tag(d).tags).collect();
    for ctl in ["no_symbols", "no_lexeme_norm", "no_pad"] {
        let mut t = Tagger::load(&spacy_dir()).unwrap();
        t.negative_control(ctl);
        let changed = docs.iter().zip(&want).filter(|(d, w)| t.tag(d).tags != **w).count();
        println!("negative control {ctl}: {changed}/{} Alice lines change tags", docs.len());
        assert!(changed > 0, "{ctl} not detected");
    }
    // special-case NORM override (e.g. "n't" -> "not") must be carried into the features
    let toks = tk.tokenize("I don't know.");
    let nt = toks.iter().find(|t| t.text == "n't").expect("n't token");
    assert_eq!(base.norm(nt), "not");
    // word_shape unit cases (spaCy lex_attrs.word_shape)
    assert_eq!(word_shape("Hello"), "Xxxxx");
    assert_eq!(word_shape("1984"), "dddd");
    assert_eq!(word_shape("C3PO-xyz!"), "XdXX-xxx!");
    assert_eq!(word_shape(&"a".repeat(100)), "LONG");
}

/// Regression (found by the fuzz corpus): the special-case matcher matches token texts regardless of
/// the whitespace between them, but spaCy only applies a rule when span.text (WITH internal spaces)
/// is a rule. ":(" is an emoticon rule; ": (" is not and must stay ":" + "(" with its space.
#[test]
fn special_case_spans_respect_internal_whitespace() {
    let tk = Tokenizer::load(&spacy_dir()).unwrap();
    let show = |s: &str| -> Vec<(String, bool)> { tk.tokenize(s).into_iter().map(|t| (t.text, t.space)).collect() };
    let v = |xs: &[(&str, bool)]| xs.iter().map(|(a, b)| (a.to_string(), *b)).collect::<Vec<_>>();
    assert_eq!(show("sad :( face"), v(&[("sad", true), (":(", true), ("face", false)]));
    assert_eq!(show("note: (aside)"), v(&[("note", false), (":", true), ("(", false), ("aside", false), (")", false)]));
    assert_eq!(show("x = (y)"), v(&[("x", true), ("=", true), ("(", false), ("y", false), (")", false)]));
    assert_eq!(show("'a' 'b'"), v(&[("'", false), ("a", false), ("'", true), ("'", false), ("b", false), ("'", false)]));
}
