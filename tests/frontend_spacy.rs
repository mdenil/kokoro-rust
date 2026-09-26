//! F2 differential tests: native spaCy tokenizer / lexeme attributes / tok2vec+tagger vs the pinned
//! en_core_web_sm 3.8.0 oracle (oracle/spacy_oracle.py) on misaki-preprocessed lines.
use kokoro::frontend::spacy_tag::{word_shape, Tagger};
use kokoro::frontend::spacy_tok::Tokenizer;
use std::path::PathBuf;

fn data() -> PathBuf {
    PathBuf::from(std::env::var("KOKORO_DATA").unwrap_or_else(|_| "/data/mdenil/code/kokoro-rust".into()))
}

fn spacy_dir() -> PathBuf {
    data().join("frontend/spacy-en_core_web_sm-3.8.0")
}

struct Line {
    line: u64,
    text: String,
    toks: Vec<serde_json::Value>,
}

fn load(file: &PathBuf) -> Vec<Line> {
    let s = std::fs::read_to_string(file).unwrap_or_else(|_| panic!("{} missing — NOT a pass", file.display()));
    s.lines()
        .map(|l| {
            let r: serde_json::Value = serde_json::from_str(l).unwrap();
            Line { line: r["line"].as_u64().unwrap(), text: r["text"].as_str().unwrap().into(), toks: r["tokens"].as_array().unwrap().clone() }
        })
        .collect()
}

fn check_tokens(file: PathBuf, private: bool) {
    let tk = Tokenizer::load(&spacy_dir()).unwrap();
    let lines = load(&file);
    let mut bad = vec![];
    for l in &lines {
        let got: Vec<(String, String)> = tk.tokenize(&l.text).into_iter().map(|t| (t.text, if t.space { " ".into() } else { String::new() })).collect();
        let want: Vec<(String, String)> =
            l.toks.iter().map(|t| (t["text"].as_str().unwrap().to_string(), t["ws"].as_str().unwrap().to_string())).collect();
        if got != want {
            bad.push(if private { format!("line {}", l.line) } else { format!("line {}:\n  got  {:?}\n  want {:?}", l.line, got, want) });
        }
    }
    println!("{}: {}/{} lines token-exact", file.display(), lines.len() - bad.len(), lines.len());
    for b in bad.iter().take(10) {
        println!("{b}");
    }
    assert!(!lines.is_empty());
    assert!(bad.is_empty(), "{} lines differ", bad.len());
}

#[test]
fn tokenizer_matches_oracle_public() {
    check_tokens(data().join("fixtures/frontend/edge.spacy.tokens.jsonl"), false);
    check_tokens(data().join("fixtures/frontend/alice.spacy.tokens.jsonl"), false);
}

#[test]
#[ignore = "private chapter (local only)"]
fn tokenizer_matches_oracle_private() {
    check_tokens(data().join("evidence/private/frontend/chapter.spacy.tokens.jsonl"), true);
}

/// Feature ids / strings exact; tok2vec tensor within max|d| <= 1e-4 (fixed before judging; f32 BLAS
/// vs f64-accumulated dots); tags EXACT. Fed the NATIVE tokenizer output (already proven exact).
fn check_tagger(prefix: &str, private: bool) {
    let tk = Tokenizer::load(&spacy_dir()).unwrap();
    let tg = Tagger::load(&spacy_dir()).unwrap();
    let lines = load(&data().join(format!("{prefix}.tokens.jsonl")));
    let seams = kokoro::st::load(&data().join(format!("{prefix}.seams.safetensors"))).unwrap();
    let (mut ntok, mut tag_bad, mut feat_bad, mut max_d, mut bad_lines) = (0usize, 0usize, 0usize, 0f32, vec![]);
    for l in &lines {
        let toks = tk.tokenize(&l.text);
        assert_eq!(toks.len(), l.toks.len(), "line {}: token count", l.line);
        let mut why = vec![];
        for (t, w) in toks.iter().zip(&l.toks) {
            let ids = tg.features(t);
            let want: Vec<u64> = w["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            let strings_ok = tg.norm(t) == w["norm"].as_str().unwrap() && word_shape(&t.text) == w["shape"].as_str().unwrap();
            if ids[..] != want[..] || !strings_ok {
                feat_bad += 1;
                why.push(if private { "features".to_string() } else { format!("features {:?}: got {:?} norm {:?} want {:?} {}", t.text, ids, tg.norm(t), want, w) });
            }
        }
        let out = tg.tag(&toks);
        let tensor = seams.get(&format!("l{}.tensor", l.line)).unwrap().f32().unwrap();
        assert_eq!(tensor.len(), out.tensor.len());
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
    println!("{prefix}: {} lines, {ntok} tokens; feature mismatches {feat_bad}; tag mismatches {tag_bad}; tensor max|d| {max_d:.3e}; lines with any diff {}", lines.len(), bad_lines.len());
    for b in bad_lines.iter().take(10) {
        println!("  {b}");
    }
    assert!(bad_lines.is_empty());
}

#[test]
fn tagger_matches_oracle_public() {
    check_tagger("fixtures/frontend/edge.spacy", false);
    check_tagger("fixtures/frontend/alice.spacy", false);
}

#[test]
#[ignore = "private chapter (local only)"]
fn tagger_matches_oracle_private() {
    check_tagger("evidence/private/frontend/chapter.spacy", true);
}

/// Negative controls: every subtle rule the port depends on must be load-bearing on the public corpora.
#[test]
fn tagger_negative_controls() {
    let tk = Tokenizer::load(&spacy_dir()).unwrap();
    let lines = load(&data().join("fixtures/frontend/alice.spacy.tokens.jsonl"));
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
