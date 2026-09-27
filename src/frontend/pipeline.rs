//! The complete native English text frontend = production KPipeline(lang_code='a') text path:
//! strip -> split r'\n+' -> per segment misaki G2P (preprocess link features, spaCy tokenizer +
//! tagger, feature alignment, fold_left, lexicon / number / espeak fallback) -> en_tokenize chunks.

use super::espeak::{Espeak, EspeakSource};
use super::g2p::{en_tokenize, g2p_tokens_unresolved, Fallback};
use super::lexicon::Lexicon;
use super::pystr;
use super::spacy_tag::Tagger;
use super::spacy_tok::{py_isspace, Tokenizer};
use super::MToken;
use anyhow::{bail, Context, Result};
use fancy_regex::Regex;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Where the frontend's language data lives (hashed into `ident`); eSpeak NG is the system install
/// unless overridden.
#[derive(Clone, Debug)]
pub struct FrontendPaths {
    /// misaki 0.9.4 lexicon JSONs (us_gold.json, us_silver.json, ...)
    pub lexicon_dir: PathBuf,
    /// exported en_core_web_sm 3.8.0 tokenizer/tagger data
    pub spacy_dir: PathBuf,
    /// eSpeak NG library and data (default: the system installation)
    pub espeak: EspeakSource,
}

impl FrontendPaths {
    /// Standard layout under one frontend data directory, with the system eSpeak NG.
    pub fn under(frontend_dir: &Path) -> Self {
        let f = frontend_dir;
        Self { lexicon_dir: f.join("misaki-0.9.4"), spacy_dir: f.join("spacy-en_core_web_sm-3.8.0"), espeak: EspeakSource::default() }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub graphemes: String,
    pub phonemes: String,
}

pub struct EnglishFrontend {
    lex: Lexicon,
    tok: Tokenizer,
    tagger: Tagger,
    espeak: &'static Espeak,
    ident: String,
}

#[derive(Clone, Debug, PartialEq)]
enum Feature {
    Num(f64),
    Str(String),
}

fn link_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\[([^\]]+)\]\(([^\)]*)\)").unwrap())
}

fn py_split_ws(s: &str) -> Vec<String> {
    s.split(py_isspace).filter(|w| !w.is_empty()).map(String::from).collect()
}

fn is_ascii_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// misaki G2P.preprocess: [text](feature) links -> (text, whitespace-split words, {word index: feature}).
fn preprocess(text: &str) -> (String, Vec<String>, Vec<(usize, Feature)>) {
    let text = text.trim_start_matches(py_isspace);
    let (mut result, mut tokens, mut features) = (String::new(), vec![], vec![]);
    let mut last_end = 0;
    for m in link_regex().captures_iter(text).filter_map(|c| c.ok()) {
        let whole = m.get(0).unwrap();
        let between = &text[last_end..whole.start()];
        result.push_str(between);
        tokens.extend(py_split_ws(between));
        let f = m.get(2).unwrap().as_str();
        let signless = if f.starts_with('-') || f.starts_with('+') { &f[1..] } else { f };
        let fchars: Vec<char> = f.chars().collect();
        let feat = if is_ascii_digits(signless) {
            Some(Feature::Num(f.parse::<f64>().unwrap_or(0.0)))
        } else if f == "0.5" || f == "+0.5" {
            Some(Feature::Num(0.5))
        } else if f == "-0.5" {
            Some(Feature::Num(-0.5))
        } else if fchars.len() > 1 && fchars[0] == '/' && fchars[fchars.len() - 1] == '/' {
            Some(Feature::Str(format!("/{}", f[1..].trim_end_matches('/'))))
        } else if fchars.len() > 1 && fchars[0] == '#' && fchars[fchars.len() - 1] == '#' {
            Some(Feature::Str(format!("#{}", f[1..].trim_end_matches('#'))))
        } else {
            None
        };
        if let Some(feat) = feat {
            features.push((tokens.len(), feat));
        }
        let g1 = m.get(1).unwrap().as_str();
        result.push_str(g1);
        tokens.push(g1.to_string());
        last_end = whole.end();
    }
    if last_end < text.len() {
        result.push_str(&text[last_end..]);
        tokens.extend(py_split_ws(&text[last_end..]));
    }
    (result, tokens, features)
}

/// spacy.training.align.get_alignments(A, B) -> y2x (per B token, sorted A indices). Python code-point
/// semantics, including the lower()-length quirks and the E949 error.
fn get_alignments_y2x(a: &[String], b: &[String]) -> Result<Vec<Vec<usize>>> {
    let expand = |toks: &[String]| -> (Vec<usize>, Vec<char>) {
        let mut map = vec![];
        let mut chars = vec![];
        for (i, x) in toks.iter().enumerate() {
            let low: Vec<char> = pystr::lower(x).chars().collect();
            map.extend(std::iter::repeat(i).take(low.len()));
            chars.extend(low);
        }
        (map, chars)
    };
    let (c2a, sa) = expand(a);
    let (c2b, sb) = expand(b);
    let strip = |v: &[char]| -> String { v.iter().filter(|c| !py_isspace(**c)).collect() };
    if strip(&sa) != strip(&sb) {
        bail!("spaCy alignment E949: link-feature tokens do not match the tokenization");
    }
    let alen = |i: usize| a[i].chars().count();
    let blen = |i: usize| b[i].chars().count();
    let (mut ia, mut ib) = (0usize, 0usize);
    let (mut prev_a, mut prev_b): (Option<usize>, Option<usize>) = (None, None);
    let mut a2b: Vec<Vec<usize>> = vec![];
    let mut b2a: Vec<Vec<usize>> = vec![];
    while ia < sa.len() && ib < sb.len() {
        let ta = c2a[ia];
        let tb = c2b[ib];
        if prev_a != Some(ta) {
            a2b.push(vec![]);
        }
        if prev_b != Some(tb) {
            b2a.push(vec![]);
        }
        if a[ta] == b[tb] && (ia == 0 || c2a[ia - 1] < ta) && (ib == 0 || c2b[ib - 1] < tb) {
            a2b.last_mut().unwrap().push(tb);
            b2a.last_mut().unwrap().push(ta);
            ia += alen(ta);
            ib += blen(tb);
        } else if sa[ia] == sb[ib] {
            a2b.last_mut().unwrap().push(tb);
            b2a.last_mut().unwrap().push(ta);
            ia += 1;
            ib += 1;
        } else if py_isspace(sa[ia]) {
            ia += 1;
        } else if py_isspace(sb[ib]) {
            ib += 1;
        } else {
            bail!("spaCy alignment E949 (unalignable characters)");
        }
        prev_a = Some(ta);
        prev_b = Some(tb);
    }
    let mut rest: Vec<usize> = c2b[ib.min(c2b.len())..].to_vec();
    rest.sort();
    rest.dedup();
    b2a.extend(std::iter::repeat(vec![]).take(rest.len()));
    Ok(b2a
        .into_iter()
        .map(|mut v| {
            v.sort();
            v.dedup();
            v
        })
        .collect())
}

impl EnglishFrontend {
    pub fn load(p: &FrontendPaths) -> Result<Self> {
        let lex = Lexicon::load(&p.lexicon_dir, false).context("misaki lexicon")?;
        let tok = Tokenizer::load(&p.spacy_dir).context("spaCy tokenizer data")?;
        let tagger = Tagger::load(&p.spacy_dir).context("spaCy tagger data")?;
        let espeak = Espeak::get(&p.espeak).context("eSpeak NG (required for text input)")?;
        let mut h = Sha256::new();
        for (dir, files) in [
            (&p.lexicon_dir, &["us_gold.json", "us_silver.json"][..]),
            (&p.spacy_dir, &["tokenizer.json", "lookups.json", "base_norms.json", "symbols.json", "model_structure.json", "tagger_weights.safetensors"][..]),
        ] {
            for f in files {
                let bytes = std::fs::read(dir.join(f)).with_context(|| format!("hashing {}", dir.join(f).display()))?;
                h.update(f.as_bytes());
                h.update(Sha256::digest(&bytes));
            }
        }
        let data_hash: String = h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect();
        let ident = format!(
            "native misaki-0.9.4 en-us G2P (KPipeline lang a) + spaCy en_core_web_sm-3.8.0 tokenizer/tagger [data {data_hash}] + {} fallback",
            espeak.describe()
        );
        Ok(Self { lex, tok, tagger, espeak, ident })
    }

    pub fn ident(&self) -> &str {
        &self.ident
    }

    /// The loaded eSpeak NG backend (version, library and data actually selected).
    pub fn espeak(&self) -> &Espeak {
        self.espeak
    }

    /// misaki G2P.__call__(text) -> (phonemes, tokens).
    pub fn g2p(&self, text: &str) -> Result<(String, Vec<MToken>)> {
        let (pre, words, features) = preprocess(text);
        let toks = self.tok.tokenize(&pre);
        let tags = self.tagger.tag(&toks).tags;
        let mut mts: Vec<MToken> = toks
            .iter()
            .zip(tags)
            .map(|(t, tag)| MToken { text: t.text.clone(), tag, whitespace: if t.space { " ".into() } else { String::new() }, is_head: true, ..Default::default() })
            .collect();
        if !features.is_empty() {
            let texts: Vec<String> = mts.iter().map(|t| t.text.clone()).collect();
            let y2x = get_alignments_y2x(&words, &texts)?;
            let flat: Vec<usize> = y2x.into_iter().flatten().collect();
            for (k, v) in &features {
                for (i, j) in flat.iter().enumerate().filter(|(_, x)| **x == *k).map(|(pos, _)| pos).enumerate() {
                    if j >= mts.len() {
                        continue;
                    }
                    match v {
                        Feature::Num(x) => mts[j].stress = Some(*x),
                        Feature::Str(s) if s.starts_with('/') => {
                            mts[j].is_head = i == 0;
                            mts[j].phonemes = Some(if i == 0 { s.trim_start_matches('/').to_string() } else { String::new() });
                            mts[j].rating = Some(5);
                        }
                        Feature::Str(s) if s.starts_with('#') => mts[j].num_flags = s.trim_start_matches('#').to_string(),
                        Feature::Str(_) => {}
                    }
                }
            }
        }
        let fb: &dyn Fallback = self.espeak;
        let (ps, toks, missing) = g2p_tokens_unresolved(&self.lex, mts, Some(fb), "")?;
        if !missing.is_empty() {
            let list: Vec<String> = missing.iter().map(|w| format!("{w:?}")).collect();
            bail!("unresolved word(s) {}: no pronunciation from the lexicon or eSpeak NG {} (the line is not synthesized rather than dropping words)", list.join(", "), self.espeak.version);
        }
        Ok((ps, toks))
    }

    /// KPipeline.__call__ text path for one input line: every non-empty (graphemes, phonemes) chunk,
    /// in order. Chunks longer than 510 are returned intact (the caller refuses them explicitly).
    pub fn line_chunks(&self, line: &str) -> Result<Vec<Chunk>> {
        let mut out = vec![];
        let stripped = line.trim_matches(py_isspace);
        for seg in stripped.split('\n') {
            if seg.chars().all(py_isspace) {
                continue;
            }
            let (_, mut tokens) = self.g2p(seg)?;
            for (gs, ps) in en_tokenize(&mut tokens) {
                if !ps.is_empty() {
                    out.push(Chunk { graphemes: gs, phonemes: ps });
                }
            }
        }
        Ok(out)
    }
}
