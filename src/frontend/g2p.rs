//! misaki 0.9.4 `G2P.__call__` (en.py:568-712) over spaCy tokens, plus KPipeline.en_tokenize
//! chunking (pipeline.py:174-221). Spec: docs/frontend/MISAKI_EN_SPEC.md S1, S4.

use super::lexicon::{apply_stress, is_vowel, Ctx, Lexicon, PRIMARY};
use super::pystr::*;
use super::{MToken, SpacyToken};
use fancy_regex::Regex;
use std::sync::OnceLock;

const SUBTOKEN_JUNKS: &str = "',-._‘’/";
const PUNCTS: &str = ";:,.!?—…\"“”";
const NON_QUOTE_PUNCTS: &str = ";:,.!?—…";
const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const PUNCT_TAGS: [&str; 11] = [".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"];

fn punct_tag_phonemes(tag: &str) -> Option<&'static str> {
    match tag {
        "-LRB-" => Some("("),
        "-RRB-" => Some(")"),
        "``" => Some("\u{201C}"),
        "\"\"" | "''" => Some("\u{201D}"),
        _ => None,
    }
}

fn subtoken_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$")
            .expect("subtoken regex")
    })
}

pub fn subtokenize(word: &str) -> Vec<String> {
    subtoken_regex().find_iter(word).filter_map(|m| m.ok()).map(|m| m.as_str().to_string()).collect()
}

/// Fallback G2P for words the lexicon cannot resolve (espeak-ng in production; owner decision).
pub trait Fallback {
    fn g2p(&self, tk: &MToken) -> (Option<String>, Option<i32>);
}

pub enum Word {
    One(MToken),
    Many(Vec<MToken>),
}

pub fn merge_tokens(tokens: &[MToken], unk: Option<&str>) -> MToken {
    let stresses: Vec<f64> = {
        let mut v: Vec<f64> = tokens.iter().filter_map(|t| t.stress).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup();
        v
    };
    let currency = tokens.iter().filter_map(|t| t.currency.clone()).max();
    let rating = if tokens.iter().any(|t| t.rating.is_none()) { None } else { tokens.iter().filter_map(|t| t.rating).min() };
    let phonemes = unk.map(|unk| {
        let mut p = String::new();
        for tk in tokens {
            if tk.prespace && !p.is_empty() && !super::pystr::char_isspace(p.chars().last().unwrap()) && tk.phonemes.as_deref().map(|x| !x.is_empty()).unwrap_or(false) {
                p.push(' ');
            }
            p.push_str(tk.phonemes.as_deref().unwrap_or(unk));
        }
        p
    });
    let weight = |t: &MToken| t.text.chars().map(|c| if lower(&c.to_string()) == c.to_string() { 1 } else { 2 }).sum::<usize>();
    let mut best = &tokens[0];
    for t in tokens {
        if weight(t) > weight(best) {
            best = t;
        }
    }
    let mut flags: Vec<char> = tokens.iter().flat_map(|t| t.num_flags.chars()).collect();
    flags.sort();
    flags.dedup();
    let n = tokens.len();
    MToken {
        text: tokens[..n - 1].iter().map(|t| format!("{}{}", t.text, t.whitespace)).collect::<String>() + &tokens[n - 1].text,
        tag: best.tag.clone(),
        whitespace: tokens[n - 1].whitespace.clone(),
        phonemes,
        rating,
        is_head: tokens[0].is_head,
        alias: None,
        stress: if stresses.len() == 1 { Some(stresses[0]) } else { None },
        currency,
        num_flags: flags.into_iter().collect(),
        prespace: tokens[0].prespace,
    }
}

fn stress_weight(ps: &str) -> usize {
    ps.chars().map(|c| if "AIOQWYʤʧ".contains(c) { 2 } else { 1 }).sum()
}

/// misaki `all(97 <= ord(c.lower()) <= 122 for c in text)` with Python's short-circuit: a char whose
/// lower() is not exactly one code point raises TypeError in the reference (e.g. 'İ' -> 'i̇') unless an
/// earlier char already made the predicate false. The reference then fails the whole line; so do we.
fn all_ascii_lower_ord(text: &str) -> anyhow::Result<bool> {
    for c in text.chars() {
        let l: Vec<char> = lower(&c.to_string()).chars().collect();
        if l.len() != 1 {
            anyhow::bail!("reference misaki raises TypeError here (ord() of {c:?}.lower(), {} code points): line not synthesizable by the pinned frontend", l.len());
        }
        if !l[0].is_ascii_lowercase() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn retokenize(tokens: &[MToken]) -> anyhow::Result<Vec<Word>> {
    let mut words: Vec<Word> = vec![];
    let mut currency: Option<String> = None;
    for (i, token) in tokens.iter().enumerate() {
        let mut tks: Vec<MToken> = if token.alias.is_none() && token.phonemes.is_none() {
            subtokenize(&token.text)
                .into_iter()
                .map(|t| MToken {
                    text: t,
                    whitespace: String::new(),
                    is_head: true,
                    num_flags: token.num_flags.clone(),
                    stress: token.stress,
                    prespace: false,
                    ..token.clone()
                })
                .collect()
        } else {
            vec![token.clone()]
        };
        if tks.is_empty() {
            continue; // Python would IndexError on tks[-1]; subtokenize never returns empty for real tokens
        }
        let last = tks.len() - 1;
        tks[last].whitespace = token.whitespace.clone();
        let n = tks.len();
        for j in 0..n {
            let (prev_last, next_first) = (
                if j > 0 { last_char(&tks[j - 1].text) } else { None },
                if j + 1 < n { tks[j + 1].text.chars().next() } else { None },
            );
            let tk = &mut tks[j];
            if tk.alias.is_some() || tk.phonemes.is_some() {
            } else if tk.tag == "$" && matches!(tk.text.as_str(), "$" | "£" | "€") {
                currency = Some(tk.text.clone());
                tk.phonemes = Some(String::new());
                tk.rating = Some(4);
            } else if tk.tag == ":" && (tk.text == "-" || tk.text == "–") {
                tk.phonemes = Some("—".into());
                tk.rating = Some(3);
            } else if PUNCT_TAGS.contains(&tk.tag.as_str()) && !all_ascii_lower_ord(&tk.text)? {
                tk.phonemes = Some(punct_tag_phonemes(&tk.tag).map(String::from).unwrap_or_else(|| tk.text.chars().filter(|c| PUNCTS.contains(*c)).collect()));
                tk.rating = Some(4);
            } else if currency.is_some() {
                if tk.tag != "CD" {
                    currency = None;
                } else if j + 1 == n && (i + 1 == tokens.len() || tokens[i + 1].tag != "CD") {
                    tk.currency = currency.clone();
                }
            } else if 0 < j && j + 1 < n && tk.text == "2" {
                if let (Some(a), Some(b)) = (prev_last, next_first) {
                    if char_isalpha(a) && char_isalpha(b) {
                        tk.alias = Some("to".into());
                    }
                }
            }
        }
        for mut tk in tks {
            if tk.alias.is_some() || tk.phonemes.is_some() {
                words.push(Word::One(tk));
            } else if let Some(Word::Many(list)) = words.last_mut().filter(|w| matches!(w, Word::Many(l) if l.last().map(|t| t.whitespace.is_empty()).unwrap_or(false))) {
                tk.is_head = false;
                list.push(tk);
            } else if !tk.whitespace.is_empty() {
                words.push(Word::One(tk));
            } else {
                words.push(Word::Many(vec![tk]));
            }
        }
    }
    Ok(words
        .into_iter()
        .map(|w| match w {
            Word::Many(mut l) if l.len() == 1 => Word::One(l.pop().unwrap()),
            w => w,
        })
        .collect())
}

fn token_context(ctx: Ctx, ps: Option<&str>, token: &MToken) -> Ctx {
    let mut vowel = ctx.future_vowel;
    if let Some(ps) = ps.filter(|p| !p.is_empty()) {
        for c in ps.chars() {
            if is_vowel(c) || CONSONANTS.contains(c) || NON_QUOTE_PUNCTS.contains(c) {
                vowel = if NON_QUOTE_PUNCTS.contains(c) { None } else { Some(is_vowel(c)) };
                break;
            }
        }
    }
    let future_to = matches!(token.text.as_str(), "to" | "To") || (token.text == "TO" && (token.tag == "TO" || token.tag == "IN"));
    Ctx { future_vowel: vowel, future_to }
}

fn resolve_tokens(tokens: &mut [MToken]) {
    let n = tokens.len();
    let text = tokens[..n - 1].iter().map(|t| format!("{}{}", t.text, t.whitespace)).collect::<String>() + &tokens[n - 1].text;
    let classes: std::collections::BTreeSet<u8> = text
        .chars()
        .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
        .map(|c| if char_isalpha(c) { 0 } else if c.is_ascii_digit() { 1 } else { 2 })
        .collect();
    let prespace = text.contains(' ') || text.contains('/') || classes.len() > 1;
    for (i, tk) in tokens.iter_mut().enumerate() {
        if tk.phonemes.is_none() {
            if i == n - 1 && tk.text.chars().count() == 1 && NON_QUOTE_PUNCTS.contains(tk.text.as_str()) {
                tk.phonemes = Some(tk.text.clone());
                tk.rating = Some(3);
            } else if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                tk.phonemes = Some(String::new());
                tk.rating = Some(3);
            }
        } else if i > 0 {
            tk.prespace = prespace;
        }
    }
    if prespace {
        return;
    }
    let mut indices: Vec<(bool, usize, usize)> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.phonemes.as_deref().map(|p| !p.is_empty()).unwrap_or(false))
        .map(|(i, t)| {
            let p = t.phonemes.as_deref().unwrap();
            (p.contains(PRIMARY), stress_weight(p), i)
        })
        .collect();
    if indices.len() == 2 && tokens[indices[0].2].text.chars().count() == 1 {
        let i = indices[1].2;
        tokens[i].phonemes = apply_stress(tokens[i].phonemes.clone(), Some(-0.5));
        return;
    }
    if indices.len() < 2 || indices.iter().filter(|x| x.0).count() <= (indices.len() + 1) / 2 {
        return;
    }
    indices.sort();
    let half = indices.len() / 2;
    for &(_, _, i) in &indices[..half] {
        tokens[i].phonemes = apply_stress(tokens[i].phonemes.clone(), Some(-0.5));
    }
}

/// G2P over one (already preprocessed) text's spaCy tokens. Returns (phoneme string, tokens).
/// `unk` is the unknown-token phoneme placeholder ('' in production).
pub fn g2p(lex: &Lexicon, spacy: &[SpacyToken], fallback: Option<&dyn Fallback>, unk: &str) -> anyhow::Result<(String, Vec<MToken>)> {
    let toks: Vec<MToken> = spacy
        .iter()
        .map(|t| MToken { text: t.text.clone(), tag: t.tag.clone(), whitespace: t.ws.clone(), is_head: true, ..Default::default() })
        .collect();
    g2p_tokens(lex, toks, fallback, unk)
}

/// G2P.fold_left: a non-head token (from a multi-token /phoneme/ link feature) merges into its predecessor.
pub fn fold_left(tokens: Vec<MToken>, unk: &str) -> Vec<MToken> {
    let mut result: Vec<MToken> = vec![];
    for tk in tokens {
        if !result.is_empty() && !tk.is_head {
            let prev = result.pop().unwrap();
            result.push(merge_tokens(&[prev, tk], Some(unk)));
        } else {
            result.push(tk);
        }
    }
    result
}

/// G2P.__call__ from the tokenize() output onwards (fold_left, retokenize, lexicon/fallback, merge).
pub fn g2p_tokens(lex: &Lexicon, toks: Vec<MToken>, fallback: Option<&dyn Fallback>, unk: &str) -> anyhow::Result<(String, Vec<MToken>)> {
    let toks = fold_left(toks, unk);
    let mut words = retokenize(&toks)?;
    let mut ctx = Ctx::default();
    for w in words.iter_mut().rev() {
        match w {
            Word::One(tk) => {
                if tk.phonemes.is_none() {
                    let (p, r) = lex.call(tk, ctx);
                    tk.phonemes = p;
                    tk.rating = r;
                }
                if tk.phonemes.is_none() {
                    if let Some(fb) = fallback {
                        let (p, r) = fb.g2p(tk);
                        tk.phonemes = p;
                        tk.rating = r;
                    }
                }
                ctx = token_context(ctx, tk.phonemes.as_deref(), tk);
            }
            Word::Many(list) => {
                let (mut left, mut right) = (0usize, list.len());
                let mut should_fallback = false;
                while left < right {
                    let span = &list[left..right];
                    let found = if span.iter().any(|t| t.alias.is_some() || t.phonemes.is_some()) {
                        None
                    } else {
                        let tk = merge_tokens(span, None);
                        let (ps, r) = lex.call(&tk, ctx);
                        ps.map(|p| (p, r, tk))
                    };
                    if let Some((ps, r, tk)) = found {
                        list[left].phonemes = Some(ps.clone());
                        list[left].rating = r;
                        for x in &mut list[left + 1..right] {
                            x.phonemes = Some(String::new());
                        }
                        ctx = token_context(ctx, Some(&ps), &tk);
                        right = left;
                        left = 0;
                    } else if left + 1 < right {
                        left += 1;
                    } else {
                        right -= 1;
                        let tk = &mut list[right];
                        if tk.phonemes.is_none() {
                            if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                                tk.phonemes = Some(String::new());
                                tk.rating = Some(3);
                            } else if fallback.is_some() {
                                should_fallback = true;
                                break;
                            }
                        }
                        left = 0;
                    }
                }
                if should_fallback {
                    let tk = merge_tokens(list, None);
                    let (p, r) = fallback.unwrap().g2p(&tk);
                    list[0].phonemes = p;
                    list[0].rating = r;
                    for x in &mut list[1..] {
                        x.phonemes = Some(String::new());
                        x.rating = r;
                    }
                } else {
                    resolve_tokens(list);
                }
            }
        }
    }
    let mut out: Vec<MToken> = words
        .into_iter()
        .map(|w| match w {
            Word::One(t) => t,
            Word::Many(l) => merge_tokens(&l, Some(unk)),
        })
        .collect();
    for t in &mut out {
        if let Some(p) = &t.phonemes {
            if !p.is_empty() {
                t.phonemes = Some(p.replace('ɾ', "T").replace('ʔ', "t"));
            }
        }
    }
    let result = out.iter().map(|t| format!("{}{}", t.phonemes.as_deref().unwrap_or(unk), t.whitespace)).collect::<String>();
    Ok((result, out))
}

// ------------------------------------------------------------------ KPipeline chunking

pub const MAX_CHUNK: usize = 510;

fn tokens_to_ps(tokens: &[MToken]) -> String {
    let joined: String = tokens.iter().map(|t| format!("{}{}", t.phonemes.as_deref().unwrap_or(""), if t.whitespace.is_empty() { "" } else { " " })).collect();
    super::pystr::strip(&joined).to_string()
}

fn tokens_to_text(tokens: &[MToken]) -> String {
    super::pystr::strip(&tokens.iter().map(|t| format!("{}{}", t.text, t.whitespace)).collect::<String>()).to_string()
}

fn waterfall_last(tokens: &[MToken], next_count: usize) -> usize {
    for w in ["!.?…", ":;", ",—"] {
        let z = tokens.iter().rposition(|t| {
            let p = t.phonemes.as_deref().unwrap_or("");
            p.chars().count() == 1 && w.contains(p)
        });
        let Some(mut z) = z else { continue };
        z += 1;
        if z < tokens.len() && matches!(tokens[z].phonemes.as_deref(), Some(")") | Some("”")) {
            z += 1;
        }
        if next_count as i64 - nchars(&tokens_to_ps(&tokens[..z])) as i64 <= MAX_CHUNK as i64 {
            return z;
        }
    }
    tokens.len()
}

/// KPipeline.en_tokenize: (graphemes, phonemes) chunks; empty-phoneme chunks are dropped (as
/// KPipeline.__call__ does). Chunks longer than MAX_CHUNK are returned intact (caller refuses them).
pub fn en_tokenize(tokens: &mut [MToken]) -> Vec<(String, String)> {
    let mut out = vec![];
    let mut tks: Vec<MToken> = vec![];
    let mut pcount = 0usize;
    for t in tokens.iter_mut() {
        if t.phonemes.is_none() {
            t.phonemes = Some(String::new());
        }
        let mut next_ps = format!("{}{}", t.phonemes.as_deref().unwrap(), if t.whitespace.is_empty() { "" } else { " " });
        let next_pcount = pcount + nchars(super::pystr::rstrip(&next_ps));
        if next_pcount > MAX_CHUNK {
            let z = waterfall_last(&tks, next_pcount);
            let (text, ps) = (tokens_to_text(&tks[..z]), tokens_to_ps(&tks[..z]));
            out.push((text, ps));
            tks = tks[z..].to_vec();
            pcount = nchars(&tokens_to_ps(&tks));
            if tks.is_empty() {
                next_ps = super::pystr::lstrip(&next_ps).to_string();
            }
        }
        tks.push(t.clone());
        pcount += nchars(&next_ps);
    }
    if !tks.is_empty() {
        out.push((tokens_to_text(&tks), tokens_to_ps(&tks)));
    }
    out.into_iter().filter(|(_, ps)| !ps.is_empty()).collect()
}
