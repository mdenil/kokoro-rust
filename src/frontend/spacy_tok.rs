//! spaCy 3.8 English tokenizer (tokenizer.pyx), driven by the pinned en_core_web_sm 3.8.0 data
//! (prefix/suffix/infix/url patterns + special-case rules, exported by oracle/export_spacy.py).
//! Algorithm: whitespace runs -> per span: special case | affix splitting with special-case checks
//! -> infix split; then special cases re-applied over the token sequence (PhraseMatcher pass).

use anyhow::{Context, Result};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Tok {
    pub text: String,
    /// trailing single space (spaCy `spacy` flag)
    pub space: bool,
    /// NORM override from a special-case rule
    pub norm: Option<String>,
}

pub struct Tokenizer {
    prefix: Option<Regex>,
    suffix: Option<Regex>,
    infix: Option<Regex>,
    url: Option<Regex>,
    rules: HashMap<String, Vec<(String, Option<String>)>>,
    /// special-case patterns re-applied after affix splitting: token-text sequence -> rule key
    matcher: HashMap<String, Vec<(Vec<String>, String)>>,
}

pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

impl Tokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json")).context("tokenizer.json")?)?;
        let re = |k: &str| -> Result<Option<Regex>> {
            match d[k].as_str() {
                Some(p) => Ok(Some(Regex::new(p).with_context(|| format!("compiling {k} pattern"))?)),
                None => Ok(None),
            }
        };
        let mut rules = HashMap::new();
        for (k, v) in d["rules"].as_object().context("rules")? {
            let subs = v
                .as_array()
                .context("rule")?
                .iter()
                .map(|s| (s["ORTH"].as_str().unwrap_or_default().to_string(), s["NORM"].as_str().map(String::from)))
                .collect();
            rules.insert(k.clone(), subs);
        }
        let mut t = Self { prefix: re("prefix")?, suffix: re("suffix")?, infix: re("infix")?, url: re("url_match")?, rules, matcher: HashMap::new() };
        anyhow::ensure!(d["token_match"].is_null(), "token_match patterns are not supported");
        anyhow::ensure!(d["faster_heuristics"].as_bool() == Some(true), "faster_heuristics expected");
        // add_special_case: rules whose string is affected by affixes/infixes/spaces go to the matcher,
        // with the pattern = its tokenization WITHOUT special cases.
        let mut keys: Vec<String> = t.rules.keys().cloned().collect();
        keys.sort();
        let mut matcher: HashMap<String, Vec<(Vec<String>, String)>> = HashMap::new();
        for k in keys {
            if t.find_prefix(&k) > 0 || !t.find_infix(&k).is_empty() || t.find_suffix(&k) > 0 || k.contains(' ') {
                let pat: Vec<String> = t.tokenize_affixes(&k, false).into_iter().map(|x| x.text).collect();
                if let Some(first) = pat.first().cloned() {
                    matcher.entry(first).or_default().push((pat, k));
                }
            }
        }
        t.matcher = matcher;
        Ok(t)
    }

    fn find_prefix(&self, s: &str) -> usize {
        self.prefix.as_ref().and_then(|r| r.find(s).ok().flatten()).map(|m| m.end() - m.start()).unwrap_or(0)
    }

    fn find_suffix(&self, s: &str) -> usize {
        self.suffix.as_ref().and_then(|r| r.find(s).ok().flatten()).map(|m| m.end() - m.start()).unwrap_or(0)
    }

    fn find_infix(&self, s: &str) -> Vec<(usize, usize)> {
        match &self.infix {
            None => vec![],
            Some(r) => r.find_iter(s).filter_map(|m| m.ok()).map(|m| (m.start(), m.end())).collect(),
        }
    }

    fn special(&self, s: &str) -> Option<Vec<Tok>> {
        self.rules.get(s).map(|subs| subs.iter().map(|(o, n)| Tok { text: o.clone(), space: false, norm: n.clone() }).collect())
    }

    /// _tokenize_affixes (tokenizer.pyx:165-210); byte offsets throughout (patterns are char-based
    /// but all slicing uses match byte boundaries).
    pub fn tokenize_affixes(&self, string: &str, specials: bool) -> Vec<Tok> {
        let mut doc: Vec<Tok> = vec![];
        if string.is_empty() {
            return doc;
        }
        let chars: Vec<(usize, char)> = string.char_indices().collect();
        let mut in_ws = py_isspace(chars[0].1);
        let mut start = 0usize; // byte offset
        for &(i, uc) in &chars {
            if py_isspace(uc) != in_ws {
                if start < i {
                    self.tokenize_span(&string[start..i], specials, &mut doc);
                }
                if uc == ' ' {
                    if let Some(last) = doc.last_mut() {
                        last.space = true;
                    }
                    start = i + 1;
                } else {
                    start = i;
                }
                in_ws = !in_ws;
            }
        }
        if start < string.len() {
            self.tokenize_span(&string[start..], specials, &mut doc);
            let last_char = chars.last().unwrap().1;
            if let Some(last) = doc.last_mut() {
                last.space = last_char == ' ' && !in_ws;
            }
        }
        doc
    }

    fn tokenize_span(&self, span: &str, specials: bool, doc: &mut Vec<Tok>) {
        if specials {
            if let Some(toks) = self.special(span) {
                doc.extend(toks);
                return;
            }
        }
        // _split_affixes (tokenizer.pyx:406-452)
        let mut prefixes: Vec<String> = vec![];
        let mut suffixes: Vec<String> = vec![];
        let mut s = span.to_string();
        let mut last_size = usize::MAX;
        while !s.is_empty() && s.chars().count() != last_size {
            if specials && self.rules.contains_key(&s) {
                break;
            }
            last_size = s.chars().count();
            let pre_len = self.find_prefix(&s);
            let (mut prefix, mut minus_pre) = (String::new(), String::new());
            if pre_len != 0 {
                prefix = s[..pre_len].to_string();
                minus_pre = s[pre_len..].to_string();
                if !minus_pre.is_empty() && specials && self.rules.contains_key(&minus_pre) {
                    s = minus_pre;
                    prefixes.push(prefix);
                    break;
                }
            }
            let suf_len = self.find_suffix(&s[pre_len..]);
            let (mut suffix, mut minus_suf) = (String::new(), String::new());
            if suf_len != 0 {
                suffix = s[s.len() - suf_len..].to_string();
                minus_suf = s[..s.len() - suf_len].to_string();
                if !minus_suf.is_empty() && specials && self.rules.contains_key(&minus_suf) {
                    s = minus_suf;
                    suffixes.push(suffix);
                    break;
                }
            }
            if pre_len != 0 && suf_len != 0 && pre_len + suf_len <= s.len() {
                s = s[pre_len..s.len() - suf_len].to_string();
                prefixes.push(std::mem::take(&mut prefix));
                suffixes.push(std::mem::take(&mut suffix));
            } else if pre_len != 0 {
                s = minus_pre;
                prefixes.push(prefix);
            } else if suf_len != 0 {
                s = minus_suf;
                suffixes.push(suffix);
            }
        }
        // _attach_tokens (tokenizer.pyx:454-512)
        for p in prefixes {
            doc.push(Tok { text: p, space: false, norm: None });
        }
        if !s.is_empty() {
            if let Some(toks) = specials.then(|| self.special(&s)).flatten() {
                doc.extend(toks);
            } else if self.url.as_ref().map(|r| r.is_match(&s).unwrap_or(false) && url_full(r, &s)).unwrap_or(false) {
                doc.push(Tok { text: s, space: false, norm: None });
            } else {
                let matches = self.find_infix(&s);
                if matches.is_empty() {
                    doc.push(Tok { text: s, space: false, norm: None });
                } else {
                    let mut start = 0usize;
                    let start_before = 0usize;
                    for (a, b) in matches {
                        if a == start_before {
                            continue;
                        }
                        if a != start {
                            doc.push(Tok { text: s[start..a].to_string(), space: false, norm: None });
                        }
                        if a != b {
                            doc.push(Tok { text: s[a..b].to_string(), space: false, norm: None });
                        }
                        start = b;
                    }
                    if start < s.len() {
                        doc.push(Tok { text: s[start..].to_string(), space: false, norm: None });
                    }
                }
            }
        }
        for suf in suffixes.into_iter().rev() {
            doc.push(Tok { text: suf, space: false, norm: None });
        }
    }

    /// Tokenizer.__call__: affix tokenization + special cases re-applied over token sequences.
    pub fn tokenize(&self, text: &str) -> Vec<Tok> {
        let doc = self.tokenize_affixes(text, true);
        // find all (possibly overlapping) matches of matcher patterns
        let mut matches: Vec<(usize, usize, String)> = vec![];
        for i in 0..doc.len() {
            if let Some(pats) = self.matcher.get(&doc[i].text) {
                for (pat, key) in pats {
                    if i + pat.len() <= doc.len() && doc[i..i + pat.len()].iter().zip(pat).all(|(t, p)| t.text == *p) {
                        matches.push((i, i + pat.len(), key.clone()));
                    }
                }
            }
        }
        if matches.is_empty() {
            return doc;
        }
        // _filter_special_spans: sort by (length asc, start desc) then take from the end (longest
        // first; among equal length the smallest start first); endpoint-only overlap check.
        matches.sort_by(|a, b| {
            let (la, lb) = (a.1 - a.0, b.1 - b.0);
            la.cmp(&lb).then(b.0.cmp(&a.0))
        });
        matches.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        let mut seen = std::collections::HashSet::new();
        let mut filtered = vec![];
        for m in matches.iter().rev() {
            if !seen.contains(&m.0) && !seen.contains(&(m.1 - 1)) {
                filtered.push(m.clone());
            }
            for k in m.0..m.1 {
                seen.insert(k);
            }
        }
        filtered.sort_by_key(|m| m.0);
        let mut out = vec![];
        let mut i = 0;
        let mut fi = 0;
        while i < doc.len() {
            if fi < filtered.len() && filtered[fi].0 == i {
                let (s, e, ref key) = filtered[fi];
                // span.text of doc[s:e] must be the rule key (it is, by construction of the pattern)
                let final_space = doc[e - 1].space;
                let mut toks = self.special(key).unwrap_or_else(|| doc[s..e].to_vec());
                if let Some(last) = toks.last_mut() {
                    last.space = final_space;
                }
                out.extend(toks);
                i = e;
                fi += 1;
            } else {
                out.push(doc[i].clone());
                i += 1;
            }
        }
        out
    }
}

/// Python `re.match` semantics for url_match (anchored at the start; the pattern itself ends with `$`).
fn url_full(r: &Regex, s: &str) -> bool {
    r.find(s).ok().flatten().map(|m| m.start() == 0).unwrap_or(false)
}
