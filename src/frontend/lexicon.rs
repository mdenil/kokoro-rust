//! misaki 0.9.4 `Lexicon` (en.py:123-493): gold/silver dictionaries, special cases, stress,
//! morphology (-s/-ed/-ing) and number reading. Spec: docs/frontend/MISAKI_EN_SPEC.md S2.

use super::num2words;
use super::pystr::*;
use super::MToken;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub const PRIMARY: char = 'ˈ';
pub const SECONDARY: char = 'ˌ';
const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";

pub fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

fn in_lexicon_ords(c: char) -> bool {
    c == '\'' || c == '-' || c.is_ascii_alphabetic()
}

#[derive(Clone, Debug)]
pub enum Entry {
    S(String),
    D(HashMap<String, Option<String>>),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx {
    /// None = unknown, Some(true/false) = next word starts with a vowel
    pub future_vowel: Option<bool>,
    pub future_to: bool,
}

pub type Lookup = (Option<String>, Option<i32>);

pub struct Lexicon {
    british: bool,
    golds: HashMap<String, Entry>,
    silvers: HashMap<String, Entry>,
}

fn load_dict(path: &Path) -> Result<HashMap<String, Entry>> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?)?;
    let obj = v.as_object().context("lexicon is not a JSON object")?;
    let mut d = HashMap::with_capacity(obj.len() * 2);
    for (k, v) in obj {
        let e = match v {
            serde_json::Value::String(s) => Entry::S(s.clone()),
            serde_json::Value::Object(m) => Entry::D(m.iter().map(|(t, p)| (t.clone(), p.as_str().map(String::from))).collect()),
            other => anyhow::bail!("bad lexicon entry for {k}: {other}"),
        };
        d.insert(k.clone(), e);
    }
    // grow_dictionary (en.py:125-136): capitalized/lowercased variants; originals override
    let mut grown: HashMap<String, Entry> = HashMap::with_capacity(d.len() * 2);
    for (k, v) in &d {
        if nchars(k) < 2 {
            continue;
        }
        if *k == lower(k) {
            let c = capitalize(k);
            if *k != c {
                grown.insert(c, v.clone());
            }
        } else if *k == capitalize(&lower(k)) {
            grown.insert(lower(k), v.clone());
        }
    }
    grown.extend(d);
    Ok(grown)
}

pub fn apply_stress(ps: Option<String>, stress: Option<f64>) -> Option<String> {
    let ps = ps?;
    let Some(stress) = stress else { return Some(ps) };
    let has_p = ps.contains(PRIMARY);
    let has_any = ps.contains(PRIMARY) || ps.contains(SECONDARY);
    let has_vowel = ps.chars().any(is_vowel);
    let restress = |ps: String| -> String {
        // move the (single, leading) stress mark right before the first vowel after it
        let chars: Vec<char> = ps.chars().collect();
        let mut out: Vec<char> = vec![];
        let mut pending: Option<char> = None;
        for (i, &c) in chars.iter().enumerate() {
            if c == PRIMARY || c == SECONDARY {
                if chars[i..].iter().any(|&v| is_vowel(v)) {
                    pending = Some(c);
                    continue;
                }
            }
            if let Some(p) = pending {
                if is_vowel(c) {
                    out.push(p);
                    pending = None;
                }
            }
            out.push(c);
        }
        out.into_iter().collect()
    };
    Some(if stress < -1.0 {
        ps.replace(PRIMARY, "").replace(SECONDARY, "")
    } else if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && has_p) {
        ps.replace(SECONDARY, "").replace(PRIMARY, &SECONDARY.to_string())
    } else if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_any {
        if !has_vowel { ps } else { restress(format!("{SECONDARY}{ps}")) }
    } else if stress >= 1.0 && !has_p && ps.contains(SECONDARY) {
        ps.replace(SECONDARY, &PRIMARY.to_string())
    } else if stress > 1.0 && !has_any {
        if !has_vowel { ps } else { restress(format!("{PRIMARY}{ps}")) }
    } else {
        ps
    })
}

pub fn parent_tag(tag: Option<&str>) -> Option<String> {
    let t = tag?;
    Some(if t.starts_with("VB") {
        "VERB".into()
    } else if t.starts_with("NN") {
        "NOUN".into()
    } else if t.starts_with("ADV") || t.starts_with("RB") {
        "ADV".into()
    } else if t.starts_with("ADJ") || t.starts_with("JJ") {
        "ADJ".into()
    } else {
        t.into()
    })
}

fn is_digit_str(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

const ORDINALS: [&str; 4] = ["st", "nd", "rd", "th"];

fn currency_units(c: &str) -> Option<(&'static str, &'static str)> {
    match c {
        "$" => Some(("dollar", "cent")),
        "£" => Some(("pound", "pence")),
        "€" => Some(("euro", "cent")),
        _ => None,
    }
}

fn symbol(w: &str) -> Option<&'static str> {
    match w {
        "%" => Some("percent"),
        "&" => Some("and"),
        "+" => Some("plus"),
        "@" => Some("at"),
        _ => None,
    }
}

impl Lexicon {
    pub fn load(dir: &Path, british: bool) -> Result<Self> {
        let p = if british { "gb" } else { "us" };
        Ok(Self { british, golds: load_dict(&dir.join(format!("{p}_gold.json")))?, silvers: load_dict(&dir.join(format!("{p}_silver.json")))? })
    }

    fn gold_str(&self, k: &str) -> Option<String> {
        match self.golds.get(k)? {
            Entry::S(s) => Some(s.clone()),
            Entry::D(d) => d.get("DEFAULT").cloned().flatten(),
        }
    }

    pub fn get_nnp(&self, word: &str) -> Lookup {
        let mut ps = String::new();
        for c in word.chars().filter(|c| char_isalpha(*c)) {
            match self.golds.get(&upper(&c.to_string())) {
                Some(Entry::S(s)) => ps.push_str(s),
                _ => return (None, None),
            }
        }
        let ps = apply_stress(Some(ps), Some(0.0)).unwrap();
        // rsplit(SECONDARY, 1) then join with PRIMARY == replace the LAST secondary with primary
        let ps = match ps.rfind(SECONDARY) {
            Some(i) => format!("{}{PRIMARY}{}", &ps[..i], &ps[i + SECONDARY.len_utf8()..]),
            None => ps,
        };
        (Some(ps), Some(3))
    }

    fn get_special_case(&self, word: &str, tag: &str, stress: Option<f64>, ctx: Ctx) -> Lookup {
        if tag == "ADD" && (word == "." || word == "/") {
            return self.lookup(if word == "." { "dot" } else { "slash" }, None, Some(-0.5), Some(ctx));
        }
        if let Some(s) = symbol(word) {
            return self.lookup(s, None, None, Some(ctx));
        }
        let stripped = word.trim_matches('.');
        let nodots = word.replace('.', "");
        if stripped.contains('.') && isalpha(&nodots) && word.split('.').map(nchars).max().unwrap_or(0) < 3 {
            return self.get_nnp(word);
        }
        match word {
            "a" | "A" => return (Some(if tag == "DT" { "ɐ" } else { "ˈA" }.into()), Some(4)),
            "am" | "Am" | "AM" => {
                if tag.starts_with("NN") {
                    return self.get_nnp(word);
                } else if ctx.future_vowel.is_none() || word != "am" || stress.map(|s| s > 0.0).unwrap_or(false) {
                    return (self.gold_str("am"), Some(4));
                }
                return (Some("ɐm".into()), Some(4));
            }
            "an" | "An" | "AN" => {
                if word == "AN" && tag.starts_with("NN") {
                    return self.get_nnp(word);
                }
                return (Some("ɐn".into()), Some(4));
            }
            _ => {}
        }
        if word == "I" && tag == "PRP" {
            return (Some(format!("{SECONDARY}I")), Some(4));
        }
        if matches!(word, "by" | "By" | "BY") && parent_tag(Some(tag)).as_deref() == Some("ADV") {
            return (Some("bˈI".into()), Some(4));
        }
        if matches!(word, "to" | "To") || (word == "TO" && (tag == "TO" || tag == "IN")) {
            return (
                match ctx.future_vowel {
                    None => self.gold_str("to"),
                    Some(false) => Some("tə".into()),
                    Some(true) => Some("tʊ".into()),
                },
                Some(4),
            );
        }
        if matches!(word, "in" | "In") || (word == "IN" && tag != "NNP") {
            let s = if ctx.future_vowel.is_none() || tag != "IN" { PRIMARY.to_string() } else { String::new() };
            return (Some(format!("{s}ɪn")), Some(4));
        }
        if matches!(word, "the" | "The") || (word == "THE" && tag == "DT") {
            return (Some(if ctx.future_vowel == Some(true) { "ði" } else { "ðə" }.into()), Some(4));
        }
        if tag == "IN" && (lower(word) == "vs" || lower(word) == "vs.") {
            return self.lookup("versus", None, None, Some(ctx));
        }
        if matches!(word, "used" | "Used" | "USED") {
            if let Some(Entry::D(d)) = self.golds.get("used") {
                let key = if (tag == "VBD" || tag == "JJ") && ctx.future_to { "VBD" } else { "DEFAULT" };
                return (d.get(key).cloned().flatten(), Some(4));
            }
        }
        (None, None)
    }

    fn is_known(&self, word: &str, _tag: Option<&str>) -> bool {
        if self.golds.contains_key(word) || symbol(word).is_some() || self.silvers.contains_key(word) {
            return true;
        }
        if !isalpha(word) || !word.chars().all(in_lexicon_ords) {
            return false;
        }
        if nchars(word) == 1 {
            return true;
        }
        if word == upper(word) && self.golds.contains_key(&lower(word)) {
            return true;
        }
        let rest = slice(word, 1, usize::MAX);
        rest == upper(&rest)
    }

    pub fn lookup(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<Ctx>) -> Lookup {
        let mut word = word.to_string();
        let mut is_nnp = false;
        if word == upper(&word) && !self.golds.contains_key(&word) {
            word = lower(&word);
            is_nnp = tag == Some("NNP");
        }
        let (mut entry, mut rating) = (self.golds.get(&word), 4);
        if entry.is_none() && !is_nnp {
            entry = self.silvers.get(&word);
            rating = 3;
        }
        let ps: Option<String> = match entry {
            None => None,
            Some(Entry::S(s)) => Some(s.clone()),
            Some(Entry::D(d)) => {
                let mut t: Option<String> = tag.map(String::from);
                if ctx.map(|c| c.future_vowel.is_none()).unwrap_or(false) && d.contains_key("None") {
                    t = Some("None".into());
                } else if !t.as_ref().map(|t| d.contains_key(t)).unwrap_or(false) {
                    t = parent_tag(tag);
                }
                match t.as_ref().and_then(|t| d.get(t)) {
                    Some(v) => v.clone(),
                    None => d.get("DEFAULT").cloned().flatten(),
                }
            }
        };
        if ps.is_none() || (is_nnp && !ps.as_ref().unwrap().contains(PRIMARY)) {
            let (p, r) = self.get_nnp(&word);
            if p.is_some() {
                return (p, r);
            }
        }
        (apply_stress(ps, stress), Some(rating))
    }

    fn suffix_s(&self, stem: Option<String>) -> Option<String> {
        let stem = stem?;
        let last = last_char(&stem)?;
        Some(if "ptkfθ".contains(last) {
            format!("{stem}s")
        } else if "szʃʒʧʤ".contains(last) {
            format!("{stem}{}z", if self.british { "ɪ" } else { "ᵻ" })
        } else {
            format!("{stem}z")
        })
    }

    pub fn stem_s(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<Ctx>) -> Lookup {
        let n = nchars(word);
        if n < 3 || !word.ends_with('s') {
            return (None, None);
        }
        let stem = if !word.ends_with("ss") && self.is_known(&drop_last(word, 1), tag) {
            drop_last(word, 1)
        } else if (word.ends_with("'s") || (n > 4 && word.ends_with("es") && !word.ends_with("ies"))) && self.is_known(&drop_last(word, 2), tag) {
            drop_last(word, 2)
        } else if n > 4 && word.ends_with("ies") && self.is_known(&format!("{}y", drop_last(word, 3)), tag) {
            format!("{}y", drop_last(word, 3))
        } else {
            return (None, None);
        };
        let (ps, r) = self.lookup(&stem, tag, stress, ctx);
        (self.suffix_s(ps), r)
    }

    fn suffix_ed(&self, stem: Option<String>) -> Option<String> {
        let stem = stem?;
        let chars: Vec<char> = stem.chars().collect();
        let last = *chars.last()?;
        Some(if "pkfθʃsʧ".contains(last) {
            format!("{stem}t")
        } else if last == 'd' {
            format!("{stem}{}d", if self.british { "ɪ" } else { "ᵻ" })
        } else if last != 't' {
            format!("{stem}d")
        } else if self.british || chars.len() < 2 {
            format!("{stem}ɪd")
        } else if US_TAUS.contains(chars[chars.len() - 2]) {
            format!("{}ɾᵻd", drop_last(&stem, 1))
        } else {
            format!("{stem}ᵻd")
        })
    }

    pub fn stem_ed(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<Ctx>) -> Lookup {
        let n = nchars(word);
        if n < 4 || !word.ends_with('d') {
            return (None, None);
        }
        let stem = if !word.ends_with("dd") && self.is_known(&drop_last(word, 1), tag) {
            drop_last(word, 1)
        } else if n > 4 && word.ends_with("ed") && !word.ends_with("eed") && self.is_known(&drop_last(word, 2), tag) {
            drop_last(word, 2)
        } else {
            return (None, None);
        };
        let (ps, r) = self.lookup(&stem, tag, stress, ctx);
        (self.suffix_ed(ps), r)
    }

    fn suffix_ing(&self, stem: Option<String>) -> Option<String> {
        let stem = stem?;
        let chars: Vec<char> = stem.chars().collect();
        let last = *chars.last()?;
        if self.british {
            if last == 'ə' || last == 'ː' {
                return None;
            }
        } else if chars.len() > 1 && last == 't' && US_TAUS.contains(chars[chars.len() - 2]) {
            return Some(format!("{}ɾɪŋ", drop_last(&stem, 1)));
        }
        Some(format!("{stem}ɪŋ"))
    }

    pub fn stem_ing(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<Ctx>) -> Lookup {
        let n = nchars(word);
        if n < 5 || !word.ends_with("ing") {
            return (None, None);
        }
        let chars: Vec<char> = word.chars().collect();
        let doubled = n > 5 && (word.ends_with("cking") || ("bcdgklmnprstvxz".contains(chars[n - 4]) && chars[n - 4] == chars[n - 5]));
        let stem = if n > 5 && self.is_known(&drop_last(word, 3), tag) {
            drop_last(word, 3)
        } else if self.is_known(&format!("{}e", drop_last(word, 3)), tag) {
            format!("{}e", drop_last(word, 3))
        } else if doubled && self.is_known(&drop_last(word, 4), tag) {
            drop_last(word, 4)
        } else {
            return (None, None);
        };
        let (ps, r) = self.lookup(&stem, tag, stress, ctx);
        (self.suffix_ing(ps), r)
    }

    pub fn get_word(&self, word: &str, tag: &str, stress: Option<f64>, ctx: Ctx) -> Lookup {
        let (ps, r) = self.get_special_case(word, tag, stress, ctx);
        if ps.is_some() {
            return (ps, r);
        }
        let wl = lower(word);
        let mut word = word.to_string();
        let cond = nchars(&word) > 1
            && isalpha(&word.replace('\'', ""))
            && word != wl
            && (tag != "NNP" || nchars(&word) > 7)
            && !self.golds.contains_key(&word)
            && !self.silvers.contains_key(&word)
            && (word == upper(&word) || slice(&word, 1, usize::MAX) == lower(&slice(&word, 1, usize::MAX)))
            && (self.golds.contains_key(&wl)
                || self.silvers.contains_key(&wl)
                || self.stem_s(&wl, Some(tag), stress, Some(ctx)).0.map(|s| !s.is_empty()).unwrap_or(false)
                || self.stem_ed(&wl, Some(tag), stress, Some(ctx)).0.map(|s| !s.is_empty()).unwrap_or(false)
                || self.stem_ing(&wl, Some(tag), stress, Some(ctx)).0.map(|s| !s.is_empty()).unwrap_or(false));
        if cond {
            word = wl;
        }
        if self.is_known(&word, Some(tag)) {
            return self.lookup(&word, Some(tag), stress, Some(ctx));
        }
        if word.ends_with("s'") && self.is_known(&format!("{}'s", drop_last(&word, 2)), Some(tag)) {
            return self.lookup(&format!("{}'s", drop_last(&word, 2)), Some(tag), stress, Some(ctx));
        }
        if word.ends_with('\'') && self.is_known(&drop_last(&word, 1), Some(tag)) {
            return self.lookup(&drop_last(&word, 1), Some(tag), stress, Some(ctx));
        }
        let r = self.stem_s(&word, Some(tag), stress, Some(ctx));
        if r.0.is_some() {
            return r;
        }
        let r = self.stem_ed(&word, Some(tag), stress, Some(ctx));
        if r.0.is_some() {
            return r;
        }
        let r = self.stem_ing(&word, Some(tag), Some(stress.unwrap_or(0.5)), Some(ctx));
        if r.0.is_some() {
            return r;
        }
        (None, None)
    }

    fn is_currency(word: &str) -> bool {
        if !word.contains('.') {
            return true;
        }
        if word.matches('.').count() > 1 {
            return false;
        }
        let cents = word.split('.').nth(1).unwrap_or("");
        nchars(cents) < 3
    }

    fn get_number(&self, word: &str, currency: Option<&str>, is_head: bool, num_flags: &str) -> Lookup {
        // suffix = trailing [a-z']+
        let chars: Vec<char> = word.chars().collect();
        let mut k = chars.len();
        while k > 0 && (chars[k - 1].is_ascii_lowercase() || chars[k - 1] == '\'') {
            k -= 1;
        }
        let suffix: Option<String> = if k < chars.len() { Some(chars[k..].iter().collect()) } else { None };
        let mut word: String = chars[..k].iter().collect();
        let mut result: Vec<(Option<String>, Option<i32>)> = vec![];
        if word.starts_with('-') {
            result.push(self.lookup("minus", None, None, None));
            word = word[1..].to_string();
        }
        let extend = |num: &str, first: bool, escape: bool, result: &mut Vec<(Option<String>, Option<i32>)>| {
            let text = if escape { num.to_string() } else { num2words::cardinal(num.parse::<i128>().unwrap_or(0)) };
            let splits: Vec<&str> = text.split(|c: char| !c.is_ascii_lowercase()).filter(|s| !s.is_empty()).collect();
            // Python re.split keeps empty strings only at the ends; an empty split word is never
            // looked up meaningfully, so empties are dropped.
            for (i, w) in splits.iter().enumerate() {
                if *w != "and" || num_flags.contains('&') {
                    if first && i == 0 && splits.len() > 1 && *w == "one" && num_flags.contains('a') {
                        result.push((Some("ə".into()), Some(4)));
                    } else {
                        result.push(self.lookup(w, None, if *w == "point" { Some(-2.0) } else { None }, None));
                    }
                } else if *w == "and" && num_flags.contains('n') && !result.is_empty() {
                    let last = result.last_mut().unwrap();
                    last.0 = last.0.as_ref().map(|p| format!("{p}ən"));
                }
            }
        };
        let suffix_is_ord = suffix.as_deref().map(|s| ORDINALS.contains(&s)).unwrap_or(false);
        if is_digit_str(&word) && suffix_is_ord {
            extend(&num2words::ordinal(word.parse().unwrap_or(0)), true, true, &mut result);
        } else if result.is_empty() && nchars(&word) == 4 && currency.and_then(currency_units).is_none() && is_digit_str(&word) {
            extend(&num2words::year(word.parse().unwrap_or(0)), true, true, &mut result);
        } else if !is_head && !word.contains('.') {
            let num = word.replace(',', "");
            let nc: Vec<char> = num.chars().collect();
            if nc.first() == Some(&'0') || nc.len() > 3 {
                for c in &nc {
                    extend(&c.to_string(), false, false, &mut result);
                }
            } else if nc.len() == 3 && !num.ends_with("00") {
                extend(&nc[0].to_string(), true, false, &mut result);
                if nc[1] == '0' {
                    result.push(self.lookup("O", None, Some(-2.0), None));
                    extend(&nc[2].to_string(), false, false, &mut result);
                } else {
                    extend(&nc[1..].iter().collect::<String>(), false, false, &mut result);
                }
            } else {
                extend(&num, true, false, &mut result);
            }
        } else if word.matches('.').count() > 1 || !is_head {
            let mut first = true;
            for num in word.replace(',', "").split('.') {
                let nc: Vec<char> = num.chars().collect();
                if nc.is_empty() {
                } else if nc[0] == '0' || (nc.len() != 2 && nc[1..].iter().any(|&n| n != '0')) {
                    for c in &nc {
                        extend(&c.to_string(), false, false, &mut result);
                    }
                } else {
                    extend(num, first, false, &mut result);
                }
                first = false;
            }
        } else if let (Some(units), true) = (currency.and_then(currency_units), Self::is_currency(&word)) {
            let w2 = word.replace(',', "");
            let mut pairs: Vec<(i128, &str)> = w2.split('.').zip([units.0, units.1]).map(|(n, u)| (if n.is_empty() { 0 } else { n.parse().unwrap_or(0) }, u)).collect();
            if pairs.len() > 1 {
                if pairs[1].0 == 0 {
                    pairs.truncate(1);
                } else if pairs[0].0 == 0 {
                    pairs.remove(0);
                }
            }
            for (i, (num, unit)) in pairs.iter().enumerate() {
                if i > 0 {
                    result.push(self.lookup("and", None, None, None));
                }
                extend(&num.to_string(), i == 0, false, &mut result);
                result.push(if num.abs() != 1 && *unit != "pence" { self.stem_s(&format!("{unit}s"), None, None, None) } else { self.lookup(unit, None, None, None) });
            }
        } else {
            let text = if is_digit_str(&word) {
                num2words::cardinal(word.parse().unwrap_or(0))
            } else if !word.contains('.') {
                let n: i128 = word.replace(',', "").parse().unwrap_or(0);
                if suffix_is_ord { num2words::ordinal(n.max(0) as u128) } else { num2words::cardinal(n) }
            } else {
                let w2 = word.replace(',', "");
                if w2.starts_with('.') {
                    let digits: Vec<String> = w2[1..].chars().map(|c| num2words::cardinal(c.to_digit(10).unwrap_or(0) as i128)).collect();
                    format!("point {}", digits.join(" "))
                } else {
                    num2words::float_str(&w2).unwrap_or_default()
                }
            };
            extend(&text, true, true, &mut result);
        }
        if result.is_empty() || result.iter().any(|r| r.0.is_none()) {
            return (None, None);
        }
        let ps = result.iter().map(|r| r.0.clone().unwrap()).collect::<Vec<_>>().join(" ");
        let rating = result.iter().filter_map(|r| r.1).min();
        let ps = Some(ps);
        match suffix.as_deref() {
            Some("s") | Some("'s") => (self.suffix_s(ps), rating),
            Some("ed") | Some("'d") => (self.suffix_ed(ps), rating),
            Some("ing") => (self.suffix_ing(ps), rating),
            _ => (ps, rating),
        }
    }

    fn append_currency(&self, ps: String, currency: Option<&str>) -> String {
        match currency.and_then(currency_units) {
            None => ps,
            Some((unit, _)) => match self.stem_s(&format!("{unit}s"), None, None, None).0 {
                Some(c) => format!("{ps} {c}"),
                None => ps,
            },
        }
    }

    fn is_number(word: &str, is_head: bool) -> bool {
        if !word.chars().any(|c| c.is_ascii_digit()) {
            return false;
        }
        let mut w = word.to_string();
        for s in ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"] {
            if w.ends_with(s) {
                w = w[..w.len() - s.len()].to_string();
                break;
            }
        }
        w.chars().enumerate().all(|(i, c)| c.is_ascii_digit() || c == ',' || c == '.' || (is_head && i == 0 && c == '-'))
    }

    /// Lexicon.__call__ (en.py:476-493)
    pub fn call(&self, tk: &MToken, ctx: Ctx) -> Lookup {
        let raw = tk.alias.clone().unwrap_or_else(|| tk.text.clone()).replace('\u{2018}', "'").replace('\u{2019}', "'");
        let nfkc: String = unicode_normalization::UnicodeNormalization::nfkc(raw.as_str()).collect();
        let word: String = nfkc
            .chars()
            .map(|c| if char_isdigit(c) { digit_value(c).map(|d| char::from_digit(d, 10).unwrap()).unwrap_or(c) } else { c })
            .collect();
        let stress = if word == lower(&word) { None } else if word == upper(&word) { Some(2.0) } else { Some(0.5) };
        let (ps, r) = self.get_word(&word, &tk.tag, stress, ctx);
        if let Some(ps) = ps {
            return (apply_stress(Some(self.append_currency(ps, tk.currency.as_deref())), tk.stress), r);
        }
        if Self::is_number(&word, tk.is_head) {
            let (ps, r) = self.get_number(&word, tk.currency.as_deref(), tk.is_head, &tk.num_flags);
            return (apply_stress(ps, tk.stress), r);
        }
        (None, None)
    }
}
