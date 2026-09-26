//! Python `str` semantics the misaki / spaCy ports depend on, EXACT for the pinned reference
//! interpreter (Python 3.12.3, Unicode 15.0.0) via generated tables (pyunicode.rs), not Rust's own
//! (newer-Unicode) char properties. Verified exhaustively over every scalar value by
//! tests/frontend_pystr.rs.

use super::pyunicode as t;

/// (Python version, Unicode version) the tables were generated from.
pub fn table_basis() -> (&'static str, &'static str) {
    (t::PYTHON, t::UNICODE)
}

fn in_ranges(r: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    match r.binary_search_by(|&(a, b)| {
        if b < cp {
            std::cmp::Ordering::Less
        } else if a > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(_) => true,
        Err(_) => false,
    }
}

fn mapped(table: &'static [(u32, &'static str)], c: char) -> Option<&'static str> {
    table.binary_search_by_key(&(c as u32), |&(k, _)| k).ok().map(|i| table[i].1)
}

/// Python str.isalpha() for one char.
pub fn char_isalpha(c: char) -> bool {
    in_ranges(t::ISALPHA, c)
}

/// Python str.isalpha(): non-empty and every char alphabetic.
pub fn isalpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char_isalpha)
}

/// Python str.isdigit() for one char (Numeric_Type Digit or Decimal).
pub fn char_isdigit(c: char) -> bool {
    digit_value(c).is_some()
}

/// int(unicodedata.numeric(c)) for chars where c.isdigit() (misaki numeric_if_needed); else None.
pub fn digit_value(c: char) -> Option<u32> {
    let cp = c as u32;
    let i = t::DIGITS.partition_point(|&(_, end, _)| end < cp);
    t::DIGITS.get(i).filter(|&&(start, _, _)| start <= cp).map(|&(start, _, v)| v + (cp - start))
}

/// Python str.isspace() for one char.
pub fn char_isspace(c: char) -> bool {
    in_ranges(t::ISSPACE, c)
}

/// Python str.isupper() for one char.
pub fn char_isupper(c: char) -> bool {
    in_ranges(t::ISUPPER, c)
}

/// Python str.strip() / lstrip() / rstrip() (no argument: Python whitespace).
pub fn strip(s: &str) -> &str {
    s.trim_matches(char_isspace)
}

pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(char_isspace)
}

pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(char_isspace)
}

/// CPython handle_capital_sigma: U+03A3 lowercases to final ς when preceded (skipping
/// case-ignorable chars) by a cased char and not followed (skipping case-ignorables) by one.
fn sigma(chars: &[char], i: usize) -> char {
    let mut j = i as isize - 1;
    while j >= 0 && in_ranges(t::SIGMA_IGNORABLE, chars[j as usize]) {
        j -= 1;
    }
    let mut fin = j >= 0 && in_ranges(t::SIGMA_CASED, chars[j as usize]);
    if fin && i + 1 < chars.len() {
        let mut k = i + 1;
        while k < chars.len() && in_ranges(t::SIGMA_IGNORABLE, chars[k]) {
            k += 1;
        }
        fin = k == chars.len() || !in_ranges(t::SIGMA_CASED, chars[k]);
    }
    if fin {
        'ς'
    } else {
        'σ'
    }
}

fn push_lower(chars: &[char], i: usize, out: &mut String) {
    let c = chars[i];
    if c == 'Σ' {
        out.push(sigma(chars, i));
    } else {
        match mapped(t::LOWER, c) {
            Some(m) => out.push_str(m),
            None => out.push(c),
        }
    }
}

/// Python str.lower() (full case mapping incl. Final_Sigma).
pub fn lower(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for i in 0..chars.len() {
        push_lower(&chars, i, &mut out);
    }
    out
}

/// Python str.upper() (full case mapping).
pub fn upper(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match mapped(t::UPPER, c) {
            Some(m) => out.push_str(m),
            None => out.push(c),
        }
    }
    out
}

/// Python str.capitalize(): first char TITLE-cased (full mapping), the rest lower() (Final_Sigma
/// evaluated over the whole string).
pub fn capitalize(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    if let Some(&f) = chars.first() {
        match mapped(t::TITLE, f) {
            Some(m) => out.push_str(m),
            None => out.push(f),
        }
    }
    for i in 1..chars.len() {
        push_lower(&chars, i, &mut out);
    }
    out
}

pub fn nchars(s: &str) -> usize {
    s.chars().count()
}

/// Python s[a..b] on code points (clamped like Python slicing for non-negative bounds).
pub fn slice(s: &str, a: usize, b: usize) -> String {
    s.chars().skip(a).take(b.saturating_sub(a)).collect()
}

/// Python s[:-n]
pub fn drop_last(s: &str, n: usize) -> String {
    let k = nchars(s);
    slice(s, 0, k.saturating_sub(n))
}

pub fn last_char(s: &str) -> Option<char> {
    s.chars().last()
}
