//! Python `str` semantics the misaki port depends on (isalpha, isdigit, capitalize, ...).

use unicode_general_category::{get_general_category, GeneralCategory as G};

/// Python str.isalpha(): non-empty and every char in Lu/Ll/Lt/Lm/Lo.
pub fn isalpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char_isalpha)
}

pub fn char_isalpha(c: char) -> bool {
    matches!(get_general_category(c), G::UppercaseLetter | G::LowercaseLetter | G::TitlecaseLetter | G::ModifierLetter | G::OtherLetter)
}

/// Python str.isdigit() for one char: Numeric_Type Decimal or Digit. Approximation: Nd plus the
/// superscript/subscript/circled digit forms (Numeric_Type=Digit) that occur in English text.
pub fn char_isdigit(c: char) -> bool {
    get_general_category(c) == G::DecimalNumber
        || matches!(c, '²' | '³' | '¹' | '⁰' | '⁴'..='⁹' | '₀'..='₉' | '①'..='⑨' | '⓪')
}

/// Numeric value for a digit char (used by misaki's numeric_if_needed after NFKC). ASCII and the
/// common decimal-digit blocks; None otherwise (misaki then keeps the char).
pub fn digit_value(c: char) -> Option<u32> {
    if c.is_ascii_digit() {
        return c.to_digit(10);
    }
    let cp = c as u32;
    // zero code points of common Nd blocks (Arabic-Indic, Extended, Devanagari, Bengali, fullwidth)
    for z in [0x0660u32, 0x06F0, 0x0966, 0x09E6, 0xFF10, 0x2070, 0x2080] {
        if (z..z + 10).contains(&cp) {
            return Some(cp - z);
        }
    }
    match c {
        '¹' => Some(1),
        '²' => Some(2),
        '³' => Some(3),
        '①'..='⑨' => Some(c as u32 - '①' as u32 + 1),
        _ => None,
    }
}

pub fn lower(s: &str) -> String {
    s.to_lowercase()
}

pub fn upper(s: &str) -> String {
    s.to_uppercase()
}

/// Python str.capitalize(): first char upper(title)case, rest lowercase.
pub fn capitalize(s: &str) -> String {
    let mut it = s.chars();
    match it.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + &it.as_str().to_lowercase(),
    }
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
