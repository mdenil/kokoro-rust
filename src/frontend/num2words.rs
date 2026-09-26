//! English number words as consumed by misaki (cardinal, ordinal, year, float). Behavioural
//! reimplementation validated by differential test against num2words 0.5.14
//! (fixtures/frontend/num2words_en.oracle.jsonl); no code copied (num2words is LGPL).

const LOW: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

/// (value, word) cards in descending order, the unit table the split/merge algorithm walks.
fn cards() -> Vec<(u128, String)> {
    let mut c: Vec<(u128, String)> = vec![];
    let highs = ["m", "b", "tr", "quadr", "quint", "sext", "sept", "oct", "non", "dec"];
    for (i, h) in highs.iter().enumerate().rev() {
        c.push((10u128.pow(6 + 3 * i as u32), format!("{h}illion")));
    }
    c.push((1000, "thousand".into()));
    c.push((100, "hundred".into()));
    for t in (2..10).rev() {
        c.push((t * 10, TENS[t as usize].into()));
    }
    for n in (0..20).rev() {
        c.push((n, LOW[n as usize].into()));
    }
    c
}

#[derive(Clone, Debug)]
enum Node {
    Pair(String, u128),
    List(Vec<Node>),
}

fn splitnum(value: u128, cards: &[(u128, String)]) -> Vec<Node> {
    for (elem, word) in cards {
        if *elem > value {
            continue;
        }
        let (div, m) = if value == 0 { (1, 0) } else { (value / elem, value % elem) };
        let mut out = vec![];
        if div == 1 {
            out.push(Node::Pair("one".into(), 1));
        } else {
            out.push(Node::List(splitnum(div, cards)));
        }
        out.push(Node::Pair(word.clone(), *elem));
        if m != 0 {
            out.push(Node::List(splitnum(m, cards)));
        }
        return out;
    }
    unreachable!("zero card always matches")
}

fn merge(l: (String, u128), r: (String, u128)) -> (String, u128) {
    let ((lt, ln), (rt, rn)) = (l, r);
    if ln == 1 && rn < 100 {
        (rt, rn)
    } else if 100 > ln && ln > rn {
        (format!("{lt}-{rt}"), ln + rn)
    } else if ln >= 100 && 100 > rn {
        (format!("{lt} and {rt}"), ln + rn)
    } else if rn > ln {
        (format!("{lt} {rt}"), ln * rn)
    } else {
        (format!("{lt}, {rt}"), ln + rn)
    }
}

fn clean(val: Vec<Node>) -> (String, u128) {
    let mut val = val;
    while val.len() != 1 {
        let mut out = vec![];
        match (&val[0], &val[1]) {
            (Node::Pair(a, an), Node::Pair(b, bn)) => {
                let m = merge((a.clone(), *an), (b.clone(), *bn));
                out.push(Node::Pair(m.0, m.1));
                if val.len() > 2 {
                    out.push(Node::List(val[2..].to_vec()));
                }
            }
            _ => {
                for e in &val {
                    match e {
                        Node::List(l) if l.len() == 1 => out.push(l[0].clone()),
                        Node::List(l) => {
                            let (t, n) = clean(l.clone());
                            out.push(Node::Pair(t, n));
                        }
                        p => out.push(p.clone()),
                    }
                }
            }
        }
        val = out;
    }
    match val.pop().unwrap() {
        Node::Pair(t, n) => (t, n),
        Node::List(l) => clean(l),
    }
}

/// num2words(n) for integers (negative -> "minus ...").
pub fn cardinal(n: i128) -> String {
    let neg = n < 0;
    let v = n.unsigned_abs();
    let (w, _) = clean(splitnum(v, &cards()));
    if neg { format!("minus {w}") } else { w }
}

pub fn ordinal(n: u128) -> String {
    let card = cardinal(n as i128);
    let mut outwords: Vec<String> = card.split(' ').map(String::from).collect();
    let last = outwords.pop().unwrap();
    let mut lastwords: Vec<String> = last.split('-').map(String::from).collect();
    let lw = lastwords.pop().unwrap().to_lowercase();
    let irregular = [
        ("one", "first"), ("two", "second"), ("three", "third"), ("four", "fourth"), ("five", "fifth"), ("six", "sixth"),
        ("seven", "seventh"), ("eight", "eighth"), ("nine", "ninth"), ("ten", "tenth"), ("eleven", "eleventh"), ("twelve", "twelfth"),
    ];
    let lw = match irregular.iter().find(|(c, _)| *c == lw) {
        Some((_, o)) => o.to_string(),
        None if lw.ends_with('y') => format!("{}ieth", &lw[..lw.len() - 1]),
        None => format!("{lw}th"),
    };
    lastwords.push(lw);
    outwords.push(lastwords.join("-"));
    outwords.join(" ")
}

pub fn year(val: i128) -> String {
    let (v, suffix) = if val < 0 { (-val, Some("BC")) } else { (val, None) };
    let (high, low) = (v / 100, v % 100);
    let text = if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
        cardinal(v)
    } else {
        let lowtext = if low == 0 { "hundred".to_string() } else if low < 10 { format!("oh-{}", cardinal(low)) } else { cardinal(low) };
        format!("{} {lowtext}", cardinal(high))
    };
    match suffix {
        Some(s) => format!("{text} {s}"),
        None => text,
    }
}

/// num2words(float(s)) where `s` is the decimal text misaki passes (e.g. "3.14", "12.50").
/// Integral values go the integer path; otherwise "<int> point <digit> <digit> ...", with the
/// number of fraction digits taken from Python's shortest float repr.
pub fn float_str(s: &str) -> Option<String> {
    let v: f64 = s.parse().ok()?;
    if v.fract() == 0.0 && v.abs() < 1e15 {
        return Some(cardinal(v as i128));
    }
    let repr = format!("{v}"); // Rust Display = shortest round-trip, like Python repr (no exponent here)
    let precision = repr.split_once('.').map(|(_, f)| f.len()).unwrap_or(0) as i32;
    let pre = v.trunc();
    let post_f = (v - pre).abs() * 10f64.powi(precision);
    let post = if (post_f.round() - post_f).abs() < 0.01 { post_f.round() as u64 } else { post_f.floor() as u64 };
    let mut post_s = post.to_string();
    while (post_s.len() as i32) < precision {
        post_s.insert(0, '0');
    }
    let mut out = vec![cardinal(pre as i128)];
    if precision > 0 {
        out.push("point".into());
    }
    for ch in post_s.chars().take(precision as usize) {
        out.push(cardinal(ch.to_digit(10)? as i128));
    }
    Some(out.join(" "))
}
