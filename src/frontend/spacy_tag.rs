//! en_core_web_sm 3.8.0 lexical features + tok2vec (HashEmbed CNN) + tagger softmax, from the pinned
//! weights exported by oracle/export_spacy.py. Semantics follow spaCy 3.8 / thinc 8:
//! - features [NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE]; strings -> StringStore.add ids
//!   (symbol id if the string is a spaCy symbol, else MurmurHash64A(utf8, seed 1); "" -> 0);
//! - NORM = special-case NORM | lexeme_norm table (keyed by string id) | BASE_NORMS | lower();
//! - HashEmbed: MurmurHash3_x86_128_uint64(id, seed) -> 4 u32 keys % nV, rows summed in key order;
//! - maxout = max_p (x W[o,p]^T + b[o,p]); layernorm var + 1e-8;
//! - the 4 residual expand_window blocks run over the doc flattened with 4 zero pad rows on each
//!   side (thinc with_array pad=4): the pad rows evolve through the blocks and feed the edge tokens.

use super::pystr;
use super::spacy_tok::{py_isspace, Tok};
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub const WIDTH: usize = 96;
const NP: usize = 3;
const PAD: usize = 4;
const SEEDS: [u32; 6] = [8, 9, 10, 11, 12, 13];

pub fn murmurhash64a(key: &[u8], seed: u64) -> u64 {
    const M: u64 = 0xc6a4a7935bd1e995;
    const R: u32 = 47;
    let len = key.len();
    let mut h = seed ^ (len as u64).wrapping_mul(M);
    let mut chunks = key.chunks_exact(8);
    for c in &mut chunks {
        let mut k = u64::from_le_bytes(c.try_into().unwrap());
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }
    let tail = chunks.remainder();
    if !tail.is_empty() {
        for (i, &b) in tail.iter().enumerate() {
            h ^= (b as u64) << (8 * i);
        }
        h = h.wrapping_mul(M);
    }
    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

/// thinc numpy_ops.pyx MurmurHash3_x86_128_uint64 (verbatim arithmetic).
pub fn murmur3_u64(val: u64, seed: u32) -> [u32; 4] {
    let mut h1: u64 = val;
    h1 = h1.wrapping_mul(0x87c37b91114253d5);
    h1 = h1.rotate_left(31);
    h1 = h1.wrapping_mul(0x4cf5ad432745937f);
    h1 ^= seed as u64;
    h1 ^= 8;
    let mut h2: u64 = seed as u64;
    h2 ^= 8;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    let fmix = |mut h: u64| {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    };
    h1 = fmix(h1);
    h2 = fmix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    [h1 as u32, (h1 >> 32) as u32, h2 as u32, (h2 >> 32) as u32]
}

/// spaCy lex_attrs.word_shape.
pub fn word_shape(text: &str) -> String {
    if text.chars().count() >= 100 {
        return "LONG".into();
    }
    let mut shape = String::new();
    let mut last: Option<char> = None;
    let mut seq = 0;
    for c in text.chars() {
        let sc = if pystr::char_isalpha(c) {
            if c.is_uppercase() {
                'X'
            } else {
                'x'
            }
        } else if pystr::char_isdigit(c) {
            'd'
        } else {
            c
        };
        if Some(sc) == last {
            seq += 1;
        } else {
            seq = 0;
            last = Some(sc);
        }
        if seq < 4 {
            shape.push(sc);
        }
    }
    shape
}

struct Maxout {
    w: Vec<f32>, // [nO, nP, nI]
    b: Vec<f32>, // [nO, nP]
    ni: usize,
}

impl Maxout {
    fn forward(&self, x: &[f32], out: &mut [f32]) {
        for o in 0..WIDTH {
            let mut best = f32::NEG_INFINITY;
            for p in 0..NP {
                let w = &self.w[(o * NP + p) * self.ni..(o * NP + p + 1) * self.ni];
                let mut acc = 0f64;
                for (a, b) in x.iter().zip(w) {
                    acc += (*a as f64) * (*b as f64);
                }
                let v = acc as f32 + self.b[o * NP + p];
                if v > best {
                    best = v;
                }
            }
            out[o] = best;
        }
    }
}

struct LayerNorm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    fn forward(&self, x: &mut [f32]) {
        let n = x.len() as f64;
        let mu = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = x.iter().map(|&v| (v as f64 - mu).powi(2)).sum::<f64>() / n + 1e-8;
        let inv = var.powf(-0.5);
        for (i, v) in x.iter_mut().enumerate() {
            *v = (((*v as f64 - mu) * inv) as f32) * self.g[i] + self.b[i];
        }
    }
}

pub struct Tagger {
    pub labels: Vec<String>,
    symbols: HashMap<String, u64>,
    /// lexeme_norm Lookups table: keys are get_string_id(word)
    lexeme_norm: HashMap<u64, String>,
    base_norms: HashMap<String, String>,
    embed: Vec<(Vec<f32>, usize)>, // (E [nV, 96], nV) per column
    reduce: (Maxout, LayerNorm),
    blocks: Vec<(Maxout, LayerNorm)>,
    softmax_w: Vec<f32>, // [nL, 96]
    softmax_b: Vec<f32>,
    pad: usize,
}

/// Per-token tagger output.
pub struct Tagged {
    pub tags: Vec<String>,
    /// tok2vec output [T, 96] (the doc.tensor seam)
    pub tensor: Vec<f32>,
    /// tagger scores (softmax) [T, nL]
    pub scores: Vec<f32>,
}

impl Tagger {
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |f: &str| -> Result<serde_json::Value> {
            serde_json::from_str(&std::fs::read_to_string(dir.join(f)).with_context(|| format!("reading {f}"))?).with_context(|| format!("parsing {f}"))
        };
        let structure = read("model_structure.json")?;
        let labels: Vec<String> = structure["labels"].as_array().context("labels")?.iter().map(|l| l.as_str().unwrap_or_default().to_string()).collect();
        let symbols = read("symbols.json")?.as_object().context("symbols")?.iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0))).collect();
        let strmap = |v: &serde_json::Value| -> HashMap<String, String> {
            v.as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string())).collect()).unwrap_or_default()
        };
        let lexeme_norm = strmap(&read("lookups.json")?["lexeme_norm"])
            .into_iter()
            .map(|(k, v)| Ok((k.parse::<u64>().context("lexeme_norm key")?, v)))
            .collect::<Result<HashMap<u64, String>>>()?;
        let base_norms = strmap(&read("base_norms.json")?);
        let st = crate::st::load(&dir.join("tagger_weights.safetensors"))?;
        let get = |k: &str| -> Result<Vec<f32>> { Ok(st.get(k).with_context(|| format!("missing {k}"))?.f32()?.to_vec()) };
        let mut embed = vec![];
        for (node, nv) in [(28, 5000), (30, 1000), (32, 2500), (34, 2500), (36, 50), (38, 50)] {
            let e = get(&format!("tok2vec.{node}.hashembed.E"))?;
            ensure!(e.len() == nv * WIDTH, "hashembed {node} shape");
            embed.push((e, nv));
        }
        let maxout = |node: usize, ni: usize| -> Result<Maxout> {
            let w = get(&format!("tok2vec.{node}.maxout.W"))?;
            ensure!(w.len() == WIDTH * NP * ni, "maxout {node} shape");
            Ok(Maxout { w, b: get(&format!("tok2vec.{node}.maxout.b"))?, ni })
        };
        let ln = |node: usize| -> Result<LayerNorm> {
            Ok(LayerNorm { g: get(&format!("tok2vec.{node}.layernorm.G"))?, b: get(&format!("tok2vec.{node}.layernorm.b"))? })
        };
        let reduce = (maxout(39, 6 * WIDTH)?, ln(40)?);
        let mut blocks = vec![];
        for (m, l) in [(57, 58), (59, 60), (61, 62), (63, 64)] {
            blocks.push((maxout(m, 3 * WIDTH)?, ln(l)?));
        }
        let softmax_w = get("tagger.3.softmax.W")?;
        let softmax_b = get("tagger.3.softmax.b")?;
        ensure!(softmax_w.len() == labels.len() * WIDTH && softmax_b.len() == labels.len(), "softmax shape");
        Ok(Self { labels, symbols, lexeme_norm, base_norms, embed, reduce, blocks, softmax_w, softmax_b, pad: PAD })
    }

    /// Test-only negative controls (each must break oracle agreement): "no_symbols", "no_lexeme_norm",
    /// "no_pad" (thinc with_array pad=4 -> 0).
    #[doc(hidden)]
    pub fn negative_control(&mut self, which: &str) {
        match which {
            "no_symbols" => self.symbols.clear(),
            "no_lexeme_norm" => self.lexeme_norm.clear(),
            "no_pad" => self.pad = 0,
            _ => panic!("unknown negative control {which}"),
        }
    }

    /// StringStore.add(s)
    pub fn string_id(&self, s: &str) -> u64 {
        if s.is_empty() {
            return 0;
        }
        if let Some(&id) = self.symbols.get(s) {
            return id;
        }
        murmurhash64a(s.as_bytes(), 1)
    }

    pub fn norm(&self, t: &Tok) -> String {
        if let Some(n) = &t.norm {
            return n.clone();
        }
        if let Some(n) = self.lexeme_norm.get(&self.string_id(&t.text)) {
            return n.clone();
        }
        if let Some(n) = self.base_norms.get(&t.text) {
            return n.clone();
        }
        pystr::lower(&t.text)
    }

    pub fn features(&self, t: &Tok) -> [u64; 6] {
        let chars: Vec<char> = t.text.chars().collect();
        let prefix: String = chars.iter().take(1).collect();
        let suffix: String = chars[chars.len().saturating_sub(3)..].iter().collect();
        let is_space = !chars.is_empty() && chars.iter().all(|&c| py_isspace(c));
        [
            self.string_id(&self.norm(t)),
            self.string_id(&prefix),
            self.string_id(&suffix),
            self.string_id(&word_shape(&t.text)),
            t.space as u64,
            is_space as u64,
        ]
    }

    pub fn tag(&self, toks: &[Tok]) -> Tagged {
        let ids: Vec<[u64; 6]> = toks.iter().map(|t| self.features(t)).collect();
        self.forward_ids(&ids)
    }

    pub fn forward_ids(&self, ids: &[[u64; 6]]) -> Tagged {
        let n = ids.len();
        if n == 0 {
            return Tagged { tags: vec![], tensor: vec![], scores: vec![] };
        }
        let rows = n + 2 * self.pad;
        let mut x = vec![0f32; rows * WIDTH];
        let mut emb = vec![0f32; 6 * WIDTH];
        for (i, f) in ids.iter().enumerate() {
            for (c, (table, nv)) in self.embed.iter().enumerate() {
                let dst = &mut emb[c * WIDTH..(c + 1) * WIDTH];
                dst.fill(0.0);
                for k in murmur3_u64(f[c], SEEDS[c]) {
                    let row = (k as usize) % nv;
                    for (d, s) in dst.iter_mut().zip(&table[row * WIDTH..(row + 1) * WIDTH]) {
                        *d += *s;
                    }
                }
            }
            let out = &mut x[(self.pad + i) * WIDTH..(self.pad + i + 1) * WIDTH];
            self.reduce.0.forward(&emb, out);
            self.reduce.1.forward(out);
        }
        let mut win = vec![0f32; 3 * WIDTH];
        let mut y = vec![0f32; WIDTH];
        for (m, l) in &self.blocks {
            let prev = x.clone();
            for r in 0..rows {
                win.fill(0.0);
                if r > 0 {
                    win[..WIDTH].copy_from_slice(&prev[(r - 1) * WIDTH..r * WIDTH]);
                }
                win[WIDTH..2 * WIDTH].copy_from_slice(&prev[r * WIDTH..(r + 1) * WIDTH]);
                if r + 1 < rows {
                    win[2 * WIDTH..].copy_from_slice(&prev[(r + 1) * WIDTH..(r + 2) * WIDTH]);
                }
                m.forward(&win, &mut y);
                l.forward(&mut y);
                for (d, v) in x[r * WIDTH..(r + 1) * WIDTH].iter_mut().zip(&y) {
                    *d += *v;
                }
            }
        }
        let tensor = x[self.pad * WIDTH..(self.pad + n) * WIDTH].to_vec();
        let nl = self.labels.len();
        let mut scores = vec![0f32; n * nl];
        let mut tags = Vec::with_capacity(n);
        for i in 0..n {
            let h = &tensor[i * WIDTH..(i + 1) * WIDTH];
            let mut logits = vec![0f64; nl];
            for (j, lg) in logits.iter_mut().enumerate() {
                let w = &self.softmax_w[j * WIDTH..(j + 1) * WIDTH];
                *lg = h.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() + self.softmax_b[j] as f64;
            }
            let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = logits.iter().map(|v| (v - mx).exp()).sum();
            let mut best = 0;
            for j in 0..nl {
                scores[i * nl + j] = ((logits[j] - mx).exp() / z) as f32;
                if logits[j] > logits[best] {
                    best = j;
                }
            }
            tags.push(self.labels[best].clone());
        }
        Tagged { tags, tensor, scores }
    }
}
