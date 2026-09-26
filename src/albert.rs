//! PL-BERT (transformers AlbertModel, one shared layer applied 12 times, post-LN, gelu_new).

use crate::nn::Linear;
use crate::ops;
use crate::weights::Weights;
use anyhow::{ensure, Result};

pub const EMB: usize = 128;
pub const HID: usize = 768;
pub const HEADS: usize = 12;
pub const HEAD_DIM: usize = 64;
pub const FFN: usize = 2048;
pub const LAYERS: usize = 12;
pub const MAX_POS: usize = 512;
const LN_EPS: f32 = 1e-12;

pub struct Albert {
    word: Vec<f32>,
    pos: Vec<f32>,
    tok_type0: Vec<f32>,
    emb_ln: (Vec<f32>, Vec<f32>),
    map_in: Linear,
    q: Linear,
    k: Linear,
    v: Linear,
    dense: Linear,
    attn_ln: (Vec<f32>, Vec<f32>),
    ffn: Linear,
    ffn_out: Linear,
    full_ln: (Vec<f32>, Vec<f32>),
    n_token: usize,
}

impl Albert {
    pub fn load(w: &Weights, n_token: usize) -> Result<Self> {
        let l = "bert.encoder.albert_layer_groups.0.albert_layers.0";
        let tt = w.get("bert.embeddings.token_type_embeddings.weight", &[2, EMB])?;
        Ok(Self {
            word: w.get("bert.embeddings.word_embeddings.weight", &[n_token, EMB])?,
            pos: w.get("bert.embeddings.position_embeddings.weight", &[MAX_POS, EMB])?,
            tok_type0: tt[..EMB].to_vec(),
            emb_ln: (
                w.get("bert.embeddings.LayerNorm.weight", &[EMB])?,
                w.get("bert.embeddings.LayerNorm.bias", &[EMB])?,
            ),
            map_in: Linear::load(w, "bert.encoder.embedding_hidden_mapping_in", EMB, HID, true)?,
            q: Linear::load(w, &format!("{l}.attention.query"), HID, HID, true)?,
            k: Linear::load(w, &format!("{l}.attention.key"), HID, HID, true)?,
            v: Linear::load(w, &format!("{l}.attention.value"), HID, HID, true)?,
            dense: Linear::load(w, &format!("{l}.attention.dense"), HID, HID, true)?,
            attn_ln: (
                w.get(&format!("{l}.attention.LayerNorm.weight"), &[HID])?,
                w.get(&format!("{l}.attention.LayerNorm.bias"), &[HID])?,
            ),
            ffn: Linear::load(w, &format!("{l}.ffn"), HID, FFN, true)?,
            ffn_out: Linear::load(w, &format!("{l}.ffn_output"), FFN, HID, true)?,
            full_ln: (
                w.get(&format!("{l}.full_layer_layer_norm.weight"), &[HID])?,
                w.get(&format!("{l}.full_layer_layer_norm.bias"), &[HID])?,
            ),
            n_token,
        })
    }

    /// Embedding output after LayerNorm: [t, 128]
    pub fn embeddings(&self, ids: &[i64]) -> Result<Vec<f32>> {
        let t = ids.len();
        ensure!(t <= MAX_POS, "sequence length {t} exceeds {MAX_POS}");
        let mut x = vec![0.0f32; t * EMB];
        for (i, &id) in ids.iter().enumerate() {
            ensure!(id >= 0 && (id as usize) < self.n_token, "token id {id} out of range");
            let we = &self.word[id as usize * EMB..(id as usize + 1) * EMB];
            let pe = &self.pos[i * EMB..(i + 1) * EMB];
            for j in 0..EMB {
                x[i * EMB + j] = (we[j] + self.tok_type0[j]) + pe[j];
            }
        }
        ops::layer_norm_rows(&mut x, EMB, Some(&self.emb_ln.0), Some(&self.emb_ln.1), LN_EPS);
        Ok(x)
    }

    /// One application of the shared layer: h [t, 768] -> [t, 768]
    pub fn layer(&self, h: &[f32], t: usize) -> Vec<f32> {
        let q = self.q.forward(h, t);
        let k = self.k.forward(h, t);
        let v = self.v.forward(h, t);
        let scale = (HEAD_DIM as f32).powf(-0.5);
        let mut ctx = vec![0.0f32; t * HID];
        let mut scores = vec![0.0f32; t * t];
        for hd in 0..HEADS {
            let off = hd * HEAD_DIM;
            // scores[i, j] = q[i, off..] . k[j, off..]
            ops::gemm(t, HEAD_DIM, t, &q, off, HID, 1, &k, off, 1, HID, 0.0, &mut scores, 0, t, 1);
            for row in scores.chunks_exact_mut(t) {
                let mut mx = f32::NEG_INFINITY;
                for s in row.iter_mut() {
                    *s *= scale;
                    mx = mx.max(*s);
                }
                let mut sum = 0.0f32;
                for s in row.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for s in row.iter_mut() {
                    *s *= inv;
                }
            }
            ops::gemm(t, t, HEAD_DIM, &scores, 0, t, 1, &v, off, HID, 1, 0.0, &mut ctx, off, HID, 1);
        }
        let mut a = self.dense.forward(&ctx, t);
        for (x, y) in a.iter_mut().zip(h) {
            *x += *y;
        }
        ops::layer_norm_rows(&mut a, HID, Some(&self.attn_ln.0), Some(&self.attn_ln.1), LN_EPS);
        let mut f = self.ffn.forward(&a, t);
        ops::gelu_new(&mut f);
        let mut o = self.ffn_out.forward(&f, t);
        for (x, y) in o.iter_mut().zip(&a) {
            *x += *y;
        }
        ops::layer_norm_rows(&mut o, HID, Some(&self.full_ln.0), Some(&self.full_ln.1), LN_EPS);
        o
    }

    /// last_hidden_state [t, 768]
    pub fn forward(&self, ids: &[i64]) -> Result<Vec<f32>> {
        let t = ids.len();
        let e = self.embeddings(ids)?;
        let mut h = self.map_in.forward(&e, t);
        for _ in 0..LAYERS {
            h = self.layer(&h, t);
        }
        Ok(h)
    }
}
