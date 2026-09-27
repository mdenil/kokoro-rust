//! PL-BERT weights (transformers AlbertModel, one shared layer applied 12 times, post-LN,
//! gelu_new; the layers run on the GPU) and the host-side embedding lookup + LayerNorm.

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
    pub(crate) word: Vec<f32>,
    pub(crate) pos: Vec<f32>,
    pub(crate) tok_type0: Vec<f32>,
    pub(crate) emb_ln: (Vec<f32>, Vec<f32>),
    pub(crate) map_in: Linear,
    pub(crate) q: Linear,
    pub(crate) k: Linear,
    pub(crate) v: Linear,
    pub(crate) dense: Linear,
    pub(crate) attn_ln: (Vec<f32>, Vec<f32>),
    pub(crate) ffn: Linear,
    pub(crate) ffn_out: Linear,
    pub(crate) full_ln: (Vec<f32>, Vec<f32>),
    pub(crate) n_token: usize,
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
}
