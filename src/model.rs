//! KModel definition: weights of every stage (loaded once, uploaded by the GPU engine), vocabulary,
//! and the host-side pieces of the forward (phoneme -> id mapping, duration rounding).

use crate::albert::Albert;
use crate::nn::{AdainResBlk1d, BiLstm, Conv1d, Linear};
use crate::ops;
use crate::vocoder::Decoder;
use crate::weights::Weights;
use anyhow::{ensure, Result};
use std::path::Path;

pub const STYLE_DIM: usize = 128;
pub const HIDDEN: usize = 512;
pub const MAX_DUR: usize = 50;
pub const SAMPLE_RATE: usize = 24000;
/// Output samples per predicted duration frame (2 × 10 × 6 × hop 5).
pub const SAMPLES_PER_FRAME: usize = 600;
/// Max input_ids length including the two boundary tokens.
pub const CONTEXT_LEN: usize = 512;

pub struct TextEncoder {
    pub(crate) embedding: Vec<f32>,
    pub(crate) cnn: Vec<(Conv1d, Vec<f32>, Vec<f32>)>,
    pub(crate) lstm: BiLstm,
    pub(crate) n_token: usize,
}

impl TextEncoder {
    fn load(w: &Weights, n_token: usize) -> Result<Self> {
        let mut cnn = vec![];
        for i in 0..3 {
            cnn.push((
                Conv1d::load(w, &format!("text_encoder.cnn.{i}.0"), HIDDEN, HIDDEN, 5, 1, 2, 1, true)?,
                w.get(&format!("text_encoder.cnn.{i}.1.gamma"), &[HIDDEN])?,
                w.get(&format!("text_encoder.cnn.{i}.1.beta"), &[HIDDEN])?,
            ));
        }
        Ok(Self {
            embedding: w.get("text_encoder.embedding.weight", &[n_token, HIDDEN])?,
            cnn,
            lstm: BiLstm::load(w, "text_encoder.lstm", HIDDEN, HIDDEN / 2)?,
            n_token,
        })
    }
}

pub struct Predictor {
    pub(crate) dur_lstms: Vec<BiLstm>,
    pub(crate) dur_norms: Vec<Linear>,
    pub(crate) lstm: BiLstm,
    pub(crate) duration_proj: Linear,
    pub(crate) shared: BiLstm,
    pub(crate) f0: Vec<AdainResBlk1d>,
    pub(crate) n: Vec<AdainResBlk1d>,
    pub(crate) f0_proj: Conv1d,
    pub(crate) n_proj: Conv1d,
}

impl Predictor {
    fn load(w: &Weights) -> Result<Self> {
        let cin = HIDDEN + STYLE_DIM;
        let mut dur_lstms = vec![];
        let mut dur_norms = vec![];
        for i in 0..3 {
            dur_lstms.push(BiLstm::load(w, &format!("predictor.text_encoder.lstms.{}", 2 * i), cin, HIDDEN / 2)?);
            dur_norms.push(Linear::load(w, &format!("predictor.text_encoder.lstms.{}.fc", 2 * i + 1), STYLE_DIM, 2 * HIDDEN, true)?);
        }
        let blocks = |name: &str| -> Result<Vec<AdainResBlk1d>> {
            Ok(vec![
                AdainResBlk1d::load(w, &format!("predictor.{name}.0"), HIDDEN, HIDDEN, STYLE_DIM, false)?,
                AdainResBlk1d::load(w, &format!("predictor.{name}.1"), HIDDEN, HIDDEN / 2, STYLE_DIM, true)?,
                AdainResBlk1d::load(w, &format!("predictor.{name}.2"), HIDDEN / 2, HIDDEN / 2, STYLE_DIM, false)?,
            ])
        };
        Ok(Self {
            dur_lstms,
            dur_norms,
            lstm: BiLstm::load(w, "predictor.lstm", cin, HIDDEN / 2)?,
            duration_proj: Linear::load(w, "predictor.duration_proj.linear_layer", HIDDEN, MAX_DUR, true)?,
            shared: BiLstm::load(w, "predictor.shared", cin, HIDDEN / 2)?,
            f0: blocks("F0")?,
            n: blocks("N")?,
            f0_proj: Conv1d::load(w, "predictor.F0_proj", HIDDEN / 2, 1, 1, 1, 0, 1, true)?,
            n_proj: Conv1d::load(w, "predictor.N_proj", HIDDEN / 2, 1, 1, 1, 0, 1, true)?,
        })
    }
}

/// Round-half-to-even of sigmoid-sum durations divided by speed, clamped to >= 1.
pub fn durations_from_logits(logits: &[f32], t: usize, speed: f32) -> Vec<i64> {
    (0..t)
        .map(|i| {
            let row = &logits[i * MAX_DUR..(i + 1) * MAX_DUR];
            let s: f32 = row.iter().map(|&v| ops::sigmoid(v)).sum();
            let d = (s / speed).round_ties_even();
            (d as i64).max(1)
        })
        .collect()
}

pub struct Output {
    pub audio: Vec<f32>,
    pub pred_dur: Vec<i64>,
}

pub struct Kokoro {
    pub albert: Albert,
    pub bert_encoder: Linear,
    pub predictor: Predictor,
    pub text_encoder: TextEncoder,
    pub decoder: Decoder,
    pub vocab: std::collections::HashMap<char, i64>,
}

impl Kokoro {
    pub fn load(weights_path: &Path, config_path: &Path) -> Result<Self> {
        let w = Weights::load(weights_path)?;
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        Self::from_weights(&w, &cfg)
    }

    pub fn from_weights(w: &Weights, cfg: &serde_json::Value) -> Result<Self> {
        let n_token = cfg["n_token"].as_u64().unwrap_or(178) as usize;
        let mut vocab = std::collections::HashMap::new();
        if let Some(obj) = cfg["vocab"].as_object() {
            for (k, v) in obj {
                let mut chars = k.chars();
                let c = chars.next().unwrap();
                ensure!(chars.next().is_none(), "vocab key {k:?} is not a single char");
                vocab.insert(c, v.as_i64().unwrap());
            }
        }
        ensure!(!vocab.is_empty(), "config has no vocab");
        Ok(Self {
            albert: Albert::load(w, n_token)?,
            bert_encoder: Linear::load(w, "bert_encoder", 768, HIDDEN, true)?,
            predictor: Predictor::load(w)?,
            text_encoder: TextEncoder::load(w, n_token)?,
            decoder: Decoder::load(w)?,
            vocab,
        })
    }

    /// Phoneme string -> input ids with boundary zeros. Unknown chars are dropped (reference
    /// semantics); the dropped characters are returned so callers can surface them.
    pub fn phonemes_to_ids(&self, ps: &str) -> (Vec<i64>, Vec<char>) {
        let mut ids = vec![0i64];
        let mut dropped = vec![];
        for c in ps.chars() {
            match self.vocab.get(&c) {
                Some(&id) => ids.push(id),
                None => dropped.push(c),
            }
        }
        ids.push(0);
        (ids, dropped)
    }
}
