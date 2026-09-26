//! KModel: phoneme ids + style vector -> 24 kHz waveform.
//! Stage functions are public so each can be proven against oracle seam tensors in isolation.

use crate::albert::Albert;
use crate::nn::{AdainResBlk1d, BiLstm, Conv1d, Linear};
use crate::ops;
use crate::vocoder::{Decoder, NoiseSource};
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
    embedding: Vec<f32>,
    cnn: Vec<(Conv1d, Vec<f32>, Vec<f32>)>,
    lstm: BiLstm,
    n_token: usize,
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

    /// ids -> [512, t]
    pub fn forward(&self, ids: &[i64]) -> Result<Vec<f32>> {
        let t = ids.len();
        let mut e = vec![0.0f32; t * HIDDEN];
        for (i, &id) in ids.iter().enumerate() {
            ensure!(id >= 0 && (id as usize) < self.n_token, "token id {id} out of range");
            e[i * HIDDEN..(i + 1) * HIDDEN].copy_from_slice(&self.embedding[id as usize * HIDDEN..(id as usize + 1) * HIDDEN]);
        }
        let mut x = ops::transpose(&e, t, HIDDEN);
        for (conv, g, b) in &self.cnn {
            let (y, _) = conv.forward(&x, t);
            let mut yt = ops::transpose(&y, HIDDEN, t);
            ops::layer_norm_rows(&mut yt, HIDDEN, Some(g), Some(b), 1e-5);
            ops::leaky_relu(&mut yt, 0.2);
            x = ops::transpose(&yt, t, HIDDEN);
        }
        let xt = ops::transpose(&x, HIDDEN, t);
        let h = self.lstm.forward(&xt, t);
        Ok(ops::transpose(&h, t, HIDDEN))
    }
}

pub struct Predictor {
    dur_lstms: Vec<BiLstm>,
    dur_norms: Vec<Linear>,
    lstm: BiLstm,
    duration_proj: Linear,
    shared: BiLstm,
    f0: Vec<AdainResBlk1d>,
    n: Vec<AdainResBlk1d>,
    f0_proj: Conv1d,
    n_proj: Conv1d,
}

fn cat_style(x: &[f32], t: usize, d: usize, s: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(t * (d + s.len()));
    for row in x.chunks_exact(d) {
        out.extend_from_slice(row);
        out.extend_from_slice(s);
    }
    out
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

    /// DurationEncoder: d_en [t, 512], s [128] -> d [t, 640]
    pub fn duration_encoder(&self, d_en: &[f32], t: usize, s: &[f32]) -> Vec<f32> {
        let mut x = cat_style(d_en, t, HIDDEN, s);
        for (lstm, fc) in self.dur_lstms.iter().zip(&self.dur_norms) {
            let mut h = lstm.forward(&x, t);
            let gb = fc.forward(s, 1);
            let (gamma, beta) = gb.split_at(HIDDEN);
            ops::layer_norm_rows(&mut h, HIDDEN, None, None, 1e-5);
            for row in h.chunks_exact_mut(HIDDEN) {
                for j in 0..HIDDEN {
                    row[j] = (1.0 + gamma[j]) * row[j] + beta[j];
                }
            }
            x = cat_style(&h, t, HIDDEN, s);
        }
        x
    }

    /// predictor.lstm: d [t, 640] -> [t, 512]
    pub fn dur_lstm(&self, d: &[f32], t: usize) -> Vec<f32> {
        self.lstm.forward(d, t)
    }

    /// duration_proj logits [t, 50]
    pub fn duration_logits(&self, x: &[f32], t: usize) -> Vec<f32> {
        self.duration_proj.forward(x, t)
    }

    /// F0Ntrain: en [N, 640] (frame-major), s [128] -> (F0 [2N], N [2N])
    pub fn f0n(&self, en: &[f32], nf: usize, s: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let x = self.shared.forward(en, nf);
        let x = ops::transpose(&x, nf, HIDDEN);
        let run = |blocks: &[AdainResBlk1d], proj: &Conv1d| {
            let mut h = x.clone();
            let mut t = nf;
            for b in blocks {
                let (y, t2) = b.forward(&h, t, s);
                h = y;
                t = t2;
            }
            proj.forward(&h, t).0
        };
        (run(&self.f0, &self.f0_proj), run(&self.n, &self.n_proj))
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

/// Frame -> token index map from durations.
pub fn alignment(pred_dur: &[i64]) -> Vec<usize> {
    let mut idx = vec![];
    for (tok, &d) in pred_dur.iter().enumerate() {
        for _ in 0..d {
            idx.push(tok);
        }
    }
    idx
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

    pub fn forward_ids(&self, ids: &[i64], ref_s: &[f32], speed: f32, noise: &mut dyn NoiseSource) -> Result<Output> {
        ensure!(ref_s.len() == 2 * STYLE_DIM, "ref_s must have 256 values");
        ensure!(ids.len() >= 2 && ids.len() <= CONTEXT_LEN, "input_ids length {} outside [2, {CONTEXT_LEN}]", ids.len());
        ensure!(speed.is_finite() && speed > 0.0, "speed must be positive and finite");
        let t = ids.len();
        let s_dec = &ref_s[..STYLE_DIM];
        let s = &ref_s[STYLE_DIM..];

        let bert = self.albert.forward(ids)?;
        let d_en = self.bert_encoder.forward(&bert, t);
        let d = self.predictor.duration_encoder(&d_en, t, s);
        let x = self.predictor.dur_lstm(&d, t);
        let logits = self.predictor.duration_logits(&x, t);
        let pred_dur = durations_from_logits(&logits, t, speed);
        let aln = alignment(&pred_dur);
        let nf = aln.len();

        let dd = HIDDEN + STYLE_DIM;
        let mut en = vec![0.0f32; nf * dd];
        for (f, &tok) in aln.iter().enumerate() {
            en[f * dd..(f + 1) * dd].copy_from_slice(&d[tok * dd..(tok + 1) * dd]);
        }
        let (f0, n) = self.predictor.f0n(&en, nf, s);

        let t_en = self.text_encoder.forward(ids)?;
        let mut asr = vec![0.0f32; HIDDEN * nf];
        for c in 0..HIDDEN {
            for (f, &tok) in aln.iter().enumerate() {
                asr[c * nf + f] = t_en[c * t + tok];
            }
        }
        let audio = self.decoder.forward(&asr, nf, &f0, &n, s_dec, noise)?;
        Ok(Output { audio, pred_dur })
    }
}
