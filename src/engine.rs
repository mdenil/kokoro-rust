//! Resident synthesis engine: model loaded once, voices cached, one call per phoneme chunk.

use crate::model::{Kokoro, CONTEXT_LEN, STYLE_DIM};
use crate::torchpt;
use crate::vocoder::RngNoise;
use crate::weights::Weights;
use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const MAX_PHONEMES: usize = CONTEXT_LEN - 2;

pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(hex(&Sha256::digest(&bytes)))
}

pub fn sha256_bytes(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Voice pack [510, 256] (f32, row per phoneme-length bucket).
pub struct Voice {
    pub pack: Vec<f32>,
    pub rows: usize,
    pub sha256: String,
}

impl Voice {
    /// Reference rule: ref_s = pack[len(phonemes) - 1] (pipeline.py:232).
    pub fn ref_s(&self, n_phoneme_chars: usize) -> Result<&[f32]> {
        ensure!(n_phoneme_chars >= 1, "empty phoneme string");
        let r = n_phoneme_chars - 1;
        ensure!(r < self.rows, "phoneme length {n_phoneme_chars} exceeds voice pack rows {}", self.rows);
        Ok(&self.pack[r * 2 * STYLE_DIM..(r + 1) * 2 * STYLE_DIM])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
pub enum Device {
    Cpu,
    Cuda,
}

pub struct Engine {
    pub model: Kokoro,
    #[cfg(feature = "cuda")]
    pub gpu: Option<crate::gpu::GpuKokoro>,
    pub device: Device,
    pub model_dir: PathBuf,
    pub model_sha256: String,
    pub config_sha256: String,
    voices: HashMap<String, Voice>,
}

pub struct Synth {
    pub audio: Vec<f32>,
    pub pred_dur: Vec<i64>,
    pub dropped: Vec<char>,
}

impl Engine {
    /// `model_dir` is an HF snapshot dir containing config.json, kokoro-v1_0.pth, voices/.
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::load_on(model_dir, Device::Cpu, 0)
    }

    /// Load for a device. `cuda_ordinal` is the CUDA device index (after CUDA_VISIBLE_DEVICES).
    pub fn load_on(model_dir: &Path, device: Device, cuda_ordinal: usize) -> Result<Self> {
        let weights = model_dir.join("kokoro-v1_0.pth");
        let config = model_dir.join("config.json");
        let w = Weights::load_pth(&weights)?;
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config).with_context(|| format!("reading {}", config.display()))?)?;
        let model = Kokoro::from_weights(&w, &cfg)?;
        #[cfg(feature = "cuda")]
        let gpu = match device {
            Device::Cuda => Some(crate::gpu::GpuKokoro::new(&model, cuda_ordinal)?),
            Device::Cpu => None,
        };
        #[cfg(not(feature = "cuda"))]
        {
            let _ = cuda_ordinal;
            if device == Device::Cuda {
                bail!("this binary was built without CUDA support (cargo build --features cuda)");
            }
        }
        Ok(Self {
            model,
            #[cfg(feature = "cuda")]
            gpu,
            device,
            model_dir: model_dir.to_path_buf(),
            model_sha256: sha256_file(&weights)?,
            config_sha256: sha256_file(&config)?,
            voices: HashMap::new(),
        })
    }

    /// Voice by name (voices/<name>.pt), explicit .pt path, or comma-separated mean of several
    /// (reference: torch.mean(torch.stack(packs), dim=0), pipeline.py:166).
    pub fn voice(&mut self, spec: &str) -> Result<&Voice> {
        if !self.voices.contains_key(spec) {
            let parts: Vec<&str> = spec.split(',').collect();
            let mut packs = vec![];
            let mut hashes = vec![];
            for p in &parts {
                let path = if p.ends_with(".pt") { PathBuf::from(p) } else { self.model_dir.join("voices").join(format!("{p}.pt")) };
                let t = torchpt::load_voice_pack(&path)?;
                hashes.push(sha256_file(&path)?);
                packs.push(t);
            }
            let rows = packs[0].shape[0];
            ensure!(packs.iter().all(|p| p.shape == packs[0].shape), "voice packs have different shapes");
            let pack = if packs.len() == 1 {
                packs.pop().unwrap().data
            } else {
                let n = packs.len() as f32;
                (0..packs[0].data.len()).map(|i| packs.iter().map(|p| p.data[i]).sum::<f32>() / n).collect()
            };
            self.voices.insert(spec.to_string(), Voice { pack, rows, sha256: hashes.join(",") });
        }
        Ok(&self.voices[spec])
    }

    /// Synthesize one phoneme chunk. Oversize and empty inputs are errors, never truncated.
    pub fn synth_phonemes(&mut self, phonemes: &str, voice: &str, speed: f32, seed: u64) -> Result<Synth> {
        let n = phonemes.chars().count();
        if n == 0 {
            bail!("empty phoneme string");
        }
        if n > MAX_PHONEMES {
            bail!("oversize: {n} phoneme chars > {MAX_PHONEMES}");
        }
        let (ids, dropped) = self.model.phonemes_to_ids(phonemes);
        let ref_s = self.voice(voice)?.ref_s(n)?.to_vec();
        let mut noise = RngNoise::new(seed);
        #[cfg(feature = "cuda")]
        let out = match &self.gpu {
            Some(g) => g.forward_ids(&self.model, &ids, &ref_s, speed, &mut noise)?,
            None => self.model.forward_ids(&ids, &ref_s, speed, &mut noise)?,
        };
        #[cfg(not(feature = "cuda"))]
        let out = self.model.forward_ids(&ids, &ref_s, speed, &mut noise)?;
        Ok(Synth { audio: out.audio, pred_dur: out.pred_dur, dropped })
    }
}

/// splitmix64 for per-item seeds
pub fn mix_seed(seed: u64, index: u64) -> u64 {
    let mut z = seed ^ index.wrapping_mul(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
