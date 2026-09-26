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

/// SHA-256 of a file, streamed. Uses ring's assembly implementation (AVX2 on this host; the
/// pure-Rust sha2 crate only accelerates via SHA-NI, which e.g. Broadwell lacks: ~2.6x slower).
pub fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }
        ctx.update(&buf[..n]);
    }
    Ok(hex(ctx.finish().as_ref()))
}

pub fn sha256_bytes(b: &[u8]) -> String {
    hex(ring::digest::digest(&ring::digest::SHA256, b).as_ref())
}

/// Reference implementation (sha2 crate) kept for the equivalence test.
#[doc(hidden)]
pub fn sha256_bytes_sha2(b: &[u8]) -> String {
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
        // hash the checkpoint on a helper thread while it is parsed/uploaded (identical result)
        let hash = std::thread::spawn({
            let weights = weights.clone();
            move || sha256_file(&weights)
        });
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
            model_sha256: hash.join().map_err(|_| anyhow::anyhow!("weight hashing thread panicked"))??,
            config_sha256: sha256_file(&config)?,
            voices: HashMap::new(),
        })
    }

    /// Numerical identity of this engine instance (part of output provenance / resume keys).
    pub fn identity(&self) -> String {
        match self.device {
            Device::Cpu => "cpu f32".into(),
            #[cfg(feature = "cuda")]
            Device::Cuda => format!("cuda f32 kernels={}", crate::gpu::KERNEL_ROUNDING),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda => "cuda (unavailable)".into(),
        }
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

/// Batching budget for `synth_batch` (length-bucketed microbatches).
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct BatchPolicy {
    /// Max total phoneme characters per microbatch (proxy for frames / VRAM).
    pub max_phonemes: usize,
    /// Max items per microbatch.
    pub max_items: usize,
}

impl Default for BatchPolicy {
    fn default() -> Self {
        Self { max_phonemes: 4000, max_items: 64 }
    }
}

/// Plan microbatches: sort by phoneme length (length bucketing), then greedily fill each batch up
/// to the policy budget. Returns batches of indices into `lens`.
pub fn plan_batches(lens: &[usize], p: BatchPolicy) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by_key(|&i| (lens[i], i));
    let mut out: Vec<Vec<usize>> = vec![];
    let mut cur = vec![];
    let mut sum = 0;
    for i in order {
        if !cur.is_empty() && (cur.len() >= p.max_items || sum + lens[i] > p.max_phonemes) {
            out.push(std::mem::take(&mut cur));
            sum = 0;
        }
        sum += lens[i];
        cur.push(i);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

impl Engine {
    /// Synthesize many phoneme chunks; results are returned in input order. On the CUDA device the
    /// chunks run as length-bucketed batches (per-item semantics unchanged; each item keeps its own
    /// seed-keyed noise stream); a failing batch is split in half down to single items (OOM safety).
    pub fn synth_batch(&mut self, reqs: &[(String, u64)], voice: &str, speed: f32, policy: BatchPolicy) -> Vec<Result<Synth>> {
        #[cfg(feature = "cuda")]
        if self.gpu.is_some() {
            return self.synth_batch_gpu(reqs, voice, speed, policy);
        }
        let _ = policy;
        reqs.iter().map(|(p, seed)| self.synth_phonemes(p, voice, speed, *seed)).collect()
    }

    #[cfg(feature = "cuda")]
    fn synth_batch_gpu(&mut self, reqs: &[(String, u64)], voice: &str, speed: f32, policy: BatchPolicy) -> Vec<Result<Synth>> {
        let mut results: Vec<Option<Result<Synth>>> = (0..reqs.len()).map(|_| None).collect();
        // validate + prepare per item (identical rules to synth_phonemes)
        let mut prepared: Vec<(usize, Vec<i64>, Vec<char>, Vec<f32>)> = vec![];
        for (i, (p, _)) in reqs.iter().enumerate() {
            let n = p.chars().count();
            let r: Result<()> = (|| {
                if n == 0 {
                    bail!("empty phoneme string");
                }
                if n > MAX_PHONEMES {
                    bail!("oversize: {n} phoneme chars > {MAX_PHONEMES}");
                }
                let (ids, dropped) = self.model.phonemes_to_ids(p);
                let ref_s = self.voice(voice)?.ref_s(n)?.to_vec();
                prepared.push((i, ids, dropped, ref_s));
                Ok(())
            })();
            if let Err(e) = r {
                results[i] = Some(Err(e));
            }
        }
        let lens: Vec<usize> = prepared.iter().map(|x| x.1.len()).collect();
        let mut stack: Vec<Vec<usize>> = plan_batches(&lens, policy).into_iter().rev().collect();
        while let Some(batch) = stack.pop() {
            let items: Vec<crate::gpu::BatchItem> = batch
                .iter()
                .map(|&k| {
                    let (i, ids, _, ref_s) = &prepared[k];
                    crate::gpu::BatchItem { ids, ref_s, speed, noise: crate::gpu::ItemNoise::Counter(reqs[*i].1) }
                })
                .collect();
            match self.gpu.as_ref().unwrap().forward_batch(&self.model, &items) {
                Ok(outs) => {
                    for (&k, o) in batch.iter().zip(outs) {
                        let (i, _, dropped, _) = &prepared[k];
                        results[*i] = Some(Ok(Synth { audio: o.audio, pred_dur: o.pred_dur, dropped: dropped.clone() }));
                    }
                }
                Err(e) if batch.len() > 1 => {
                    let mid = batch.len() / 2;
                    eprintln!("batch of {} failed ({e:#}); splitting", batch.len());
                    stack.push(batch[mid..].to_vec());
                    stack.push(batch[..mid].to_vec());
                }
                Err(e) => {
                    results[prepared[batch[0]].0] = Some(Err(e));
                }
            }
        }
        results.into_iter().map(|r| r.unwrap_or_else(|| Err(anyhow::anyhow!("internal: missing batch result")))).collect()
    }
}
