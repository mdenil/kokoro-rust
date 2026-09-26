//! RIFF/WAVE writer. Default PCM_16 matches the production path (`soundfile.write(path, f32, sr)`
//! uses libsndfile's default WAV subtype PCM_16: sample * 0x7FFF, rounded to nearest-even).
//! Samples outside [-1, 1] are clamped and COUNTED (libsndfile without clipping would wrap).

use anyhow::Result;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
pub enum Format {
    Pcm16,
    Float32,
}

pub struct Encoded {
    pub bytes: Vec<u8>,
    pub clipped: usize,
}

pub fn encode(audio: &[f32], sample_rate: u32, fmt: Format) -> Encoded {
    let (bits, tag) = match fmt {
        Format::Pcm16 => (16u16, 1u16),
        Format::Float32 => (32u16, 3u16),
    };
    let data_len = audio.len() * (bits as usize / 8);
    let mut b = Vec::with_capacity(44 + data_len);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&tag.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&(sample_rate * bits as u32 / 8).to_le_bytes());
    b.extend_from_slice(&(bits / 8).to_le_bytes());
    b.extend_from_slice(&bits.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(data_len as u32).to_le_bytes());
    let mut clipped = 0;
    match fmt {
        Format::Pcm16 => {
            for &s in audio {
                let v = (s * 32767.0).round_ties_even();
                if !(-32768.0..=32767.0).contains(&v) {
                    clipped += 1;
                }
                b.extend_from_slice(&(v.clamp(-32768.0, 32767.0) as i16).to_le_bytes());
            }
        }
        Format::Float32 => {
            for &s in audio {
                b.extend_from_slice(&s.to_le_bytes());
            }
        }
    }
    Encoded { bytes: b, clipped }
}

/// Write atomically (temp file + rename) so a crash never leaves a truncated WAV behind.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("partial");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}
