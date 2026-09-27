//! RIFF/WAVE writer. Default PCM_16 matches the production path (`soundfile.write(path, f32, sr)`
//! uses libsndfile's default WAV subtype PCM_16: sample * 0x7FFF, rounded to nearest-even).
//! Samples outside [-1, 1] are clamped and COUNTED (libsndfile without clipping would wrap).

use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
pub enum Format {
    Pcm16,
    Float32,
}

pub struct Encoded {
    pub bytes: Vec<u8>,
    pub clipped: usize,
}

/// Size of the canonical header written here (RIFF, 16-byte fmt chunk, data chunk header).
pub const HEADER_LEN: usize = 44;
/// Largest data chunk a RIFF/WAVE file can describe (its RIFF size field is 36 + data, in 32 bits).
pub const MAX_DATA_BYTES: u64 = u32::MAX as u64 - 36;

fn bits_and_tag(fmt: Format) -> (u16, u16) {
    match fmt {
        Format::Pcm16 => (16, 1),
        Format::Float32 => (32, 3),
    }
}

/// Refuses data sizes a RIFF/WAVE file cannot represent (never wraps the 32-bit size fields).
pub fn check_riff_size(data_bytes: u64) -> Result<()> {
    ensure!(
        data_bytes <= MAX_DATA_BYTES,
        "the audio ({data_bytes} bytes) exceeds the 4 GiB limit of a WAV file; split the input into smaller files"
    );
    Ok(())
}

/// The canonical 44-byte header for `data_bytes` of mono audio in `fmt`.
pub fn header(fmt: Format, sample_rate: u32, data_bytes: u64) -> Result<[u8; HEADER_LEN]> {
    check_riff_size(data_bytes)?;
    let (bits, tag) = bits_and_tag(fmt);
    let mut b = [0u8; HEADER_LEN];
    let mut put = |at: usize, v: &[u8]| b[at..at + v.len()].copy_from_slice(v);
    put(0, b"RIFF");
    put(4, &((36 + data_bytes) as u32).to_le_bytes());
    put(8, b"WAVEfmt ");
    put(16, &16u32.to_le_bytes());
    put(20, &tag.to_le_bytes());
    put(22, &1u16.to_le_bytes());
    put(24, &sample_rate.to_le_bytes());
    put(28, &(sample_rate * bits as u32 / 8).to_le_bytes());
    put(32, &(bits / 8).to_le_bytes());
    put(34, &bits.to_le_bytes());
    put(36, b"data");
    put(40, &(data_bytes as u32).to_le_bytes());
    Ok(b)
}

pub fn encode(audio: &[f32], sample_rate: u32, fmt: Format) -> Encoded {
    let (bits, _) = bits_and_tag(fmt);
    let data_len = audio.len() * (bits as usize / 8);
    let mut b = Vec::with_capacity(HEADER_LEN + data_len);
    // one line is far below the WAV size limit (at most 510 phonemes)
    b.extend_from_slice(&header(fmt, sample_rate, data_len as u64).expect("line audio fits a WAV file"));
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
    write_atomic_opt(path, bytes, true)
}

/// Write to `<path>.partial` then rename (readers never see a partial file); `sync` = fsync first.
pub fn write_atomic_opt(path: &Path, bytes: &[u8], sync: bool) -> Result<()> {
    let tmp = path.with_extension("partial");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        if sync {
            f.sync_all()?;
        }
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Data size of a WAV file written by `encode` (the canonical 44-byte header), after checking that
/// its format, channel count and sample rate are the expected ones and that its size is consistent.
pub fn read_canonical(path: &Path, fmt: Format, sample_rate: u32) -> Result<u64> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = [0u8; HEADER_LEN];
    f.read_exact(&mut h).with_context(|| format!("{}: shorter than a WAV header", path.display()))?;
    let data = u32::from_le_bytes(h[40..44].try_into().unwrap()) as u64;
    let want = header(fmt, sample_rate, data)?;
    ensure!(h == want, "{}: not a mono {sample_rate} Hz {fmt:?} WAV as written by kokoro", path.display());
    let len = f.metadata()?.len();
    ensure!(len == HEADER_LEN as u64 + data, "{}: size {len} does not match its header ({} bytes of audio)", path.display(), data);
    Ok(data)
}

pub struct Assembled {
    pub sha256: String,
    pub data_bytes: u64,
    pub samples: u64,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Concatenates the audio of `parts` (WAV files written by `encode`, each with its expected sha256)
/// into one WAV at `dest`: one header, then every part's samples exactly, in order. Streams through
/// a temporary file next to `dest` and renames it into place only when everything checked out, so
/// `dest` is either the complete new file or left as it was.
pub fn assemble(parts: &[(PathBuf, String)], fmt: Format, sample_rate: u32, dest: &Path, sync: bool) -> Result<Assembled> {
    let mut total = 0u64;
    for (p, _) in parts {
        total += read_canonical(p, fmt, sample_rate)?;
    }
    let head = header(fmt, sample_rate, total)?;
    let name = dest.file_name().context("output path has no file name")?.to_string_lossy().into_owned();
    let tmp = dest.with_file_name(format!(".{name}.partial"));
    let result = (|| -> Result<Assembled> {
        let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?);
        let mut whole = Sha256::new();
        out.write_all(&head)?;
        whole.update(head);
        let mut buf = vec![0u8; 1 << 20];
        for (p, expected) in parts {
            let mut f = std::fs::File::open(p).with_context(|| format!("opening {}", p.display()))?;
            let mut part = Sha256::new();
            let mut h = [0u8; HEADER_LEN];
            f.read_exact(&mut h)?;
            part.update(h);
            loop {
                let n = f.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                part.update(&buf[..n]);
                whole.update(&buf[..n]);
                out.write_all(&buf[..n])?;
            }
            let got = hex(&part.finalize());
            if &got != expected {
                bail!("{} changed since it was synthesized (sha256 {got}, expected {expected})", p.display());
            }
        }
        let f = out.into_inner().map_err(|e| e.into_error())?;
        if sync {
            f.sync_all()?;
        }
        let (bits, _) = bits_and_tag(fmt);
        Ok(Assembled { sha256: hex(&whole.finalize()), data_bytes: total, samples: total / (bits as u64 / 8) })
    })();
    match result {
        Ok(a) => {
            std::fs::rename(&tmp, dest).with_context(|| format!("moving the finished WAV to {}", dest.display()))?;
            Ok(a)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kokoro-wav-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn part(d: &Path, i: usize, audio: &[f32], fmt: Format) -> (PathBuf, String) {
        let e = encode(audio, 24000, fmt);
        let p = d.join(format!("p{i}.wav"));
        std::fs::write(&p, &e.bytes).unwrap();
        (p, crate::engine::sha256_bytes(&e.bytes))
    }

    fn lines() -> Vec<Vec<f32>> {
        vec![
            (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect(),
            vec![0.25, -1.5, 1.5, 0.0, -0.3333], // includes clipped samples
            (0..777).map(|i| ((i % 17) as f32 - 8.0) / 9.0).collect(),
        ]
    }

    #[test]
    fn assembly_equals_encoding_the_concatenated_samples() {
        for fmt in [Format::Pcm16, Format::Float32] {
            let d = tmpdir(&format!("exact-{fmt:?}"));
            let ls = lines();
            let parts: Vec<_> = ls.iter().enumerate().map(|(i, a)| part(&d, i, a, fmt)).collect();
            let dest = d.join("book.wav");
            let a = assemble(&parts, fmt, 24000, &dest, false).unwrap();
            let all: Vec<f32> = ls.concat();
            let want = encode(&all, 24000, fmt).bytes;
            let got = std::fs::read(&dest).unwrap();
            assert_eq!(got, want, "{fmt:?}: assembled file != encode(concatenated samples)");
            assert_eq!(a.samples, all.len() as u64);
            assert_eq!(a.sha256, crate::engine::sha256_bytes(&want));
            assert_eq!(read_canonical(&dest, fmt, 24000).unwrap(), a.data_bytes);
            assert!(!d.join(".book.wav.partial").exists());
        }
    }

    #[test]
    fn riff_size_limit_is_enforced_not_wrapped() {
        assert!(check_riff_size(MAX_DATA_BYTES).is_ok());
        assert!(check_riff_size(MAX_DATA_BYTES + 1).is_err());
        assert!(check_riff_size(u32::MAX as u64 + 10).is_err());
        let h = header(Format::Pcm16, 24000, MAX_DATA_BYTES).unwrap();
        assert_eq!(u32::from_le_bytes(h[4..8].try_into().unwrap()), u32::MAX);
        assert!(header(Format::Float32, 24000, MAX_DATA_BYTES + 2).is_err());
    }

    #[test]
    fn corrupt_or_mismatched_parts_leave_the_destination_untouched() {
        let d = tmpdir("bad");
        let ls = lines();
        let parts: Vec<_> = ls.iter().enumerate().map(|(i, a)| part(&d, i, a, Format::Pcm16)).collect();
        let dest = d.join("book.wav");
        std::fs::write(&dest, b"previous good output").unwrap();
        let unchanged = |what: &str| {
            assert_eq!(std::fs::read(&dest).unwrap(), b"previous good output", "{what}: destination changed");
            assert!(!d.join(".book.wav.partial").exists(), "{what}: temporary file left");
        };
        // a sample changed after synthesis (same size, same header): caught by the hash
        let mut b = std::fs::read(&parts[1].0).unwrap();
        b[HEADER_LEN] ^= 1;
        std::fs::write(&parts[1].0, &b).unwrap();
        assert!(assemble(&parts, Format::Pcm16, 24000, &dest, false).is_err());
        unchanged("changed sample");
        // truncated part
        let parts: Vec<_> = ls.iter().enumerate().map(|(i, a)| part(&d, i, a, Format::Pcm16)).collect();
        let b = std::fs::read(&parts[2].0).unwrap();
        std::fs::write(&parts[2].0, &b[..b.len() - 3]).unwrap();
        assert!(assemble(&parts, Format::Pcm16, 24000, &dest, false).is_err());
        unchanged("truncated part");
        // missing part
        let parts: Vec<_> = ls.iter().enumerate().map(|(i, a)| part(&d, i, a, Format::Pcm16)).collect();
        std::fs::remove_file(&parts[0].0).unwrap();
        assert!(assemble(&parts, Format::Pcm16, 24000, &dest, false).is_err());
        unchanged("missing part");
        // wrong format / sample rate
        let parts: Vec<_> = ls.iter().enumerate().map(|(i, a)| part(&d, i, a, Format::Pcm16)).collect();
        assert!(assemble(&parts, Format::Float32, 24000, &dest, false).is_err());
        assert!(assemble(&parts, Format::Pcm16, 22050, &dest, false).is_err());
        unchanged("format mismatch");
    }
}
