//! `kokoro` command line: `synth` (narration lines -> WAV + JSON sidecars) and `bench`.

use crate::engine::{mix_seed, sha256_bytes, sha256_file, Engine, MAX_PHONEMES};
use crate::model::SAMPLE_RATE;
use crate::wav::{self, Format};
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

pub const ENGINE_VERSION: &str = concat!("kokoro-rust ", env!("CARGO_PKG_VERSION"));

#[derive(Parser)]
#[command(name = "kokoro", version, about = "Native Rust inference for hexgrad/Kokoro-82M")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
enum InputFormat {
    /// One utterance of narration text per line (needs a frontend).
    Text,
    /// One phoneme string (Kokoro vocabulary, <= 510 chars) per line; fully native.
    Phonemes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
enum Frontend {
    /// No text frontend: only --input-format phonemes is accepted.
    None,
    /// DEV-ONLY bridge to the pinned Python misaki G2P (oracle/frontend_bridge.py).
    PythonBridge,
}

#[derive(clap::Args)]
struct Common {
    /// HF snapshot dir with config.json, kokoro-v1_0.pth and voices/ [env: KOKORO_MODEL_DIR]
    #[arg(long, env = "KOKORO_MODEL_DIR")]
    model_dir: PathBuf,
    /// Voice name (voices/<name>.pt), a .pt path, or comma-separated names to average.
    #[arg(long, default_value = "af_heart")]
    voice: String,
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    /// Worker threads for the math kernels (0 = library default).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, value_enum, default_value = "cpu")]
    device: crate::engine::Device,
    /// CUDA device index (after CUDA_VISIBLE_DEVICES).
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
}

#[derive(Subcommand)]
enum Cmd {
    /// One WAV + JSON sidecar per input line (<stem>_<1-based line>.wav) plus <stem>.manifest.json.
    Synth {
        #[command(flatten)]
        common: Common,
        /// UTF-8 input file, one utterance per line ("-" for stdin).
        #[arg(long)]
        input: PathBuf,
        #[arg(long, value_enum, default_value = "text")]
        input_format: InputFormat,
        #[arg(long, value_enum, default_value = "none")]
        frontend: Frontend,
        /// Language code for the frontend bridge (a = American English, b = British English).
        #[arg(long, default_value = "a")]
        lang: String,
        #[arg(long, env = "KOKORO_BRIDGE_PYTHON")]
        bridge_python: Option<PathBuf>,
        #[arg(long)]
        bridge_script: Option<PathBuf>,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, value_enum, default_value = "pcm16")]
        format: Format,
        /// Base seed for the excitation noise (per item: splitmix(seed, index)).
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Also encode each WAV with ffmpeg to this extension (e.g. flac, mp3, opus).
        #[arg(long)]
        encode: Option<String>,
        /// Re-synthesize even if a matching completed output exists.
        #[arg(long)]
        force: bool,
        /// Blank-line policy (canonical audiobook files contain none).
        #[arg(long, value_enum, default_value = "error")]
        blank_lines: BlankLines,
    },
    /// Time pure inference over pre-computed phoneme chunks (JSONL with a "phonemes" field).
    Bench {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        chunks: PathBuf,
        #[arg(long, default_value_t = 5)]
        reps: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn set_threads(n: usize) {
    if n > 0 {
        // matrixmultiply reads this on first use; rayon is configured explicitly.
        std::env::set_var("MATMUL_NUM_THREADS", n.to_string());
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }
}

struct Bridge {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    ident: String,
}

impl Bridge {
    fn spawn(python: &Path, script: &Path, lang: &str) -> Result<Self> {
        let mut child = Command::new(python)
            .arg(script)
            .arg(lang)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("spawning frontend bridge {} {}", python.display(), script.display()))?;
        let stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        stdout.read_line(&mut line)?;
        let v: serde_json::Value = serde_json::from_str(&line).context("frontend bridge did not start")?;
        let ident = v["frontend"].as_str().unwrap_or("unknown").to_string();
        Ok(Self { child, stdin, stdout, ident: format!("{ident} [python bridge, DEV-ONLY]") })
    }

    fn phonemize(&mut self, text: &str) -> Result<Vec<String>> {
        writeln!(self.stdin, "{}", serde_json::json!({ "text": text }))?;
        self.stdin.flush()?;
        let mut line = String::new();
        self.stdout.read_line(&mut line)?;
        let v: serde_json::Value = serde_json::from_str(&line).context("bad bridge response")?;
        if let Some(e) = v["error"].as_str() {
            bail!("frontend error: {e}");
        }
        Ok(v["chunks"].as_array().context("bridge: no chunks")?.iter().map(|c| c["phonemes"].as_str().unwrap_or("").to_string()).collect())
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[derive(Serialize, serde::Deserialize, Clone, PartialEq)]
struct Config {
    engine: String,
    model_sha256: String,
    config_sha256: String,
    voice: String,
    voice_sha256: String,
    speed: f32,
    seed: u64,
    format: String,
    input_format: String,
    frontend: String,
    sample_rate: usize,
}

#[derive(Serialize, serde::Deserialize)]
struct Sidecar {
    /// 1-based line number in the input file (the line's identity)
    line: usize,
    input_file: String,
    input_sha256: String,
    text: String,
    text_sha256: String,
    status: String, // ok | blank | oversize | invalid | error
    error: Option<String>,
    phonemes: Vec<String>,
    dropped_phoneme_chars: Vec<String>,
    config: Config,
    wav: Option<String>,
    audio_sha256: Option<String>,
    samples: usize,
    duration_s: f64,
    clipped_samples: usize,
    encoded: Option<String>,
    encoded_sha256: Option<String>,
    elapsed_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
enum BlankLines {
    /// A blank (empty/whitespace-only) line is a per-line failure (canonical files have none).
    Error,
    /// A blank line is recorded with status "blank" and produces no audio; not a failure.
    Skip,
}

/// Parsed input file. Line identity = 1-based line number; nothing is merged, reordered or dropped.
struct InputFile {
    display: String,
    stem: String,
    sha256: String,
    lines: Vec<String>,
    bom_stripped: bool,
    crlf_lines: usize,
}

/// Reads the input as bytes. Invalid UTF-8 is a job-level error naming the exact 1-based line and
/// byte offset (nothing is synthesized). A leading UTF-8 BOM and per-line trailing CR are removed
/// (recorded in the manifest). A final newline terminates the last line (no phantom empty line).
fn read_input(input: &Path) -> Result<InputFile> {
    let bytes = if input == Path::new("-") {
        let mut b = vec![];
        std::io::Read::read_to_end(&mut std::io::stdin(), &mut b)?;
        b
    } else {
        std::fs::read(input).with_context(|| format!("reading {}", input.display()))?
    };
    let sha256 = sha256_bytes(&bytes);
    let (body, bom_stripped) = match bytes.strip_prefix(b"\xEF\xBB\xBF") {
        Some(rest) => (rest, true),
        None => (&bytes[..], false),
    };
    let text = match std::str::from_utf8(body) {
        Ok(t) => t,
        Err(e) => {
            let off = e.valid_up_to();
            let line = body[..off].iter().filter(|&&b| b == b'\n').count() + 1;
            let col = off - body[..off].iter().rposition(|&b| b == b'\n').map(|p| p + 1).unwrap_or(0);
            bail!("input is not valid UTF-8: line {line}, byte {} of the line (file byte {}); nothing synthesized", col + 1, off + bom_stripped as usize * 3);
        }
    };
    let mut crlf_lines = 0;
    let mut lines: Vec<String> = text
        .split('\n')
        .map(|l| match l.strip_suffix('\r') {
            Some(x) => {
                crlf_lines += 1;
                x.to_string()
            }
            None => l.to_string(),
        })
        .collect();
    if lines.last().map(|l| l.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    let stem = if input == Path::new("-") {
        "stdin".to_string()
    } else {
        input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "input".into())
    };
    Ok(InputFile { display: input.display().to_string(), stem, sha256, lines, bom_stripped, crlf_lines })
}

/// Control characters other than TAB make a line malformed (explicit per-line failure).
fn control_char(text: &str) -> Option<(usize, char)> {
    text.chars().enumerate().find(|(_, c)| c.is_control() && *c != '\t').map(|(i, c)| (i + 1, c))
}

/// Outputs are keyed by line identity + exact text + config (not the whole-file hash, so editing
/// one line only invalidates that line).
fn resume_ok(json_path: &Path, wav_path: &Path, line: usize, text_sha: &str, cfg: &Config) -> bool {
    let Ok(s) = std::fs::read_to_string(json_path) else { return false };
    let Ok(sc) = serde_json::from_str::<Sidecar>(&s) else { return false };
    if sc.line != line || sc.text_sha256 != text_sha || sc.config != *cfg {
        return false;
    }
    match sc.status.as_str() {
        "ok" => sc.audio_sha256.as_deref().map(|h| sha256_file(wav_path).map(|x| x == h).unwrap_or(false)).unwrap_or(false),
        _ => false, // blank/failed lines are always re-evaluated (cheap, and never counted as done output)
    }
}

/// Exit status: 0 every line done, 1 some line failed/incomplete, 2 job-level error.
#[allow(clippy::too_many_arguments)]
fn synth(
    common: Common,
    input: PathBuf,
    input_format: InputFormat,
    frontend: Frontend,
    lang: String,
    bridge_python: Option<PathBuf>,
    bridge_script: Option<PathBuf>,
    out_dir: PathBuf,
    format: Format,
    seed: u64,
    encode: Option<String>,
    force: bool,
    blank_lines: BlankLines,
) -> Result<i32> {
    if input_format == InputFormat::Text && frontend == Frontend::None {
        bail!("--input-format text needs a frontend: no native G2P yet. Use --input-format phonemes, or the DEV-ONLY --frontend python-bridge");
    }
    set_threads(common.threads);
    let inp = read_input(&input)?;
    std::fs::create_dir_all(&out_dir)?;
    let t_load = Instant::now();
    let mut engine = Engine::load_on(&common.model_dir, common.device, common.cuda_device)?;
    let voice_sha = engine.voice(&common.voice)?.sha256.clone();
    let load_s = t_load.elapsed().as_secs_f64();
    eprintln!("loaded model + voice in {load_s:.2}s");
    let mut bridge = match (input_format, frontend) {
        (InputFormat::Text, Frontend::PythonBridge) => {
            let py = bridge_python.context("--bridge-python (or KOKORO_BRIDGE_PYTHON) required for the python bridge")?;
            let script = bridge_script.unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("oracle/frontend_bridge.py"));
            Some(Bridge::spawn(&py, &script, &lang)?)
        }
        _ => None,
    };
    let cfg = Config {
        engine: ENGINE_VERSION.into(),
        model_sha256: engine.model_sha256.clone(),
        config_sha256: engine.config_sha256.clone(),
        voice: common.voice.clone(),
        voice_sha256: voice_sha,
        speed: common.speed,
        seed,
        format: format!("{format:?}"),
        input_format: format!("{input_format:?}"),
        frontend: bridge.as_ref().map(|b| b.ident.clone()).unwrap_or_else(|| "none (phoneme input)".into()),
        sample_rate: SAMPLE_RATE,
    };
    let width = inp.lines.len().max(1).to_string().len().max(5);
    let name = |line: usize, ext: &str| format!("{}_{line:0width$}.{ext}", inp.stem);
    let (mut n_ok, mut n_skip, mut n_bad, mut audio_total) = (0usize, 0usize, 0usize, 0.0f64);
    let mut manifest_lines = Vec::with_capacity(inp.lines.len());
    let t_all = Instant::now();
    for (i, text) in inp.lines.iter().enumerate() {
        let line = i + 1;
        let (wav_path, json_path) = (out_dir.join(name(line, "wav")), out_dir.join(name(line, "json")));
        let text_sha = sha256_bytes(text.as_bytes());
        if !force && resume_ok(&json_path, &wav_path, line, &text_sha, &cfg) {
            n_skip += 1;
            let sc: Sidecar = serde_json::from_str(&std::fs::read_to_string(&json_path)?)?;
            audio_total += sc.duration_s;
            manifest_lines.push(serde_json::json!({"line": line, "status": sc.status, "resumed": true, "wav": sc.wav, "sidecar": name(line, "json"),
                "text_sha256": sc.text_sha256, "audio_sha256": sc.audio_sha256, "duration_s": sc.duration_s}));
            continue;
        }
        let t0 = Instant::now();
        let mut sc = Sidecar {
            line,
            input_file: inp.display.clone(),
            input_sha256: inp.sha256.clone(),
            text: text.clone(),
            text_sha256: text_sha,
            status: "ok".into(),
            error: None,
            phonemes: vec![],
            dropped_phoneme_chars: vec![],
            config: cfg.clone(),
            wav: None,
            audio_sha256: None,
            samples: 0,
            duration_s: 0.0,
            clipped_samples: 0,
            encoded: None,
            encoded_sha256: None,
            elapsed_s: 0.0,
        };
        let result: Result<Option<Vec<f32>>> = (|| {
            if text.trim().is_empty() {
                sc.status = "blank".into();
                if blank_lines == BlankLines::Error {
                    bail!("blank line (canonical input has none; use --blank-lines skip to allow)");
                }
                return Ok(None);
            }
            if let Some((col, c)) = control_char(text) {
                sc.status = "invalid".into();
                bail!("control character U+{:04X} at character {col}", c as u32);
            }
            let chunks = match &mut bridge {
                Some(b) => b.phonemize(text)?,
                None => vec![text.clone()],
            };
            sc.phonemes = chunks.clone();
            if chunks.is_empty() {
                sc.status = "error".into();
                bail!("frontend produced no phonemes for a non-blank line");
            }
            if let Some(c) = chunks.iter().find(|c| c.chars().count() > MAX_PHONEMES) {
                sc.status = "oversize".into();
                bail!("chunk of {} phoneme chars exceeds {MAX_PHONEMES}; refusing to truncate", c.chars().count());
            }
            let mut audio = vec![];
            for (k, c) in chunks.iter().enumerate() {
                let s = engine.synth_phonemes(c, &common.voice, common.speed, mix_seed(seed, (line as u64) << 16 | k as u64))?;
                sc.dropped_phoneme_chars.extend(s.dropped.iter().map(|ch| ch.to_string()));
                audio.extend_from_slice(&s.audio);
            }
            Ok(Some(audio))
        })();
        match result {
            Ok(None) => {
                let _ = std::fs::remove_file(&wav_path);
            }
            Ok(Some(audio)) => {
                let enc = wav::encode(&audio, SAMPLE_RATE as u32, format);
                wav::write_atomic(&wav_path, &enc.bytes)?;
                sc.wav = Some(name(line, "wav"));
                sc.audio_sha256 = Some(sha256_bytes(&enc.bytes));
                sc.samples = audio.len();
                sc.duration_s = audio.len() as f64 / SAMPLE_RATE as f64;
                sc.clipped_samples = enc.clipped;
                audio_total += sc.duration_s;
                if let Some(ext) = &encode {
                    let dst = out_dir.join(name(line, ext));
                    let st = Command::new("ffmpeg").args(["-nostdin", "-y", "-loglevel", "error", "-i"]).arg(&wav_path).arg(&dst).status();
                    match st {
                        Ok(s) if s.success() => {
                            sc.encoded_sha256 = Some(sha256_file(&dst)?);
                            sc.encoded = Some(name(line, ext));
                        }
                        other => {
                            sc.status = "error".into();
                            sc.error = Some(format!("ffmpeg encode failed: {other:?}"));
                        }
                    }
                }
            }
            Err(e) => {
                if sc.status == "ok" {
                    sc.status = "error".into();
                }
                sc.error = Some(format!("{e:#}"));
                let _ = std::fs::remove_file(&wav_path);
            }
        }
        sc.elapsed_s = t0.elapsed().as_secs_f64();
        let done = sc.status == "ok" || (sc.status == "blank" && blank_lines == BlankLines::Skip);
        if done {
            n_ok += 1;
        } else {
            n_bad += 1;
            eprintln!("line {line}: {} — {}", sc.status, sc.error.as_deref().unwrap_or(""));
        }
        wav::write_atomic(&json_path, serde_json::to_string_pretty(&sc)?.as_bytes())?;
        manifest_lines.push(serde_json::json!({"line": line, "status": sc.status, "resumed": false, "wav": sc.wav, "sidecar": name(line, "json"),
            "text_sha256": sc.text_sha256, "audio_sha256": sc.audio_sha256, "duration_s": sc.duration_s, "error": sc.error}));
    }
    let wall = t_all.elapsed().as_secs_f64();
    let complete = n_bad == 0;
    let manifest = serde_json::json!({
        "input_file": inp.display, "input_sha256": inp.sha256, "input_lines": inp.lines.len(),
        "bom_stripped": inp.bom_stripped, "crlf_lines": inp.crlf_lines, "blank_lines_policy": format!("{blank_lines:?}"),
        "naming": format!("{}_<1-based line, width {width}>.wav / .json", inp.stem),
        "config": cfg, "complete": complete, "counts": {"done": n_ok, "resumed": n_skip, "failed": n_bad},
        "audio_s": audio_total, "load_s": load_s, "synth_wall_s": wall, "lines": manifest_lines,
    });
    wav::write_atomic(&out_dir.join(format!("{}.manifest.json", inp.stem)), serde_json::to_string_pretty(&manifest)?.as_bytes())?;
    eprintln!(
        "{} lines: {n_ok} done, {n_skip} resumed/skipped, {n_bad} failed; {audio_total:.1}s audio in {wall:.2}s (RTF {:.4})",
        inp.lines.len(),
        if audio_total > 0.0 { wall / audio_total } else { 0.0 }
    );
    if !complete {
        eprintln!("INCOMPLETE: {n_bad} line(s) failed; see {}.manifest.json and sidecars", inp.stem);
        return Ok(1);
    }
    Ok(0)
}

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn bench(common: Common, chunks: PathBuf, reps: usize, out: Option<PathBuf>) -> Result<()> {
    set_threads(common.threads);
    let t0 = Instant::now();
    let mut engine = Engine::load_on(&common.model_dir, common.device, common.cuda_device)?;
    engine.voice(&common.voice)?;
    let load_s = t0.elapsed().as_secs_f64();
    let items: Vec<String> = std::fs::read_to_string(&chunks)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<serde_json::Value>(l).map(|v| v["phonemes"].as_str().unwrap_or("").to_string()))
        .collect::<std::result::Result<_, _>>()?;
    // warmup: one full uncounted pass
    let mut audio_s = vec![0.0f64; items.len()];
    for (i, p) in items.iter().enumerate() {
        audio_s[i] = engine.synth_phonemes(p, &common.voice, common.speed, i as u64)?.audio.len() as f64 / SAMPLE_RATE as f64;
    }
    let mut per = vec![vec![]; items.len()];
    let mut totals = vec![];
    for _ in 0..reps {
        let tr = Instant::now();
        for (i, p) in items.iter().enumerate() {
            let t = Instant::now();
            let s = engine.synth_phonemes(p, &common.voice, common.speed, i as u64)?;
            per[i].push(t.elapsed().as_secs_f64());
            std::hint::black_box(&s.audio);
        }
        totals.push(tr.elapsed().as_secs_f64());
    }
    let stats = |xs: &[f64]| {
        let mut v = xs.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len().max(2) - 1) as f64).sqrt();
        serde_json::json!({"n": v.len(), "median": v[v.len() / 2], "min": v[0], "max": v[v.len() - 1], "mean": mean, "cv_pct": 100.0 * sd / mean})
    };
    let audio_total: f64 = audio_s.iter().sum();
    let tot = stats(&totals);
    let prof = crate::prof::report();
    if !prof.is_empty() {
        println!("stage profile (warmup + {reps} reps):\n{prof}");
    }
    let rec = serde_json::json!({
        "engine": ENGINE_VERSION, "device": format!("{:?}", common.device), "threads": common.threads, "voice": common.voice, "speed": common.speed,
        "reps": reps, "chunks_file": chunks, "chunks_sha256": sha256_file(&chunks)?, "n_chunks": items.len(),
        "model_sha256": engine.model_sha256, "load_s": load_s, "total_s": tot, "audio_s": audio_total,
        "rtf_median": tot["median"].as_f64().unwrap() / audio_total, "peak_rss_mb": peak_rss_mb(),
        "per_chunk": items.iter().enumerate().map(|(i, p)| { let mut s = stats(&per[i]); s["phonemes"] = p.chars().count().into(); s["audio_s"] = audio_s[i].into(); s }).collect::<Vec<_>>(),
        "loadavg": std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
    });
    println!("total median {:.3}s cv {:.1}% audio {:.1}s RTF {:.4} peak RSS {:.0} MB",
        tot["median"].as_f64().unwrap(), tot["cv_pct"].as_f64().unwrap(), audio_total, rec["rtf_median"].as_f64().unwrap(), peak_rss_mb());
    if let Some(o) = out {
        std::fs::write(&o, serde_json::to_string_pretty(&rec)?)?;
        println!("receipt: {}", o.display());
    }
    Ok(())
}

/// Returns the process exit status (0 complete, 1 incomplete); Err = job-level failure (exit 2).
pub fn cli_main() -> Result<i32> {
    match Cli::parse().cmd {
        Cmd::Synth { common, input, input_format, frontend, lang, bridge_python, bridge_script, out_dir, format, seed, encode, force, blank_lines } => {
            synth(common, input, input_format, frontend, lang, bridge_python, bridge_script, out_dir, format, seed, encode, force, blank_lines)
        }
        Cmd::Bench { common, chunks, reps, out } => bench(common, chunks, reps, out).map(|_| 0),
    }
}
