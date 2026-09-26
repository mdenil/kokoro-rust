//! `kokoro` command line: `synth` (narration lines -> WAV + JSON sidecars) and `bench`.

use crate::engine::{mix_seed, sha256_bytes, sha256_file, BatchPolicy, Engine, MAX_PHONEMES};
use crate::model::SAMPLE_RATE;
use crate::wav::{self, Format};
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use std::path::{Path, PathBuf};
use crate::frontend::pipeline::{EnglishFrontend, FrontendPaths};
use std::process::Command;
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
    /// Native English frontend (misaki 0.9.4 G2P + spaCy tokenizer/tagger + espeak-ng fallback).
    Native,
    /// No text frontend: only --input-format phonemes is accepted.
    None,
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
    #[cfg_attr(feature = "cuda", arg(long, value_enum, default_value = "cuda"))]
    #[cfg_attr(not(feature = "cuda"), arg(long, value_enum, default_value = "cpu"))]
    device: crate::engine::Device,
    /// CUDA device index (after CUDA_VISIBLE_DEVICES).
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    /// Batched synthesis budget: max phoneme chars per microbatch (0 = one chunk at a time).
    #[arg(long, default_value_t = 8000)]
    batch_phonemes: usize,
    /// Max chunks per microbatch.
    #[arg(long, default_value_t = 64)]
    batch_items: usize,
    /// Chunks collected before scheduling batches in `synth` (window for length bucketing).
    #[arg(long, default_value_t = 256)]
    batch_window: usize,
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
        #[arg(long, value_enum, default_value = "native")]
        frontend: Frontend,
        /// Directory with the pinned frontend data (misaki-0.9.4/, spacy-en_core_web_sm-3.8.0/,
        /// espeak-ng-1.52.0/) [env: KOKORO_FRONTEND_DIR]
        #[arg(long, env = "KOKORO_FRONTEND_DIR")]
        frontend_dir: Option<PathBuf>,
        /// libespeak-ng 1.52.0 shared library (default: <frontend-dir>/espeak-ng-1.52.0/libespeak-ng.so.1.52.0)
        #[arg(long)]
        espeak_lib: Option<PathBuf>,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, value_enum, default_value = "pcm16")]
        format: Format,
        /// Base seed for the excitation noise (per item: splitmix(seed, index)).
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// OPTIONAL, off by default: also encode each WAV by running the EXTERNAL `ffmpeg` executable
        /// (a helper subprocess) to this extension (e.g. flac, mp3, opus).
        #[arg(long)]
        encode: Option<String>,
        /// Re-synthesize even if a matching completed output exists.
        #[arg(long)]
        force: bool,
        /// Blank-line policy (canonical audiobook files contain none).
        #[arg(long, value_enum, default_value = "error")]
        blank_lines: BlankLines,
        /// Frontend worker threads for the prepare stage (0 = auto: half the CPUs, at most 16).
        #[arg(long, default_value_t = 0)]
        prep_threads: usize,
        /// Output writer threads (WAV encode + hash + sidecar).
        #[arg(long, default_value_t = 4)]
        write_threads: usize,
        /// fsync every WAV/sidecar/manifest before its atomic rename. Off by default (like the production
        /// Python path): resume re-verifies every output against its recorded audio sha256, so a file
        /// lost to a power failure is re-synthesized; renames still keep partial files invisible.
        #[arg(long)]
        fsync: bool,
        /// BENCHMARK ONLY: after loading once, run 1 warm-up + N timed full passes (frontend, GPU,
        /// WAV/sidecar/manifest writes) into <out-dir>/pass<k> (fresh dirs, no resume skips).
        #[arg(long, default_value_t = 0, hide = true)]
        bench_passes: usize,
        /// Write raw per-thread stage spans + per-pass summaries (overlap-aware timing) to this JSON.
        #[arg(long)]
        timeline: Option<PathBuf>,
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
    /// grapheme text of each synthesized chunk (native frontend), aligned with `phonemes`
    #[serde(default)]
    graphemes: Vec<String>,
    phonemes: Vec<String>,
    dropped_phoneme_chars: Vec<String>,
    config: Config,
    /// batching policy the line was synthesized under (provenance; not a resume key — batched and
    /// batch-1 outputs are both owner-accepted)
    #[serde(default)]
    synthesis: serde_json::Value,
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
    frontend_dir: Option<PathBuf>,
    espeak_lib: Option<PathBuf>,
    out_dir: PathBuf,
    format: Format,
    seed: u64,
    encode: Option<String>,
    force: bool,
    blank_lines: BlankLines,
    bench_passes: usize,
    timeline: Option<PathBuf>,
    prep_threads: usize,
    write_threads: usize,
    fsync: bool,
) -> Result<i32> {
    let prep_threads = if prep_threads > 0 {
        prep_threads
    } else {
        std::thread::available_parallelism().map(|n| n.get() / 2).unwrap_or(4).clamp(1, 16)
    };
    use crate::timeline::{now, Timeline};
    let tl = Timeline::default();
    let t_main = now();
    if input_format == InputFormat::Text && frontend == Frontend::None {
        bail!("--input-format text needs --frontend native (or use --input-format phonemes)");
    }
    set_threads(common.threads);
    let ts = now();
    let inp = read_input(&input)?;
    tl.push(0, "main", "input_read", ts, inp.lines.len());
    std::fs::create_dir_all(&out_dir)?;
    // Model and frontend load concurrently (independent; the frontend is CPU/disk, the model is
    // parse + GPU upload).
    let tl_ref = &tl;
    let load_frontend = move || -> Result<(Option<EnglishFrontend>, f64)> {
        let ts = now();
        let fe = match (input_format, frontend) {
            (InputFormat::Text, Frontend::Native) => {
                let dir = frontend_dir.context("--frontend-dir (or KOKORO_FRONTEND_DIR) is required for text input")?;
                let mut paths = FrontendPaths::under(&dir);
                if let Some(lib) = espeak_lib {
                    paths.espeak_lib = lib;
                }
                Some(EnglishFrontend::load(&paths).context("loading the native text frontend")?)
            }
            _ => None,
        };
        tl_ref.push(0, "loader", "frontend_load", ts, 1);
        Ok((fe, now() - ts))
    };
    let (engine_res, fe_res) = std::thread::scope(|s| {
        let fe_h = s.spawn(load_frontend);
        let ts = now();
        let r = (|| -> Result<(Engine, String, f64)> {
            let mut engine = Engine::load_on(&common.model_dir, common.device, common.cuda_device)?;
            let voice_sha = engine.voice(&common.voice)?.sha256.clone();
            Ok((engine, voice_sha, now() - ts))
        })();
        tl_ref.push(0, "main", "model_load", ts, 1);
        (r, fe_h.join())
    });
    let (mut engine, voice_sha, load_s) = engine_res?;
    let (fe, frontend_load_s) = fe_res.map_err(|_| anyhow::anyhow!("frontend loader panicked"))??;
    eprintln!("loaded model + voice in {load_s:.2}s (frontend {frontend_load_s:.2}s, concurrently)");
    let cfg = Config {
        engine: format!("{ENGINE_VERSION} [{}]", engine.identity()),
        model_sha256: engine.model_sha256.clone(),
        config_sha256: engine.config_sha256.clone(),
        voice: common.voice.clone(),
        voice_sha256: voice_sha,
        speed: common.speed,
        seed,
        format: format!("{format:?}"),
        input_format: format!("{input_format:?}"),
        frontend: fe.as_ref().map(|f| f.ident().to_string()).unwrap_or_else(|| "none (phoneme input)".into()),
        sample_rate: SAMPLE_RATE,
    };
    let npasses = if bench_passes > 0 { bench_passes + 1 } else { 1 };
    let base_out = out_dir.clone();
    let mut code = 0;
    let mut pass_records = vec![];
    for pass in 0..npasses {
    let out_dir: PathBuf = if bench_passes > 0 {
        let d = base_out.join(format!("pass{pass}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d)?;
        d
    } else {
        base_out.clone()
    };
    let force = force || bench_passes > 0;
    let t_pass = now();
    let width = inp.lines.len().max(1).to_string().len().max(5);
    let name = |line: usize, ext: &str| format!("{}_{line:0width$}.{ext}", inp.stem);
    let new_sidecar = |line: usize, text: &str, text_sha: String| Sidecar {
        line,
        input_file: inp.display.clone(),
        input_sha256: inp.sha256.clone(),
        text: text.to_string(),
        text_sha256: text_sha,
        status: "ok".into(),
        error: None,
        graphemes: vec![],
        phonemes: vec![],
        dropped_phoneme_chars: vec![],
        config: cfg.clone(),
        synthesis: serde_json::json!({"batch_phonemes": common.batch_phonemes, "batch_items": common.batch_items, "batch_window": common.batch_window}),
        wav: None,
        audio_sha256: None,
        samples: 0,
        duration_s: 0.0,
        clipped_samples: 0,
        encoded: None,
        encoded_sha256: None,
        elapsed_s: 0.0,
    };

    // Three pipelined stages (bounded channels, strict per-line identity):
    //   prepare (resume check, validation, frontend) -> GPU synthesis -> writer (encode, hash,
    //   atomic writes, sidecar). Host work overlaps GPU work; output order/naming is unchanged.
    enum Job {
        Resumed(usize, serde_json::Value, f64),
        Final(Box<Sidecar>),                       // no synthesis (blank/invalid/oversize/frontend error)
        Synth(Box<Sidecar>, Vec<String>, Instant), // chunks to synthesize
    }
    enum Out {
        Resumed(usize, serde_json::Value, f64),
        Write(Box<Sidecar>, Option<Vec<f32>>, Instant),
    }
    let depth = 8;
    let (inp_r, out_r, enc_r, cfg_r) = (&inp, &out_dir, &encode, &cfg);
    let t_all = Instant::now();
    let tl_r = &tl;
    let (manifest_lines, n_ok, n_skip, n_bad, audio_total) = std::thread::scope(|sc_scope| -> Result<_> {
        let (prep_tx, prep_rx) = std::sync::mpsc::sync_channel::<Job>(depth);
        let (out_tx, out_rx) = std::sync::mpsc::sync_channel::<Out>(depth);
        let fe_ref = fe.as_ref();
        // Prepare stage: `prep_threads` workers run resume checks + the frontend on lines in parallel;
        // a sequencer forwards jobs to the GPU stage strictly in line order.
        let make_job = move |i: usize, text: &String| -> Result<Job> {
                let line = i + 1;
                let ts = now();
                let (wav_path, json_path) = (out_r.join(name(line, "wav")), out_r.join(name(line, "json")));
                let text_sha = sha256_bytes(text.as_bytes());
                if !force && resume_ok(&json_path, &wav_path, line, &text_sha, cfg_r) {
                    let sc: Sidecar = serde_json::from_str(&std::fs::read_to_string(&json_path)?)?;
                    let entry = serde_json::json!({"line": line, "status": sc.status, "resumed": true, "wav": sc.wav, "sidecar": name(line, "json"),
                        "text_sha256": sc.text_sha256, "audio_sha256": sc.audio_sha256, "duration_s": sc.duration_s});
                    tl_r.push(pass, "prepare", "resume_check", ts, 1);
                    return Ok(Job::Resumed(line, entry, sc.duration_s));
                }
                let t0 = Instant::now();
                let mut sc = new_sidecar(line, text, text_sha);
                let chunks: Result<Option<Vec<String>>> = (|| {
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
                    let chunks = match fe_ref {
                        Some(f) => {
                            let ch = f.line_chunks(text)?;
                            sc.graphemes = ch.iter().map(|c| c.graphemes.clone()).collect();
                            ch.into_iter().map(|c| c.phonemes).collect()
                        }
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
                    Ok(Some(chunks))
                })();
                let job = match chunks {
                    Ok(Some(c)) => Job::Synth(Box::new(sc), c, t0),
                    Ok(None) => Job::Final(Box::new(sc)),
                    Err(e) => {
                        if sc.status == "ok" {
                            sc.status = "error".into();
                        }
                        sc.error = Some(format!("{e:#}"));
                        Job::Final(Box::new(sc))
                    }
                };
                tl_r.push(pass, "prepare", "frontend", ts, 1);
                Ok(job)
                    };
        let prep = sc_scope.spawn(move || -> Result<()> {
            let n = inp_r.lines.len();
            // bounded look-ahead, strictly ordered emission (see crate::ordered)
            let window = (4 * prep_threads.max(1)).max(8);
            let emitted = crate::ordered::ordered_parallel_map(n, prep_threads, window, |i| make_job(i, &inp_r.lines[i]), |_, job| {
                let ws = now();
                if prep_tx.send(job).is_err() {
                    return false;
                }
                tl_r.push(pass, "prepare", "send_wait", ws, 1);
                true
            })?;
            // emitted < n only when the GPU stage closed its receiver (it then reports its own error)
            let _ = emitted;
            Ok(())
        });
        let out_rx = std::sync::Arc::new(std::sync::Mutex::new(out_rx));
        let writer_fn = move |out_rx: std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Receiver<Out>>>| -> Result<(Vec<(usize, serde_json::Value)>, usize, usize, usize, f64)> {
            let (mut n_ok, mut n_skip, mut n_bad, mut audio_total) = (0usize, 0usize, 0usize, 0.0f64);
            let mut entries = vec![];
            loop {
                let ws = now();
                let msg = out_rx.lock().unwrap().recv();
                let Ok(msg) = msg else { break };
                tl_r.push(pass, "writer", "recv_wait", ws, 1);
                let ts = now();
                match msg {
                    Out::Resumed(line, e, d) => {
                        n_skip += 1;
                        audio_total += d;
                        entries.push((line, e));
                    }
                    Out::Write(mut sc, audio, t0) => {
                        let line = sc.line;
                        let (wav_path, json_path) = (out_r.join(name(line, "wav")), out_r.join(name(line, "json")));
                        match audio {
                            Some(audio) => {
                                let enc = wav::encode(&audio, SAMPLE_RATE as u32, format);
                                wav::write_atomic_opt(&wav_path, &enc.bytes, fsync)?;
                                sc.wav = Some(name(line, "wav"));
                                sc.audio_sha256 = Some(sha256_bytes(&enc.bytes));
                                sc.samples = audio.len();
                                sc.duration_s = audio.len() as f64 / SAMPLE_RATE as f64;
                                sc.clipped_samples = enc.clipped;
                                audio_total += sc.duration_s;
                                if let Some(ext) = enc_r {
                                    let dst = out_r.join(name(line, ext));
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
                            None => {
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
                        wav::write_atomic_opt(&json_path, serde_json::to_string_pretty(&*sc)?.as_bytes(), fsync)?;
                        entries.push((line, serde_json::json!({"line": line, "status": sc.status, "resumed": false, "wav": sc.wav, "sidecar": name(line, "json"),
                            "text_sha256": sc.text_sha256, "audio_sha256": sc.audio_sha256, "duration_s": sc.duration_s, "error": sc.error})));
                    }
                }
                tl_r.push(pass, "writer", "write", ts, 1);
            }
            Ok((entries, n_ok, n_skip, n_bad, audio_total))
        };
        let writers: Vec<_> = (0..write_threads.max(1))
            .map(|_| {
                let rx = out_rx.clone();
                sc_scope.spawn(move || writer_fn(rx))
            })
            .collect();
        drop(out_rx);

        // GPU stage on this thread (the engine is not shared). With --batch-phonemes > 0, chunks
        // are collected in a window and synthesized as length-bucketed batches; each line is sent
        // to the writer only after all of its chunks are joined in order.
        let policy = BatchPolicy { max_phonemes: common.batch_phonemes, max_items: common.batch_items.max(1) };
        let batching = common.batch_phonemes > 0;
        let mut pending: Vec<(Box<Sidecar>, Vec<String>, Instant)> = vec![];
        let mut pending_chunks = 0usize;
        let flush = |engine: &mut Engine, pending: &mut Vec<(Box<Sidecar>, Vec<String>, Instant)>, out_tx: &std::sync::mpsc::SyncSender<Out>| -> bool {
            let mut reqs = vec![];
            for (sc, chunks, _) in pending.iter() {
                for (k, c) in chunks.iter().enumerate() {
                    reqs.push((c.clone(), mix_seed(seed, (sc.line as u64) << 16 | k as u64)));
                }
            }
            let ts = now();
            let mut res = engine.synth_batch(&reqs, &common.voice, common.speed, policy).into_iter();
            tl_r.push(pass, "gpu", "synth", ts, reqs.len());
            for (mut sc, chunks, t0) in pending.drain(..) {
                let mut audio = vec![];
                let mut err = None;
                for _ in 0..chunks.len() {
                    match res.next().expect("result per chunk") {
                        Ok(s) if err.is_none() => {
                            sc.dropped_phoneme_chars.extend(s.dropped.iter().map(|ch| ch.to_string()));
                            audio.extend_from_slice(&s.audio);
                        }
                        Ok(_) => {}
                        Err(e) => {
                            if err.is_none() {
                                err = Some(format!("{e:#}"));
                            }
                        }
                    }
                }
                let msg = match err {
                    None => Out::Write(sc, Some(audio), t0),
                    Some(e) => {
                        sc.status = "error".into();
                        sc.error = Some(e);
                        Out::Write(sc, None, t0)
                    }
                };
                let ws = now();
                if out_tx.send(msg).is_err() {
                    return false;
                }
                tl_r.push(pass, "gpu", "send_wait", ws, 1);
            }
            true
        };
        loop {
            let ws = now();
            let Ok(job) = prep_rx.recv() else { break };
            tl_r.push(pass, "gpu", "recv_wait", ws, 1);
            let msg = match job {
                Job::Resumed(line, e, d) => Out::Resumed(line, e, d),
                Job::Final(sc) => Out::Write(sc, None, Instant::now()),
                Job::Synth(sc, chunks, t0) if batching => {
                    pending_chunks += chunks.len();
                    pending.push((sc, chunks, t0));
                    if pending_chunks >= common.batch_window {
                        pending_chunks = 0;
                        if !flush(&mut engine, &mut pending, &out_tx) {
                            break;
                        }
                    }
                    continue;
                }
                Job::Synth(mut sc, chunks, t0) => {
                    let mut audio = vec![];
                    let mut err = None;
                    let ts = now();
                    for (k, c) in chunks.iter().enumerate() {
                        match engine.synth_phonemes(c, &common.voice, common.speed, mix_seed(seed, (sc.line as u64) << 16 | k as u64)) {
                            Ok(s) => {
                                sc.dropped_phoneme_chars.extend(s.dropped.iter().map(|ch| ch.to_string()));
                                audio.extend_from_slice(&s.audio);
                            }
                            Err(e) => {
                                err = Some(format!("{e:#}"));
                                break;
                            }
                        }
                    }
                    tl_r.push(pass, "gpu", "synth", ts, chunks.len());
                    match err {
                        None => Out::Write(sc, Some(audio), t0),
                        Some(e) => {
                            sc.status = "error".into();
                            sc.error = Some(e);
                            Out::Write(sc, None, t0)
                        }
                    }
                }
            };
            let ws = now();
            if out_tx.send(msg).is_err() {
                break;
            }
            tl_r.push(pass, "gpu", "send_wait", ws, 1);
        }
        if !pending.is_empty() {
            flush(&mut engine, &mut pending, &out_tx);
        }
        drop(out_tx);
        prep.join().map_err(|_| anyhow::anyhow!("prepare stage panicked"))??;
        let (mut entries, mut n_ok, mut n_skip, mut n_bad, mut audio_total) = (vec![], 0, 0, 0, 0.0);
        for w in writers {
            let (e, a, b, c, d) = w.join().map_err(|_| anyhow::anyhow!("writer stage panicked"))??;
            entries.extend(e);
            n_ok += a;
            n_skip += b;
            n_bad += c;
            audio_total += d;
        }
        entries.sort_by_key(|e| e.0);
        Ok((entries.into_iter().map(|e| e.1).collect::<Vec<_>>(), n_ok, n_skip, n_bad, audio_total))
    })?;
    let wall = t_all.elapsed().as_secs_f64();
    let complete = n_bad == 0;
    let manifest = serde_json::json!({
        "input_file": inp.display, "input_sha256": inp.sha256, "input_lines": inp.lines.len(),
        "bom_stripped": inp.bom_stripped, "crlf_lines": inp.crlf_lines, "blank_lines_policy": format!("{blank_lines:?}"),
        "naming": format!("{}_<1-based line, width {width}>.wav / .json", inp.stem),
        "config": cfg, "complete": complete, "counts": {"done": n_ok, "resumed": n_skip, "failed": n_bad},
        "audio_s": audio_total, "load_s": load_s, "frontend_load_s": frontend_load_s, "synth_wall_s": wall, "lines": manifest_lines,
    });
    let ts = now();
    wav::write_atomic_opt(&out_dir.join(format!("{}.manifest.json", inp.stem)), serde_json::to_string_pretty(&manifest)?.as_bytes(), fsync)?;
    tl.push(pass, "main", "manifest", ts, 1);
    tl.push(pass, "main", "pass", t_pass, inp.lines.len());
    eprintln!(
        "{} lines: {n_ok} done, {n_skip} resumed/skipped, {n_bad} failed; {audio_total:.1}s audio in {wall:.2}s (RTF {:.4})",
        inp.lines.len(),
        if audio_total > 0.0 { wall / audio_total } else { 0.0 }
    );
    if !complete {
        eprintln!("INCOMPLETE: {n_bad} line(s) failed; see {}.manifest.json and sidecars", inp.stem);
        code = 1;
    }
    pass_records.push(serde_json::json!({"pass": pass, "warmup": bench_passes > 0 && pass == 0, "out_dir": out_dir,
        "pass_wall_s": now() - t_pass, "synth_wall_s": wall, "audio_s": audio_total, "complete": complete,
        "counts": {"done": n_ok, "resumed": n_skip, "failed": n_bad}, "timing": tl.summary(pass)}));
    }
    if let Some(path) = timeline {
        let rec = serde_json::json!({
            "engine": cfg.engine, "frontend": cfg.frontend, "input_file": inp.display, "input_sha256": inp.sha256,
            "input_lines": inp.lines.len(), "bench_passes": bench_passes,
            "synthesis": {"batch_phonemes": common.batch_phonemes, "batch_items": common.batch_items, "batch_window": common.batch_window,
                "prep_threads": prep_threads, "write_threads": write_threads, "fsync": fsync},
            "main_entered_s": t_main, "exit_s": now(), "load_s": load_s, "frontend_load_s": frontend_load_s,
            "passes": pass_records, "spans": tl.spans(),
        });
        std::fs::write(&path, serde_json::to_string(&rec)?).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(code)
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
    let policy = BatchPolicy { max_phonemes: common.batch_phonemes, max_items: common.batch_items.max(1) };
    let reqs: Vec<(String, u64)> = items.iter().enumerate().map(|(i, p)| (p.clone(), i as u64)).collect();
    // warmup: one full uncounted pass
    let mut audio_s = vec![0.0f64; items.len()];
    if common.batch_phonemes > 0 {
        for (i, r) in engine.synth_batch(&reqs, &common.voice, common.speed, policy).into_iter().enumerate() {
            audio_s[i] = r?.audio.len() as f64 / SAMPLE_RATE as f64;
        }
    } else {
        for (i, p) in items.iter().enumerate() {
            audio_s[i] = engine.synth_phonemes(p, &common.voice, common.speed, i as u64)?.audio.len() as f64 / SAMPLE_RATE as f64;
        }
    }
    let mut per = vec![vec![]; items.len()];
    let mut totals = vec![];
    for _ in 0..reps {
        let tr = Instant::now();
        if common.batch_phonemes > 0 {
            for r in engine.synth_batch(&reqs, &common.voice, common.speed, policy) {
                std::hint::black_box(&r?.audio);
            }
        } else {
            for (i, p) in items.iter().enumerate() {
                let t = Instant::now();
                let s = engine.synth_phonemes(p, &common.voice, common.speed, i as u64)?;
                per[i].push(t.elapsed().as_secs_f64());
                std::hint::black_box(&s.audio);
            }
        }
        totals.push(tr.elapsed().as_secs_f64());
    }
    if common.batch_phonemes > 0 {
        per = vec![vec![0.0]; items.len()]; // per-chunk latency is not defined for batched passes
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
        "engine": format!("{ENGINE_VERSION} [{}]", engine.identity()), "device": format!("{:?}", common.device), "threads": common.threads, "batch_policy": if common.batch_phonemes > 0 { serde_json::json!(policy) } else { serde_json::json!("batch-1") }, "voice": common.voice, "speed": common.speed,
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
        Cmd::Synth { common, input, input_format, frontend, frontend_dir, espeak_lib, out_dir, format, seed, encode, force, blank_lines, bench_passes, timeline, prep_threads, write_threads, fsync } => {
            synth(common, input, input_format, frontend, frontend_dir, espeak_lib, out_dir, format, seed, encode, force, blank_lines, bench_passes, timeline, prep_threads, write_threads, fsync)
        }
        Cmd::Bench { common, chunks, reps, out } => bench(common, chunks, reps, out).map(|_| 0),
    }
}
