# kokoro-rust — compact milestone plan

Goal: native pure-Rust inference for hexgrad/Kokoro-82M (StyleTTS2-derived TTS), proven
against a pinned PyTorch oracle, then measured fast on this host (CPU baseline + RTX 4090),
with a final performance report vs the CURRENT production implementation (kokoro==0.9.4,
torch==2.12.1, CUDA). Owner brief: HERMES_BRIEF.md. Skill: ai-model-into-rust-mega-fused-hyper-kernel.

## Architecture (to be verified against pinned source — OQ register tracks each claim)

Pipeline: text --misaki G2P--> phoneme string --vocab--> ids (≤510) → KModel forward → 24kHz f32 waveform.

KModel stages (from kokoro 0.9.4 `model.py`, StyleTTS2 lineage):
1. ALBERT (`bert`): custom PLBert; token ids → hidden states [T, 768].
2. `bert_encoder`: Linear 768→512.
3. `predictor` (ProsodyPredictor): DurationEncoder (LSTMs + AdaLayerNorm w/ style s=ref_s[:,128:]),
   LSTM → `duration_proj` → per-token durations (sigmoid sum, /speed, round, clamp≥1);
   alignment expansion (pred_aln_trg); shared BiLSTM → F0 + Noise predictors (AdainResBlk1d stacks).
4. `text_encoder` (CNN+BiLSTM) over ids → asr features; expanded by alignment.
5. `decoder` (iSTFTNet Decoder): AdainResBlk1d encode/decode stacks conditioned on style
   s=ref_s[:,:128], F0/N convs, Generator: upsampling ConvTranspose1d + resblocks +
   harmonic source module (SineGen from F0) + iSTFT head (custom STFT class) → waveform.
6. Voice pack: [510, 1, 256] style vectors; ref_s = pack[len(phonemes)-1].

Kernel families needed (beyond OCR-exemplar transformer set): bidirectional LSTM, Conv1d
(+weight-norm), ConvTranspose1d, InstanceNorm/AdaIN1d, AdaLayerNorm, reflection pad, leaky-relu,
sine-source generation, STFT/iSTFT (FFT), interpolation/alignment matmul, plus ALBERT attention/GEMM/LayerNorm/GELU.

## Milestones

- M0 truth pack: pins+hashes for HF snapshot f3ff357…, kokoro/misaki 0.9.4 (PyPI authoritative),
  exact runtime env (see brief), licenses; OQ register resolved from source. Artifact: docs/truth-pack/*.
- M1 oracle: pinned venv runner; per-stage activation dumps (hooks) on CPU-f32 (correctness
  anchor) AND CUDA (deployment behavior); nondeterminism floor (2 runs × 2 thread counts × both
  devices; identify/freeze stochastic noise paths in SineGen/SourceModuleHnNSF if present);
  frozen corpus: both voices (af_heart, am_adam), speeds {0.8,1.0,1.25}, short/long/boundary,
  punctuation/numbers/Unicode, invalid inputs. Baseline perf protocol + measured production
  baseline receipts (CUDA & CPU; cold vs warm; batch-1 latency + throughput/RTF + peak mem).
- M2 Rust skeleton: workspace, safetensors/config/vocab loaders, voices loader (via converted
  fixture first; native .pt reader or converted-voices artifact decided later — documented),
  tensor census test vs oracle-dumped names/shapes.
- M3 f32 forward, seam-by-seam vs oracle tensors in pipeline order (each stage fed the oracle's
  exact input): ALBERT → bert_encoder → duration path (durations EXACT ints) → alignment →
  F0/N → text_encoder → decoder → generator → iSTFT → waveform. e2e gates set from measured
  floor (waveform max/RMS err, correlation, spectral, sample-count exact). Perturbation-proof the comparator.
- M4 text frontend: dev bridge = fixtures with misaki-precomputed phonemes (never shipped as
  product); then native English G2P subset with honest parity accounting vs misaki (dictionary
  hit-rate, espeak-fallback rate, punctuation/number handling). Report separately from core parity.
- M5 CLI/library per brief contract: lines-in → per-line WAV + JSON sidecars (hashes, sample
  rate, duration, status, elapsed, errors), resident model (no per-sentence weight loads),
  voice/speed/language/model-path flags, resumability, explicit empty/error/oversize handling.
- M6 perf: CPU pass (profile → one-lever ritual; rayon/autovec first, SIMD only where measured);
  CUDA backend for 4090 (custom kernels via cudarc; go/no-go doc first). Interleaved paired A/B
  vs pinned production PyTorch on matched inputs/voices/speeds/devices/thread budgets.
- M7 report + ship: docs/PERFORMANCE_REPORT.md (baseline vs final, per-case + aggregate,
  scopes separated: pure inference / frontend / persistent batch / production e2e; uncertainty,
  correctness gates, repro commands, identities). Release engineering deferred until review.

## Explicit unresolved assumptions (tracked as OQs; no kernel against an unresolved one)

- A1 Exact module graph & shapes of kokoro 0.9.4 KModel (verify from installed source, not memory).
- A2 Stochastic paths: SineGen phase noise / SourceModuleHnNSF additive noise — deterministic or
  RNG-driven in 0.9.4? Determines noise-freezing harness design.
- A3 Voice `.pt` files: exact tensor layout + how ref_s is indexed (pack[len(ps)-1]).
- A4 misaki en G2P: dictionary + espeak fallback split; spacy usage; determinism.
- A5 torch.stft/istft parameters (n_fft/hop/win, center, normalization, window) in custom STFT.
- A6 Production "current implementation" scope for the headline comparison: KPipeline(en-us) on
  CUDA:0, af_heart/am_adam, warm resident model (matches hermes-tts-workshop usage).
- A7 License: Kokoro-82M weights Apache-2.0 (verify in snapshot); kokoro/misaki code Apache-2.0 (verify).
- A8 Duration rounding semantics: torch.round(...).clamp(min=1) — exact integer parity required.

## Decisions taken without asking (per brief authority)

- Single model Kokoro-82M @ f3ff357…; English (a/b) voices first; af_heart + am_adam defaults.
- CPU f32 = correctness baseline; RTX 4090 = deployment perf target.
- Python allowed ONLY for oracle/fixtures/comparison. No LibTorch/ONNX in the product.
- Voices conversion: oracle env exports voices .pt → .safetensors fixture for Rust dev; a
  self-contained acquisition story (native reader or converter tool) decided at M5, documented.


## Roadmap update (2026-09-26): native English frontend (owner scope #6) + CUDA + publish

Production English frontend = misaki 0.9.4 `en.G2P(trf=False, fallback=EspeakFallback, unk='')`
+ KPipeline `en_tokenize`/waterfall chunking. Components and native plan:
| component | reference | native plan | license note |
|---|---|---|---|
| text preprocess, tokenization | misaki regex preprocessing + spaCy 3.8 tokenizer (en_core_web_sm 3.8.0) | port spaCy English tokenizer rules (prefix/suffix/infix/exceptions from the pinned model) | spaCy/model MIT |
| POS tags (drive heteronyms, e.g. read/live/used) | en_core_web_sm tok2vec + tagger (thinc CNN) | port the tagger inference (hash embed + maxout CNN + softmax) from the pinned weights; measure tag agreement; tags only matter where lexicon entries are POS-keyed | MIT |
| lexicon | us_gold/us_silver (gb_*) JSON | load natively (embed or data file) | misaki Apache-2.0; provenance of lexicon data to verify |
| numbers, currency, ordinals, years, decimals | num2words 0.5.14 (English) | reimplement English behaviour natively from differential tests (no code copy) | num2words LGPL — avoided by reimplementation |
| OOV words | espeak-ng via phonemizer-fork (GPL-3) + misaki post-mapping | **OWNER DECISION NEEDED**: (a) optional runtime-loaded libespeak-ng (GPL-3, user-installed), (b) link espeak-ng (product becomes GPL-3), (c) native non-GPL OOV G2P with measured divergence | phonemizer-fork + libespeak-ng GPL-3 |
| chunking | KPipeline.en_tokenize / waterfall_last (510) | port exactly (small) | Apache-2.0 |

Sequence (sole coder; CUDA work interleaved at natural boundaries):
- F0 frontend truth pack: pin/hash lexicons, spaCy model, espeak-ng lib/data; frontend OQ register;
  frontend oracle dumping misaki tokens (text, whitespace, tag, phonemes, rating, source path) and
  KPipeline chunks for a large corpus (full Alice + curated edge-case suite); immutable fixtures.
- F1 native lexicon + preprocessing + number normalization + chunking, with differential tests;
  first measure how often POS and the espeak fallback actually decide the output on the corpus.
- F2 spaCy tokenizer + tagger port (bit/label agreement tests).
- F3 OOV fallback per owner decision.
- F4 wire into CLI (`--frontend native` default), remove the bridge from the shipped path, verify
  with no Python on PATH (strace execve audit), both voices, durations/coverage; benchmarks.
- CUDA Tier A levers (conv_direct tiling, chan_stats, LSTM step, fusions) interleaved; Tier B later.
- Then publish/install phase (brief #8): reproducible build, pinned model acquisition incl. frontend
  data, release binaries + checksums, clean-install smoke — deferred until review.

### Audiobook line-file interface (owner #7) — placement
- I0 (now, small, no frontend dependency): extend `synth` — 1-based line identity + input-file
  provenance in sidecars, run manifest (manifest.json), `--blank-lines error|skip` (default error:
  canonical files have none; never shifts numbering), whole-file UTF-8 validation with exact
  line/byte of any invalid sequence, duplicate lines kept by identity, filename scheme confirmed with
  Hermes against audiobook/audio_chunks/ conventions; tests for restart, invalidation on changed
  text/settings/model/frontend, long-line (multi-chunk) completeness.
- I1 (with F4): default `--frontend native`; Python-free acceptance run (no python on PATH, execve
  audit), both voices; cold + resident benchmarks of the same line-file interface.

### Work order after owner #8 (throughput/batching), natural-boundary sequencing
1. I0 line-file interface on `synth` (1-based line ids, manifest, blank/malformed policy, resume/
   invalidation tests) — prerequisite for the primary file->WAVs benchmark.
2. Primary-workload baselines on the private chapter (private evidence under /data only):
   production Python/CUDA per-line WAVs; Rust batch-1 (dev bridge, labelled); measured phoneme /
   frame / chunk length distributions (aggregate stats only in Git).
3. B1 batched CUDA forward: masked padded batches (per-item lengths through ALBERT attention, LSTMs,
   AdaIN/instance-norm statistics, convs with re-zeroed padding, per-item SineGen/noise streams),
   length-bucketed scheduler with VRAM-aware batch sizing + single-item/OOM fallback, pipelined
   host work (frontend, WAV/hash writes). Correctness: batch-vs-single bounds fixed in advance,
   padding-contamination and cross-item negative controls, reorder invariance, tail batches.
4. Native frontend F0–F4 continues interleaved (required for the headline number).
5. CUDA kernel levers re-profiled under batching (Tier A/B reassessed).
