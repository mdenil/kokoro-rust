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
