# PORT_STATE — kokoro-rust   (read this first on any resume; then re-verify pins)

## Where we are (2026-09-26)
- Deliverable: **native Rust + CUDA on RTX 4090 (GPU-first, owner decision)**. CPU path = oracle/debug baseline.
- Correctness: **NOT ACCEPTED as a whole.** Binding original gates (HERMES_BRIEF owner log #2).
  CUDA ladder `ladder-gpu-1790426627`: all 135 stage-seam rows (9 seams × 15 cases) PASS; ids/durations/
  sample counts exact; **8 enforced E2E rows FAIL** (v1 max ×5, G-SPEC ×3).
  - s02_fox/am_adam/s1.0: **OWNER-ACCEPTED by listening (DISC-003)**.
  - 7 others: OPEN, unaccepted (DISC-004) — escalate to owner; do not self-authorize.
  - Attainability evidence: exact-f64 reference passes v1 on 6/15; production torch CUDA 3/15
    (docs/conformance/TOLERANCE_HISTORY.md #9).
- Performance: levers PL-001 (device noise, 1.39× process wall) and PL-002 (persistent LSTM, bit-identical,
  −2.5% inference) KEPT; NE-003 (im2col) reverted. Regression policy RB-1 (bounded variation, owner
  clarification) in force: `gpu_regression_bounded`; bounds fixed in tests/pinned/regression_bounds.json.
- Interim checkpoint DONE (docs/PERFORMANCE_REPORT.md CURRENT CHECKPOINT; evidence
  /data/mdenil/code/kokoro-rust/evidence/interim-comparison/20260926-141335/, sealed SHA256SUMS):
  warm core Rust 2.462 s vs production torch 6.986 s (2.84×, af_heart; 3.09× am_adam; reference cv
  7–11% → PROVISIONAL); cold text→WAV 25.10 s vs 14.84 s (Rust uses DEV-ONLY Python bridge).
- FRONTEND PROGRESS: F0 done (oracle dumps, coverage findings docs/frontend/COVERAGE.md). F1 done:
  native misaki logic (lexicon, stress, stemming, numbers incl. native num2words, retokenize, context
  resolution, KPipeline chunking) EXACT vs oracle given oracle spaCy tokens+tags and recorded espeak
  outputs: edge 65/65, Alice 1402/1402, private chapter 316/316 lines (tests/frontend_g2p.rs; negative
  controls: flattened tags 400/400 lines differ, no-fallback 34, british lexicon 397). Lexicon data
  pinned under /data/.../frontend/misaki-0.9.4 (hashes in COVERAGE/oracle meta). Next: F2 spaCy
  tokenizer + tagger port (tags affect nearly every line), then F3 fallback (owner GPL decision).
- NEW SCOPE (owner #6): native English text frontend, Python-free shipped path — roadmap in
  COMPREHENSIVE_PLAN_FOR_kokoro.md (F0–F4). Open owner decision: espeak-ng OOV fallback is GPL-3.
  Next single action: F0 frontend truth pack + oracle token/chunk dump.
- Levers since checkpoint: PL-004 tiled conv (bit-identical, KEEP); PL-003 batching (opt-in, ~5%,
  quality PROVISIONAL, escalated: evidence/listening/batch-worst-s04_alice_am_adam); PL-005 FMA (opt-in
  build KOKORO_FMA=1, +1.6%, authoritative RB-1 FAILS: 9 drift + new s03_moon/am_adam G-SPEC; escalated:
  evidence/listening/fma-worst-s03_moon_am_adam). Default build = strict (owner-accepted baseline).
  Authoritative regression test = gpu_regression_bounded vs the STRICT baseline; FMA snapshot is a
  PROVISIONAL diagnostic only. ABBA receipts: evidence/ab/ (scripts/ab.sh tees raw output).
- Owner #13: headline = original production Python (unchanged) vs fastest Rust; no Python optimization.
- Owner #12: representative subsets OK for fast iteration (label them); full-chapter runs for milestones.
- Owner #11: speed + close match that sounds good; FMA contraction allowed (strict -fmad=false = optional diagnostic build).
- Owner #10: host-side parallelism allowed/encouraged for GPU chapter throughput (bounded pools, record CPU budgets).
- NEW SCOPE (owner #8): chapter throughput + RTX 4090 batching; primary workload = private chapter
  (316 lines, sha256 8129112a…, under /data/mdenil/code/kokoro-rust/bench/private/ — NEVER in Git).
  Work order: I0 -> chapter baselines -> B1 batching -> (frontend F0–F4 interleaved) -> kernel levers.
- I0 DONE (line-file interface): `<stem>_<1-based line>.wav/.json` + `<stem>.manifest.json`; UTF-8 validated
  up front (exact line/byte), BOM/CRLF recorded, control chars / blank (default error) / oversize explicit,
  resume = line identity + text + config + audio hash, exit 0/1/2; tests/cli_linefile.rs (5 tests, real
  binary, no Python on PATH). Remaining for I1: text-path long-line multi-chunk completeness + Python-free
  acceptance with the native frontend.
- SCOPE (owner #7): audiobook line-file interface on `kokoro synth` (I0 done, I1 with F4) —
  HERMES_AUDIOBOOK_INTERFACE_BRIEF.md; open question to Hermes: exact audio_chunks filename convention.
- Optimization continues (owner follow-up), interleaved with F0–F4: Tier A levers under RB-1 — tiled conv_direct,
  multi-block chan_stats, cheaper LSTM step, elementwise fusions; then Tier B implicit-GEMM conv.
- Process lesson: never wait with `pgrep -f <pattern>` from a shell whose own command line contains
  the pattern (self-match hung a waiter); wait on the tracked background task instead.

## Environment (ALWAYS `source scripts/env.sh` first)
- Data root: /data/mdenil/code/kokoro-rust (relocated; docs/RELOCATION_2026-09-26.md).
- Cargo target: /home/mdenil/code/kokoro-rust/target. CUDA build: `cargo build --release --features cuda`
  (nvcc 12.9 → PTX compute_89, -fmad=false). `CUDA_VISIBLE_DEVICES=0` (RTX 4090; never GPU 1).
- Push via SSH origin, ALWAYS after `python3 scripts/check_private_leaks.py` passes (owner #9). Shell `grep` is a ugrep wrapper honoring .gitignore; use `command grep`.
- serde_json needs `float_roundtrip` (default parser was 1-ulp lossy — caught by the pin test).

## Pins
- Model: hexgrad/Kokoro-82M @ f3ff357…; weights sha256 496dba11…; loaded natively from .pth (bitwise = reference load).
- Reference: kokoro==0.9.4, misaki==0.9.4, torch==2.12.1 (CUDA 13.0), Python 3.12.3 (exact prod pins).
- Production comparison scope: KPipeline('a'), af_heart/am_adam, CUDA:0, warm resident.

## Tests (skip-honest; missing fixtures or empty filters FAIL)
- `cargo test --release --test parity` — CPU ladder (v1+G-SPEC enforced; v2 = UNAPPROVED diagnostic), f64 truth, perturbations.
- `cargo test --release --features cuda --test gpu_parity` — CUDA ladder (enforced), device-noise parity,
  GPU negative controls, `gpu_regression_pinned` (owner-approved baseline).
- `cargo test --release --test native_load` — .pth/.pt reader bitwise vs reference.
- Ladders currently FAIL on the enforced E2E rows listed above; that is the truthful state.

## Measured baselines (receipts under evidence/baseline/)
- Production torch CUDA, 65-line Alice ch.1 corpus (656.2 s audio): pure inference 6.97 s (cv 2.9%,
  RTF 0.0106); resident batch 7.41 s; cold process 25.5 s; frontend 0.32 s.
- torch CPU 8 threads: inference 171.1 s (RTF 0.261). NOTE: that run overlapped with my niced
  compiles and GPU runs (see PERF_LEDGER) — supporting data only.
- Rust CUDA timings so far are INVALID as evidence (contended by the concurrent CPU baseline).

## Session log
- truth pack; oracle + fixtures (15 cases × {cpu-t1, cuda-t1}); OQs resolved.
- CPU f32 forward, seam ladder; native .pth loader; CLI; production CUDA baseline.
- CUDA backend (ceef72f) — seams green; E2E judged under v2 (unapproved) → hold.
- Owner ruling: v1 binding; ladders restored to enforce v1 + G-SPEC (implemented late — was specified
  originally but missing); attainability evidence; listening pair; owner acceptance of that pair; pin.
