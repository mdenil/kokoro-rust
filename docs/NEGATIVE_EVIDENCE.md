# Negative Evidence Ledger — kokoro

<!-- Paste as docs/NEGATIVE_EVIDENCE.md on day 0, BEFORE any engine code.
     Machine-lint this file in CI: unique ids, required field tokens, every named
     kill-switch env var must be a source literal. -->

Outcome taxonomy (the CLOSED set — a lint should refuse any other label):
- **WIN** — head-to-head measured ratio vs the REAL pinned reference + correctness proof.
- **PROVISIONAL_LOCAL_WIN** — survived a local bit-exact A/B but has NOT cleared the strict
  current-tree + reference gates. Stays here, not in PERF_LEDGER, until re-certified.
  Sweep these periodically: an unpromoted provisional win behind an opt-in flag silently rots.
- **NEGATIVE(reverted)** — measured loss/neutral; source reverted same day.
- **NEGATIVE(retained-for-proof)** — kept in-tree (flagged off) because the code IS the proof.
- **NO_EVIDENCE** — the precommitted gate could not be evaluated or was failed incoherently
  (e.g. a split verdict across workloads); nothing is claimed either way.
- **VOID** — the measurement could not have DETECTED the lever (dead flag, wrong arm, harness
  bug); not a rejection — re-run with a fixed harness.

Rules: sweep this ledger BEFORE starting any perf lever; a lever is re-attemptable only if its
do-not-retry predicate is satisfied; NEUTRAL is a revert too; record honest LOSING baselines.

---

## Entry template

### NE-<ID> — <lever name>   [<DATE> | <OUTCOME>]
- Claim id / evidence id: `<claim>` / `artifacts/perf/<issue>/` (SHA-256 manifest)
- Provenance: model commit `<hash>`, fixture sha `<hash>`, artifact sha `<hash>`
- Dispatched CPU features: `<string>`   (divergence and perf are arch-specific)
- Exact command + env: `<cmd>`
- Kill-switch state: `<ENV_VAR>=<val>` (default: <on/off>)
- Measured: <before> → <after> (<ratio>), cv% <val>
- Correctness proof: <byte-identical | argmax-exact | DISC-NNN ledgered | FAILED: how>
- Disposition: <kept-default | reverted | flag-retained>
- **Do-not-retry unless:** <explicit predicate — new silicon / new LLVM major / changed call
  shape / a native packed dot exists / deterministic EOS fallback lands first. Never "later".>
- Per-lever tally: W<i>/L<j>/N<k>
- Agent: <name>

---

## Inherited priors (pre-truth-pack — re-confirm on THIS model's shapes; not local evidence)

### NE-INH-1 — hand-rolled wide-SIMD int8 dot vs LLVM autovec  [inherited | NEGATIVE]
~5× SLOWER than autovectorized scalar in frankensearch/frankentorch; later confirmed on
franken_ocr where autovec beat even the hand-SDOT micro-tile 4.4× at m=1 (22% real e2e).
Do-not-retry unless: per-toolchain-major re-proof shows otherwise for a specific shape.

### NE-INH-2 — plain SDOT/VNNI dot vs ONNX/MLAS
Still ~1.5–2.4× behind ONNX/MLAS without register blocking — the gap a bespoke engine exists
to close. Register-block or lose.

### NE-INH-3 — un-blocked SMMLA
Load-bound (≈2 loads : 1 SMMLA), SLOWER than SDOT. "The instruction is not the lever; the
blocking is." Also: SMMLA/i8mm is HALF-RATE on Apple M-series (0.994× SDOT MACs) — prefer
SDOT on macOS; Neoverse may be full-rate.

### NE-INH-4 — accelerator-f32 (Apple AMX via Accelerate) vs CPU-int8
Bandwidth-bound loss; AMX not directly programmable (C dep). Do-not-retry unless int8.

### NE-INH-5 — naive fused tape-free forward with scalar-f32 ops
Regressed 3–10×. Framework overhead is NOT the gap; kernels-below-peak is. Fusion pays only
with every op at peak.

### NE-INH-6 — vision/encoder-tower int8
Broke end-metric catastrophically (CER 0.0094 → 0.3679): encoder outputs feed the decoder as
inputs; per-layer error compounds. Do-not-retry naive full-tower; if ever: MLP-only + cosine
gate per item.

### NE-INH-7 — int4 via unpack-to-memory
2.5× int8's memory traffic (read 0.5B packed + write/read 1B int8) → 5.8× SLOWER on the most
bandwidth-favorable tensor. Do-not-retry unless an in-register nibble→MAC dot exists.

### NE-INH-8 — f32-reordering decode attention (GEMM/int8-KV)
Real 12–15% wins but EOS timing tips on long repetitive inputs → runaway generation, end-metric
collapse from ONE item. Bit-exactness is MANDATORY for decode attention, optional for vision.
Do-not-retry unless a deterministic repetition/EOS fallback ships first.

### NE-INH-9 — SIMD/polynomial exp in the encoder path
Unit-proven 2.1-ulp accuracy still flipped greedy tokens at ≤1.1e-8 probability drift; wall
win was already retired by head-parallelism. Ulp proofs do NOT transfer to token-exactness.

### NE-INH-10 — thermal/measurement traps
Sequential A/B arms drift ~28% (35.8→46 ms) on unchanged code on heat-soaking hosts;
oversubscribed reference benches
(torch@64) fake wins; wall totals under load invert real results (0.52× for a true win);
cv%>5 rows are noise. Interleaved same-window stage pairs only.

---

## Local evidence (this model)

### NE-001 — SineGen `rand_ini` initial-phase draw is dead code  [2026-09-26 | VOID-for-parity]
- `torch.rand(1, 9)` is added to `rad_values[:, 0, :]` only (`istftnet.py:150-152`); the
  following ×1/300 linear downsample samples indices 300d+149/150 and never reads sample 0.
- Proof: `tests/parity.rs::perturbation_detected` asserts bit-identical output after perturbing
  rand_ini. Consequence: only the [S,9] Gaussian draw affects audio; harness keeps recording
  rand_ini anyway (cheap, and future model revisions might read it).
- Do-not-retry unless: upsample_scale or interpolation mode changes.

### NE-002 — End-to-end waveform comparison has low power  [2026-09-26 | limitation]
- A 1.5× scale of harmonic-0 excitation noise moves the waveform only 2.27% rel on
  s02_fox/af_heart — inside that case's own torch reorder envelope (2·1.29e-2), so G-E2E-v2
  PASSES it. The stage-isolated `har_source` seam (gate 1e-4) catches it (rel 1.44e-2).
- Rule: correctness claims rest on the seam ladder + exact discrete gates; the E2E waveform
  gate is a coarse backstop, never sufficient on its own. F0 ×1.001 is caught by E2E (rel 1.3).

### NE-003 — CUDA conv as im2col + single GEMM (K = Cin·taps)   [2026-09-26 | NEGATIVE(reverted)]
- Hypothesis: per-tap GEMM accumulation (beta=1, C read-modify-write per tap) was the generator
  bottleneck (profile: generator 67% of GPU time at ~15-17 TFLOPS f32).
- Correctness (not bit-identical): all seams PASS; enforced E2E failures 8→6; vs pinned envelope
  34 metric comparisons better / 26 worse (e.g. s01_hello/am_adam max 0.0088→0.0141); listened
  case not bitwise → would have required owner escalation. Receipt ladder-gpu-1790427458.json.
- Speed (quiet host): ABBA n=5 whole-process 14.28 s → 14.71 s (0.97×, B faster 1/5);
  inference 2.526 s → 2.630 s (receipts evidence/rust/cuda-im2col{0,1}-*.json). LOSS → reverted.
- Do-not-retry unless: a custom implicit-GEMM kernel (no im2col materialization) or a cuBLASLt
  epilogue-fused path is available; plain im2col+SGEMM is dead on this shape set.
- Tally: W0/L1/N0

## NE-004 — conv1d_igemm with a 128-out-channel tile (256 threads)   [2026-09-27 | REVERTED]
- Hypothesis: a 128×128 tile halves the input-window loads per FLOP versus 64×128 (PL-008).
- Result (bench, Alice 69 chunks, in-process forward, 3 reps each):
  - with the Cin ≤ 128 policy: 64-tile 1.918 s vs 128-tile 2.173 s;
  - with the Cin ≤ 256 policy: 1.964 s vs 2.217 s.
  The 128-tile is 13% slower. No spills; likely occupancy-bound (256 threads × 94 regs, up to
  45 KB smem per block for K = 11 → 1–2 blocks/SM), with no load/compute overlap inside a block.
- Do-not-retry predicate: a larger tile WITHOUT asynchronous (cp.async) double buffering.

## NE-005 — larger synthesis batches (--batch-phonemes 16000 / 32000)   [2026-09-27 | REJECTED]
- Private chapter, warm pass (2 rounds × 2 passes, private evidence dir):
  - 4000 → 9.20 s; 8000 (default) → 9.02 s; 16000 → 12.27 s; 32000 → 15.22 s.
- Cause of the slowdown: CUDA_ERROR_OUT_OF_MEMORY on the large chapter batches. synth_batch splits
  the batch and recomputes it, so work is wasted (stderr: "batch of 91 failed … splitting").
- Even when a batch fits (Alice, 16000 vs 8000), the per-stage GPU profile is equal (2.353 vs
  2.359 s): larger batches bring no kernel efficiency.
- Do-not-retry predicate: raising the batch size without first cutting peak activation memory, and
  even then there is no expected gain beyond ~8000 phonemes on this model.

## NE-006 — AdaIN+Snake fused into the conv1d_igemm prologue   [2026-09-27 | REVERTED; bitwise identical but slower]
- Idea: compute the conv input (AdaIN with per-item stats/style + Snake) while loading the tile,
  removing adain_apply's write + re-read.
- Result: bitwise identical WAVs (Alice 65/65), but the in-process forward was SLOWER:
  1.907 / 1.920 s vs 1.887 / 1.884 s. Each input element is loaded 2–3× (two out-channel tiles,
  plus halo overlap), so the prologue recomputes AdaIN + sinf per load and costs more than the
  memory traffic it saves.
- Kept instead: the residual epilogue only (PL-009).
- Do-not-retry predicate: a prologue fusion that recomputes transcendental activations per tile
  load, without a single-producer staging step.

## NE-007 — cp.async double-buffered conv1d_igemm   [2026-09-27 | REVERTED; neutral]
- Two smem stages filled with cp.async (zero-fill for halo / out-of-range). The dynamic smem limit
  was raised to 96 KB so the strided variant still fits.
- Bitwise identical (Alice 65/65).
- Speed:
  - forward sweep: BK 2 / 4 / 8 → 1.724 / 1.665 / 2.091 s (vs 1.677 single-buffered at BK 4);
  - sealed whole-system A/B (`/data/mdenil/code/kokoro-rust/evidence/ab/20260927-020426-L14-cpasync-alice`): 1.913 → 1.871 s median (1.023×). The ranges overlap (base
    min 1.851 < db min 1.861; cv 2.0% / 3.6%), so it is neutral within noise.
- Likely cause: the doubled smem lowers occupancy by as much as the load/compute overlap gains.
- Do-not-retry predicate: double buffering without also cutting per-stage smem (e.g. staging
  weights per tap group) or raising compute per block.

## NE-009 — sliding-window fused conv for the 256-channel generator stage (Cin ≤ 256)   [2026-09-27 | NOT DEFAULT; exceeds RB-1]
- Speed: forward 1.486 → 1.412 s (Alice 69 chunks).
- Numerics: all 135 stage seams pass (max rel 1.43e-5), but end-to-end drift vs the strict baseline
  exceeds RB-1: 11 drift violations and a new binding failure s06_long/af_heart v1. The original-gate
  fail set changed (1 new; 3 historical rows resolved).
- Why the difference from stage 1 (PL-008 had 0 RB-1 violations): stage 0 is earlier in the
  generator, so reordering differences are amplified through more layers and the F0 phase.
- Status: available via KOKORO_CONV_IGEMM_MAX_CIN=256; not approximately lossless under RB-1.
  Phase-2 / owner-listening material only; no default change.

## NE-010 — fused-conv input-channel chunk (IG_BK) sweeps, raw values   [2026-09-27 | BK = 4 kept]
All values: in-process forward, Alice 69 chunks, bench --reps 3 median; recorded to justify the
kept setting.
- Single-buffered generic kernel (PL-010): BK 2 / 4 / 8 = 1.845 / 1.766 / 1.828 s (Cin ≤ 128);
  1.926 / 1.775 / 1.836 s (Cin ≤ 256).
- cp.async double-buffered (NE-007): BK 2 / 4 / 8 = 1.724 / 1.665 / 2.091 s (≤ 128); 1.760 /
  1.668 / 2.199 s (≤ 256).
- Sliding-window kernel (PL-016): BK 2 → 1.572 s, BK 4 → 1.501 s; BK 8 with the ≤ 128 policy
  → 1.578 s. Cin ≤ 256 at BK 4 → 1.412 s is NE-009 (exceeds RB-1).

## PHASE-1 FREEZE (2026-09-27, tree 3dc5a35)
Phase 1 (approximately lossless) stopped at the owner-#22 criterion; reasons in
docs/PERFORMANCE_REPORT.md "Phase-1 stopping decision". Further gains go to phase 2 (lossy, branch).
