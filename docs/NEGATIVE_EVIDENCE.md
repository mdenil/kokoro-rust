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
