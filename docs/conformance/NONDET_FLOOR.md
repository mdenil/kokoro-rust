# Reference nondeterminism floor (Phase 0 gate)

Command: `CUDA_VISIBLE_DEVICES=0 $VENV/bin/python oracle/nondet_floor.py`
Raw: `/data/mdenil/code/kokoro-rust/fixtures/nondet_floor.json` (pins asserted in-run).
Case: "The quick brown fox jumps over the lazy dog." / af_heart / speed 1.0 / 78,000 samples,
reference waveform RMS 0.0472.

## Measured (2026-09-26, pinned env, RTX 4090 GPU0)

| comparison (identical frozen noise) | max\|Δ\| | RMS(Δ) | RMS(Δ)/RMS(ref) | pred_dur |
|---|---|---|---|---|
| cpu-t1 run-to-run (×3) | 0 | 0 | 0 | identical |
| cpu-t8 run-to-run (×3) | 0 | 0 | 0 | identical |
| cuda run-to-run (×3) | 0 | 0 | 0 | identical |
| cpu-t1 vs cpu-t8 | 1.65e-2 | 4.38e-4 | 0.93% | identical |
| cpu-t1 vs cuda (TF32 convs, prod default) | 7.06e-2 | 2.67e-3 | 5.65% | identical |
| **free-running noise**, seed 1 vs seed 2 (cpu-t1) | 1.19e-1 | 4.46e-3 | 9.45% | identical |

Across all 16 corpus fixtures: input ids and integer durations are IDENTICAL cpu-t1 vs cuda.

## Interpretation → tolerance derivation

1. The reference is bit-repeatable within a fixed (device, threads) config once the RNG draws
   are frozen. So the frozen-noise harness is sound; noise is not "measurement noise".
2. Merely changing torch's CPU thread count (reduction order) moves the waveform by 0.93% RMS
   with 1.65e-2 peak. This is the *reference's own f32 reordering floor*: a correct f32 subject
   with a different reduction order is expected to land at roughly this distance, not at 0.
   The large peak-vs-RMS ratio reflects phase-sensitive amplification (sin/cumsum of large
   phases, exp(spec)) in the vocoder.
3. Production CUDA differs from CPU by 5.65% RMS because cuDNN TF32 convolutions are enabled by
   default. That is the production incumbent's own precision; it is NOT a correctness anchor.
4. Free-running noise is 9.45% RMS: parity is only meaningful with injected noise.

## Gates (frozen BEFORE any subject result exists)

Discrete (EXACT): input_ids, pred_dur (every element), sample count.

Continuous, Rust CPU f32 vs oracle cpu-t1 with frozen noise, per case:
- G-WAVE-RMS: RMS(Δ)/RMS(ref) ≤ 2 × (cpu-t1 vs cpu-t8 floor) = **1.9%**
- G-WAVE-MAX: max|Δ| ≤ 2 × 1.65e-2 = **3.3e-2**
- G-CORR: Pearson correlation ≥ **0.9995** (floor pair: to be recorded by the comparator)
- G-SPEC: log-magnitude STFT (n_fft 1024, hop 256) mean |Δ dB| ≤ 2 × floor-pair value
  (computed by the same comparator on the t1/t8 pair; recorded in the ladder receipt)

Per-seam (fed the oracle's exact input tensor, stage-isolated): relative L2 ≤ 1e-4 for
linear-algebra stages (ALBERT, LSTMs, convs), and per-seam max|Δ| ledgered. Isolated-stage
error should be far below the whole-pipeline floor; if a seam cannot meet 1e-4 the cause is
investigated and ledgered in DISCREPANCIES.md — the gate is not widened.

The factor 2× is the analysis margin for a subject whose reduction orders differ from BOTH
reference configs (the t1/t8 pair measures one such reorder; ours is another). Any gate
change requires a DISCREPANCIES entry and re-measured floor evidence.

Rust CUDA (later) vs oracle cuda: same discrete gates; continuous gates re-derived from a
CUDA-specific floor measurement (TF32 on/off pair) before any GPU claim.
