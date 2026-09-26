# Regression policy RB-1 (CUDA optimization levers)

Owner clarification (Misha, 1553392534095527999): bit identity is NOT required; small bounded
numerical differences are allowed; a changed sample/hash is not by itself a blocker. Bounds must be
explicit, nonzero, grounded in measured reference arithmetic sensitivity and in the listening
comparison Misha accepted, and FIXED BEFORE judging a new optimization (no ratcheting).

Fixed 2026-09-26 in `tests/pinned/regression_bounds.json` (created once; the test refuses to
overwrite). Approved baseline audio: `/data/mdenil/code/kokoro-rust/evidence/pinned-baseline/*.f32le`
(the owner-approved CUDA outputs, hash-verified against `tests/pinned/gpu_envelope.json`).

For every one of the 15 fixture cases (frozen cpu-t1 inputs + recorded noise):
1. Exact invariants: input ids, durations, sample count.
2. Stage seams ≤ 1e-4 rel-L2 (gpu_ladder), complete coverage (15 cases × 11 seams), GPU negative controls.
3. Drift of the candidate CUDA output from the approved baseline output, on rel-L2, max|Δ| and
   spectral mean|ΔdB| (G-SPEC metric), must be ≤ per-case bound =
   min( reference arithmetic sensitivity of that case  [pinned torch CPU f32 at 2/4/8 threads vs 1 thread],
        owner-accepted listened divergence            [rel 0.01522, max 0.04636, spectral 0.0797 dB] ).
4. The set of cases failing the binding original gates (v1 / G-SPEC vs the reference fixture) may not grow.
5. Bit identity to the baseline is reported as optional evidence only.

Beyond-bound drift, new binding-gate failures, or suspected audible quality loss → escalate to the
owner with raw paired samples (examples/listening_pair.rs pattern); never widen RB-1 to pass a lever.
Historical original-gate verdicts are unchanged (TOLERANCE_HISTORY.md).
Test: `cargo test --release --features cuda --test gpu_parity gpu_regression_bounded`.
