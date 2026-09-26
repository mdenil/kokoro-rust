# Performance Ledger — kokoro

<!-- Paste as docs/PERF_LEDGER.md on day 0. Only WINs live here — a local A/B that has not
     cleared the strict current-tree + reference gates is a PROVISIONAL_LOCAL_WIN and stays
     in NEGATIVE_EVIDENCE.md until promoted. -->

Row admission rules (ALL mandatory):
1. Ratio is vs an EXTERNAL pinned reference (never a self-relative number).
2. Thread parity (never benchmark the reference oversubscribed), allocator parity,
   OMP/MKL-class env exported and recorded.
3. Precision annotated per row (`ours-int8` vs `ref-bf16` — a raw ratio across numerics is
   meaningless without it).
4. Best-of-N warm with warmup discard; **cv% ≤ 5 or the row is refused** (noise cannot land).
   (Under an incumbent contract with dual-null verdict gates, cv% demotes to provenance —
   the nulls decide; see INCUMBENT-CONTRACT-TEMPLATE. Solo-host rows keep this refusal.)
5. stdout/output identity checked EVERY run on BOTH sides.
6. Load-inclusion bias stated (who counts model load/artifact hydration).
7. Quiet-window conditions recorded (or documented FORCE bypass — cv% is the arbiter).
8. Parity receipt re-stated (the lever's correctness proof, linked).
9. Evidence dir with raw per-run stdout/stderr/meta + SHA-256 manifest.
10. Roofline column: compute floor, memory floor, distance-above-floor (a stage >1.3× above
    floor is a named attackable lever, never an excuse).

---

## Row template

| date | stage | ours | reference | ratio | threads | precision | cv% | evidence | notes |
|---|---|---|---|---|---|---|---|---|---|
| <d> | <e2e/vision/prefill/decode-per-tok> | <ms> | <ms> | <×> | <n>/<n> | <ours-vs-ref> | <v> | `artifacts/perf/<id>/` | <load-bias, window, FORCE?> |

## Frozen bench baseline
Maintain `benches/.bench-history/baseline.json` + a guardrail script: frozen-baseline compare,
ratchet-only advance, cv>5% ineligible, posture-mismatch refusal, parity-receipt-required.
The frozen baseline is hash-pinned core evidence that moves ONLY via an explicit reviewed
ratchet; run-generated evidence is subject to the certification max-age gate (exemplar: 24h) —
the frozen baseline is not.

---

## Status notes (2026-09-26)
- Perf work was BLOCKED ON CORRECTNESS (supervisor hold) and resumed after the owner's listening
  acceptance (DISC-003), under the pinned owner-approved envelope (tests/pinned/gpu_envelope.json).
- INVALID / contended measurements (not evidence, kept for honesty):
  - First Rust CUDA bench (6.29 s) and the device-noise ABBA (A 17.30 s / B 11.80 s, 4/4 B faster)
    ran while the torch CPU baseline (8 threads) was running → contended; must be re-run quiet.
  - torch CPU baseline receipt `prod-cpu-20260926-131415` overlapped my niced `-j2` compiles and GPU
    benches; recorded as supporting data only.
- Admissible so far: production torch CUDA baseline receipts `prod-cuda-20260926-130924` (batch,
  frontend, cold) and `prod-cuda-inference-20260926-131240` (inference, cv 2.9%).

### PL-001 — SineGen excitation noise generated on device (counter-based RNG)   [2026-09-26 | WIN (local)]
- Lever: `gen_noise` kernel + `RngNoise` counter-based (splitmix64 + f32 Box-Muller), identical
  integer stream on CPU and GPU. Kill switch: `KOKORO_GPU_HOST_NOISE=1`.
- Correctness: FixedNoise (parity/fixture) path untouched — `gpu_regression_pinned` 15/15 bitwise.
  Device vs CPU noise stream max |Δ| 4.8e-7 (80.7% bit-exact; transcendental ulps); E2E product
  path device-vs-host noise rel 1.5e-6. Noise mean 0.0002 var 0.9999.
- A/B (quiet host, ABBA n=5, whole-process wall, bench --reps 2, 69 chunks): A 16.877 s (cv 4.99%)
  → B 12.153 s (cv 2.75%), 1.389×, B faster 5/5. Keep, default on.

### PL-002 — BiLSTM as one cooperative persistent kernel per sequence   [2026-09-26 | WIN (local, small)]
- Lever: `lstm_seq` (grid (H,2)×128, grid.sync per step) replaces T per-step launches; identical
  per-step arithmetic. Kill switch `KOKORO_LSTM_PERSISTENT=0`.
- Correctness: pinned envelope 15/15 BITWISE identical (optional evidence; bit identity is not the
  acceptance requirement per owner clarification 1553392534095527999).
- A/B (quiet host, ABBA n=5, whole process, bench --reps 3): 14.102 s → 13.818 s (1.021×, B faster
  4/5, cv 2.2%/2.0%); inference receipts 2.541 s → 2.477 s (evidence/rust/cuda-lstmpersist{0,1}-*.json). Keep.
