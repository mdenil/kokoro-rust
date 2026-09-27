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
- EVIDENCE CAVEAT: rule 9 (raw per-run output + SHA-256 manifest) was NOT met for the PL-001..PL-005
  ABBA runs. Only terminal summaries were transcribed (evidence/ab/README.txt), plus the in-process
  bench JSON receipts in evidence/rust/. Future ABBA runs go through scripts/ab.sh, which tees the raw output.
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

### PL-003 — B1 batched forward (length-bucketed, ragged gap layout)   [2026-09-26 | WIN ~5%; quality OWNER-ACCEPTED (owner #14); opt-in pending default choice]
- `--batch-phonemes N --batch-items M` (default 0 = batch-1). Exploratory (Alice 69 chunks, in-process
  pass): before PL-004 batching was 2.40–2.78 s vs batch-1 2.465 s (bigger batches SLOWER: naive
  conv_direct lost L2 reuse, 0.17 → 0.57 s/pass); after PL-004: 2.30–2.32 s vs 2.43 s (~5%).
- Correctness: masking/noise/style negative controls detected; F0 batch-vs-single rel ≤ 1.6e-6
  (reorder level); waveform drift exceeds RB-1 on several cases (phase amplification); distance to
  the reference equivalent to batch-1 (mean rel .0121 vs .0118; binding-gate fails 5/15 vs 8/15);
  worst peak s04_alice/am_adam 0.041 → 0.071 vs reference ESCALATED to owner (listening triple in
  evidence/listening/batch-worst-s04_alice_am_adam). OWNER-ACCEPTED 2026-09-26 ("All of these are fine",
  1553428873020973171; candidate sha256 d90a653a…9d4f, manifest b30607c7…2034). Hold removed; default
  to be chosen on actual speed + remaining functional validation (after the single-binary milestone, owner #15).

### PL-004 — tiled direct conv (shared-memory input window, 16 out-channels per block)   [2026-09-26 | WIN]
- Same per-output arithmetic order as conv_direct → BITWISE identical (gpu_regression_bounded 15/15).
  Kill switch `KOKORO_CONV_TILED=0`.
- ABBA n=5 whole process: batch-1 14.01 → 14.06 s (0.997×, 2/5: neutral); batched(8000) 15.19 → 13.70 s
  (1.109×, 5/5, cv 2.9%/0.5%). Keep (enables batching; neutral single-item).

### PL-005 — FMA contraction in CUDA kernels (drop -fmad=false; owner #11)   [2026-09-26 | speed WIN; quality OWNER-ACCEPTED (owner #14); opt-in pending default choice]
- Opt-in build `KOKORO_FMA=1` (default stays strict -fmad=false, the owner-accepted baseline). Engine identity
  records the rounding mode (`cuda f32 kernels=fma|strict`), so resume never mixes builds.
- Quality: all 135 GPU stage-seam rows pass; mean distance to the reference similar (rel .0123 vs
  .0118, spectral .096 vs .090 dB). Binding-gate FAILURE SET CHANGED (count 8 → 8 rows): s03_moon/
  am_adam G-SPEC NEWLY FAILS; s05_word/am_adam v1 resolves (ladder-gpu-1790435169 vs -1790426627).
  Authoritative RB-1 vs the owner-accepted strict baseline FAILS: 9 drift violations + the new
  (s03_moon/am_adam, G-SPEC) failure. Worst peak s03_moon/am_adam 0.0542 vs strict 0.0353
  (rel .0198 vs .0149) → listening triple evidence/listening/fma-worst-s03_moon_am_adam, OWNER-ACCEPTED 2026-09-26
  (candidate sha256 c6a01bc2…1e6e, manifest 895d2822…81ef). The raw RB-1 failures above stay recorded
  (accepted variation, not erased); the strict baseline remains the regression reference.
  A separate FMA snapshot exists only as a labelled PROVISIONAL diagnostic (never parity/approval).
- Speed: ABBA n=5 whole process 13.948 → 13.730 s (1.016×, 5/5); batched in-process 2.318 → 2.285 s.

### PL-006 — whole-system pipeline: parallel frontend + parallel writers + fsync off + concurrent load   [2026-09-27 | KEEP]
- Measured bottleneck (whole-system baseline, owner #21): the Rust warm pass was serial.
  - frontend 6.4 s on 1 thread, with the GPU starved 6.25 s behind the 256-chunk batch window;
  - GPU 10.1 s;
  - writer 3.7 s, dominated by two fsyncs per line on ZFS;
  - cold start: model load, frontend load and weight hashing all serial.
- Lever (one structural change to the host pipeline; GPU numerics untouched):
  - `--prep-threads` (default half the CPUs, at most 16) with an in-order sequencer and a bounded
    look-ahead window (4×threads; stop flag on any exit);
  - `--write-threads 4`;
  - fsync off by default (`--fsync` restores; durability tradeoff in PERFORMANCE_REPORT);
  - model load concurrent with frontend load; weight sha256 on a helper thread.
- Correctness:
  - WAVs bit-identical to the pre-change binary (Alice, 65/65, 2 passes);
  - cli_linefile 5/5 and cli_text_native 7/7, incl. the private chapter, the fuzz corpus through
    the binary and the 15 output negative controls.
  - The exec audit was fixed to count strace "resumed" continuation lines once: parallel ffmpeg
    execs produced them, and it was a test artifact, not an extra process.
- A/B, sealed (`/data/mdenil/code/kokoro-rust/evidence/ab/20260926-235743-L1-sealed-alice`): Alice, 4 interleaved rounds; identities, host/GPU state and coverage digests
  per run.
  - warm pass: base (commit 6e306b7) 4.766 s → new 2.575 s (**1.85×, PROVISIONAL**: the base warm
    process medians are 11.409 / 4.809 / 4.722 / 4.721 s, cv 51.9% from one outlier, kept; new cv 1.6%);
  - cold process: 9.749 s → 5.504 s (**1.77×**; cv 0.9% / 4.1%);
  - same-binary single-worker ablation: warm 5.037 s, cold 7.410 s.
  - Earlier exploratory rounds under a concurrent root `zfs receive`: fsync amplified write
    stalls up to 27 s; with fsync off, the remaining fsync (on the manifest) stalled 2–4 s, so the
    manifest now follows `--fsync` too.
- Baseline binaries:
  - $KOKORO_DATA/bin/kokoro-6e306b7 (sha256 3fdd8374…, used in the A/B);
  - target/baseline-6e306b7/release/kokoro (b1bb54cd…).
  They are the same commit built from different checkout paths; debuginfo paths make the builds
  not byte-reproducible across locations.
- Identity addendum (audit):
  - the measured "new" binary sha256 c1815fa2… was NOT retained (later builds overwrote target/);
  - its source is identical to commit f20090a: tree 1eb73fc + the uncommitted src/cli.rs,
    engine.rs and wav.rs edits, committed unchanged as f20090a;
  - ab_synth now copies every measured binary to $KOKORO_DATA/bin/kokoro-<sha12>;
  - the L1 evidence dirs now carry SHA256SUMS; the regression-suite log is archived in the
    sealed dir (L1_regression_suites.log).

### PL-007 — cold-start load: ring SHA-256 + contiguous .pth fast path   [2026-09-27 | KEEP]
- Measured (examples/load_breakdown.rs; this host is a Broadwell E5-2698 v4 without SHA-NI):
  - weight sha256 2.6–2.8 s with the pure-Rust sha2 (no SHA-NI path) — the longest load item,
    even on its helper thread;
  - load_pth 1.0–1.25 s, mostly a per-element strided copy of 82M floats.
- Lever:
  - SHA-256 via ring (assembly; 0.92 s for the 327 MB file). tests/hashing.rs checks it is
    byte-identical to sha2 on 14 lengths and to the known file hash 496dba11…
  - contiguous tensor views are converted straight from the zip bytes (load_pth → 0.48 s); the
    generic strided walk is kept for other views. tests/native_load bitwise vs reference: pass.
  - Also in this commit: the prepare-stage logic moved to crate::ordered with unit tests (see
    below).
- Correctness:
  - WAVs bit-identical (Alice 65/65) and model_sha256 unchanged;
  - regression suites: see the commit.
- A/B, sealed (`/data/mdenil/code/kokoro-rust/evidence/ab/20260927-001711-L2-load-alice`): cold 5.463 → 4.704 s (**1.16×**, cv 5.2% / 2.4%); warm 2.586 → 2.548 s
  (1.015×, neutral as expected).
- crate::ordered (bounded ordered parallel map): unit tests cover a delayed first item (no
  run-ahead beyond the window), sink cancellation, error propagation, and empty/single-thread
  inputs, plus a negative control (an unbounded window DOES run ahead).
  - Found and fixed on the way: an f() error could be raised before the earlier items were
    emitted. It surfaced in 4/20 runs. Errors are now raised at their ordered position (0/30
    failures afterwards).

### PL-008 — fused implicit-GEMM dilated conv1d (Cin ≤ 128)   [2026-09-27 | KEEP; approximately lossless]
- Measured bottleneck: the batched GPU forward, per-stage profile (KOKORO_PROFILE scopes added to
  forward_batch). Generator stage 1 (128 ch, 120× frame length) was 58% and stage 0 (256 ch) 22%.
  It was dominated by CUTLASS SIMT SGEMM, issued as one GEMM per conv tap: each tap re-reads and
  re-writes the full output, so it is memory-bound at K = Cin = 128.
- Lever: kernel `conv1d_igemm`.
  - 64 out-ch × 128 time tile, 128 threads × (8×8) register block; the input window with halo
    for 8 input channels is loaded once per chunk; all taps accumulate in registers; the output is
    written once with bias.
  - Explicit `__fmaf_rn`, like the cuBLAS path it replaces; the strict build's -fmad=false still
    governs every other kernel. Without explicit FMA the kernel was only 2.7% faster.
  - Shape policy: Cin ≤ 128 only. Measured: all shapes 1.942 s, ≤256 1.913 s, ≤128 1.888 s vs off
    2.291 s. The wider layers are still better on cuBLAS.
  - Kill switch KOKORO_CONV_IGEMM=0; KOKORO_CONV_IGEMM_MAX_CIN for A/B.
- Correctness (summation order differs; approximately lossless):
  - all 135 GPU stage seams pass (max rel 1.42e-5, gate 1e-4);
  - original-gate fail set IDENTICAL (the same 8 historical rows, ladder-gpu-1790465810);
  - RB-1 vs the strict baseline: 0 drift violations (drift rel ≈ 6–9e-6), no new binding-gate
    failures;
  - GPU negative controls detected;
  - batched-vs-reference diagnostic unchanged to 4 digits (mean rel 0.0121 batched / 0.0118
    single; gate fails 5/15 / 8/15);
  - the batch-vs-single RB-1 test fails exactly as before (accepted PL-003);
  - cli_linefile 5/5 and cli_text_native 7/7 incl. the private chapter and the fuzz corpus.
- A/B (same binary, kill switch; sealed `/data/mdenil/code/kokoro-rust/evidence/ab/20260927-004628-L8-igemm-alice`): Alice warm pass 2.567 → 2.137 s (**1.20×**, cv
  0.6% / 1.2%); cold 4.862 → 4.532 s (1.07×, cv 8.4% / 7.1% → provisional). In-process forward
  (bench, Alice 69 chunks): 2.291 → 1.888 s.

### PL-009 — residual epilogue in the fused conv (snake blocks), mask skip   [2026-09-27 | KEEP, marginal; bitwise identical]
- In each Snake residual step, conv2 now accumulates into x in its epilogue (x + conv, the
  add_inplace order). The mask before the snake convs is skipped because the AdaIN output already
  has zero gaps.
- Correctness:
  - WAVs BITWISE identical to the unfused path (Alice 65/65);
  - batching / mapping negative controls still detected;
  - the batch-vs-single RB-1 test fails exactly as before (accepted PL-003).
- Speed:
  - forward 1.880 / 1.914 → 1.856 / 1.838 s;
  - sealed whole-system A/B (`/data/mdenil/code/kokoro-rust/evidence/ab/20260927-010949-L9-resfuse-alice`): Alice warm 2.149 → 2.101 s (1.023×; the "on" arm has one
    2.331 s outlier, cv 5.6%). Marginal.
- Also tried and reverted: the AdaIN+Snake prologue fusion (NE-006, slower).

### PL-010 — conv1d_igemm input-channel chunk 8 → 4 (occupancy)   [2026-09-27 | KEEP; bitwise identical]
- Hypothesis: the fused conv is occupancy-bound; ~27 KB smem/block at K = 11 allows ~3 blocks/SM.
- Sweep (forward, Alice 69 chunks, 3 reps):
  - IG_BK = 2 → 1.845 s; 4 → 1.766 s; 8 → 1.828 s.
  - Cin policy 256 vs 128 at BK = 4: 1.775 vs 1.766 s, so 128 is kept.
- The chunk only regroups the input-channel loop; the accumulation order (ci-major, tap-minor) is
  unchanged, so WAVs are bitwise identical to BK = 8 (Alice 65/65).
- The chunk size is now a build knob (KOKORO_IG_BK, default 4), passed by build.rs to both nvcc
  (-DIG_BK) and the Rust launcher, so they cannot disagree.
- Sealed A/B (`/data/mdenil/code/kokoro-rust/evidence/ab/20260927-011907-L10-igbk4-alice`; PL-009 binary bd4e0cbd… vs 5dc9783e…): Alice warm 2.077 → 2.022 s (1.027×).

### PL-011 — smaller first batch window (--batch-first-window 32)   [2026-09-27 | KEEP, marginal]
- The GPU thread waited for a full 256-chunk window before its first flush, so on short files it
  idled while the whole frontend ran.
- Now the first flush of a pass happens at 32 chunks; later flushes still use 256.
- Batch composition changes → audio changes within the accepted batching variation (PL-003):
  - durations and sample counts are exact (batched vs batch-1 structure test);
  - integrated suites pass incl. the private chapter and the fuzz corpus.
- A/B:
  - Alice (`/data/mdenil/code/kokoro-rust/evidence/ab/20260927-012259-L11-firstwindow-alice`) warm: fw0 2.036 → fw16 1.969, fw32 1.968 (1.035×), fw64 2.010 s;
  - private chapter: fw0 8.566 → fw32 8.557 s (neutral; fw64 8.622).
  Gain on short files, neutral on the chapter.

### PL-012 — strided implicit-GEMM conv for the generator noise conv (stride 6, k 12)   [2026-09-27 | KEEP; approximately lossless]
- Measured: conv_direct_tiled was ~0.1 s per Alice pass (6% of GPU time) at ~0.7 TFLOP/s for the
  stage-0 noise conv (Cin 22 → 256, k 12, stride 6): 64-thread blocks with 34 KB smem, ~2 blocks
  per SM.
- Lever: `conv1d_igemm_s`, a STRIDED instantiation of the fused kernel. It reads the input
  window at t·stride + k·dil and allows Tout ≠ T. Kill switch KOKORO_CONV_IGEMM_STRIDED=0. The
  stride-1 kernels are separate instantiations and compile to the previous code.
- Correctness:
  - the switch-off path is bitwise identical to PL-010 (Alice 65/65, first window 0);
  - strided path: 135 GPU seams pass (max rel 1.41e-5); original-gate fail set IDENTICAL; RB-1
    0 drift violations, no new binding failures; GPU negative controls detected;
  - batched-vs-reference diagnostic unchanged (0.0121 / 0.0118; 5/15 / 8/15);
  - integrated suites pass; the accepted batch-vs-single RB-1 test fails as before.
- Speed: forward 1.768 / 1.777 → 1.676 / 1.677 s; sealed whole-system A/B (`/data/mdenil/code/kokoro-rust/evidence/ab/20260927-013937-L12-strided-alice`) Alice warm
  1.964 → 1.861 s (**1.055×**, cv 1.3% / 0.6%).
