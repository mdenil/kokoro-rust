# Tolerance history and verdicts (append-only; receipts are never rewritten)

Receipts live in `/data/mdenil/code/kokoro-rust/evidence/ladder/` (immutable). Times 2026-09-26.

## Binding contract (owner ruling, 2026-09-26)
The ORIGINAL precommitted gates are binding: per stage-isolated seam rel-L2 ≤ 1e-4; input
ids / durations / sample count exact; E2E waveform vs the frozen pinned-reference cpu-t1
fixture with identical injected noise: rel-RMS ≤ 0.019, max|Δ| ≤ 0.033, corr ≥ 0.9995; and the
original spectral requirement G-SPEC. No widening, per-case relaxation, dropped
cases/seams/metrics, baseline substitution, precision change, post-processing or waivers.
G-E2E-v2 confers NO acceptance. The ladders enforce v1 + G-SPEC as assertions.

## Chronology

| # | When | Commit / receipt | Event | Verdict as produced |
|---|---|---|---|---|
| 1 | 12:28 | `0693219` NONDET_FLOOR.md | Original gates written BEFORE any Rust subject existed. Floor measured on ONE case (s02_fox/af_heart): cpu-t1 vs cpu-t8 0.93% rel, 1.65e-2 max; gates = 2× → 1.9%, 3.3e-2, corr 0.9995; G-SPEC = 2× the t1/t8 pair mean-|ΔdB| (stated, **not implemented until #8**). | — |
| 2 | ~12:45 | `ladder-cpu-t1-1790422984` | First CPU run, one case. | stft.phase + har_source seams FAIL → fixed (FMA interpolation semantics; wrapped angle metric DISC-001). |
| 3 | 12:49 | `ladder-cpu-t1-1790423348` | CPU, 15 cases. | All seams PASS; **E2E v1 FAIL 5/15** (am_adam cases + s06, mostly max). |
| 4 | 12:52 | `76ce93f` NONDET_FLOOR Rev 1 | **G-E2E-v2 (per-case 2× reorder floor) written AFTER seeing #3's failures.** It was committed before per-case floors were measured, but it was a post-failure revision. | — |
| 5 | 13:07 | `ladder-cpu-t1-1790424439`, `fe895f3` | CPU, v2 enforced (v1 rows were then forced `pass=true` "info" — a presentation error, since corrected). | v2 14/15 (s01_hello/am_adam FAIL, DISC-002); **v1 still 5/15 FAIL**. |
| 6 | 13:30 | `ladder-gpu-1790425822`, `ceef72f` | First full GPU ladder, v2 gate. | v2 15/15; **v1 5/15 FAIL** (max: s02 am 0.8/1.0, s03 am, s04 am, s05 am). Recomputed independently by supervisor. |
| 7 | ~13:38 | supervisor hold + owner ruling | v2 not approved; v1 binding; perf blocked. | Correctness UNACCEPTED. |
| 8 | ~13:50 | `ladder-gpu-1790426627` (this tree) | Ladders now ENFORCE v1 + G-SPEC (literal reading fixed before first evaluation: Hann 1024 / hop 256, no padding, dB=20log10(max(|X|,1e-5)), mean |ΔdB|, floor pair = reference cpu-t1 vs cpu-t8 on s02_fox/af_heart → gate 0.139485 dB). Real FAIL flags; v2 printed as UNAPPROVED diagnostic only; coverage asserted (15 cases × 11 seams). | All 135 stage-seam rows (9 seams × 15 cases) PASS [corrected 2026-09-26: earlier text said 150]; **E2E enforced: 8 FAIL** — v1 max ×5 (am_adam: s02 0.8, s02 1.0, s03, s04, s05), G-SPEC ×3 (s02_fox/af_heart/0.8 0.147 dB, s03_moon/af_heart 0.143, s06_long 0.202). |
| 9 | ~13:45 | `fixtures/attainability.json` | Reference-only waveforms vs the SAME cpu-t1 fixture under the v1 gates. | exact-f64 reference 6/15; torch CPU t2/t4/t8 10/9/9; **production torch CUDA (TF32) 3/15**; torch CUDA full-f32 6/15; Rust CUDA 10/15; Rust CPU 10/15. |
| 10 | ~13:55 | owner listening verdict | Misha auditioned `evidence/listening/` pair for s02_fox/am_adam/s1.0 (the worst-peak case) and accepted it: "These are indistinguishable to me I am happy with this." | DISC-003 (owner-accepted, this case only). Perf hold lifted. |

| 11 | ~14:05 | `ladder-cpu-t1-1790427009` | CPU (supporting oracle/debug path) under enforced v1 + G-SPEC. | All seams PASS (15 cases × 16 required seams); **E2E enforced: 6 FAIL** — v1 ×5 (s02 am 0.8/1.0, s03 am, s04 am, s06), G-SPEC ×1 (s06 0.2 dB class). |
| 12 | ~14:10 | `tests/pinned/gpu_envelope.json` | Owner-approved CUDA baseline pinned (DISC-003): listened case bitwise = accepted WAV; per-case metrics pinned exactly; any regression escalates. | 15/15 bitwise identical to pin at pin time. |

| 13 | ~16:05 | `ladder-gpu-1790435169` (FMA build) | Owner #11 FMA exploration. | 135 stage seams PASS; binding-gate failure SET changed (8 → 8): s03_moon/am_adam G-SPEC NEW FAIL, s05_word/am_adam v1 resolved. Authoritative RB-1 (strict baseline): FAIL (9 drift + 1 new gate failure). Provisional; escalated for listening. Default build reverted to strict. |
| 14 | ~15:50 | B1 batched forward | Batch vs single / vs reference. | Batch-vs-single RB-1 drift exceeded on several cases; batched peak s04_alice/am_adam 0.0706 vs single 0.0411 vs reference; provisional, escalated (listening triple). Batching opt-in. |

## Correction of wording
The reference is **exactly repeatable** for a fixed (device, thread count, noise) configuration
(0.0 difference, NONDET_FLOOR.md table). The thread-count and device differences are
reduction-order sensitivity of a deterministic program, NOT run-to-run noise; earlier uses of
"nondeterminism floor" for them are inaccurate and are superseded by this note (file name kept
for link stability).

## Current status of the enforced E2E rows (GPU, receipt #8)
- s02_fox/am_adam/s1.0: FAIL v1 max (0.0464) — **OWNER-ACCEPTED by listening (DISC-003)**.
- 7 other enforced failures: FAIL, **not auditioned, not accepted** (DISC-004). Evidence #9
  evaluated the v1 waveform gate only (not G-SPEC): for all 7 of these cases the exact-f64
  reference fails v1 against the same fixture; the pinned production torch CUDA reference fails
  v1 on 6 of the 7 and PASSES on s05_word/am_adam (max 0.026), where Rust CUDA fails (max 0.0395).
  G-SPEC attainability for reference waveforms has not been evaluated yet.
