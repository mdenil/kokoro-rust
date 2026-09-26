# Kokoro-82M: Rust CUDA vs production PyTorch — performance report

> Status: **in-progress snapshot** (not a final acceptance gate, not optimization completion).
> Correctness status is tracked separately (PORT_STATE.md, docs/conformance/TOLERANCE_HISTORY.md):
> all stage seams pass on 15/15 fixture cases; 8 enforced end-to-end original-gate rows fail on the
> CUDA path (1 owner-accepted by listening, DISC-003; 7 open, DISC-004).

> **Which numbers are current.**
> - The WHOLE-SYSTEM BASELINE section below measures the current native binary (tree 6e306b7) and
>   later lever receipts.
> - Everything after "Benchmark policy" (the interim checkpoint, the private-chapter baseline, the
>   lever receipts) is HISTORICAL. It was measured on earlier trees, when the text path used the
>   DEV-ONLY Python bridge and batch-1 was the default.

## WHOLE-SYSTEM BASELINE — current native binary vs ORIGINAL production Python (owner #21; 2026-09-26 ~23:30)

Scope: prepared line file -> ALL per-line WAVs.
- Rust also writes JSON sidecars, a manifest and SHA-256 hashes (more work than Python).
- Python: kokoro 0.9.4 `KPipeline(lang_code='a')` per line, concatenated chunks, `soundfile.write`,
  unchanged (default torch threads = 40, CUDA). Harness: bench/system_reference.py.
- Rust: tree 6e306b7, default `synth` (native frontend, batched 8000, strict build), plus the
  owner-accepted FMA build.
- af_heart, speed 1.0, RTX 4090 (GPU 0).
- Cold = fresh process per replicate (3 replicates, interleaved across engines).
- Warm = resident process, 1 untimed warm-up pass + 3 timed passes, ×2 processes.
- Attribution = separate labelled runs (Python per-call timers; Rust per-thread timeline spans).
- No resume skips; GPU 0 idle before every run. Host load avg 4–16 from other users (recorded
  per run; not disturbed).
- Output coverage verified for every pass (a WAV per line; Rust manifests complete).
- Raw evidence: every process's stdout/stderr, timelines, identity.json, raw.jsonl, SHA256SUMS.
  WAVs were hashed, then deleted.
  - Alice: `/data/mdenil/code/kokoro-rust/evidence/system-baseline/20260926-224902-alice`
  - private chapter: `/data/mdenil/code/kokoro-rust/evidence/private/system-baseline/20260926-224902-chapter` (private; aggregates only here)

| workload | metric | production Python | Rust strict | Rust FMA | Python / Rust strict |
|---|---|---|---|---|---|
| private chapter (316 lines, 2915 s audio) | cold process wall | 67.08 s (cv 8.4%) | 24.47 s (cv 2.8%) | 26.16 s (cv 4.2%) | **2.74×** |
| private chapter | warm resident pass | 51.12 s (cv 4.9%, n=6) | 19.44 s (cv 3.0%, n=6) | 19.70 s (cv 5.7%) | **2.63×** |
| Alice ch.1 (65 lines, 656 s audio) | cold process wall | 25.16 s (cv 5.4%) | 9.94 s (cv 6.8%) | 9.91 s (cv 2.4%) | 2.53× |
| Alice ch.1 | warm resident pass | 8.10 s (cv 7.5%) | 4.70 s (cv 1.7%) | 4.80 s (cv 7.6%) | 1.73× |

Scope and noise disclosures for this baseline:
- Harness scope. The Python cold wall contains everything outside its timed body: interpreter
  start, the harness's own in-process WAV hashing for coverage, and exit. That adds 1.2–1.8 s per
  process (raw: wall_s − record.process_body_s). The Rust run's coverage hashing happened outside
  the process, in the driver; Rust's own sidecar hashes are part of the product and included.
  From now on the Python harness no longer hashes in-process: the driver hashes both engines
  externally.
- Output format is identical: both write PCM_16 24 kHz mono WAV (verified with soundfile.info on
  actual outputs). Rust also writes sidecars and a manifest.
- Noise. Ratios are provisional where cv > 5%: Python chapter cold cv 8.4%, Alice warm cv 7.5%;
  Rust FMA warm cv 5.7–7.6%.
- Host. Other users' jobs were active (load avg 4–16), including a root `zfs receive` writing to the
  same ZFS pool as the outputs from ~23:10. They were recorded, not disturbed.

Attribution, private chapter.
- Python warm pass 54.2 s = g2p 1.5 + inference 50.6 (incl. its `.cpu()` sync) + WAV write 1.7
  + rest. Cold adds import 5.5 s + load 3.2 s.
- Rust warm pass 19.9 s (timeline spans): the three stages are effectively SERIAL.
  - frontend 6.4 s on 1 thread; the GPU starves 6.25 s, because the 256-chunk batch window waits
    for the frontend;
  - GPU synth 10.1 s;
  - writer 3.7 s busy, 3.4 s of it as backpressure on the GPU. Two fsyncs per line on ZFS
    dominate.
- Rust cold = 0.64 s outside main + model load 4.1 s + frontend load 0.6 s (serial) + pass 19.1 s.
- FMA vs strict is within noise at the whole-system level; the GPU kernels are not the only
  bottleneck.

## Durability tradeoff (lever L1, fsync)
From L1 on, `synth` no longer fsyncs each WAV, sidecar and manifest; `--fsync` restores that.
- Files are still written to `<name>.partial` and atomically renamed, so after a PROCESS crash no
  reader sees a partial file.
- An atomic rename is NOT power-loss durability. After a power loss or OS crash, recently written
  files may be empty or missing.
- Resume re-verifies every WAV against the audio sha256 recorded in its sidecar, and every sidecar
  must parse and match the line, text and config. So such files are detected and re-synthesized;
  they are never trusted.
- The production Python path (`soundfile.write`) never fsyncs either.

## Benchmark policy (owner #13)
Headline = the ORIGINAL pinned production Python usage (kokoro 0.9.4 KPipeline, one line at a time,
production defaults incl. cuDNN TF32) vs the FASTEST Rust configuration (batching, host parallelism,
fusion, CUDA tuning). Unequal batching / host scheduling is intended. Matched-precision runs are
supporting diagnostics only. Deployed-wrapper overhead is not measured unless stated.

## CURRENT CHECKPOINT (2026-09-26)

**Identity.** Rust tree `5581c57` (clean), binary sha256 `3fea4c20…f5a0ae`; reference = production pins
(kokoro 0.9.4 / torch 2.12.1 CUDA 13.0 / Python 3.12.3), weights sha256 `496dba11…`; RTX 4090, driver 580.178.04.
Evidence: `/data/mdenil/code/kokoro-rust/evidence/interim-comparison/20260926-141335/`
(`raw.jsonl` every process: command, wall, rc, host state — sha256 `d7ebdf28…`; `summary.json` `0a430784…`;
`pins.json` `b03366b5…`; per-process receipts; `SHA256SUMS` `58efe06c…`; nsys capture of the Rust tree).
Regenerate the summary: `python3 bench/summarize_interim.py <evidence dir>`. Driver: `bench/interim_compare.py`.

### Warm core inference (69 identical production phoneme chunks, speed 1.0; text frontend excluded)

| voice | engine (precision) | median warm core s | replicate values s | audio s | RTF | × realtime | reference / Rust |
|---|---|---|---|---|---|---|---|
| af_heart | torch reference, production (cuDNN TF32) | 6.986 | 8.361 7.507 6.986 6.882 6.933 (cv 8.5%) | 656.175 | 0.0106 | 94× | **2.84×** |
| af_heart | torch reference, full f32 (not production) | 7.375 | 7.720 7.088 7.260 7.375 8.404 (cv 6.9%) | 656.175 | 0.0112 | 89× | 3.00× |
| af_heart | **Rust CUDA, full f32** | **2.462** | 2.487 2.461 2.464 2.462 2.461 (cv 0.4%) | 656.175 | 0.0038 | 267× | — |
| am_adam | torch reference, production (TF32) | 7.464 | 7.464 6.937 8.341 (cv 9.3%) | 636.925 | 0.0117 | 85× | **3.09×** |
| am_adam | torch reference, full f32 (not production) | 6.880 | 6.880 8.106 6.650 (cv 10.9%) | 636.925 | 0.0108 | 93× | 2.85× |
| am_adam | **Rust CUDA, full f32** | **2.415** | 2.442 2.405 2.415 (cv 0.8%) | 636.925 | 0.0038 | 264× | — |

Each replicate = median of 3 timed passes in its own process after a full warm-up pass. Both engines
produced 69 chunks with identical total audio duration per voice. **Reference-side cv 7–11% exceeds
the 5% gate → the ratios are PROVISIONAL.** Reference replicates drift with shared-host CPU load
(load avg 5–15 from other users; torch here is Python/launch-bound: ~70–80 ms per chunk almost
independent of length); the Rust replicates are flat. Conservative min-vs-min: 6.882 / 2.461 = 2.80×
(af_heart). Consistent with the earlier stand-alone reference measurement 6.973 s (cv 2.9%).

### Cold process, text → WAV files (65 corpus lines, af_heart, speed 1.0; 3 interleaved replicates)

| engine | median wall s | replicates s | breakdown (medians) | output |
|---|---|---|---|---|
| torch reference (KPipeline + soundfile, production precision) | 25.10 | 27.45 25.10 24.08 | import 5.62, model+voice load 3.16, synth+write 15.64, first audio 9.79 | 65 WAVs, 656.17 s |
| Rust CUDA CLI + **DEV-ONLY Python misaki bridge** | 14.84 (**1.69×**) | 14.73 14.84 15.28 | Rust load 3.99; synth loop (bridge G2P + inference + WAV + JSON sidecars + hashes) 4.03; remainder ≈ bridge interpreter startup | 65 WAVs, 656.2 s |
| Rust CUDA CLI, native, **pre-phonemized chunks (NOT text→WAV)** | 7.45 | 7.67 7.45 7.19 | load 4.03; synth loop 3.27 | 69 WAVs (one per chunk), 656.2 s |

Python frontend bridge alone (the part the Rust product still borrows from Python): interpreter +
misaki/spaCy startup 7.26 s, G2P of all 65 lines 0.38 s (median of 3; 69 chunks). At the time, the Rust
text path was therefore **not Python-free**; the native phoneme path was Python-free but skipped G2P.
(Historical: the bridge has since been removed. The current binary's text path is native and
Python-free, and it has not been benchmarked.)
Resident batch (model already loaded): Rust text-mode synth loop 4.03 s (from a cold process, incl.
first-call GPU warm-up, bridge G2P, WAV + sidecar writes) vs the earlier reference resident KPipeline
batch 7.41 s (in-memory, no WAV writes; 13:09 run) — different harnesses, indicative only (~1.8×).
Not measured: the deployed openclaw wrapper's own overhead.

### Where the Rust time goes (nsys, current tree, per 69-chunk pass; profiled pass 2.674 s)

Summed kernel durations 2.324 s + memcpy 0.061 s per pass in the PROFILED run (88 k launches/pass). [Corrected: an earlier version divided these by the wall time of a different, unprofiled run to claim "≈97% busy"; that comparison is invalid and was removed.]
cuBLAS SGEMM 1.466 s (63%; almost all generator convolutions issued as one GEMM per tap) ·
`lstm_seq` 0.234 s (10%; 82,132 sequential steps/pass ≈ 2.85 µs/step) · `conv_direct` 0.173 s (7%;
the stride-6 noise conv, naive kernel) · `chan_stats` 0.165 s (7%; AdaIN statistics, one block per
channel) · elementwise (AdaIN apply, adds, bias fills, …) ≈ 0.23 s (10%).

### Headroom estimate (assumption-labelled)

Analytic work model (`bench/flop_model.py`, from layer shapes): **36.4 TFLOP per pass** (generator
32.3, decoder 1.8, ALBERT 1.6, prosody/text 0.8). Current effective rate ≈ 14.8 TFLOP/s overall,
≈ 24.8 TFLOP/s inside the SGEMMs. The per-tap GEMM design moves ≈ **1.20 TB** of HBM traffic per pass
(A read + C read/write per tap), vs ≈ 0.10 TB for an implicit-GEMM convolution that reads each input
and writes each output once; at ~1.0 TB/s the per-tap design is bandwidth-bound (≥ ~1.2 s if it all
went to DRAM; the measured 1.47 s is consistent, with L2 absorbing part of it). This also explains
why im2col (NE-003, which materializes the same bytes) did not help.

Theoretical bounds (warm core; NOT expected to be reached):
- (Removed: an Amdahl bound on host/launch idle derived from summed profiled kernel durations vs a different unprofiled wall. GPU idle fraction is UNMEASURED; a trace-timeline busy/idle measurement is needed before any such bound.)
- Compute roofline, full f32 on CUDA cores (assumptions: 128 SMs × 128 FP32 lanes × 2 × ~2.7 GHz
  observed SM clock ≈ 88 TFLOP/s, 100% efficiency, FLOP model exact, elementwise/LSTM free):
  ≥ 0.41 s → ≤ ~6× over current Rust, ≤ ~17× over the production reference. Adding the sequential
  LSTM path at an assumed ≥ 1 µs/step (unmeasured minimum) gives ≥ ~0.49 s → ≤ ~5× / ≤ ~14×.
- Unknown: attainable efficiency of a custom f32 implicit-GEMM conv on this GPU (not yet measured),
  the true LSTM step-latency floor, and L2 behaviour of a fused design.

Practical range, tied to concrete untried optimizations (each must pass RB-1 bounds + seams):

| tier | levers | est. saving / pass | warm core → | over current Rust | over reference (6.99 s) | confidence |
|---|---|---|---|---|---|---|
| A | tiled `conv_direct` (−~0.15 s), multi-block `chan_stats` (−~0.13), cheaper LSTM step without grid-wide sync (−~0.1), bias/AdaIN elementwise fusions (−~0.05–0.1) | 0.4–0.5 s | ~1.95–2.05 s | ~1.2–1.26× | ~3.4–3.6× | moderate–high |
| B | custom f32 implicit-GEMM convolution for the generator (no per-tap C traffic), assuming 45–65% of f32 peak | a further 0.65–0.9 s | ~1.1–1.4 s | ~1.75–2.2× | ~5–6.3× | moderate–low (significant kernel engineering; bounded-variation numerics) |
| out of scope now | FP16/BF16 tensor cores (TF32 has no dense-rate advantage over FP32 on the 4090) | unknown | — | — | — | precision change not authorized by the owner rules |

End-to-end headroom (HISTORICAL, bridge-era; superseded by the native frontend and not re-measured):
in the Rust text→WAV cold path (14.84 s) the DEV-ONLY Python bridge startup is
≈ 7.3 s and the Rust model load ≈ 4.0 s (checkpoint parse, weight-norm, upload, CUDA context/PTX JIT,
plus a full sha256 of the 327 MB weights for the sidecars). Practical: load 4.0 → ~1–1.5 s (cache
hydrated weights / skip or cache hashing), which would put native-phoneme cold runs at ~4.5–5 s; the
text path stayed bridge-bound (~7 s) until a native G2P existed. It now exists; its cold-start and
throughput costs are unmeasured.
Warm-core gains above translate to end-to-end only through the synth-loop share (≈ 3–4 s here).


## Private chapter baseline (aggregate numbers only; completed 2026-09-26 15:40)

Workload: the approved private chapter (316 lines, sha256 8129112a…; text never in Git), 317 chunks,
phoneme lengths min 14 / median 134 / p90 280 / max 501, one line split into 2 chunks. Evidence:
/data/mdenil/code/kokoro-rust/evidence/private/chapter-baseline/20260926-145712/ (private).
Rust = strict (-fmad=false) build, batch-1, pre-PL-004 — i.e. BEFORE the later levers, with the
DEV-ONLY Python bridge for text.

**All ratios in this table are HISTORICAL and PROVISIONAL.** Variability disqualifies several rows
under the 5% cv rule:
- Cold Rust replicates are bimodal: text path with the bridge 60.29 / 29.45 / 30.28 s (cv 43.9%);
  pre-phonemized path 41.99 / 20.02 / 21.12 s (cv 44.7%). The first replicate of each was about
  2× slower, which is unexplained.
- Production cold is 70.56 / 65.66 / 63.38 s (cv 5.5%).
- Warm-core production: the af_heart replicate medians have cv 4.7%, but the inner repeat cv
  inside each replicate reaches 8.1% (am_adam: 11.0%, with only n=2 replicates).
- The raw inner per-pass samples were NOT retained: the receipts keep only n / median / cv per
  replicate, so the inner spread cannot be re-analysed.
Receipts are unchanged in the evidence directory (summary.json, raw.jsonl, per-replicate JSONs).

| scope | production Python (unchanged, TF32) | torch full-f32 (diag) | Rust CUDA | production / Rust |
|---|---|---|---|---|
| warm core, af_heart (2915 s audio) | 50.17 s (cv 4.7% across replicates, n=3; inner cv up to 8.1%) | 46.89 s | 10.95 s (cv 1.0%) | 4.58× (provisional) |
| warm core, am_adam (2837 s) | 49.01 s (cv 10.5%, n=2 → provisional) | 46.75 s | 10.71 s (cv 0.5%) | 4.58× (provisional) |
| cold file → 316 WAVs, af_heart | 65.66 s, cv 5.5% (import 5.82 + load 3.16 + synth/write 55.40) | — | 30.28 s, **cv 43.9%**, with the DEV-ONLY Python G2P bridge (bridge startup ≈6.3 s, G2P 1.3 s) | 2.17× (**provisional**; Rust cv fails the gate) |
| cold, pre-phonemized chunks (NOT text→WAV) | — | — | 21.12 s, **cv 44.7%** (317 WAVs) | — |

Scope: "production Python" = pinned kokoro 0.9.4 KPipeline through bench/bench_reference.py (the
library path production uses), NOT the deployed openclaw wrapper. Rust's total audio (2915.025 s)
equals full-f32 torch exactly; production TF32 is 0.05 s longer (duration differences somewhere).

## Lever receipts after the checkpoint (Alice 69 chunks, public; exploratory subset per owner #12)
ABBA evidence for PL-001..005 is NOT retained raw output. evidence/ab/README.txt holds summaries
transcribed from the terminal (per-arm medians, cv, ratio, wins; only some per-pair values survive).
The ABBA script wrote no files at the time. scripts/ab.sh now tees raw output for any future run.
In-process pass receipts (bench --out JSON) are retained under evidence/rust/. In-process pass (bench --reps 3):
batch-1 strict 2.465 s → +PL-004 tiled conv 2.426 s → batched(8000)+PL-004 2.298 s → +FMA (opt-in,
provisional) 2.285 s. Batching and FMA quality: OWNER-ACCEPTED by listening (owner #14, 2026-09-26);
at the time both were opt-in. Since the F4 change (commit 169c9e9), batching (`--batch-phonemes 8000`)
is the DEFAULT for `synth` and `bench`, which share the option. The historical `bench` receipts above
were batch-1 by default; reproducing them now needs `--batch-phonemes 0`. FMA remains an opt-in build
(`KOKORO_FMA=1`); the default build is strict.

**Headroom status (owner question 1553429320150810717, then PAUSED by owner priority change #15).** The
headroom section above predates batching and is STALE: its Tier A (1.95–2.05 s) and Tier B (1.1–1.4 s)
figures were single-item analytic estimates, and the measured tiled-conv win (neutral batch-1; ~5% only
when batched) was smaller than its −0.15 s forecast. Measured so far: batch-1+tiled 2.426 s → batched
2.298 s (~5.6%) → batched+FMA 2.285 s (~7.9% vs 2.465 s). Nsys captures of the current tree (batch-1 and
batched) were taken but are deliberately UNANALYSED (evidence/profiles/20260926-1640-paused/); the
single-item roofline/traffic model is not a batched bound. The refresh is superseded by the next
performance phase (owner #16): whole-system chapter file → verified WAVs, cold and resident, with
overlap-aware stage attribution — after the complete/correct single-binary milestone.

## Method (applies to the checkpoint)

- Hardware: RTX 4090 (GPU 0, `CUDA_VISIBLE_DEVICES=0`), 2× Xeon E5-2698 v4 (shared host; host load
  recorded per process in raw.jsonl). No other jobs of mine ran during measurement.
- Reference = the production pin set: kokoro 0.9.4, misaki 0.9.4, torch 2.12.1 (CUDA 13.0,
  cuDNN 9.20), Python 3.12.3, hexgrad/Kokoro-82M @ f3ff357 (weights sha256 496dba11…). Measured
  through `bench/bench_reference.py` (KModel / KPipeline harness) — this is the upstream library path
  production uses, **not** the deployed openclaw wrapper, whose own overheads are not measured here.
- Precision regimes (labelled in every table):
  - `ref-prod`: torch defaults = production (cuDNN convolutions in **TF32**, matmuls f32).
  - `ref-f32`: same, cuDNN TF32 disabled — matched full-f32 comparison, **not production**.
  - `rust`: native Rust + CUDA kernels + cuBLAS SGEMM in full **f32** (no TF32), kernels `-fmad=false`.
- Inputs: public-domain *Alice in Wonderland* ch. 1 (65 lines, bench/CORPUS.md); the 69 phoneme
  chunks production's own frontend produces from it (bench/corpus_alice_ch1.chunks.jsonl) are the
  identical inputs for the warm-core scope; voices af_heart and am_adam; speed 1.0.
- Thread budget: 8 CPU threads for both engines. GPU work CUDA-synchronized before timing stops.
- Warm core: per process, one full uncounted warm-up pass over all 69 chunks, then 3 timed passes;
  a replicate = the median of those 3; engines run in separate processes, interleaved with a
  rotating order (af_heart 5 replicates per engine, am_adam 3). Includes host→device inputs and
  audio device→host copy on both sides; excludes text frontend.
- Cold: a fresh process that imports/loads everything, synthesizes the 65 lines and writes one WAV
  per line (3 interleaved replicates). Rust text mode uses the **DEV-ONLY Python misaki bridge**
  (the Rust product has no native G2P yet), timed separately below; Rust phoneme mode is fully
  native but consumes pre-phonemized chunks.
- Noise/output variation is permitted (owner policy RB-1); both engines draw their own excitation
  noise; no bit-identity prerequisite.

## Earlier baselines (kept for continuity)
- Production torch CUDA, same corpus (13:09 run): pure inference 6.97 s (cv 2.9%, RTF 0.0106),
  resident KPipeline batch 7.41 s, cold process 25.5 s, frontend 0.32 s
  (evidence/baseline/prod-cuda-20260926-130924, prod-cuda-inference-20260926-131240).
- torch CPU 8 threads: inference 171 s (RTF 0.261) — supporting data only (contended run).
