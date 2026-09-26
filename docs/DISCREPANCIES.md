# Accepted-Divergence Ledger (DISC) — kokoro

<!-- Paste as docs/DISCREPANCIES.md on day 0. Machine-lint: unique DISC ids + exact field
     tokens. Doctrine: "A discrepancy is only recorded once its impact has been MEASURED —
     the cost of a divergence must be a real number tied to a real test." -->

Rules:
- Every accepted divergence carries a kill-switch restoring reference behavior.
- Tests for a divergence are **XFAIL, never SKIP** (a SKIP silently drops coverage).
- A widened tolerance is a DISC entry, never a silent epsilon bump.
- Divergences can be arch-specific — always record the dispatched CPU feature string.
- This ledger also records upstream BUGS you deliberately did not port (measured-superior
  ports are still divergences) and reference-implementation quirks you reproduce on purpose
  (down to the reference's sort/topk internals when routing depends on them).

---

## DISC-<NNN> — <one-line title>
- Claim id / evidence id: `<...>` / `<artifacts path>`
- Provenance: model commit `<hash>`; oracle source `<file>` sha `<hash>` lines `<a-b>`;
  fixture sha `<hash>`; artifact sha `<hash>`
- Dispatched CPU features: `<string>`
- Exact command + env: `<cmd>`
- Reference behavior: "<QUOTED source line>"
- Our implementation: `<file>:<fn>`
- Fallback / kill-switch state: `<ENV_VAR>=<val>` restores reference behavior
- Measured impact: <number(s) — include a TAIL figure for accuracy divergences, not just mean>
- Tests affected: <list — XFAIL ids>
- Resolution: <ACCEPTED | INVESTIGATING | WILL-FIX>
- Review date: <date>

<!-- Canonical divergence classes to expect (from the exemplar's 7 entries):
     1. Resampling-filter substitution (image-crate vs PIL bicubic).
     2. Quant flips a NEAR-TIED greedy token.
     3. f32 ITSELF flips near-ties via summation-order drift (even f32-vs-bf16 forks).
     4. Post-deterministic-prefix f32-vs-bf16 greedy fork ("correct math, not wrong math").
     5. Quant forks on ONE degraded input with no ground truth either way (accept + retain
        the f32 artifact as the bit-exact reference).
     6. Reference-runtime internals reproduced on purpose (torch topk slot-permutation).
     7. Upstream bug deliberately not ported (measured-superior; still ledgered). -->

---

## DISC-001 — STFT phase branch flips where the reference's imaginary part is rounding noise
- Evidence: `tests/diag_source.rs` (bin 9, frame 0 on s01_hello/af_heart); ladder column
  `stft.phase [flips=N]` (0–4 flips per case out of 7e4–9e5 elements).
- Reference behavior: `torch.stft(..., center=True)` (MKL r2c) then `torch.angle`
  (`kokoro/istftnet.py:90-94`). Frame 0 is exactly even-symmetric (reflect pad + symmetric
  periodic Hann), so its imaginary parts are mathematically 0; the reference emits ±7e-9
  rounding noise whose SIGN decides +π vs −π when re < 0.
- Our implementation: `src/vocoder.rs:stft` (direct 20-point DFT, f64 accumulation).
- Measured impact: seam compared by wrapped angular distance: rel ≤ 3.3e-6 on all 15 cases;
  flips reach the waveform only through noise_convs and are inside the E2E evidence.
- Kill-switch: none needed (matching MKL's rounding bits is not achievable portably).
- Resolution: ACCEPTED (ill-conditioned by construction, not a semantic difference).
- Review date: 2026-10-26

## DISC-002 — s01_hello/am_adam end-to-end peak error above 2× per-case reorder floor
- Evidence: ladder receipt `evidence/ladder/ladder-cpu-t1-1790423348.json` (+ latest);
  `fixtures/floor_per_case.json`; `cargo test --release --test parity truth_distance`.
- Measured: subject vs torch-t1 max|Δ| 1.388e-2 vs gate 2·6.174e-3 = 1.235e-2 (rel 6.84e-3
  within its 9.70e-3 gate; corr 0.99998). All 17 stage seams pass; durations exact.
- Attribution: substituting the oracle F0/N curves → max 4.3e-4 (32× smaller), i.e. the whole
  E2E gap is F0-integration amplification of a 2.9e-7-relative F0 difference.
- Truth analysis (f64 oracle, `oracle/gen_f64.py`): subject max-to-truth 1.368e-2 is SMALLER
  than every torch f32 reorder (1.374–1.394e-2); subject rel-to-truth 1.098e-2 is 1.18× the
  worst torch reorder (9.34e-3). Subject F0 rel-to-truth 2.89e-7 vs torch 2.95e-7.
- Resolution: INVESTIGATING (reported as a FAIL of precommitted G-E2E-v2; not re-gated).
- Review date: 2026-10-26

## DISC-003 — OWNER-ACCEPTED: s02_fox/am_adam/s1.0 Rust-CUDA waveform (listening review)
- Owner decision: Misha, 2026-09-26 (relayed by Hermes; Discord 1553386472906694677), after
  listening to the raw pair: "These are indistinguishable to me I am happy with this."
- Scope: THIS case only (text "The quick brown fox jumps over the lazy dog.", voice am_adam,
  speed 1.0, frozen cpu-t1 fixture noise/style/ids). Not bit identity; not an audition of any
  other case; not permission for any tolerance relaxation.
- Artifacts: `evidence/listening/manifest.json` (sha256 `322f8013379201eefa2ea1084102dd48280b0b4a16df4fafb3844863aa0fa721`),
  `worst-peak-reference.wav` (pinned reference, CPU 1 thread; sha256 `8aeb4c6c2332a17851cae112c50de006b973cb2dc69bf090e186a5b3628b3b4f`),
  `worst-peak-rust-cuda.wav` (native CUDA full f32; sha256 `aa5de4d4c6e000e3398a56ba3048a3f717284340728b8dcbf09075976e547dea`).
  Reproduced receipt `ladder-gpu-1790425822.json` bitwise (exporter `examples/listening_pair.rs`).
- Measured: rel 0.01522, max 0.04636 (binding gate 0.033 → v1 FAIL), corr 0.99988, 82,800 samples.
- Regression fixture: `tests/gpu_parity.rs::gpu_regression_pinned` requires the current CUDA
  output for this case to stay BITWISE equal to `worst-peak-rust-cuda.wav`; any change escalates.
- Resolution: ACCEPTED BY OWNER (listening). v1 row still reported as FAIL in receipts.

## DISC-004 — Unaccepted enforced E2E failures on the CUDA path (not auditioned)
- Receipt `ladder-gpu-1790426627.json`: v1 max FAIL on s02_fox/am_adam/s0.8 (0.0398),
  s03_moon/am_adam (0.0353), s04_alice/am_adam (0.0411), s05_word/am_adam (0.0395); G-SPEC FAIL
  on s02_fox/af_heart/s0.8 (0.147 dB), s03_moon/af_heart (0.143 dB), s06_long/af_heart (0.202 dB)
  vs gate 0.1395 dB. All 150 stage seams PASS on all 15 cases; ids/durations/sample counts exact.
- Evidence (docs/conformance/TOLERANCE_HISTORY.md #9): exact-f64 reference fails v1 on all 7;
  production torch CUDA fails v1 on 6/7 (passes s05_word/am_adam).
- Resolution: OPEN — failures stand; escalate to owner (audition or other decision). Not waived.
- Review date: 2026-10-03

(DISC-002 note, 2026-09-26: the CPU path is now a supporting oracle/debug baseline per owner
priority; its differences stay documented here and are not release blockers for the CUDA product.)
