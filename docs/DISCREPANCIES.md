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
