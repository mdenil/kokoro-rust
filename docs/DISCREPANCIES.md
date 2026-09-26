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
