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
