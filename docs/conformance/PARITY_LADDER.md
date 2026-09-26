# Parity Ladder — kokoro  (paste as docs/conformance/PARITY_LADDER.md)

> **The One Rule:** "The discrete output is BIT-EXACT where the reference is deterministic;
> the continuous output is held only within the MEASURED tolerance."

## §0 Determinism map (the entire safety argument — fill per artifact)

| Artifact | Reference deterministic? | Comparison |
|---|---|---|
| Preprocess tensor / tile geometry / pad | Yes | EXACT |
| Tokenizer ids / prompt ids | Yes | EXACT |
| SIMD vs scalar integer kernels | Yes (int add associative) | EXACT i32, every tier |
| Greedy tokens | Yes, over reproducible prefix | EXACT prefix |
| Per-op / per-layer activations / logits | No (fp order) | cosine / ULP table / MEASURED budget |
| End metric | aggregate | budget + tail bound (CVaR), never mean-only |
| Cross-build outputs | No (LTO reorder) | content metric, never bytes |

Every "Yes" is held to bit/value exactness; every "No" to a number we MEASURED — never guessed,
never imported from another project.

## §1 The rungs

| Rung | Compares | Tolerance | Fixture source |
|---|---|---|---|
| L0a tokenizer | ids on conformance corpus (triple-agreement) | EXACT | upstream tokenizer |
| L0b preprocess | tensor bytes, mean/std, geometry | EXACT | oracle dump |
| L0c prompt | full assembled id stream | EXACT | oracle |
| L1 per-op | kernel vs activation .npy | cos ≥ 0.9999; ULP table | oracle hooks |
| L2 per-layer | every layer hidden + every seam | cos ≈ 1.0; max-abs LEDGERED per layer | oracle .npz |
| L3 logits | pre-sampling, all positions | measured budget + argmax MUST match | oracle |
| L4 tokens | greedy ids | EXACT reproducible prefix | oracle |
| L5 e2e | final output on frozen corpus | metric budget + tail bound | frozen corpus |

## §2 The keystone (run FIRST): oracle nondeterminism floor
Run the reference twice at two thread counts over the corpus; commit the envelope (per-token
divergence rate, first-divergence position, per-logit max-abs spread). L3/L4 budgets DERIVE
from it. "A subject divergence inside the oracle's own noise is not a bug — this sentence is
the entire reason this gate runs first." If floor = 0.0 (bit-deterministic oracle): hold
parity ~exact. NEVER import a tolerance from another model/project.

## §3 Oracle split
Correctness fixtures frozen ONCE from the unmodified model on its NATIVE device; provenance
checker refuses replays from the wrong device class. CPU perf baseline separate, proven to
reproduce the fixture tokens. The bridge is test-only, never linked into the shipping binary;
Subject/Oracle identity asserted-distinct at the comparator.

## §4 Runner semantics
Rungs in order, short-circuit; rungs above first failure = `not_meaningful`; no weights →
`skip_no_model` and all_green FORCED false (fallback pointed at `/nonexistent` to prove the
native path ran). One-command scorecard receipt (machine-readable) consumed by the ship gate.
**A skipped ladder is never mistaken for a green one.**

## §5 Re-baselining rule
If the reference's precision (bf16) can't anchor "exact" rungs: exact rungs move to a
SELF-GENERATED f32 fixture set; the native-precision oracle stays the accuracy authority in a
cross-precision ledger.
