# Frontend coverage findings (F0, 2026-09-26)

Oracle: `oracle/frontend_oracle.py` (pinned misaki 0.9.4 / spaCy 3.8.14 / en_core_web_sm 3.8.0 /
espeak-ng via espeakng-loader 0.2.4). Public fixtures (hashes):
- `fixtures/frontend/frontend_edge_cases.oracle.jsonl` sha256 3c171874… (65 self-authored lines)
- `fixtures/frontend/alice_full.oracle.jsonl` sha256 4e603df2… (1402 lines, regenerated from pinned pg11)
- private chapter dump: /data/.../evidence/private/frontend/ (never in Git; aggregates only below)

| corpus | lines | tokens | chunks (multi-chunk lines) | espeak fallback calls (lines affected) | POS/context-dependent tokens |
|---|---|---|---|---|---|
| edge cases | 65 | 768 | 67 (1) | 32 (17) = 4.17% of tokens | 135 (17.6%) |
| Alice full | 1402 | 33,575 | 1413 (10) | 111 (77) = 0.33% | 5763 (17.2%) |
| private chapter | 316 | 8,847 | 317 (1) | 88 (**72 lines = 23%**) = 0.99% | 1725 (19.5%) |

Observations
- The espeak-ng fallback (GPL-3) is rare per token but touches ~23% of the real chapter's lines.
  A large share is compounds the spaCy tokenizer leaves glued to em-dashes / quotes (e.g. public
  Alice examples: `again—before`, `course—I`, `Alice)—“and`) plus names and rare words — so any
  replacement for espeak is audible on many lines. **Owner decision needed** (options in the
  roadmap): (a) optional runtime-loaded libespeak-ng, (b) GPL-3 product, (c) native non-GPL fallback
  (e.g. a seq2seq G2P trained on the lexicons, as misaki's own TODO suggests) with measured divergence.
- ~17–20% of tokens pass through POS/context-dependent rules (a/an/the/to/used, POS-keyed entries,
  NNP handling) → the spaCy tagger must be ported (F2); a tag-free approximation would change
  pronunciations.
- No unresolved tokens (with fallback enabled) in any corpus.

Provenance (open item before any publication): misaki's us/gb gold/silver lexicons ship under the
repository's Apache-2.0; the upstream README (pinned fba1236) documents no origin for the entries.
