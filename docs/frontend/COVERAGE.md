# Frontend coverage findings (2026-09-26)

Oracle: `oracle/frontend_oracle.py` (pinned misaki 0.9.4 / spaCy 3.8.14 / en_core_web_sm 3.8.0 /
espeak-ng via espeakng-loader 0.2.4). Public fixtures (hashes):
- `fixtures/frontend/frontend_edge_cases.oracle.jsonl` sha256 3c171874… (65 self-authored lines)
- `fixtures/frontend/alice_full.oracle.jsonl` sha256 4e603df2… (1402 lines, regenerated from pinned pg11)
- private long-form text dump: under the test data root's `evidence/private/frontend/` (never in Git; aggregates only below)

| corpus | lines | tokens | chunks (multi-chunk lines) | espeak fallback calls (lines affected) | POS/context-dependent tokens |
|---|---|---|---|---|---|
| edge cases | 65 | 768 | 67 (1) | 32 (17) = 4.17% of tokens | 135 (17.6%) |
| Alice full | 1402 | 33,575 | 1413 (10) | 111 (77) = 0.33% | 5763 (17.2%) |
| private long-form text | 316 | 8,847 | 317 (1) | 88 (**72 lines = 23%**) = 0.99% | 1725 (19.5%) |

Observations
- The espeak-ng fallback (GPL-3) is rare per token but touches ~23% of the private long-form text's lines.
  A large share is compounds the spaCy tokenizer leaves glued to em-dashes / quotes (e.g. public
  Alice examples: `again—before`, `course—I`, `Alice)—“and`) plus names and rare words — so any
  replacement for espeak is audible on many lines. The product loads the system eSpeak NG at
  runtime (dynamic loading, never compiled in); see docs/DEPENDENCIES.md. Fallback pronunciations
  follow the installed eSpeak NG version; the reference comparisons use the reference's 1.52.0.
- ~17–20% of tokens pass through POS/context-dependent rules (a/an/the/to/used, POS-keyed entries,
  NNP handling) → the spaCy tagger had to be ported; a tag-free approximation would change
  pronunciations.
- Words left without a pronunciation: the reference silently drops them (for example "-12" in
  public edge case 7, the "n’t" of "won’t" on two Alice lines, and many fuzz/soup tokens in other
  scripts). The native frontend instead fails the line ("unresolved word"), so no word is ever
  dropped from the audio; tests/frontend_pipeline.rs checks that exactly those lines are refused.

Provenance (open item before any publication): misaki's us/gb gold/silver lexicons ship under the
repository's Apache-2.0; the upstream README (pinned fba1236) documents no origin for the entries.
