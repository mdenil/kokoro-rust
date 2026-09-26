# Truth Pack — kokoro  (split into docs/truth-pack/{PINNED_SOURCES,SOURCE_HASHES,OQ_INDEX}.md)

<!-- "The single highest-EV move is NOT a kernel — it is an executable truth pack that stops
     the project from optimizing the wrong model semantics." Fill BEFORE any engine code. -->

## PINNED_SOURCES.md

- HF model repo: `<org/model>` @ commit `<hash>` (via `git ls-remote`; cross-check a second
  agent's independent fetch: "the model has not moved")
- Reference code repo: `<url>` @ commit `<hash>`
- Weights shard(s): `<file>` — <bytes> bytes, sha256 `<hash>` (VERIFY against the HF LFS etag)
- **Runtime pins — from the README/model card, NOT config.json** (config carries export tags):
  `torch==<v>`, `transformers==<v>`, `Pillow==<v>`, python `<v>`
  The oracle script asserts these at runtime and refuses on mismatch.
- License: `<SPDX>` — redistribution of converted derivatives: <allowed? attribution?>
- Re-fetch: `scripts/fetch_sources.sh [--verify]` (snapshots gitignored; hashes committed)

## SOURCE_HASHES.md

| file | sha256 | why load-bearing |
|---|---|---|
| config.json | `<hash>` | shapes/eps/theta |
| tokenizer(.json/.model/.tiktoken) | `<hash>` | L0a oracle |
| modeling_<x>.py | `<hash>` | THE semantics source |
| preprocessing/processor cfg | `<hash>` | L0b oracle |
| generation_config.json | `<hash>` | eos, no_repeat_ngram, penalties |
| index.json / shard map | `<hash>` | tensor census |
| LICENSE | `<hash>` | redistribution |

## OQ_INDEX.md — Open Questions register

**Hard rule: no kernel ships against an unresolved [OPEN]. A phase exit gate cannot pass while
it depends on an unresolved OQ.** Answer each with a QUOTED, line-cited excerpt from the
pinned source. Classify RESOLVED / PARTIAL / DEFERRED(non-blocking).

| OQ | Question | Status | Answer (line-cited) |
|---|---|---|---|
| OQ-1 | Attention window/ring/eviction semantics? | [OPEN] | `modeling_<x>.py:<lines>`: "<quote>" |
| OQ-2 | RoPE variant (NEOX rotate-half vs interleaved) + theta + YARN? | [OPEN] | |
| OQ-3 | head_dim / kv-heads (GQA?) / qkv fused-or-split per tower? | [OPEN] | |
| OQ-4 | Tokenizer family + special tokens + pretok rules? | [OPEN] | |
| OQ-5 | Preprocess: resize filter, pad value+rounding, normalize, tile math? | [OPEN] | |
| OQ-6 | Prompt/token census (image slots, separators, exact id count)? | [OPEN] | |
| OQ-7 | Generation processors: no_repeat_ngram, penalties, eos list, max len? | [OPEN] | |
| OQ-8 | Sampling mode of the reference eval (greedy? temp? beams?)? | [OPEN] | |
| OQ-9 | Multi-item decode: independent or cross-item dependent? | [OPEN] | |
| OQ-10 | Device orientation of reference infer() (CUDA-hardcoded?) — oracle split plan | [OPEN] | |
| OQ-11 | Tied lm_head? (sha256 the two tensors) | [OPEN] | |
| OQ-12 | Which tensors are the validated quant set? (prior-art quants of this model) | [OPEN] | |
| OQ-13 | Norm kind + eps + where (pre/post, final)? | [OPEN] | |
| OQ-14 | Name traps: typos to preserve, irregular numbering, lookalike fns in same file | [OPEN] | |
| OQ-15 | Worst-case K for i32 overflow proof? | [OPEN] | |
| OQ-16 | Encoder→decoder splice mechanics (scatter positions, separators)? | [OPEN] | |
| OQ-17 | Reference nondeterminism sources (threads, bf16, GPU) — floor-measurement plan | [OPEN] | |
| OQ-18 | <model-specific> | [OPEN] | |

Exit gate: "Zero blocking OQs remain" — recorded with date + commit.

## EXISTING_<MODEL>_STRUCTURE.md (the SPEC)
Extract the numbered-clause behavioral spec (target: every MUST clause line-cited). Implement
FROM THE SPEC, never line-by-line from the Python. Companion docs: PROPOSED_ARCHITECTURE.md
(with [SPEC-NNN] tags binding code to clauses) + FEATURE_PARITY.md (surface scoreboard).
