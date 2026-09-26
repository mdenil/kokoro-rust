# Test receipts — single-binary milestone + hardening (2026-09-26, owner #15/#17/#20)

Tree: commit recorded in the git log entry that adds this file. Host: RTX 4090 (CUDA:0), driver
580.178.04, CUDA 12.9.

Environment for every command: `cd /home/mdenil/code/kokoro-rust && source scripts/env.sh`.
Private tests are `#[ignore]` and are run explicitly with `-- --ignored`. They print aggregates
only, and their outputs stay under /data/.../evidence/private.

## Anti-vacuity measures (owner #20)
- Every fixture the frontend and chapter checks read is pinned in `tests/support/mod.rs`:
  sha256, exact record count, lines 1..=N in order, complete final line, and per-corpus totals
  (lines / chunks / spaCy tokens / distinct fallback calls). Private fixtures are pinned by hash only.
- Array lengths are asserted before every zip:
  - tokens vs oracle tokens; tags, logits and tensor vs tokens;
  - phoneme vs grapheme chunks vs expected; manifest entries, sidecars and WAVs vs input lines.
- Totals checked are asserted equal to the pins, so a shorter corpus cannot pass.
- Negative controls on the validators (`fixture_validators_reject_damaged_fixtures[_private]`): each
  fixture is rejected when missing, when its last record is dropped, when a line is cut, when a
  record is duplicated (with and without keeping the count), or when one byte is altered. Mismatched
  cardinality pins are also rejected.
- Negative controls on the output verifier (`output_negative_controls`, on a real public run). Each
  of these damages is rejected:
  - missing / truncated / duplicated WAV; extra stray WAV;
  - missing / truncated / duplicated sidecar;
  - manifest entry dropped or duplicated; truncated manifest;
  - a chunk dropped from a multi-chunk line; an altered phoneme.

  The expectation side is checked too: a truncated or duplicated expectation, or the wrong voice,
  is rejected. The private run applies the expectation-side controls to its real outputs.

## Commands and results
| command | result |
|---|---|
| `cargo test --release --test frontend_spacy -- --include-ignored` | tokenizer edge 65/65, Alice 1402/1402, private 316/316 lines. Tagger: tags 0 mismatches over 822 + 34,473 + 9,137 tokens; feature ids exact; tok2vec tensor max\|Δ\| ≤ 3.22e-6; tagger logits max\|Δ\| ≤ 1.34e-5 (seam newly compared; gate 1e-4 fixed before judging). Negative controls: symbols 12, lexeme_norm 3, pad 46 Alice lines change. PASS (5 tests) |
| `cargo test --release --test frontend_num2words` | 37,048/37,048 rows exact (cardinal 23,032, ordinal 3,501, year 10,000, float 515). PASS |
| `cargo test --release --test frontend_espeak -- --include-ignored` | distinct fallback calls exact: edge 31/31, Alice 71/71, links 2/2, private 37/37; synthetic 83/83 (raw phonemize list + mapping). PASS |
| `cargo test --release --test frontend_espeak_negctl` | system espeak-ng 1.51 refused (version != pinned 1.52.0). PASS |
| `cargo test --release --test frontend_g2p -- --include-ignored` | misaki logic given oracle tokens: edge 65/65, Alice 1402/1402, private 316/316. Negative controls (400 Alice lines): flattened tags 400, no fallback 34, British lexicon 397 lines differ. PASS |
| `cargo test --release --test frontend_pipeline -- --include-ignored` | assembled native frontend: edge 65/65 (67 chunks), links 20/20 (20), Alice 1402/1402 (1413), private 316/316 (317), **fuzz 4000/4000 (4388 chunks; 1 reference-error line reproduced as an error)** — phonemes + chunks exact. Validator negative controls: public and private. PASS (6 tests) |
| (fuzz findings, resumed session) | `python3 bench/make_frontend_fuzz.py $KOKORO_DATA/fixtures/frontend/fuzz.txt` (sha256 ddc6e9c4…) → `$KOKORO_PY oracle/frontend_oracle.py …/fuzz.txt …/fuzz.oracle.jsonl` (sha256 6096bf12…). FIRST run: 3992/4000 exact. 7 lines had tokenizer divergences: special-case spans applied across whitespace, e.g. `: (` → `:(`, `= (` → `=(`, `' '` → `''`. 1 line where the reference raises TypeError (`ord('İ'.lower())`) and native succeeded. Both fixed (src/frontend/spacy_tok.rs span-text rule lookup; g2p.rs Python short-circuit + error). Unit regressions: `frontend_spacy::special_case_spans_respect_internal_whitespace`, `frontend_pipeline::reference_typeerror_is_reproduced_as_an_error`. All component suites re-run afterwards: unchanged, exact. |
| `cargo test --release --features cuda --test cli_text_native -- --test-threads=2` (strict build) | 98 lines / 111 chunks, both voices, pronunciations = reference, 1 exec each; batched vs batch-1 sample counts identical (worst corr 0.99933); long lines 11/24 chunks complete; failures/restart/invalidation; ffmpeg `--encode`; 12 damaged-output + 3 expectation negative controls rejected. PASS (5 tests) |
| `cargo test --release --features cuda --test cli_text_native private -- --ignored` (strict) | private chapter: 316 lines verified, 317 chunks, 2915.0 s (af_heart) / 2837.4 s (am_adam) audio, 0 mismatches, 1 exec. PASS |
| `KOKORO_BIN=$PWD/target/release/kokoro-fma cargo test --release --features cuda --test cli_text_native -- --include-ignored --test-threads=2` (FMA build) | public: same checks PASS (worst batched vs batch-1 corr 0.99941 — identical to the earlier FMA session, i.e. reproducible); 15 negative controls rejected; private chapter 316/316 lines, 317 chunks, 0 mismatches, both voices, 1 exec. PASS (6 tests) |
| `cargo test --release --test frontend_pystr` | NEW. Exhaustive over all 1,112,064 Unicode scalar values vs the pinned interpreter (Python 3.12.3, Unicode 15.0.0; oracle/pystr_oracle.py, fixture sha256 499e9808…): isalpha, isdigit + numeric value, isspace, isupper, lower (incl. Final_Sigma), upper, capitalize (titlecase first char). BEFORE the fix it FAILED: the helpers used Rust's Unicode 16 properties and approximations. Disagreeing code points: isalpha 4,924; isdigit 178; digit value 719; isupper / lower / upper 55 each; capitalize 190. Now PASS, using tables generated from the interpreter (src/frontend/pyunicode.rs, oracle/gen_pyunicode.py). Frontend corpora results unchanged (all exact). |
| (resumed session, after the fuzz fixes) `KOKORO_BIN=$PWD/target/release/{kokoro,kokoro-fma} cargo test --release --features cuda --test cli_text_native -- --include-ignored --test-threads=2` | both builds 7/7 PASS. New `fuzz_corpus_through_binary`: 4000 lines → 3997 ok with reference-identical chunks (4366), 1 `error` (reference TypeError reproduced), 2 `oversize` (refused, not truncated), exit 1, 1 exec. The rest is unchanged: both voices, batching structure (worst corr 0.99933 strict / 0.99941 FMA), long lines 11/24, 15 negative controls, private chapter 316/316 both voices. |
| `cargo test --release --features cuda --test cli_linefile` | phoneme-input line-file interface: 5/5 PASS |
| full suite `cargo test --release --features cuda --no-fail-fast -- --test-threads=2` (run before this hardening; frontend/CLI tests re-run above) | all pass except the enforced known failures, whose fail sets are identical to earlier runs: `parity::ladder_cpu_f32` (7 rows, = ladder-cpu-t1-1790427009), `gpu_parity::gpu_ladder` (8 rows, = ladder-gpu-1790426627), `gpu_batch::batched_matches_single_item_within_rb1` (accepted PL-003 drift) |

## Determinism and memory safety (GPU path)
- Two identical `synth` runs of the 98-line corpus produced bit-identical WAVs: 98/98 for batch-1 and
  98/98 batched. They also match the outputs of the concurrently-run test processes.
- Unexplained single observation: an earlier strict-build test run (engine source identical, per git
  diff) reported a worst batched-vs-batch-1 correlation of 0.99956, versus 0.99933 now. Those outputs
  were not retained, so it cannot be attributed. Current builds are deterministic; the difference is
  within owner-accepted variation.
- compute-sanitizer (evidence/sanitizer/20260926/README.txt; strict build):
  - memcheck + initcheck: 0 errors, on the 7-line input, both batched and batch-1.
  - racecheck on everything except cuBLAS SGEMM: 0 hazards, 7-line input, batched and batch-1.
  - The previous session's unfiltered racecheck was killed with the session after about 40 minutes
    and produced NO result.
  - `--kernel-name` filters do not match our PTX-JIT kernels on this toolchain; those per-kernel
    runs are uninformative and are not counted.
  - Instrumentation of our module is shown by sync traces of layer_norm_rows, softmax_rows and
    lstm_seq. The tool's detection works: a positive control (deliberately racy kernel) is reported.
  - Not covered: cuBLAS SGEMM internals, the FMA build, larger inputs.
