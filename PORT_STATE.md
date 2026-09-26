# PORT_STATE — kokoro-rust   (read this first on any resume; then re-verify pins)

## CURRENT PRIORITY (owner #15/#20, 2026-09-26): completeness + correctness. NO speed work.
- No speed campaign, lever, headroom analysis or comparative benchmark until the owner resumes speed.
  The next performance phase (owner #16), when resumed: the whole system, chapter file -> verified WAVs,
  production system as used vs the complete Rust binary, cold and resident, overlap-aware stage
  attribution.
- Accepted and not to be reopened: batching (PL-003) and FMA (PL-005) audio variation (owner #14),
  the s02_fox/am_adam listening pair (DISC-003), RB-1 bounds, and the strict baseline as the
  authoritative regression reference. libespeak-ng 1.52 accepted for now (#19); ffmpeg approved (#18).

## Single-binary milestone: COMPLETE and verified (details + exact commands: docs/conformance/TEST_RECEIPTS.md)
- `kokoro synth`: prepared line file -> one WAV + JSON sidecar per line plus `<stem>.manifest.json`.
  - Native frontend by default (misaki 0.9.4 G2P, spaCy en_core_web_sm 3.8.0 tokenizer + tagger,
    espeak-ng 1.52.0 fallback, KPipeline chunking); GPU model on CUDA:0.
  - Batched by default (`--batch-phonemes 8000`; 0 = one chunk at a time).
  - No Python and no subprocesses (strace audit: exactly 1 exec). Dependencies: docs/DEPENDENCIES.md.
- Component differential tests: all EXACT vs the pinned reference, with fixtures pinned by sha256 and
  cardinality (tests/support/mod.rs).
  - Tokenizer: 65 + 1402 + 316 lines.
  - Features / tok2vec / tagger: tags 0/44,432 mismatches; tensor ≤3.2e-6; logits ≤1.3e-5.
  - num2words: 37,048 rows.
  - espeak fallback: 139 distinct calls + 83 synthetic.
  - misaki logic given oracle tokens: 1,783 lines.
  - Assembled frontend: 1,803 lines, 1,817 chunks.
- Integrated binary tests (tests/cli_text_native.rs), passing on strict and FMA builds:
  - both voices; 98 lines / 111 chunks with pronunciations identical to the reference;
  - batched vs batch-1 sample counts identical;
  - long lines 11 / 24 chunks complete;
  - failures, restart, invalidation; optional ffmpeg `--encode`;
  - 12 damaged-output + 3 expectation negative controls rejected.
- Private chapter through the binary: 316/316 lines, 317 chunks, 0 mismatches, both voices, both builds
  (outputs under evidence/private only).
- Determinism: repeated identical runs are bit-identical (batch-1 and batched; checked in isolation
  and while other GPU tests ran). compute-sanitizer initcheck: 0 errors (batched).

## Correctness status of the neural engine (unchanged; original gates binding, owner #2)
- CUDA ladder: all 135 stage-seam rows pass; ids, durations and sample counts exact.
  8 enforced E2E rows FAIL (v1 max ×5, G-SPEC ×3). Latest ladder-gpu-1790439506 has the identical
  fail set.
  - s02_fox/am_adam/s1.0: OWNER-ACCEPTED by listening (DISC-003).
  - 7 others OPEN (DISC-004; not self-authorized).
- CPU f32 ladder: 7 enforced E2E rows FAIL. ladder-cpu-t1-1790439695 has the identical fail set.
- Attainability: exact f64 reference passes v1 on 6/15; production torch CUDA 3/15 (TOLERANCE_HISTORY #9).
- `gpu_batch::batched_matches_single_item_within_rb1` FAILS by design: batch drift beyond RB-1,
  owner-accepted as PL-003 (#14). Raw results are preserved.

## Open correctness gaps (concrete)
1. DISC-004: 7 enforced E2E rows unaccepted. Owner decision or evidence needed; not self-authorized.
2. British English (lang b) is not ported. The binary is American English only (voices af_*/am_*).
3. CLOSED 2026-09-26: the Python str semantics (isalpha/isdigit/isspace/isupper/lower/upper/capitalize/
   strip) were approximations on Rust's Unicode 16 tables and diverged on thousands of code points.
   They are now exact to the reference Python 3.12.3 (Unicode 15.0) via generated tables, verified
   exhaustively (tests/frontend_pystr.rs).
4. PARTLY CLOSED 2026-09-26: a deterministic synthetic fuzz corpus of 4000 hard-construct lines
   (bench/make_frontend_fuzz.py, seed 20260926; 127,514 tokens, 1,843 distinct fallback calls,
   143 multi-chunk lines) is differentially tested against the pinned reference.
   It found 2 real bugs, both fixed:
   (a) spaCy special-case retokenization ignored the whitespace inside a matched span
       (`: (` became the emoticon `:(`; 7/4000 lines);
   (b) the reference misaki raises TypeError on `ord('İ'.lower())` in punctuation-tagged tokens;
       native silently succeeded, and now fails that line explicitly like production.
   Result: 4000/4000 exact, plus unit regressions for both.
   Afterwards:
   - a second grammar seed (7; exploratory, unpinned) was 4000/4000 exact, so the generator
     looks saturated;
   - a pinned character-soup corpus (bench/make_frontend_soup.py; 3000 lines, 13,086 distinct
     fallback calls, 22 reference TypeError lines) was 3000/3000 exact. Every reference error is
     reproduced with the same exception class.
   Remaining: coverage is still finite
   (generated vocabulary, 5 corpora); arbitrary real-world text can still reach untested paths.
5. CLOSED (owner contract): production truncates chunks over 510 phonemes, and fails or skips some
   lines silently. The binary instead marks them `oversize` / `error` by line number, as
   HERMES_BRIEF.md (#7 and the original brief: "never silent truncation") and
   HERMES_AUDIOBOOK_INTERFACE_BRIEF.md require.
   Production KModel also silently drops phoneme characters outside the model vocabulary. The
   binary drops exactly the same characters (tested) and records them in `dropped_phoneme_chars`
   in the sidecar.
6. Misaki lexicon data provenance is open. This is a release concern, not a runtime one.

## Environment (ALWAYS `source scripts/env.sh` first)
- Data root: /data/mdenil/code/kokoro-rust (relocated; docs/RELOCATION_2026-09-26.md).
- Cargo target: /home/mdenil/code/kokoro-rust/target. CUDA build: `cargo build --release --features cuda`
  (nvcc 12.9 -> PTX compute_89; default strict -fmad=false; `KOKORO_FMA=1` = FMA build, copied to
  target/release/kokoro-fma for tests via `KOKORO_BIN`). `CUDA_VISIBLE_DEVICES=0` (RTX 4090; never GPU 1).
- Frontend data: `KOKORO_FRONTEND_DIR=$KOKORO_DATA/frontend` (misaki-0.9.4/, spacy-en_core_web_sm-3.8.0/,
  espeak-ng-1.52.0/). Rebuild with `oracle/export_spacy.py` (deterministic) and `scripts/stage_espeak.sh`.
- Push via the SSH origin, only after `git add` and then `python3 scripts/check_private_leaks.py`
  passes (owner #9). Stage first, so new files are audited.
- Shell `grep` is a ugrep wrapper that honors .gitignore; use `command grep`.
- serde_json needs `float_roundtrip`.
- Fresh shells may lack cargo on PATH: `export PATH=$HOME/.cargo/bin:$PATH`.

## Pins
- Model: hexgrad/Kokoro-82M @ f3ff357…; weights sha256 496dba11…; loaded natively from .pth (bitwise = reference load).
- Reference: kokoro==0.9.4, misaki==0.9.4, torch==2.12.1 (CUDA 13.0), Python 3.12.3 (exact prod pins);
  spaCy 3.8.14 + en_core_web_sm 3.8.0; espeakng-loader 0.2.4 (espeak-ng 1.52.0).
- Frontend fixture pins (sha256 + cardinalities): tests/support/mod.rs.

## Performance record (frozen while speed is paused)
- The interim checkpoint (docs/PERFORMANCE_REPORT.md) and levers PL-001..005 (docs/PERF_LEDGER.md) stand
  as recorded.
- The ABBA evidence for PL-001..005 is terminal summaries transcribed into evidence/ab/README.txt, NOT
  retained raw output. In-process bench receipts are in evidence/rust/.
- The pre-batching headroom estimate is stale. Nsys captures of the current tree
  (evidence/profiles/20260926-1640-paused/) are unanalysed.
- bench/interim_compare.py still contains the retired python-bridge row (historical driver).

## Process lessons
- compute-sanitizer: `--kernel-name` filters do NOT match our PTX-JIT kernels (they silently check
  nothing). `--racecheck-trace-sync` floods output (lstm_seq warp syncs, ~40 lines each). The
  informative racecheck is `--kernel-name-exclude kns=sgemm` on a short input: minutes, not hours.
- Never wait with `pgrep -f <pattern>` from a shell whose own command line contains the pattern; wait
  on the tracked background task instead.
- Run the leak audit AFTER `git add` (a pre-add audit misses new files; happened once, re-audited clean).
