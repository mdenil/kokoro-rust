# PORT_STATE — kokoro-rust   (read this first on any resume; then re-verify pins)

## CURRENT PRIORITY (owner #21-#23, 2026-09-26): SPEED — PHASE 1 (approximately lossless, main) then PHASE 2 (lossy, separate branch)
- Step 1 DONE: whole-system baseline (tree 6e306b7). Chapter Python / Rust: cold 2.74×, warm 2.63×.
- Step 2 DONE — PHASE 1 FROZEN at tree 3dc5a35 (owner #22 stopping criterion; evidence in
  docs/PERFORMANCE_REPORT.md "Phase-1 stopping decision" and NE-004..NE-010).
  - Levers PL-006..PL-016 kept.
  - Final whole-system result (private chapter, clean cv on both arms): Python / Rust = cold 7.90×
    (69.48 → 8.79 s), warm 7.13× (52.28 → 7.33 s).
  - Alice warm 4.76× is provisional (Python cv 10.1%).
  - Phase-1 binaries preserved: $KOKORO_DATA/bin/phase1-final-{strict,fma}-3dc5a35.
- PHASE 2 IN PROGRESS on branch experiment/reduced-precision (worktree target/worktrees/phase2, from
  3dc5a35): precision ladder f32 | tf32 | tf32all | fp16 | bf16 | fp16x | bf16x | int8 (docs there:
  docs/PHASE2_PRECISION.md). All non-f32 are UNREVIEWED; no merge; main defaults unchanged.
  - The morning pack must include the PL-014 changed-case pairs s03_moon/am_adam and
    s06_long/af_heart (raw rows in the PL-014 evidence dir).
- Accepted and not to be reopened: batching (PL-003) and FMA (PL-005) audio variation (#14), DISC-003,
  RB-1 bounds, and the strict baseline as the authoritative regression reference. NEW degradation
  must be escalated.
- PHASE 2 (owner #23, HERMES_PRECISION_EXPLORATION_BRIEF.md): LOSSY reduced-precision/quantization
  ladder. It starts ONLY after phase 1 reaches the #22 stopping point, on a separate branch from a
  verified phase-1 checkpoint (with a preserved baseline binary). Candidates are UNREVIEWED; there
  are no per-candidate listening holds; functional checks still fail hard. Deliverables: the matrix
  and a raw listening pack.
- libespeak-ng 1.52 accepted (#19); ffmpeg approved (#18). The private corpus stays under /data only.

## Single-binary milestone closeout (2026-09-26; HISTORICAL — superseded by the owner's speed authorization #21)
- Execution logs: /data/mdenil/code/kokoro-rust/evidence/integrated-runs/20260926/ (README + SHA256SUMS).
  - strict + FMA integrated runs, before the fuzz fixes (a296b58) and after (dbe5bba);
  - sanitizer summary lines.
  These are grep-filtered captures of stdout; the full cargo test stdout was not retained.
  Sanitizer logs: evidence/sanitizer/20260926/.
- Concrete in-scope implementation defects known: NONE open. The two fuzz-found frontend bugs are
  fixed and have regression tests.
- Not defects, but open items:
  - DISC-004: owner decision on original-gate diagnostics (see below).
  - Scope: American English only (British not requested or ported).
  - Coverage: finite corpora (7 pinned + exploratory); real-world text can reach untested paths.
  - Evidence: raw inner benchmark samples were not retained; historical chapter ratios are
    provisional (docs/PERFORMANCE_REPORT.md).
  - Release: misaki lexicon provenance; GPL-3 obligations of libespeak-ng if distributed (#19);
    no project license chosen.

## Correctness status of the neural engine (policy layers kept distinct)
Policy history, in order (HERMES_BRIEF owner log):
- #2: the original precommitted gates are binding.
- #4: bounded variation. Bit identity is not required; regression is judged by RB-1 with fixed
  nonzero bounds against the owner-accepted strict baseline.
- #11: speed plus a close match that sounds good. Exact waveform identity is not an acceptance
  criterion.
- #3 / #14: listening acceptances of the specific pairs presented.

Neither #4 nor #11 explicitly waived the rows below. Whether they still block acceptance is an owner
decision not yet made. Until then they are reported as original-gate DIAGNOSTICS with their raw
values preserved. They are neither silently waived nor treated as new defects.
- Original-gate diagnostics (historical, unchanged since first measured):
  - CUDA ladder: all 135 stage-seam rows pass; ids, durations and sample counts exact. 8 E2E rows
    FAIL the original v1 / G-SPEC gates vs the cpu-t1 fixture (v1 max ×5, G-SPEC ×3).
    - s02_fox/am_adam/s1.0 is owner-accepted by listening (DISC-003).
    - 7 are un-adjudicated (DISC-004).
  - CPU f32 ladder: 7 E2E rows FAIL the original gates.
  - Latest runs ladder-gpu-1790439506 and ladder-cpu-t1-1790439695 have fail sets IDENTICAL to the
    first measurements. There is no new degradation.
  - Attainability context: the exact f64 reference passes v1 on 6/15; production torch CUDA passes
    3/15 (TOLERANCE_HISTORY #9).
- Regression under the current policy (RB-1 vs the strict baseline): the strict default build is
  unchanged since the baseline was pinned.
  - FMA and batching exceed RB-1 bounds in the recorded ways (PL-005: 9 drift rows + one G-SPEC
    set change; PL-003: batch-vs-single drift). That variation was presented for listening and
    owner-ACCEPTED (#14).
  - The failing `gpu_batch::batched_matches_single_item_within_rb1` and the FMA RB-1 diagnostics
    are that accepted variation, with raw values preserved.
- NEW degradation would be different: a change in any fail set, new seam failures, or new
  pronunciation, duration or structure differences. That is not covered by any acceptance and
  must be flagged. None has been observed.

## Open correctness gaps (concrete)
1. DISC-004 (owner decision, not an implementation defect I can fix within the gates): 7 E2E rows fail
   the original gates, historically and unchanged. Whether policy #4/#11 supersedes them for these rows
   is for the owner to decide; I do not self-authorize it.
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
