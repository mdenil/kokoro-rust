# PORT_STATE — kokoro-rust   (read this first on any resume; then re-verify pins)

## CURRENT WORK (owner #26, 2026-09-27): BF16X-ONLY RELEASE CLEANUP on branch release/bf16x-cleanup
- Scope: HERMES_RELEASE_CLEANUP_BRIEF.md and HERMES_BRIEF #26.
- Reduce the code to the single accepted bf16x path.
- Reference artifact (immutable): $KOKORO_DATA/bin/phase2-9b39d48-6fde9d88990a (sha256
  6fde9d88990a8dc518ec1f366fb17db0869f7e7df5fc1fdea973ea0a371612ea), run as `--precision bf16x` with
  defaults. Build: strict -fmad=false, IG_BK=4, WMMA_TN=128.
- Progress and inventory: docs/RELEASE_CLEANUP.md.
- main and experiment/reduced-precision are untouched; no merge, release or deployment.

## CURRENT STATE (owner #25, 2026-09-27): PHASE 2 DECIDED — BF16X SELECTED, presented quality ACCEPTED
- Branch experiment/reduced-precision. Selected configuration: `--precision bf16x`
  (KOKORO_PRECISION=bf16x) with kernel levers P2-L2..L5 at their defaults. Final binary
  $KOKORO_DATA/bin/phase2-9b39d48-6fde9d88990a (sha256 6fde9d88…12ea), tree 9b39d48.
- Owner #25 (1553678159168151575): BF16x SELECTED and its presented quality ACCEPTED.
  - Basis: G1-G4 ladder listening plus B1-B5 focused comparisons vs production Python, on
    hash-verified raw WAVs.
  - Provenance: $KOKORO_DATA/evidence/listening/phase2-owner-decision/OWNER_DECISION.json.
  - Accepted differences are not to be re-escalated.
- Owner #24 (1553673730025197655): INT8 REJECTED (hiss/degradation). PL-014 pairs: no audible
  difference.
- Other levels (tf32, tf32all, fp16, bf16, fp16x): heard in ladder listening, not selected. No
  acceptance is recorded for them.
- bf16x measured (final matrix, private chapter; all arms cv ≤ 5%):
  - warm 4.15 s = 1.80× vs the Rust f32 control (7.45 s) and 13.08× vs production Python (54.31 s);
  - cold 5.81 s = 1.50× vs f32 (8.69 s) and 11.82× vs Python (68.71 s).
  - Alice warm 0.97 s (1.74× vs f32; its Python ratio is provisional).
- bf16x diagnostics (accepted; not to be re-escalated): public130 median spectral distance from f32
  2.05 / 2.28 dB (af_heart / am_adam); durations change on 45 / 41 of 130 lines.
- NOT done and NOT requested: merge into main, default change, deployment, release. main stays f32
  (the phase-1 engine frozen at 3dc5a35, plus docs), and the f32 baseline binaries are preserved.
- Full phase-2 report: docs/PHASE2_PRECISION.md. Phase-1 speed history: main
  docs/PERFORMANCE_REPORT.md.
- Standing: batching/FMA variation accepted (#14); private corpus stays under /data only;
  libespeak-ng 1.52 (#19) and ffmpeg (#18) approved.

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
