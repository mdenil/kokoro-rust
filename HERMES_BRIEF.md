# Kokoro Rust port: owner brief

Misha explicitly commissioned this personal project and authorized creating mdenil/kokoro-rust. Hermes supervises; YOU, this interactive Claude Code instance running in tmux, perform ALL implementation, test, benchmark, build/release-script code and project edits. Do not launch another coding agent, independent Claude, Codex, or implementation subagent. You may run your own tools and tests. Keep this session alive and attachable. Work autonomously through milestones, reporting blockers precisely.

## Location and boundaries
- Project: /home/mdenil/code/kokoro-rust. Personal GitHub remote: https://github.com/mdenil/kokoro-rust (private initially). Never use an R2 organization.
- All weights, downloaded snapshots, large fixtures, reference environments, caches and benchmark audio go under /data/mdenil/code/kokoro-rust. Gitignore generated/large/private material. Do not commit credentials or production books.
- OWNER CORRECTION (Misha, 2026-09-26): the data root mirrors the project path with ONLY the /home prefix replaced by /data: /home/mdenil/code/kokoro-rust -> /data/mdenil/code/kokoro-rust. The earlier root /data/mdenil/kokoro-rust is retired (relocation record: docs/RELOCATION_2026-09-26.md). Canonical env: `source scripts/env.sh`.
- OWNER CORRECTION (Misha, 2026-09-26; overrides the storage clause above and any launch env): /data is for downloaded model weights and large data, NOT ordinary build outputs. Cargo builds use /home/mdenil/code/kokoro-rust/target, a normal gitignored directory (not a symlink into /data). The launch environment exported CARGO_TARGET_DIR=/data/mdenil/kokoro-rust/target; every build/test subprocess must override it with CARGO_TARGET_DIR=/home/mdenil/code/kokoro-rust/target (project-local .claude/settings.local.json env + explicit command env). Verify with `cargo metadata` and `git check-ignore target/`.
- Existing DreamZero reference project: /home/mdenil/r2/code/dreamzero-rust. Read-only for storage/optimization workflow conventions. Do not copy proprietary R2 source into this personal repository or alter that project/session.
- Shared host: inspect CPU/GPU activity before benchmarks; use RTX 4090 GPU 0, not GTX 970 GPU 1. Bound compilation to 8 jobs initially. Do not stop others' processes, install system packages, modify drivers, or change global settings. If competition prevents honest timing, defer those measurements.
- Only modify this project and its /data workspace. Normal project-local dependencies, public upstream downloads, commits and non-force pushes to mdenil/kokoro-rust are authorized. No production skill/workshop changes and no release/publication/registry upload yet.

## Model identification and production contract
The audiobook-production skill delegates TTS to kokoro-local-tts; OCR itself is document extraction, not TTS. The requested model is hexgrad/Kokoro-82M, not an OCR vision model.
Observed production reference (read by Hermes using the configured openclaw account):
- /home/openclaw/hermes-tts-workshop
- installed kokoro==0.9.4 and misaki==0.9.4
- exact observed production environment: Python 3.12.3; torch==2.12.1; numpy==2.4.6; soundfile==0.14.0; transformers==5.12.1; espeakng-loader==0.2.4; spacy==3.8.14; en-core-web-sm==3.8.0; huggingface-hub==1.20.1
- OWNER FOLLOW-UP: maintain a direct speed comparison against the CURRENT production implementation, not merely an arbitrary upstream environment. Prepare docs/PERFORMANCE_REPORT.md with measured baseline vs final Rust, reproduction commands, commit/model/runtime/hardware identities, corpus coverage, uncertainty and correctness gates. Summarize how fast we managed to make it for Misha once validated. Include per-case and aggregate results, and clearly distinguish pure inference, text frontend, persistent batch and actual production end-to-end scopes. Preserve baseline receipts throughout optimization.
- cached HF model snapshot f3ff3571791e39611d31c381e3a41a3af07b4987
- upstream implementation https://github.com/hexgrad/kokoro
- CUDA_VISIBLE_DEVICES=0; 24,000 Hz speech
- default production voices af_heart and am_adam; selectable voice and speed
The openclaw home is not readable by mdenil. Do not bypass its permissions. Fetch independently the pinned public reference into /data; ask Hermes for targeted reference details if needed. Verify the snapshot contains the actual appropriate weight/config/voice artifacts, record hashes and exact runtime pins. Avoid floating latest versions. Resolve torch/reference dependencies deliberately, measuring reference behavior rather than assuming bit-exact cross-device audio.

Future workflow: canonical UTF-8 narration, one utterance per line; preserve order and exact text; WAV/PCM plus optional ffmpeg encoding; explicit voice/speed/language/model paths; resumable per-input output and JSON sidecars with text/config/model/audio hashes, sample rate, duration, status, elapsed time and errors. Empty/error/oversize cases must be explicit, never silent truncation. A resident/batch API should avoid loading weights per sentence. Do not implement semantic cleanup, sentence rewriting, OCR or M4B assembly here.

## Required skill and execution method
Invoke and FOLLOW your installed ai-model-into-rust-mega-fused-hyper-kernel skill at /home/mdenil/.claude/skills/ai-model-into-rust-mega-fused-hyper-kernel. Use its router and appropriate references, including oracle/ladder and performance ritual. This is a hybrid TTS model, so adapt the proof ladder to phonemes, durations and acoustic/vocoder tensors; do not mechanically copy OCR greedy-token or transformer-only quantization gates. The architecture may require additional kernel families; budget that work instead of substituting a different model.

Priority: demonstrably correct native Rust model inference, then measured fast inference on this host, then install/release engineering. Native implementation means the model forward is Rust/native kernels, not a Python subprocess, LibTorch or ONNX wrapper sold as a Rust port. Python/PyTorch is permitted for the independent reference oracle, fixture export and comparison only. Native CUDA/FFI kernels are permissible with bounded unsafe interfaces and explicit correctness tests. Establish a float baseline first; do not quantize as a shortcut to getting a running model. Distinguish core phoneme-to-audio parity from text frontend parity. A temporary reference frontend is a development bridge only, never presented as a finished Python-free text-to-speech product. Document remaining G2P/Misaki parity and native frontend work honestly.

## Acceptance gates
1. Pin and hash upstream code/runtime/model/config/vocabulary/voices/license. Inspect licensing before proposing eventual redistribution. Keep compact truth pack, explicit behavioral spec and PORT_STATE.md with reproducible commands and current blockers. Seed discrepancy/negative-evidence/performance ledgers. Favor a working vertical slice over huge process scaffolding.
2. Oracle BEFORE engine: execute the real pinned reference; measure repeatability (including thread/device settings and stochastic excitation/noise paths). Freeze reference inputs, RNG/noise inputs where necessary, intermediate tensors and final audio. Reference and subject must be independent. Never derive expected values from the Rust implementation.
3. Tests BEFORE corresponding implementation: cover preprocessing/phoneme IDs, tensor shapes/names, core ops, per-stage oracle-input seams, duration/alignment, style/voice conditioning and complete waveform. Exercise both default voices, speed changes, short/long and boundary-length inputs, punctuation/numbers/Unicode and invalid inputs. Synthetic diagnostics supplement real public/self-authored sentences, not replace them. No transcription/STT-based audio QC.
4. Set numerical criteria from measured reference behavior plus explicit justified floating-point analysis BEFORE optimizing. Exact IDs/shapes/discrete outputs where appropriate; report waveform max/RMS error, correlation and suitable spectral metrics, sample-count/duration agreement and earliest divergent seam. If stochastic inputs differ, fix the harness instead of loosening tests. Never widen gates simply to pass, and never claim bit identity unless actually proven. Prove comparisons fail under deliberate perturbation. A voiced WAV/ffprobe success alone proves neither model nor narration parity.
5. Build actual end-to-end native core inference and expose a useful CLI/library. Validate reference corpus and tests. Report native frontend completeness separately. Test repeated/multi-item synthesis, deterministic supported path, corrupt/missing artifacts, device/runtime dispatch and boundary conditions. No implicit ignored/skipped tests counted as passes.
6. Optimize only green paths via the skill's one-lever ritual: negative-evidence sweep, current profile, predeclared hypothesis/gates, reversible kill switch, parity first, repeated interleaved paired A/B, keep or revert including neutral results, hash-backed evidence and ledger. Compare pinned PyTorch and Rust on matched inputs/voices/speeds/precision/devices/thread budgets with warmups and synchronization. Report cold startup/load separately from warm inference, include preprocessing/transfers where claims include them, batch-1 latency + representative narration throughput/real-time factor + peak RAM/VRAM. Record contention and noise/CV; do not publish ratios that fail the skill gates. Seek useful end-to-end wins, not a faster isolated kernel masking slower total runtime.
7. Every milestone should have real command output and reviewable commits; push non-sensitive source/docs/tests to personal origin. Maintain PORT_STATE.md with what works, what remains and exact next action so Hermes/user can resume. Do not stop at a plan or skeleton. Continue until a substantive checkpoint, blocker, or completed correctness+performance implementation.
8. Eventual phase (after correctness and speed accepted): reproducible cargo build/install, versioned release binaries and checksums where appropriate, pinned model acquisition/cache, clean-install smoke and documented integration contract. Defer actual public release, crate publishing and production integration until Misha/Hermes review.

## First action
Load the named skill. Inspect available toolchains/storage and public reference. Write a compact milestone plan with explicit unresolved assumptions, execute the reference oracle and baseline, then implement with tests. Proceed without asking for routine choices: single Kokoro-82M model, existing production revision, CPU float correctness baseline plus RTX 4090 deployment performance as the main optimization target; English/default voices first with other language limitations documented. Ask only on genuine missing authority, semantic ambiguity or blocked resources. Keep all evidence grounded in executed commands.

## OWNER DECISIONS LOG (appended 2026-09-26; newest last; these override earlier text)
1. GPU-FIRST (Misha): the primary deliverable is a native Rust + CUDA engine on the RTX 4090.
   CPU is a supporting oracle/debug baseline only: no CPU performance work; CPU-specific waveform
   differences are documented (never concealed) but are not release blockers for a correctly
   validated CUDA product. Do not run CPU baseline workloads concurrently with GPU perf evidence.
2. CORRECTNESS HOLD + BINDING ORIGINAL GATES (Misha, via supervisor): the original precommitted
   thresholds are binding — per-seam rel ≤ 1e-4; ids/durations/sample counts exact; E2E vs the
   frozen cpu-t1 fixture: rel-RMS ≤ 0.019, max ≤ 0.033, corr ≥ 0.9995; original spectral gate.
   G-E2E-v2 (written after initial failures) confers NO acceptance. No widening, per-case
   relaxation, dropped cases/seams/metrics, baseline substitution, precision change,
   post-processing or waivers. If a gate is impossible/underspecified: stop and report evidence.
   History: docs/conformance/TOLERANCE_HISTORY.md.
3. OWNER LISTENING ACCEPTANCE (Misha, Discord 1553386472906694677): the worst-peak pair
   s02_fox/am_adam/s1.0 (evidence/listening/, WAV sha256 8aeb4c6c… reference cpu-t1 /
   aa5de4d4… Rust CUDA) is "indistinguishable ... I am happy with this" → DISC-003. This lifts
   the performance hold. It is NOT bit identity, NOT an audition of other cases, NOT permission
   to relax thresholds. Original-gate failures stay truthfully reported; unaccepted failures
   (DISC-004) remain open and must be escalated, not self-authorized. The owner-approved CUDA
   baseline is pinned (tests/pinned/gpu_envelope.json; listened case bitwise). New optimizations
   must not regress it or widen thresholds; any new/worse discrepancy is escalated.
4. BOUNDED VARIATION (Misha, 1553392534095527999; supersedes the bitwise/zero-slack pin rule in #3):
   bit identity is not required; small bounded numerical differences are allowed; approved audio
   is comparison evidence, not a byte-lock. Regression policy RB-1 (docs/conformance/REGRESSION_POLICY.md):
   explicit nonzero per-case drift bounds = min(reference arithmetic sensitivity, owner-accepted
   listened divergence), fixed before judging further levers; exact discrete invariants, complete
   seams/coverage/spectral/negative controls kept; escalate materially larger or audible differences
   with raw paired samples. Historical original-gate failures are never re-labelled as passes.
5. INTERIM CHECKPOINT (Misha, 1553392892624511017): pause new optimization; run and report an interim
   Rust CUDA vs production-pinned Python Kokoro CUDA comparison (docs/PERFORMANCE_REPORT.md,
   evidence/interim-comparison/), then stop for review.
6. NATIVE TEXT FRONTEND (Misha, 1553400013273563147; full text in HERMES_FRONTEND_BRIEF.md):
   plain text -> normalization/pronunciation/phonemes -> Kokoro chunking -> Rust/CUDA audio/WAV with
   NO Python interpreter, subprocess bridge or Python package on the shipped path (Python stays
   allowed for oracles/fixtures). Port the production English path (misaki 0.9.4 en.G2P as used by
   KPipeline('a'); British 'b' where applicable); document other languages as not covered.
   Differential tests vs the pinned frontend (phonemes, stress, punctuation, boundaries, chunk order;
   narration, numbers/currency/dates, abbreviations, names/OOV, contractions, hyphens, Unicode/quotes,
   empty, long paragraphs, chunk limits). Audio variation policy does NOT permit pronunciation
   differences or dropped/truncated text. License/provenance check before packaging. Verify the
   plain-text CLI with Python unavailable; benchmark cold text->WAV and warm resident text->audio vs
   reference and the existing bridge, separating load / frontend / model costs. CUDA optimization
   stays in scope.
7. AUDIOBOOK FILE INTERFACE (Misha, 1553400752284770372; full text in HERMES_AUDIOBOOK_INTERFACE_BRIEF.md
   "Audiobook file interface"): input = audiobook/normalized/<section>.txt (UTF-8, one prepared
   utterance per line, no blanks in canonical files). Extend `kokoro synth`: text file + out dir +
   voice/speed/device -> exactly one deterministically numbered WAV per input line in source order,
   manifest + sidecars (input file, 1-based line, text hash, settings/model/frontend identity, audio
   hash, duration, status, errors); load once per invocation; no Python runtime; no cleanup,
   re-sentencing, dedup or reordering; long lines chunked internally and joined to one output
   (never truncated); unsafe lines fail explicitly by line number; defined blank/malformed-input
   behaviour; resume only verified outputs; atomic promotion; nonzero exit if incomplete. Chapter
   assembly/M4B stay downstream. Acceptance: native CLI with Python unavailable on public-domain /
   synthetic fixtures, both voices, restart + invalidation, long-line completeness, cold + resident
   benchmarks; outputs usable under audiobook/audio_chunks/.
8. THROUGHPUT / BATCHING (Misha, 1553401985879904370 + 1553402264377368728; full text
   HERMES_BATCHING_BRIEF.md): unit = prepared line file in, many WAVs out; optimize whole-file
   throughput on the RTX 4090; true batching, length bucketing, reordering, pipelining encouraged
   with exact original line mapping restored. PRIMARY workload: private approved chapter
   /data/mdenil/code/kokoro-rust/bench/private/in-over-our-heads-ch01/
   002_hidden_curriculum_of_youth_whaddaya_want_from_me.txt (316 lines, 46,498 bytes, sha256
   8129112a801ffb67d8974dddaa1de492232ea896f84bd121d45f8cf115b7aaad; copyrighted — chapter text,
   sidecar, phoneme dumps, audio and text-bearing logs NEVER in Git/GitHub; only aggregate
   statistics/hashes and generic tools may be committed). Public Alice stays secondary. Batching must
   not cause padding contamination, state leakage, missing suffixes, pronunciation changes or
   cross-item noise coupling; validate short/long/tail batches, reordering, batch sizes, negative
   controls vs reference and batch-1 under stated bounds (RB-1 style, fixed before judging). Report
   file->WAVs wall time and audio-s per wall-s; cold vs resident; frontend/inference/output costs;
   vs production Python/CUDA on the same file/voice/speed/precision, vs Rust batch-1. Reference-
   derived phonemes may support labelled core-batching experiments; the headline needs true native
   text-file -> audio-files. Reassess headroom under batching (single-item ceilings do not carry over).
9. PUBLICATION POLICY (Misha, 1553403193109778493): the repo may eventually be open-sourced (NOT
   authorized now; no license chosen or changed by me). Keep the small redistributable public corpus
   (Alice ch. 1 text + phoneme chunks) tracked with provenance, transformations and source/legal
   notices; online availability != redistributable. The private In Over Our Heads chapter and every
   derivative (text, phonemes, audio, text-bearing logs) must NEVER be published; it stays local
   under /data. Before every push run the CONTENT audit `python3 scripts/check_private_leaks.py`
   (tracked + staged, worktree + index); ignore rules alone are not an audit. Generic stats / hashes /
   names are fine; no passages. Public corpus + generic reproduction commands must suffice for
   open-source users; the private chapter benchmark stays optional.
10. HOST PARALLELISM (Misha, 1553408435029147720): Rust CPU-side multithreading is explicitly permitted
   whenever it improves GPU-backed chapter throughput (parallel native G2P, batch preparation, staging/
   transfers, overlapped output writes/hashing). The neural model stays on the RTX 4090. The earlier
   8-thread setting is NOT a product cap: use bounded, configurable worker pools tuned to the host,
   avoid nested oversubscription, and record CPU budgets / load / affinity in comparisons (label
   unmatched budgets). Line mapping, correctness and resume semantics are preserved.
11. SPEED + CLOSE-MATCH-THAT-SOUNDS-GOOD (Misha): not chasing bit-identical output. Optimize for speed
   and a close match that sounds good. CPU-rounding constraints may be relaxed (e.g. drop -fmad=false,
   normal FMA contraction); strict rounding remains an optional diagnostic baseline, not a production
   requirement. Exact waveform identity is not an acceptance criterion; evaluate speed and audio quality.
   Keep functional correctness, text coverage, ordering, line identity/resume and negative controls.
   Document numerical/perceptual trade-offs honestly; never silently widen tolerances to pass.
12. REPRESENTATIVE SUBSETS (Misha): smaller representative test/benchmark sets may be used when the full
   chapter makes iteration too slow; favor fast feedback with relevant lengths and edge cases; label
   subset results accurately (never present them as full-chapter numbers); use broader validation when
   warranted. Private source content stays out of commits.
13. HEADLINE BENCHMARK = ORIGINAL PYTHON vs FASTEST RUST (Misha, 1553418452436385864): benchmark the
   original, pinned production Python usage (unchanged behaviour/settings: KPipeline per line, batch-1,
   its defaults) against the fastest Rust achievable. No batching/optimization/matched worker budgets
   for Python; unequal batching and host scheduling are intended, not an unfairness to repair. Same
   source workload / voice / speed / output contract; document settings and timing scope honestly.
   Matched-precision diagnostics are optional supporting evidence, not the headline or a blocker. Do not
   replace the original baseline with optimized Python, and do not imply deployed-wrapper overhead is
   included when unmeasured. Consult the ai-model-into-rust-mega-fused-hyper-kernel skill for the
   remaining performance work.
14. OWNER LISTENING ACCEPTANCE — BATCHING + FMA (Misha, Discord 1553428873020973171): Misha listened to all
   six raw WAVs and said "All of these are fine." ACCEPTED (presented comparisons only):
   - evidence/listening/batch-worst-s04_alice_am_adam: reference cpu-t1 ce9c3f12…d2e8, single control
     3bc6224e…d7dd, batched15 CANDIDATE d90a653a977d5c17dd038b6ced4f14ce66a0c6f43efffc7a67d7dcfaa1f29d4f.
   - evidence/listening/fma-worst-s03_moon_am_adam: reference cpu-t1 dfb2b530…e2ea, strict control
     64159e11…9c99, FMA CANDIDATE c6a01bc2b7b61a76ea09c49cf0a19cd847f477a9648d9331e23347183f0e1c6e.
   Both listening holds removed; batching (PL-003) and FMA (PL-005) are ACCEPTED optimization candidates;
   defaults chosen on actual speed + remaining functional validation. Preserve raw numerical results and
   regression evidence; no bit-identity / zero-slack requirements; do not re-escalate this accepted
   variation. NOT proof that unreviewed cases/combinations/frontends pass: keep functional invariants,
   negative controls and broader tests; flag meaningful NEW quality loss, not ordinary tiny arithmetic changes.
15. PRIORITY CHANGE — SINGLE BINARY FIRST (Misha, 1553430052400660527; supersedes the interleaved speed plan
   and the pending headroom refresh): completeness AND correctness first, THEN speed. PAUSED: new perf
   levers, conv/fusion experiments, batch-size tuning, headroom analysis (accepted batching/FMA preserved).
   Milestone: chunk-line text file -> per-line audio files via ONE Rust executable: native normalization/
   G2P/tokenizer/tagger/fallback/chunking, GPU model, ordered outputs/metadata, safe resume; NO Python
   interpreter/runtime/helper process or oracle replay at inference. Explicitly inventory external model/
   language data and CUDA runtime deps (one executable != embedded assets != static zero-dependency
   build); surface any native helper/dependency gap instead of calling it complete. Licensing/scope
   decisions go to Misha only if genuinely blocking; never choose a project license. Prove: actual binary
   path with Python unavailable and no hidden subprocess fallback; full input/output coverage;
   pronunciation fidelity vs the pinned reference; long lines; both voices; failures/restart/config
   invalidation. Agent cleanup + chapter assembly stay outside the binary. Representative tests for
   iteration, complete-chapter acceptance once assembled. Owner-approved numeric/audio variation stays
   accepted. Speed resumes AFTER this milestone.
16. NEXT PERFORMANCE PHASE = FULL SYSTEM (Misha, 1553430273801195591; after milestone #15): same prepared
   chapter file -> all verified audio files + metadata, original production system as used vs the
   complete Rust binary. Cold process and warm resident throughput measured separately, with stage
   attribution (startup/load, native text frontend, scheduling/transfers, GPU, output writes/hashes)
   accounting for overlap (no summing of concurrent stages). Profiling establishes where savings are —
   no assumed gains. Headline = whole-system wall time, not core-only. No optimization campaign before
   the complete/correct milestone.
17. TESTING SHAPE CONFIRMED (Misha, 1553431050242490532): unit tests IN ADDITION TO end-to-end tests.
   Accepted batching + FMA stay in force (NO rollback). Focus: finish the native frontend now. Keep
   component-level differential/unit tests (tokenizer, tagger, normalization/numbers, pronunciation/
   fallback, chunking) plus integrated actual text-file -> audio tests on the accepted GPU path
   (batched, FMA build exercised as well as strict). No new speed work; no re-litigation of accepted
   audio variation.
18. FFMPEG APPROVED (Misha, 1553439300211970079): the external ffmpeg dependency is explicitly FINE. The
   optional `--encode` subprocess is not a single-binary completion blocker: document the dependency and
   test it when enabled. Keep the default WAV path and the no-Python proof.
19. LIBESPEAK-NG FINE FOR NOW (Misha, 1553440298590281732): the pinned native libespeak-ng 1.52.0
   pronunciation fallback (in-process C shared library, not neural synthesis) is accepted for current
   development/runtime use. A non-GPL replacement may be desirable later; do NOT spend time replacing it
   now. Keep dependency/version/data/GPL-3.0-or-later notices truthful. NOT a project license choice and
   NOT redistribution clearance: open-source distribution obligations are a later release concern, not
   a reason to halt the milestone.
20. KEEP ON COMPLETENESS/CORRECTNESS (Misha, 1553442132952490105): no diversion to speed testing now (no
   speed campaign, no comparative benchmark). At this idle milestone: bounded test hardening — pin
   expected corpus/cardinality and chunk/tag-array counts before any zip; add missing/truncated/duplicate
   fixture and output negative controls so frontend and private-chapter checks cannot pass vacuously;
   private material stays local. Reconcile stale PORT_STATE next-actions and the raw-evidence wording in
   PERFORMANCE_REPORT (evidence/ab holds transcribed summaries, not retained raw ABBA output). Preserve
   accepted batching/FMA quality and baseline diagnostics; do not reopen listening or change bounds.
   Record exact commands/results; report remaining concrete correctness gaps.
21. SPEED PHASE AUTHORIZED (Misha, 1553520469574164483; supersedes the no-speed hold of #15/#20):
   1) FIRST measure/profile the CURRENT complete native Python-free binary end-to-end:
      - workload: prepared chapter file -> ALL audio files + metadata;
      - baseline: the ORIGINAL production Python as actually used, unchanged (no Python
        optimization or batching, no matched execution strategies);
      - cold process and warm resident measured separately;
      - overlap-aware attribution of startup/model load, frontend, scheduling/transfers, GPU, output;
      - primary workload: the private 316-line chapter (contents never published); public Alice for
        reproducibility;
      - keep raw individual replicates, commands, build/input identity, host state and output
        coverage (not summary-only or filtered logs);
      - clean comparisons have no contention, no resume skips and no profiler timings.
   2) Then the skill's perf ritual:
      - measured bottleneck -> one lever -> focused regression + negative controls -> comparable
        A/B -> commit keep/revert;
      - representative subsets for iteration, full chapter at milestones;
      - batching/FMA authorized; tune Rust to the 4090 including CPU-side work.
   3) Other terms:
      - unchanged historical numerical diagnostics are disclosed but are not a reason to idle or
        restart broad hardening;
      - speed authorization is NOT blanket listening approval: escalate meaningful NEW degradation
        or real bugs;
      - finite coverage, British support and release licensing are not speed blockers;
      - preserve the native English file contract, the approved libespeak/ffmpeg and the private
        corpus boundary;
      - report the current full-system baseline before a long optimization campaign.
22. KEEP GOING UNTIL SEVERELY DIMINISHING RETURNS (Misha, 1553522797244850308): after the full-system
   baseline, continue profile-driven optimization autonomously across ALL meaningful bottlenecks (not one
   small lever and then idle).
   - Target: chapter-file-to-audio throughput and cold/resident costs on the 4090; the current
     native binary vs the unchanged incumbent Python.
   - Keep accepted batching/FMA, regression gates and private-corpus protection.
   - Method: the skill; targeted subset tests/A-B for iteration, full-system checkpoints, raw
     receipts, incremental keep/revert commits.
   - STOPPING CRITERION (diminishing returns treated as evidence): the remaining plausible material
     levers have been explored; gains are repeatedly marginal or noise-level; or the complexity/risk
     is disproportionate to a negligible whole-system benefit. Never declare saturation from one
     failed experiment or an assumed roofline.
   - At that point report: the best measured configuration, gains vs the incumbent, levers tried or
     rejected, and residual opportunities.
   - Escalate only new meaningful quality loss or real blockers.
   - Do not disturb unrelated host workloads to get quiet measurements: record contention and
     adapt bounded runs.
23. TWO-PHASE SPEED PLAN (Misha, 1553523595337015369 + 1553524046136746120; full scope in
   HERMES_PRECISION_EXPLORATION_BRIEF.md):
   PHASE 1 = APPROXIMATELY LOSSLESS optimization on main.
   - Scope: current f32/FMA precision, within the existing accepted numerical/audio variation
     (NOT bit identity).
   - Runs until severely diminishing returns (#22).
   PHASE 2 = LOSSY optimization.
   - TRIGGER: only after phase 1 stops.
   - Setup: record the best phase-1 commit and its binary/config/asset/corpus identities, quality
     evidence and timings; preserve a runnable baseline binary artifact; commit clean; create a
     separate branch (e.g. experiment/reduced-precision; check for collisions).
   - Rules: no reduced-precision work on main; no merge or default replacement without approval.
   - Precision ladder:
     1. f32/FMA control;
     2. conservative mixed FP16/BF16 with f32 accumulation and f32 islands for sensitive paths;
     3. materially more aggressive mixed/low precision;
     4. quantized (INT8 weights and/or activations with documented calibration; INT4 only if
        feasible/useful; label weight-only/dequantized honestly).
     Record failures, unsupported paths and slower variants.
   - Policy: experimental waveform/gate deviations are DIAGNOSTICS (existing thresholds untouched),
     every candidate is UNREVIEWED until Misha listens, and there are no per-candidate listening
     holds. Functional checks still fail hard: finite, non-empty, correct line mapping, no
     dropped/duplicated input, explicit failures, safe resume identity. Broken/NaN candidates are
     isolated, recorded and skipped.
   - Deliverables: a precision/speed matrix (whole-chapter cold/resident vs incumbent Python and
     best f32; raw replicates, commands, hashes, dtype/quant details, calibration identity, peak
     memory, drift) plus a RAW listening pack (public passages, both voices, f32 control + Python
     reference, worst cases + held-out) with a manifest. No acceptance claims from metrics, no fake
     low-bit speed claims, private chapter never published.
24. OWNER LISTENING — INT8 REJECTED (Misha, 1553673730025197655): after the phase-2 ladder listening
   (G3/G4), INT8 is explicitly REJECTED for hiss/degradation. The G5/G6 PL-014 before/after pairs
   (s03_moon/am_adam, s06_long/af_heart): no audible differences reported.
25. OWNER FINAL LISTENING DECISION — BF16X SELECTED AND ACCEPTED (Misha, 1553678159168151575):
   "Okay BF16x confirmed. These are all indistinguishable to me."
   - BF16x is SELECTED and its presented quality ACCEPTED (not merely a leading/unreviewed candidate).
   - Basis: G1-G4 ladder listening, then focused B1-B5 comparisons against production Python on
     public130: lines 52/af_heart, 1/af_heart, 11/am_adam (top 3 vs_reference spectral scores over
     266 cases); 42/am_adam and 64/af_heart (worst equal-total-duration per voice).
   - Exact final binary sha256 6fde9d88990a8dc518ec1f366fb17db0869f7e7df5fc1fdea973ea0a371612ea,
     tree 9b39d48 ($KOKORO_DATA/bin/phase2-9b39d48-6fde9d88990a).
   - The owner verified the original raw paired WAVs (no normalization or other transformation) by
     hash and seal.
   - Provenance: $KOKORO_DATA/evidence/listening/phase2-owner-decision/OWNER_DECISION.json (per-case
     WAV hashes).
   - Preserve the f32 baseline and all numerical diagnostics. Do not re-escalate the accepted bf16x
     differences.
   - This records the decision only: NOT a new optimization campaign, public release, deployment or
     automatic merge. Source and corpus protections unchanged.
26. OWNER AUTHORIZES BF16X-ONLY RELEASE CLEANUP (Misha, 1553679677162389505, confirmed
   1553679934516371629; full scope in HERMES_RELEASE_CLEANUP_BRIEF.md):
   - Branch release/bf16x-cleanup from the accepted lineage (aed6a5e = accepted code tree 9b39d48 plus
     docs).
   - Simplify to ONLY the approved BF16x production path. Remove the other precision variants,
     experimental kernel alternatives and kill switches, and dead plumbing; do not merely hide them
     behind a bf16x default.
   - Keep the FP32 operations and shape fallbacks intrinsic to bf16x, operational options, and
     safety/resume behavior.
   - Freeze the exact accepted build/kernel/rounding configuration (strict -fmad=false, IG_BK=4,
     WMMA_TN=128, P2-L2..L5). No new precision or speed experiments.
   - Tests: repoint/add them for the actual bf16x product; keep the useful unit, differential,
     negative-control and end-to-end checks.
   - Verification: prove outputs are preserved vs the accepted binary; bounded pre/post speed smoke.
   - No Python runtime. Private material stays outside Git.
   - Deliverable: incremental commits, a switch-removal inventory, current test evidence and a
     simplified command.
   - NOT authorized: public release, deployment, license choice, automatic main merge.
   - Preserve the accepted binary, evidence and main.
27. OWNER AUTHORIZES THE NEXT RELEASE-PREPARATION STAGE (1553681850818502707; scope under the data root in
   supervision/portability-milestone.md): after supervisor verification of stage 1 (0d1f149 /
   d83540a), a bounded portability/configuration milestone.
   - Remove host/user/GPU-index/inherited-environment assumptions from product, build, test and
     tools.
   - State the tested scope (Linux x86_64 / RTX 4090 CC 8.9 / CUDA 12.9 / American English).
   - Verify with an isolated data view and a stripped environment.
   - No license choice (the prompt suggestion to remove the license field is NOT owner approval), no
     publication.
28. OWNER: MERGE VERIFIED CLEANUP INTO MAIN (1553703276434821171): a completed, verified cleanup is
    merged, not kept as a separate product line.
    - Integrate 0d1f149 (implementation d83540a) into main with a normal merge; resolve the docs to
      the current BF16x state.
    - Keep the unfinished, unverified portability work out of the merge, and preserve it.
    - Verify, then push main to the existing PRIVATE origin.
    - No force push, no history rewrite, no visibility/publication/deployment change, no branch
      deletion.
    - Main becomes canonical; later verified milestones integrate there. Then continue the
      portability task.
