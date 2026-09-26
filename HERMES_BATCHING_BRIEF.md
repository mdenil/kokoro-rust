# Owner directive: real chapter throughput and RTX4090 batching

Misha messages 1553401985879904370 and 1553402264377368728 explicitly establish:
- Binary unit = prepared chunk file in, many corresponding audio files out.
- Optimize whole-file throughput; true batching is encouraged, not a constraint to sequential one-line inference.
- Tune batch sizes and scheduling for RTX4090, the typical deployment GPU.
- Use an existing agent-prepared audiobook chapter as primary speed workload, preserving natural chunk lengths.

## Pinned private production chapter
Hermes inspected audiobook-production and located an approved actual prepared chapter from In Over Our Heads, chapter 1, The Hidden Curriculum of Youth: Whaddaya Want from Me? Source:
/home/hermes/ocr/outbox/ocr-20260818-in-over-our-heads/in-over-our-heads/audiobook/normalized/002_hidden_curriculum_of_youth_whaddaya_want_from_me.txt
Approval sidecar confirms 316 lines and SHA256 8129112a801ffb67d8974dddaa1de492232ea896f84bd121d45f8cf115b7aaad.

Remote staged path:
/data/mdenil/code/kokoro-rust/bench/private/in-over-our-heads-ch01/002_hidden_curriculum_of_youth_whaddaya_want_from_me.txt
Sibling source-section.json is the unchanged approval sidecar. The normalized text is 46,498 bytes, UTF-8 with terminal newline, 316 lines, no blanks. Character lengths min16 / median133 / nearest-rank p90 277 / max527. These are CHARACTER lengths, not phoneme/token lengths: measure actual phoneme/frame/padding distribution with the real frontend too.

Copyright/private benchmark: never commit or upload chapter text, source sidecar, phoneme dumps, generated narration, or text-bearing logs to GitHub. Keep all derivative private evidence outside the repo under mirrored data root. Commit only generic tools and aggregate timing/length statistics and hashes as appropriate. Retain the public Alice corpus for redistributable reproducibility; do not mislabel this chapter as public-domain. Preserve chapter bytes and all lines; no pruning troublesome inputs or replacing the natural workload with repeated synthetic lengths.

## Optimization scope
Keep one logical output and stable identity per source line, but permit internal reordering/length buckets, variable-size microbatches, batched kernels, multiple in-flight chunks, and pipelining if measurements justify them. Restore original line ordering in manifests/files; duplicates remain separate inputs. Internal chunk splitting for model limits must map back to the corresponding original line output without loss. The per-line external contract imposes NO sequential compute requirement.

Choose/tune batch size and bucketing for the actual RTX4090 VRAM budget. Expose override settings and retain a safe single-item/OOM fallback, but do not spend effort optimizing other GPU targets first. Record effective batch size, max batch workload (phonemes/frames if useful), padding waste, peak VRAM and throughput. Avoid baking in the exact chapter contents or a lookup-specific schedule. Respect user bounded audio variation policy: batching may alter floating arithmetic, but must not introduce padding contamination, state leakage between items, missing suffixes, changed pronunciations or cross-item noise coupling bugs. Validate representative short/long/tail batches, reordered batches, different batch sizes, and negative controls against the pinned reference and current single-item engine with stated bounds. No hidden post-hoc gate widening.

## Benchmark reporting
Primary: actual file -> all corresponding WAVs and metadata, total chapter wall time and generated-audio seconds/wall-second. Separate cold startup/load from resident processing and report frontend/inference/output costs as diagnostics. Compare pinned production Python/CUDA path on the SAME line file/voice/speed and explicit precision, current Rust batch1, and proposed Rust batching under comparable host conditions. Report reference batching support/settings honestly rather than pretending reference uses the Rust schedule. Preserve full chapter coverage/output count/durations, exact commands/build identities, repeated raw timing evidence. Use fresh benchmark outputs or force full regeneration so resume cannot masquerade as throughput.

Do not block this direction on completing native G2P: existing reference-derived phonemes may support clearly labeled core batching experiments while the frontend is ported, but the final headline needs true text-file -> audio-files runtime. Revisit headroom estimates: prior single-item profiling ceilings do NOT automatically apply to batched execution, which changes matrix shapes, utilization and memory reuse. Keep preliminary estimates assumption-labeled.

## Coordination
Same sole Claude codes all implementation/tests/benchmark scripts. Persist owner directive, fixture identity and work order in HERMES_BRIEF/PORT_STATE/roadmap. Natural pause sequencing; don't interrupt active benchmark. Retain native frontend deliverable and upstream agent/downstream assembly boundaries. Filename convention is not a new blocker: stable deterministic names plus explicit original 1-based line mapping satisfy integration; persist whichever agreed existing scheme is used. Do not start a new coder or delegate coding.
