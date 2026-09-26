# Owner scope addition: native text frontend

Misha explicitly requests including the existing Python text frontend in the Rust port (Discord 1553400013273563147). This extends the existing project; the same interactive Claude remains the sole coder. Hermes supervises only.

## Deliverable
Plain text -> normalization/pronunciation/phonemes -> Kokoro-compatible chunking -> native Rust/CUDA waveform/WAV, with NO Python interpreter, subprocess bridge, Python runtime or Python package dependency on the shipped inference path. Python remains permitted for oracle/fixture generation and development comparisons.

Inspect the pinned Misaki 0.9.4 / Kokoro 0.9.4 frontend actually used by oracle/frontend_bridge.py and its dependencies first. Port the production English path used by af_heart/am_adam; document other language coverage rather than silently claiming all languages. Preserve current supported English dialect behavior where applicable. Existing native libraries may be reasonable dependencies if explicit and redistributable; do not disguise a Python helper or simple incompatible phonemizer substitution as a completed port. Surface substantive pronunciation/coverage gaps.

## Proof and packaging
- Add differential tests against the pinned frontend for normalized text where accessible, phonemes, punctuation/stress, boundaries and chunk order. Cover ordinary narration, numbers/currency/dates, abbreviations, names/unknown words, contractions, hyphens, Unicode/quotes, empty input, long paragraphs and chunk limits.
- Keep source/reference fixtures immutable. Audio numerical variation policy does not automatically permit different pronunciation or silently dropped/truncated text. Explain frontend differences rather than masking them with waveform tolerances.
- Check licensing/provenance for dictionaries, language resources and any native dependencies before packaging. Large downloaded data under /data/mdenil/code/kokoro-rust; normal builds stay gitignored in the home project target/.
- Exercise actual plain-text CLI output with Python unavailable in the runtime environment; inspect for hidden interpreter calls. Verify output durations/coverage and both production voices.
- Benchmark cold text-to-WAV plus warm resident text-to-audio against the pinned reference and existing bridge using the same input. Distinguish loading, frontend and model costs; do not count precomputed phonemes as native text frontend completion.

## Coordination
Add this deliverable to HERMES_BRIEF.md, PORT_STATE.md and the publish/install roadmap so it survives session compaction. Keep CUDA optimization in scope. Finish the current experiment at a natural boundary; choose a sensible order for frontend work and CUDA improvements. Do not interrupt active timing runs. Do not delay the already requested checkpoint/headroom report for this new work. Continue using the same Claude instance; no extra coding agents. Reply with acknowledgement and placement in the work plan.
