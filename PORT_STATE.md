# PORT_STATE — kokoro-rust   (read this first on any resume; then re-verify pins)

## Where we are
- Mode: full-port   Phase: 1 (f32 forward, code-first) — Phases −1 and 0 complete.
- Ladder: none green yet (Rust forward being written)   Ship gate: n/a
- Next single action: finish `src/vocoder.rs` (Decoder/Generator/SineGen/STFT/iSTFT +
  NoiseSource), build, then seam tests in pipeline order vs `fixtures/cpu-t1/*` (tests/parity.rs).

## Environment (ALWAYS `source scripts/env.sh` first)
- Data root: /data/mdenil/code/kokoro-rust (hf/, reference/venv-prod, fixtures/, models/, evidence/, tmp/)
  — relocated 2026-09-26 from retired /data/mdenil/kokoro-rust; see docs/RELOCATION_2026-09-26.md.
- Cargo target: /home/mdenil/code/kokoro-rust/target (gitignored, real dir). Launch env exported
  CARGO_TARGET_DIR/HF_HOME/etc. at the retired root — env.sh and .claude/settings.local.json override.
- Push: origin push URL is SSH (git@github.com:mdenil/kokoro-rust.git); HTTPS has no credential.
- Shell `grep` is a ugrep wrapper honoring .gitignore (skips the venv); use `command grep`.

## Pins (docs/truth-pack/PINNED_SOURCES.md + SOURCE_HASHES.md)
- Model: hexgrad/Kokoro-82M @ f3ff3571791e39611d31c381e3a41a3af07b4987; weights sha256 496dba11…
- Reference: kokoro==0.9.4 + misaki==0.9.4; torch==2.12.1 (CUDA 13.0); Python 3.12.3 — exact prod pins, asserted in oracle.
- Production comparison scope: KPipeline('a'), af_heart/am_adam, CUDA:0 RTX 4090, warm resident.

## Phase gates
| Phase | Gate artifact | Status |
|---|---|---|
| −1 truth pack | OQ register zero-blocking + hashes | DONE (docs/truth-pack/OQ_INDEX.md; OQ-12 frontend-scoped) |
| 0 oracle | floor envelope + fixture inventory | DONE (docs/conformance/NONDET_FLOOR.md; 15 cases × {cpu-t1, cuda-t1}) |
| 1 forward | e2e waveform parity + seam table | IN PROGRESS (ops/nn/albert/model written; vocoder next) |
| 2 quant | (deferred; float first per brief) | — |
| 3 kernels | selftest battery | not started |
| 4 perf | ledger rows + baseline receipts | not started |
| 6 gpu | go/no-go doc (4090 target) | not started |
| 7 ship | certification bundle | deferred until review |

## Key facts established
- 3 RNG draws per forward (rand_ini[1,9], sine noise [1,S,9], unused noise branch [1,S,1]);
  frozen-noise replay is bit-exact per (device, threads). Free noise moves waveform 9.45% RMS.
- Floor: cpu-t1 vs cpu-t8 0.93% RMS / 1.65e-2 max; cpu vs cuda (TF32 convs) 5.65% RMS.
  Gates (frozen): ids/durations/sample count EXACT; wave RMS rel ≤1.9%, max ≤3.3e-2, corr ≥0.9995.
- ids + durations identical CPU↔CUDA on all 15 fixtures.
- Subject loads raw_state.safetensors (verbatim weight_g/weight_v) and computes weight-norm itself.

## Open threads
- Voices are .pt (torch zip pickles) — native Rust reader planned; dev bridge = oracle-exported safetensors.
- Frontend (misaki G2P) parity tracked SEPARATELY from core phoneme→audio parity (M4).
- Retire /data/mdenil/kokoro-rust (only live-session tmp/ remains) after this Claude session ends.

## Session log
- 2026-09-26 claude: scaffold + truth pack pins/hashes; plan; HF snapshot fetched+hashed; env installed (exact prod pins).
- 2026-09-26 claude: OWNER CORRECTION — cargo builds in project-local gitignored ./target (verified via cargo metadata + git check-ignore).
- 2026-09-26 claude: oracle (noise tap, seam recorder, weight export), nondeterminism floor, 32 frozen fixtures; OQs resolved.
- 2026-09-26 claude: OWNER CORRECTION — data root relocated to /data/mdenil/code/kokoro-rust; 169-file manifest byte-identical; venv shebangs rewritten; bit-exact fixture replay validated.
