# PORT_STATE — kokoro-rust   (read this first on any resume; then re-verify pins)

## Where we are
- Mode: full-port   Phase: −1 (truth pack) → entering 0 (oracle)
- Ladder: none yet   Ship gate: n/a
- Last session state: truth pack pinned+hashed; reference env installing; next: resolve OQs
  from installed kokoro/misaki source, then build oracle scripts.
- Next single action: read installed kokoro-0.9.4 source, resolve OQ-1..18 in docs/truth-pack/OQ_INDEX.md.

## Pins (see docs/truth-pack/PINNED_SOURCES.md + SOURCE_HASHES.md)
- Model: hexgrad/Kokoro-82M @ f3ff3571791e39611d31c381e3a41a3af07b4987; weights sha256 496dba11…
- Reference: kokoro==0.9.4 + misaki==0.9.4 (PyPI, hashes pinned); torch==2.12.1 CUDA; Python 3.12.3
- Data root: /data/mdenil/kokoro-rust (hf/, reference/venv-prod, evidence/, models/, tmp/)
- Production comparison scope: KPipeline('a'), af_heart/am_adam, CUDA:0 RTX 4090, warm resident.

## Phase gates
| Phase | Gate artifact | Status |
|---|---|---|
| −1 truth pack | OQ register zero-blocking + hashes | hashes DONE; OQs OPEN |
| 0 oracle | floor envelope + fixture inventory | not started |
| 1 forward | e2e waveform parity + seam table | not started |
| 2 quant | (deferred; float first per brief) | — |
| 3 kernels | selftest battery | not started |
| 4 perf | ledger rows + baseline receipts | not started |
| 6 gpu | go/no-go doc (4090 target) | not started |
| 7 ship | certification bundle | deferred until review |

## Open threads
- Reference venv install running in background (torch 2.12.1 download).
- Voices are .pt (torch pickle zips) — native Rust reader planned; dev bridge = oracle-exported safetensors.
- Frontend (misaki G2P) parity is tracked SEPARATELY from core phoneme→audio parity.

## Session log
- 2026-09-26 claude: scaffold + truth pack pins/hashes; plan written; HF snapshot fetched+hashed; env install started.
- 2026-09-26 claude: OWNER CORRECTION applied — cargo builds in project-local gitignored ./target (CARGO_TARGET_DIR pinned via .claude/settings.local.json env + explicit command env; verified via cargo metadata + git check-ignore). /data keeps weights/fixtures/reference env only.
