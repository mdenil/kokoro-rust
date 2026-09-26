# PINNED_SOURCES — kokoro-rust

Pinned 2026-09-26 (all fetches executed on this host; see SOURCE_HASHES.md).

## Model (weights + config + voices)
- HF repo: `hexgrad/Kokoro-82M` @ snapshot `f3ff3571791e39611d31c381e3a41a3af07b4987`
  (production-observed revision; matches openclaw production cache per HERMES_BRIEF.md)
- Local: `/data/mdenil/code/kokoro-rust/hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987`
- Weights: `kokoro-v1_0.pth` — 327,212,226 bytes, sha256 `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`
  (HF blob is content-addressed by this same sha256 ⇒ matches LFS etag by construction)
- License: **apache-2.0** (README front-matter). Redistribution of converted derivatives: permitted
  with attribution/license preservation. Base model lineage: yl4579/StyleTTS2-LJSpeech (MIT).

## Reference implementation (THE semantics source = installed PyPI artifacts)
- `kokoro==0.9.4` (PyPI wheel sha256 `a129dc6364a286bd6a92c396e9862459d3d3e45f2c15596ed5a94dcee5789efd`)
- `misaki==0.9.4` (PyPI wheel sha256 `90e2eeb169786c014c429e5058d2ea6bcd02d651f2a24450ba6c9ffc0f8da15a`)
- GitHub HEADs at pin time (context only, NOT the semantics source — PyPI artifacts are):
  - hexgrad/kokoro HEAD `dfb907a02bba8152ca444717ca5d78747ccb4bec`
  - hexgrad/misaki HEAD `fba1236595f2d2bf21d414ba6e57d25256afada3`
- kokoro license: Apache-2.0; misaki license: Apache-2.0 (verify in installed dist-info — OQ-L).

## Runtime pins (from OBSERVED PRODUCTION, per HERMES_BRIEF — not from config.json)
Python 3.12.3; torch==2.12.1; numpy==2.4.6; soundfile==0.14.0; transformers==5.12.1;
espeakng-loader==0.2.4; spacy==3.8.14; en-core-web-sm==3.8.0; huggingface-hub==1.20.1;
kokoro==0.9.4; misaki==0.9.4. CUDA_VISIBLE_DEVICES=0 (RTX 4090). Output: 24,000 Hz.
The oracle scripts assert these pins at runtime and refuse on mismatch.

- Reference venv: `/data/mdenil/code/kokoro-rust/reference/venv-prod` (uv-managed;
  re-create with `scripts/setup_reference_env.sh`)

## Production contract being compared against (headline comparison scope)
- KPipeline(lang_code='a'), voices af_heart / am_adam, selectable speed, CUDA:0,
  resident/warm model. Baseline receipts under `/data/mdenil/code/kokoro-rust/evidence/`.

## Host
- CPU: 2× Intel Xeon E5-2698 v4 (Broadwell, 80 HT threads; AVX2/FMA/F16C — NO AVX-512/VNNI)
- GPU0: RTX 4090 24GB (deployment target) · GPU1: GTX 970 (DO NOT USE)
- Driver 580.178.04 (CUDA 13.0) · rustc 1.98.1 · uv 0.12.19
