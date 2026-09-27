# Source hashes (sha256; verified 2026-09-26)

## HF snapshot f3ff3571791e39611d31c381e3a41a3af07b4987 (hexgrad/Kokoro-82M)

| file | sha256 | bytes | why load-bearing |
|---|---|---|---|
| config.json | `5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f` | — | shapes, istftnet params, phoneme vocab (178) |
| kokoro-v1_0.pth | `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4` | 327,212,226 | THE weights (f32) |
| voices/af_heart.pt | `0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff` | 523,425 | default voice A style pack [510,1,256] |
| voices/am_adam.pt | `ced7e284aba12472891be1da3ab34db84cc05cc02b5889535796dbf2d8b0cb34` | 523,420 | default voice B style pack |
| README.md | `91dcabced89db6f109b8786642f50402d3ee87450e8189589b6f85520e7f4d78` | — | model card: license and usage |

(HF blobs for LFS files are stored content-addressed by sha256 — the blob filename equals the
hash above, so download integrity vs the HF etag holds by construction.)

All 54 voice packs downloaded; per-file hashes = blob symlink targets in the snapshot dir
(en voices used in this project: af_*, am_*, bf_*, bm_*).

## PyPI reference artifacts

| artifact | sha256 |
|---|---|
| kokoro-0.9.4-py3-none-any.whl | `a129dc6364a286bd6a92c396e9862459d3d3e45f2c15596ed5a94dcee5789efd` |
| kokoro-0.9.4.tar.gz | `fbf633262797f8cf46fdac3315cf9cade67dc8b762c0feccf334892772fb9ac4` |
| misaki-0.9.4-py3-none-any.whl | `90e2eeb169786c014c429e5058d2ea6bcd02d651f2a24450ba6c9ffc0f8da15a` |
| misaki-0.9.4.tar.gz | `3960fa3e6de179a90ee8e628446a4a4f6b8c730b6e3410999cf396189f4d9c40` |

Developer re-fetch of the full snapshot with verification: `scripts/fetch_sources.sh --verify`. The
runtime assets alone: `scripts/fetch_assets.sh` (README.md).
