# Data-root relocation — 2026-09-26

Owner convention (Misha): data root = project path with ONLY `/home` → `/data`.
`/home/mdenil/code/kokoro-rust` ⇒ **`/data/mdenil/code/kokoro-rust`**.
Retired root: `/data/mdenil/kokoro-rust`. Cargo target stays `/home/mdenil/code/kokoro-rust/target`.

## What moved (same ZFS filesystem ⇒ atomic `mv` renames, no copy/re-download)
evidence/ fixtures/ hf/ models/ reference/ torch/ uv-cache/ → new root. Old empty `target/`
removed (cargo builds are project-local). New `tmp/` created at the new root.

## Integrity
- Pre-move sha256 manifest of 169 files (HF blobs incl. weights + voices, models/,
  fixtures/, evidence/): `evidence/relocation_2026-09-26_manifest.sha256`
  (sha256 `d11f2eb5323c5f80026b2678a9f2149688f38f141c719185e468d55237a63d64`).
  `sha256sum -c` at the new root: **all 169 OK**.
- Reference venv: 44 text scripts in `venv-prod/bin` embedded the absolute old path
  (entry-point shebangs + activate*); rewritten old→new prefix with sed. `bin/python` is a
  symlink to `/usr/bin/python3` (unaffected). `.pyc` files embedding old `co_filename` were
  left as-is (CPython fixes co_filename to the real source path on load). No other
  non-cache site-packages file referenced the old path. uv-cache interpreter msgpack entries
  and an hf/xet log still mention the old path: self-invalidating cache / historical log.
- Validation (pinned env via `scripts/env.sh`): pins asserted OK; `hf version`=1.20.1;
  `spacy info` resolves at new root; KModel loads from the relocated snapshot; replaying
  frozen fixture `cpu-t1/s05_word__af_heart__s1.0` with its injected noise reproduces the
  frozen audio **bit-exactly (max|Δ| = 0.0)** and identical durations.
- Search gotcha found during this work: the shell's `grep` is a ugrep wrapper with
  `--ignore-files`; uv writes a `.gitignore` containing `*` inside the venv, so recursive grep
  silently skipped it. Use `command grep` for such scans.

## Guards against regression
- `scripts/env.sh` hard-sets KOKORO_DATA/HF_HOME/TORCH_HOME/UV_CACHE_DIR/TMPDIR/CARGO_TARGET_DIR;
  repo scripts source it.
- Project-local `.claude/settings.local.json` env sets the same variables; Read/Edit +
  additionalDirectories granted for the new root (old grants preserved, no deny/ask added).
- `oracle/common.py` refuses to run if KOKORO_DATA/HF_HOME/TORCH_HOME/UV_CACHE_DIR point at the
  retired root (verified: exits with an explicit message).

## Temporary compatibility (recorded; to retire)
`/data/mdenil/kokoro-rust/tmp/` remains ONLY because the running Claude session inherited
`TMPDIR=/data/mdenil/kokoro-rust/tmp` at launch and holds open task-output handles and its
scratchpad there (verified with `lsof`). A `RETIRED.txt` marker sits in the old root.
Retirement: after this session ends, `rm -rf /data/mdenil/kokoro-rust` (contains only
session scratch). No canonical default points at the old root.

Historical receipts (e.g. `fixtures/*/meta.json`, `fixtures/nondet_floor.json`) are immutable
raw evidence and were not rewritten; any old paths inside them are historical.
