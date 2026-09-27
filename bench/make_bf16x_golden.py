"""Render per-line WAV hash tables for the BF16x regression and diagnostic tests.

Every case runs a binary on a public input with fixed options, twice, and records the per-line WAV
sha256 (both runs must agree: determinism). The WAVs stay under $KOKORO_DATA (outside Git).

Two tables:
- strict (default): the earlier strict-rounding artifact (phase2-9b39d48, built with -fmad=false, run
  with KOKORO_PRECISION=bf16x) -> tests/pinned/bf16x_golden.json, used by the diagnostic
  tests/strict_reference_diff.rs.
- --fma-pin BINARY --commit SHA: the current FMA-contraction build (-fmad=true) ->
  tests/pinned/bf16x_fma_golden.json, the regression pin of tests/bf16x_regression.rs. It records the
  binary hash and source commit. It is a reproducibility pin, not a quality acceptance.

Usage: python3 bench/make_bf16x_golden.py [--fma-pin BINARY --commit SHA] [--private PRIVATE_TEXT_FILE]
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if not os.environ.get("KOKORO_DATA"):
    raise SystemExit("KOKORO_DATA is not set: point it at the data root (docs/PORTABILITY.md; e.g. source scripts/env.sh)")
DATA = pathlib.Path(os.environ["KOKORO_DATA"])
SNAP = DATA / "hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987"
ACCEPTED = DATA / "bin/phase2-9b39d48-6fde9d88990a"
ACCEPTED_SHA256 = "6fde9d88990a8dc518ec1f366fb17db0869f7e7df5fc1fdea973ea0a371612ea"

# (case, input file relative to the repo, voice, extra synth args)
CASES = [
    ("alice_af", "bench/corpus_alice_ch1.txt", "af_heart", ["--format", "float32"]),
    ("alice_am", "bench/corpus_alice_ch1.txt", "am_adam", ["--format", "float32"]),
    ("edge_af", "bench/frontend_edge_cases.txt", "af_heart", ["--format", "float32"]),
    ("edge_am", "bench/frontend_edge_cases.txt", "am_adam", ["--format", "float32"]),
    ("links_af", "bench/frontend_link_features.txt", "af_heart", ["--format", "float32"]),
    ("heldout_af", "bench/corpus_heldout.txt", "af_heart", ["--format", "float32"]),
    ("heldout_am", "bench/corpus_heldout.txt", "am_adam", ["--format", "float32"]),
    # product defaults (pcm16 output)
    ("heldout_am_pcm16", "bench/corpus_heldout.txt", "am_adam", []),
    # operational controls of the same path
    ("edge_af_speed0.8", "bench/frontend_edge_cases.txt", "af_heart", ["--format", "float32", "--speed", "0.8"]),
    ("heldout_af_seed7", "bench/corpus_heldout.txt", "af_heart", ["--format", "float32", "--seed", "7"]),
    ("edge_am_items1", "bench/frontend_edge_cases.txt", "am_adam", ["--format", "float32", "--batch-items", "1"]),
]


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def render(binary, extra_env, inp, voice, extra, out):
    if out.exists():
        shutil.rmtree(out)
    env = dict(os.environ, KOKORO_FRONTEND_DIR=str(DATA / "frontend"), **extra_env)  # GPU: caller's CUDA_VISIBLE_DEVICES
    cmd = [str(binary), "synth", "--model-dir", str(SNAP), "--input", str(inp), "--out-dir", str(out), "--voice", voice] + extra
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (out.parent / f"{out.name}.stderr.txt").write_text(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"{binary} failed on {inp} ({voice} {extra}): exit {r.returncode}")
    stem = pathlib.Path(inp).stem
    return {int(w.stem.rsplit("_", 1)[1]): sha(w) for w in sorted(out.glob(f"{stem}_*.wav"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", help="optional private text file: pin its hashes under $KOKORO_DATA/evidence/private only (never in Git)")
    ap.add_argument("--fma-pin", metavar="BINARY", help="pin this FMA-contraction build instead of the strict artifact")
    ap.add_argument("--commit", help="source commit the --fma-pin binary was built from")
    a = ap.parse_args()
    if a.fma_pin:
        assert a.commit, "--fma-pin needs --commit"
        binary, extra_env = pathlib.Path(a.fma_pin), {}
        gold = DATA / "evidence/fma-golden"
        table = {"reference_binary_sha256": sha(binary), "source_commit": a.commit, "build": "nvcc -arch=compute_89 -fmad=true -O3 (FMA contraction)",
                 "engine_identity": "cuda bf16x kernels=fma(-fmad=true)", "run_as": "defaults otherwise",
                 "status": "regression pin of the FMA build (reproducibility); not a listening/quality acceptance",
                 "compare": "per-line WAV bytes (sha256); sidecars carry the engine identity and are not compared",
                 "cases": {}}
        out_public = ROOT / "tests/pinned/bf16x_fma_golden.json"
        out_private = DATA / "evidence/private/fma-golden/private_fma_golden.json"
    else:
        assert sha(ACCEPTED) == ACCEPTED_SHA256, "accepted binary hash mismatch"
        binary, extra_env = ACCEPTED, {"KOKORO_PRECISION": "bf16x"}
        gold = DATA / "evidence/release-cleanup/golden"
        table = {"accepted_binary": "$KOKORO_DATA/bin/" + ACCEPTED.name, "accepted_sha256": ACCEPTED_SHA256, "tree": "9b39d48",
                 "run_as": "KOKORO_PRECISION=bf16x, defaults otherwise (the accepted configuration)",
                 "compare": "per-line WAV bytes (sha256); sidecars carry the engine identity and are not compared",
                 "cases": {}}
        out_public = ROOT / "tests/pinned/bf16x_golden.json"
        out_private = DATA / "evidence/private/release-cleanup/golden/private_golden.json"
    cases = list(CASES)
    if a.private:
        gold = out_private.parent
        cases = [("chapter_af", a.private, "af_heart", ["--format", "float32"]), ("chapter_am", a.private, "am_adam", ["--format", "float32"])]
    for case, inp, voice, extra in cases:
        path = pathlib.Path(inp) if a.private else ROOT / inp
        h1 = render(binary, extra_env, path, voice, extra, gold / case)
        h2 = render(binary, extra_env, path, voice, extra, gold / f"{case}.rerun")
        assert h1 == h2, f"{case}: {binary} is not deterministic across runs"
        shutil.rmtree(gold / f"{case}.rerun")
        table["cases"][case] = {"input": inp if not a.private else "<private text>", "input_sha256": sha(path), "voice": voice, "args": extra,
                                "lines": len(h1), "wav_sha256": {str(k): v for k, v in sorted(h1.items())}}
        print(f"{case}: {len(h1)} lines, deterministic across 2 runs", flush=True)
    dst = out_private if a.private else out_public
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(table, indent=1) + "\n")
    print("wrote", dst)


if __name__ == "__main__":
    main()
