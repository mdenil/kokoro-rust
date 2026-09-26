"""Interim checkpoint: Rust CUDA vs production-pinned Python Kokoro CUDA reference (RTX 4090).

Every engine runs in its own process; engines are interleaved with a rotating order per replicate.
Raw per-process records (command, wall, exit code, receipt path) go to raw.jsonl; nothing is
summarized away. Run under scripts/env.sh with CUDA_VISIBLE_DEVICES=0.

Scopes:
  core   warm pure inference over the frozen 69 production phoneme chunks (full warmup pass, then
         3 timed in-process passes, CUDA-synchronized): ref-prod (cuDNN TF32 on = production),
         ref-f32 (TF32 off, matched precision, NOT production), rust (native CUDA, full f32).
  cold   fresh process text -> WAV over the 65 corpus lines: ref (import+load+KPipeline+soundfile),
         rust-text (CLI + DEV-ONLY Python misaki bridge), rust-phon (CLI, native, 69 phoneme lines).
  bridge Python frontend bridge alone (startup + G2P of the 65 lines).
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = pathlib.Path(os.environ["KOKORO_DATA"])
PY = os.environ["KOKORO_PY"]
SNAP = DATA / "hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987"
CORPUS = HERE / "corpus_alice_ch1.txt"
CHUNKS = HERE / "corpus_alice_ch1.chunks.jsonl"
RUST = ROOT / "target/release/kokoro"


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str)).stdout.strip()


def host():
    return {"loadavg": os.getloadavg(), "gpu": sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader"),
            "gpu_apps": sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")}


def main():
    out = DATA / "evidence/interim-comparison" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    raw = open(out / "raw.jsonl", "w")
    pins = {
        "git_head": sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": sh(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "rust_binary_sha256": sha(RUST), "model_pth_sha256": sha(SNAP / "kokoro-v1_0.pth"),
        "corpus_sha256": sha(CORPUS), "chunks_sha256": sha(CHUNKS),
        "reference_runtime": json.loads(sh([PY, "-c", "import sys,json; sys.path.insert(0,'oracle'); import common; common.assert_pins(); print(json.dumps(common.runtime_meta(), default=str))"])),
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i 0"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host_start": host(),
    }
    (out / "pins.json").write_text(json.dumps(pins, indent=1, default=str))

    def run(tag, cmd, rep, **meta):
        h0 = host()
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        wall = time.perf_counter() - t0
        rec = {"tag": tag, "rep": rep, "cmd": cmd, "wall_s": wall, "rc": p.returncode, "host_before": h0,
               "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-1500:], **meta}
        raw.write(json.dumps(rec, default=str) + "\n")
        raw.flush()
        print(f"{tag:28s} rep {rep} wall {wall:7.2f}s rc {p.returncode}", flush=True)
        if p.returncode != 0 and not tag.startswith("rust-text-note"):
            raise SystemExit(f"{tag} failed: {p.stderr[-800:]}")
        return rec

    # ---- core scope
    for voice, reps in (("af_heart", 5), ("am_adam", 3)):
        engines = {
            "core/ref-prod": lambda k, v=voice: [PY, str(HERE / "bench_reference.py"), "--device", "cuda", "--threads", "8", "--reps", "3", "--scopes", "inference", "--chunks", str(CHUNKS), "--voice", v, "--out", str(out / f"core-ref-prod-{v}-r{k}")],
            "core/ref-f32": lambda k, v=voice: [PY, str(HERE / "bench_reference.py"), "--device", "cuda", "--threads", "8", "--reps", "3", "--scopes", "inference", "--chunks", str(CHUNKS), "--voice", v, "--no-tf32", "--out", str(out / f"core-ref-f32-{v}-r{k}")],
            "core/rust": lambda k, v=voice: [str(RUST), "bench", "--device", "cuda", "--threads", "8", "--model-dir", str(SNAP), "--chunks", str(CHUNKS), "--voice", v, "--reps", "3", "--out", str(out / f"core-rust-{v}-r{k}.json")],
        }
        names = list(engines)
        for k in range(reps):
            order = names[k % 3:] + names[:k % 3]
            for n in order:
                run(n, engines[n](k), k, voice=voice)

    # ---- cold scope (af_heart, speed 1)
    phon_lines = out / "chunks_as_lines.txt"
    phon_lines.write_text("\n".join(json.loads(l)["phonemes"] for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()) + "\n", encoding="utf-8")
    cold = {
        "cold/ref": lambda k: [PY, str(HERE / "bench_reference.py"), "--cold-child", "--device", "cuda", "--threads", "8", "--voice", "af_heart", "--speed", "1.0", "--corpus", str(CORPUS), "--wav-dir", str(out / f"cold-ref-wavs-r{k}")],
        "cold/rust-text(py-bridge)": lambda k: [str(RUST), "synth", "--device", "cuda", "--threads", "8", "--model-dir", str(SNAP), "--input", str(CORPUS), "--frontend", "python-bridge", "--bridge-python", PY, "--voice", "af_heart", "--out-dir", str(out / f"cold-rust-text-r{k}"), "--force"],
        "cold/rust-phonemes(native)": lambda k: [str(RUST), "synth", "--device", "cuda", "--threads", "8", "--model-dir", str(SNAP), "--input", str(phon_lines), "--input-format", "phonemes", "--voice", "af_heart", "--out-dir", str(out / f"cold-rust-phon-r{k}"), "--force"],
    }
    names = list(cold)
    for k in range(3):
        order = names[k % 3:] + names[:k % 3]
        for n in order:
            run(n, cold[n](k), k, voice="af_heart")

    # ---- frontend bridge alone
    lines = [l for l in CORPUS.read_text(encoding="utf-8").split("\n") if l.strip()]
    for k in range(3):
        t0 = time.perf_counter()
        p = subprocess.Popen([PY, str(ROOT / "oracle/frontend_bridge.py"), "a"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        p.stdout.readline()
        t_ready = time.perf_counter() - t0
        n_chunks = 0
        for l in lines:
            p.stdin.write(json.dumps({"text": l}) + "\n")
            p.stdin.flush()
            n_chunks += len(json.loads(p.stdout.readline())["chunks"])
        total = time.perf_counter() - t0
        p.kill()
        raw.write(json.dumps({"tag": "bridge", "rep": k, "startup_s": t_ready, "total_s": total, "g2p_s": total - t_ready, "chunks": n_chunks}) + "\n")
        print(f"bridge rep {k}: startup {t_ready:.2f}s g2p {total - t_ready:.2f}s chunks {n_chunks}", flush=True)
    raw.close()
    (out / "host_end.json").write_text(json.dumps(host(), default=str))
    print("evidence:", out)


if __name__ == "__main__":
    main()
