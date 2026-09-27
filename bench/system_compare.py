"""Whole-system comparison driver (owner #21): line file -> ALL per-line WAVs (+ Rust metadata).

Engines:
  py      ORIGINAL production Python as used (bench/system_reference.py; unchanged KPipeline per line)
  rust    current native binary, default build (strict), default synth settings (batched)
  rustfma current native binary, FMA build (owner-accepted #14), default synth settings
Phases (separate, never mixed):
  cold    fresh process per replicate: startup + load + one pass; replicates interleaved across engines
  warm    one resident process per replicate: load, 1 untimed warm-up pass, P timed passes
  attr    one ATTRIBUTION process per engine (Python per-call timers; Rust timeline is always on) —
          stage breakdowns only, never headline timings
Evidence (raw, unfiltered): <out>/raw.jsonl (one record per process: command, env, wall, rc, host state
before/after, parsed stdout record, coverage), <out>/logs/<run>.{stdout,stderr}, Rust timelines,
manifests; WAVs are hashed then deleted (coverage + identity kept).
Usage: python bench/system_compare.py --corpus FILE --out DIR [--cold-reps 3 --warm-reps 2 --warm-passes 3]
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = pathlib.Path(os.environ.get("KOKORO_DATA", "/data/mdenil/code/kokoro-rust"))
PY = os.environ.get("KOKORO_PY", str(DATA / "reference/venv-prod/bin/python"))
SNAP = DATA / "hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987"
BINS = {"rust": pathlib.Path(os.environ.get("KOKORO_BIN_RUST", ROOT / "target/release/kokoro")),
        "rustfma": pathlib.Path(os.environ.get("KOKORO_BIN_RUSTFMA", ROOT / "target/release/kokoro-fma"))}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def host_state():
    st = {"t": time.time(), "loadavg": os.getloadavg()}
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    st["gpus"] = q.stdout.strip().splitlines()
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True)
    u = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    st["gpu_apps"] = a.stdout.strip().splitlines()
    st["gpu0_apps"] = [x for x in st["gpu_apps"] if x.startswith(u)]
    return st


def wait_quiet(max_wait=600):
    """Refuse to start a timed run while GPU 0 is busy (another user's job) — wait, then record."""
    t = time.time()
    while True:
        st = host_state()
        g0 = st["gpus"][0].split(", ")
        if int(g0[1]) <= 5 and not st["gpu0_apps"]:
            return st, time.time() - t
        if time.time() - t > max_wait:
            st["WARNING"] = "GPU0 not quiet after waiting; run is CONTENDED"
            return st, time.time() - t
        time.sleep(10)


def wav_digest(d):
    files = sorted(pathlib.Path(d).glob("*.wav"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return len(files), h.hexdigest()


def rust_coverage(out, passes):
    dirs = [out] if passes == 0 else [out / f"pass{k}" for k in range(passes + 1)]
    cov = []
    for d in dirs:
        man = next(d.glob("*.manifest.json"))
        m = json.loads(man.read_text())
        n, dig = wav_digest(d)
        cov.append({"dir": d.name, "complete": m["complete"], "counts": m["counts"], "input_lines": m["input_lines"],
                    "wav_files": n, "sidecars": len([f for f in d.glob("*.json") if not f.name.endswith(".manifest.json")]),
                    "digest_of_wav_hashes": dig, "audio_s": m["audio_s"]})
    return cov


def run(rec_file, logs, name, cmd, env, out, engine, passes):
    host_before, waited = wait_quiet()
    shutil.rmtree(out, ignore_errors=True)
    t = time.perf_counter()
    with open(logs / f"{name}.stdout", "wb") as so, open(logs / f"{name}.stderr", "wb") as se:
        rc = subprocess.run(cmd, env=env, stdout=so, stderr=se).returncode
    wall = time.perf_counter() - t
    host_after = host_state()
    rec = {"run": name, "engine": engine, "passes": passes, "cmd": cmd, "env": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "KOKORO_FRONTEND_DIR", "KOKORO_PRECISION") if k in env},
           "rc": rc, "wall_s": wall, "waited_for_quiet_s": waited, "host_before": host_before, "host_after": host_after}
    stdout = (logs / f"{name}.stdout").read_text(errors="replace").strip()
    if engine == "py":
        try:
            rec["record"] = json.loads(stdout.splitlines()[-1])
        except Exception as e:  # noqa: BLE001
            rec["record_error"] = repr(e)
        dirs = [out] if passes == 0 else [out / f"pass{k}" for k in range(passes + 1)]
        rec["coverage"] = [dict(zip(("wav_files", "digest_of_wav_hashes"), wav_digest(d)), dir=d.name) for d in dirs]
    else:
        tlp = logs / f"{name}.timeline.json"
        if tlp.exists():
            tl = json.loads(tlp.read_text())
            rec["record"] = {k: v for k, v in tl.items() if k != "spans"}
            rec["timeline_file"] = tlp.name
        try:
            rec["coverage"] = rust_coverage(out, passes)
        except Exception as e:  # noqa: BLE001
            rec["coverage_error"] = repr(e)
    # keep manifests/sidecars (Rust) for audit; drop WAV payloads after hashing
    for w in pathlib.Path(out).rglob("*.wav"):
        w.unlink()
    with open(rec_file, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{name}: rc={rc} wall={wall:.2f}s waited={waited:.0f}s load1={host_before['loadavg'][0]:.1f}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--cold-reps", type=int, default=3)
    ap.add_argument("--warm-reps", type=int, default=2)
    ap.add_argument("--warm-passes", type=int, default=3)
    ap.add_argument("--engines", default="py,rust,rustfma")
    ap.add_argument("--phases", default="cold,warm,attr")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    work = out / "work"
    rec_file = out / "raw.jsonl"
    engines = args.engines.split(",")
    corpus = pathlib.Path(args.corpus).resolve()
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    ident = {"git": git, "git_dirty_tracked": dirty.splitlines(), "corpus": str(corpus), "corpus_sha256": sha256_file(corpus),
             "corpus_lines": corpus.read_text(encoding="utf-8").count("\n"), "voice": args.voice,
             "binaries": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in BINS.items() if k in {e.split(":")[0] for e in engines}},
             "python": PY, "reference_script_sha256": sha256_file(HERE / "system_reference.py"),
             "args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename,
             "nproc": os.cpu_count(), "host_start": host_state()}
    (out / "identity.json").write_text(json.dumps(ident, indent=1))

    def env(engine=""):
        e = dict(os.environ)
        # PHASE 2: "rust:<precision>" = the rust binary with KOKORO_PRECISION=<precision>
        if ":" in engine:
            e["KOKORO_PRECISION"] = engine.split(":", 1)[1]
        else:
            e.pop("KOKORO_PRECISION", None)
        e["CUDA_VISIBLE_DEVICES"] = "0"
        e["KOKORO_FRONTEND_DIR"] = str(DATA / "frontend")
        e.pop("KOKORO_PROFILE", None)
        return e

    def cmd(engine, name, o, passes, attribute=False):
        if engine == "py":
            c = [PY, str(HERE / "system_reference.py"), "--input", str(corpus), "--out-dir", str(o), "--voice", args.voice, "--passes", str(passes)]
            return c + (["--attribute"] if attribute else [])
        c = [str(BINS[engine.split(":")[0]]), "synth", "--model-dir", str(SNAP), "--input", str(corpus), "--out-dir", str(o), "--voice", args.voice,
             "--timeline", str(logs / f"{name}.timeline.json")]
        return c + (["--bench-passes", str(passes)] if passes else [])

    phases = args.phases.split(",")
    if "cold" in phases:
        for k in range(args.cold_reps):
            order = engines if k % 2 == 0 else list(reversed(engines))
            for e in order:
                name = f"cold-{e}-r{k}"
                run(rec_file, logs, name, cmd(e, name, work / name, 0), env(e), work / name, e, 0)
    if "warm" in phases:
        for k in range(args.warm_reps):
            order = engines if k % 2 == 0 else list(reversed(engines))
            for e in order:
                name = f"warm-{e}-r{k}"
                run(rec_file, logs, name, cmd(e, name, work / name, args.warm_passes), env(e), work / name, e, args.warm_passes)
    if "attr" in phases:
        for e in engines:
            name = f"attr-{e}"
            run(rec_file, logs, name, cmd(e, name, work / name, args.warm_passes, attribute=True), env(e), work / name, e, args.warm_passes)
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
    print("done:", out)


if __name__ == "__main__":
    main()
