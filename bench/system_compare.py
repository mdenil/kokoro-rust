"""Whole-system comparison driver: line file -> ALL per-line WAVs (+ Rust metadata).

Engines:
  py      ORIGINAL production Python as used (bench/system_reference.py; unchanged KPipeline per line)
  rust     the native binary under test (target/release/kokoro or $KOKORO_BIN_RUST), default settings
  accepted the immutable accepted BF16x reference artifact ($KOKORO_BIN_ACCEPTED), run exactly
           as accepted (KOKORO_PRECISION=bf16x, default settings): the pre/post reference
Phases (separate, never mixed):
  cold    fresh process per replicate: startup + load + one pass; replicates interleaved across engines.
          With --cache-warmup, each engine first runs once untimed (recorded as cachewarm-<engine>,
          excluded from results) so the timed runs start with the disk/page cache warm.
  warm    one resident process per replicate: load, 1 untimed warm-up pass, P timed passes
  attr    one ATTRIBUTION process per engine (Python per-call timers; Rust timeline is always on) —
          stage breakdowns only, never headline timings
Evidence (raw, unfiltered): <out>/raw.jsonl (one record per process: command, env, wall, rc, host state
before/after, parsed stdout record, coverage), <out>/logs/<run>.{stdout,stderr}, Rust timelines,
manifests; WAVs are hashed then deleted (coverage + identity kept).
--output single: both engines write ONE WAV for the whole input (kokoro's default output; the Python
reference with --single-wav); coverage is read from that WAV. --output per-line (default): the
original per-line comparison.
Usage: python bench/system_compare.py --corpus FILE --out DIR [--output single] [--cold-reps 3 --warm-reps 2 --warm-passes 3]
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
if not os.environ.get("KOKORO_DATA"):
    raise SystemExit("KOKORO_DATA is not set: point it at the data root (docs/PORTABILITY.md; e.g. source scripts/env.sh)")
DATA = pathlib.Path(os.environ["KOKORO_DATA"])
PY = os.environ.get("KOKORO_PY", str(DATA / "reference/venv-prod/bin/python"))
SNAP = DATA / "hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987"
BINS = {"rust": pathlib.Path(os.environ.get("KOKORO_BIN_RUST", ROOT / "target/release/kokoro")),
        "accepted": pathlib.Path(os.environ.get("KOKORO_BIN_ACCEPTED", DATA / "bin/phase2-9b39d48-6fde9d88990a"))}


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


def wav_info(p):
    """(sample count, sha256) of a PCM16 mono 24 kHz WAV with the canonical 44-byte header."""
    b = pathlib.Path(p).read_bytes()
    assert b[:4] == b"RIFF" and b[8:16] == b"WAVEfmt " and b[36:40] == b"data", f"{p}: unexpected WAV layout"
    fmt, ch, rate, bits = int.from_bytes(b[20:22], "little"), int.from_bytes(b[22:24], "little"), int.from_bytes(b[24:28], "little"), int.from_bytes(b[34:36], "little")
    data = int.from_bytes(b[40:44], "little")
    assert (fmt, ch, rate, bits) == (1, 1, 24000, 16) and len(b) == 44 + data, f"{p}: not PCM16 mono 24 kHz"
    return data // 2, hashlib.sha256(b).hexdigest()


def single_coverage(out, passes, stem):
    dirs = [out] if passes == 0 else [out / f"pass{k}" for k in range(passes + 1)]
    cov = []
    for d in dirs:
        files = sorted(f.name for f in d.iterdir() if f.is_file())
        entry = {"dir": d.name, "files": files}
        w = d / f"{stem}.wav"
        if w.exists():
            entry["samples"], entry["wav_sha256"] = wav_info(w)
            entry["audio_s"] = entry["samples"] / 24000
        cov.append(entry)
    return cov


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


def run(rec_file, logs, name, cmd, env, out, engine, passes, single_stem=None):
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
    if single_stem is not None:
        if engine == "py":
            try:
                rec["record"] = json.loads(stdout.splitlines()[-1])
            except Exception as e:  # noqa: BLE001
                rec["record_error"] = repr(e)
        else:
            tlp = logs / f"{name}.timeline.json"
            if tlp.exists():
                tl = json.loads(tlp.read_text())
                rec["record"] = {k: v for k, v in tl.items() if k != "spans"}
                rec["timeline_file"] = tlp.name
        try:
            rec["coverage"] = single_coverage(out, passes, single_stem)
        except Exception as e:  # noqa: BLE001
            rec["coverage_error"] = repr(e)
    elif engine == "py":
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
    ap.add_argument("--engines", default="py,rust")
    ap.add_argument("--phases", default="cold,warm,attr")
    ap.add_argument("--output", choices=["per-line", "single"], default="per-line")
    ap.add_argument("--cache-warmup", action="store_true", help="one untimed run per engine before the cold phase")
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
             "binaries": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in BINS.items() if k in engines},
             "python": PY, "reference_script_sha256": sha256_file(HERE / "system_reference.py"),
             "compare_script_sha256": sha256_file(pathlib.Path(__file__)), "args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename,
             "nproc": os.cpu_count(), "host_start": host_state()}
    (out / "identity.json").write_text(json.dumps(ident, indent=1))

    def env(engine=""):
        e = dict(os.environ)
        # the accepted artifact selects its (accepted) numerical mode by environment
        if engine == "accepted":
            e["KOKORO_PRECISION"] = "bf16x"
        else:
            e.pop("KOKORO_PRECISION", None)
        e["KOKORO_FRONTEND_DIR"] = str(DATA / "frontend")
        return e

    single = args.output == "single"

    def cmd(engine, name, o, passes, attribute=False):
        if engine == "py":
            c = [PY, str(HERE / "system_reference.py"), "--input", str(corpus), "--out-dir", str(o), "--voice", args.voice, "--passes", str(passes)]
            return c + (["--attribute"] if attribute else []) + (["--single-wav"] if single else [])
        if single:
            # the default command; the timeline (a small JSON) only where per-pass times are needed
            c = [str(BINS[engine]), "synth", str(corpus), "--model-dir", str(SNAP), "--out-dir", str(o), "--voice", args.voice]
            if passes or attribute:
                c += ["--timeline", str(logs / f"{name}.timeline.json")]
            return c + (["--bench-passes", str(passes)] if passes else [])
        # current binaries write one WAV per input by default; this harness reads per-line outputs
        per_line = ["--per-line", "--diagnostics"] if engine == "rust" else []
        c = [str(BINS[engine]), "synth"] + per_line + ["--model-dir", str(SNAP), "--input", str(corpus), "--out-dir", str(o), "--voice", args.voice,
             "--timeline", str(logs / f"{name}.timeline.json")]
        return c + (["--bench-passes", str(passes)] if passes else [])

    phases = args.phases.split(",")
    if "cold" in phases:
        if args.cache_warmup:
            for e in engines:
                name = f"cachewarm-{e}"
                run(rec_file, logs, name, cmd(e, name, work / name, 0), env(e), work / name, e, 0, corpus.stem if single else None)
        for k in range(args.cold_reps):
            order = engines if k % 2 == 0 else list(reversed(engines))
            for e in order:
                name = f"cold-{e}-r{k}"
                run(rec_file, logs, name, cmd(e, name, work / name, 0), env(e), work / name, e, 0, corpus.stem if single else None)
    if "warm" in phases:
        for k in range(args.warm_reps):
            order = engines if k % 2 == 0 else list(reversed(engines))
            for e in order:
                name = f"warm-{e}-r{k}"
                run(rec_file, logs, name, cmd(e, name, work / name, args.warm_passes), env(e), work / name, e, args.warm_passes, corpus.stem if single else None)
    if "attr" in phases:
        for e in engines:
            name = f"attr-{e}"
            run(rec_file, logs, name, cmd(e, name, work / name, args.warm_passes, attribute=True), env(e), work / name, e, args.warm_passes, corpus.stem if single else None)
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
    print("done:", out)


if __name__ == "__main__":
    main()
