"""Interleaved A/B(/C...) of `kokoro synth` configurations on the warm resident path (and optionally
cold), for lever decisions. Each configuration runs in its own process (`--bench-passes P`), order
alternates per round (ABBA...). Raw stdout/stderr + timelines + a raw.jsonl are kept.

Usage: python bench/ab_synth.py --corpus FILE --out DIR --rounds 4 --passes 3 \
         --config A="--prep-threads 1" --config B="" [--bin A=path] [--cold]
"""
import argparse
import json
import os
import pathlib
import shlex
import shutil
import statistics
import subprocess
import time

import system_compare as sc

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = pathlib.Path(os.environ.get("KOKORO_DATA", "/data/mdenil/code/kokoro-rust"))
SNAP = DATA / "hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--config", action="append", required=True, help="NAME=extra synth args")
    ap.add_argument("--bin", action="append", default=[], help="NAME=binary path (default target/release/kokoro)")
    ap.add_argument("--cold", action="store_true", help="also time cold single-pass processes")
    ap.add_argument("--label", action="append", default=[], help="NAME=human description (e.g. 'same binary, single-worker ablation')")
    a = ap.parse_args()
    cfgs = dict(c.split("=", 1) for c in a.config)
    bins = {k: str(ROOT / "target/release/kokoro") for k in cfgs}
    bins.update(dict(b.split("=", 1) for b in a.bin))
    out = pathlib.Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", KOKORO_FRONTEND_DIR=str(DATA / "frontend"))
    # preserve every measured binary under its content hash (later rebuilds overwrite target/)
    keep = DATA / "bin"
    keep.mkdir(exist_ok=True)
    for k, b in list(bins.items()):
        h = sc.sha256_file(b)
        dst = keep / f"kokoro-{h[:12]}"
        if not dst.exists():
            shutil.copy2(b, dst)
        bins[k] = str(dst)
    corpus = pathlib.Path(a.corpus).resolve()
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.splitlines()
    (out / "identity.json").write_text(json.dumps({
        "git": git, "git_dirty_tracked": dirty, "corpus": str(corpus), "corpus_sha256": sc.sha256_file(corpus),
        "corpus_lines": corpus.read_text(encoding="utf-8").count("\n"), "configs": cfgs, "labels": dict(l.split("=", 1) for l in a.label),
        "binaries": {k: {"path": v, "sha256": sc.sha256_file(v)} for k, v in bins.items()}, "rounds": a.rounds, "passes": a.passes,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host_start": sc.host_state()}, indent=1))
    raw = open(out / "raw.jsonl", "a")
    names = list(cfgs)
    res = {k: {"warm": [], "cold": []} for k in names}
    for r in range(a.rounds):
        order = names if r % 2 == 0 else names[::-1]
        for mode in (["warm", "cold"] if a.cold else ["warm"]):
            for k in order:
                tag = f"{mode}-{k}-r{r}"
                work = DATA / "tmp/ab_synth_work" / tag
                shutil.rmtree(work, ignore_errors=True)
                tl = out / "logs" / f"{tag}.timeline.json"
                cmd = [bins[k], "synth", "--model-dir", str(SNAP), "--input", a.corpus, "--out-dir", str(work), "--timeline", str(tl)]
                cmd += (["--bench-passes", str(a.passes)] if mode == "warm" else []) + shlex.split(cfgs[k])
                host_before, waited = sc.wait_quiet()
                t = time.perf_counter()
                with open(out / "logs" / f"{tag}.stdout", "wb") as so, open(out / "logs" / f"{tag}.stderr", "wb") as se:
                    rc = subprocess.run(cmd, env=env, stdout=so, stderr=se).returncode
                wall = time.perf_counter() - t
                host_after = sc.host_state()
                d = json.loads(tl.read_text())
                cov = sc.rust_coverage(work, a.passes if mode == "warm" else 0)
                n_lines = corpus.read_text(encoding="utf-8").count("\n")
                passes = [p["pass_wall_s"] for p in d["passes"] if not (mode == "warm" and p["pass"] == 0)]
                ok = rc == 0 and all(p["complete"] for p in d["passes"]) and all(c["wav_files"] == n_lines and c["complete"] for c in cov)
                rec = {"tag": tag, "config": k, "args": cfgs[k], "bin": bins[k], "mode": mode, "rc": rc, "ok": ok, "wall_s": wall,
                       "pass_walls": passes, "load_s": d["load_s"], "frontend_load_s": d["frontend_load_s"],
                       "coverage": cov, "host_before": host_before, "host_after": host_after, "waited_for_quiet_s": waited, "synthesis": d.get("synthesis")}
                raw.write(json.dumps(rec) + "\n")
                raw.flush()
                res[k][mode].append(wall if mode == "cold" else statistics.median(passes))
                shutil.rmtree(work, ignore_errors=True)
                print(f"{tag}: ok={ok} wall={wall:.2f} passes={[round(x, 3) for x in passes]}", flush=True)
    summ = {}
    for k in names:
        for mode in res[k]:
            xs = res[k][mode]
            if xs:
                m = statistics.mean(xs)
                summ[f"{k}.{mode}"] = {"median": statistics.median(xs), "min": min(xs), "cv_pct": 100 * statistics.stdev(xs) / m if len(xs) > 1 else 0, "values": xs}
    base = names[0]
    for k in names:
        for mode in ("warm", "cold"):
            if f"{k}.{mode}" in summ:
                s = summ[f"{k}.{mode}"]
                ratio = summ[f"{base}.{mode}"]["median"] / s["median"]
                print(f"{k:>10} {mode}: median {s['median']:.3f}s cv {s['cv_pct']:.1f}% (vs {base}: {ratio:.3f}x) {[round(v, 3) for v in s['values']]}")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    raw.close()
    (out / "SHA256SUMS").write_text("".join(f"{sc.sha256_file(p)}  {p.relative_to(out)}\n" for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))


if __name__ == "__main__":
    main()
