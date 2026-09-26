"""Summarize an interim-comparison evidence dir (raw.jsonl + per-process receipts) -> summary.json.

Core scope: each process reports the median of its 3 timed in-process passes; the table reports
the median (and min/max, cv) across processes. Durations and chunk counts are cross-checked.
"""
import glob
import json
import pathlib
import statistics
import sys


def stats(xs):
    xs = list(xs)
    m = statistics.mean(xs)
    return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs),
            "cv_pct": 100 * statistics.stdev(xs) / m if len(xs) > 1 else 0.0, "values": xs}


def main():
    d = pathlib.Path(sys.argv[1])
    raw = [json.loads(l) for l in open(d / "raw.jsonl")]
    out = {"core": {}, "cold": {}, "bridge": {}}
    for voice in ("af_heart", "am_adam"):
        for eng in ("ref-prod", "ref-f32", "rust"):
            meds, audio, chunks, load = [], set(), set(), []
            for f in sorted(glob.glob(str(d / f"core-{eng}-{voice}-r*"))):
                p = pathlib.Path(f)
                rec = json.load(open(p if p.is_file() else next(p.glob("*.json"))))
                if eng == "rust":
                    meds.append(rec["total_s"]["median"])
                    audio.add(round(rec["audio_s"], 3))
                    chunks.add(rec["n_chunks"])
                    load.append(rec["load_s"])
                else:
                    inf = rec["scopes"]["inference"]
                    meds.append(inf["total_s"]["median"])
                    audio.add(round(inf["audio_s"], 3))
                    chunks.add(rec["n_chunks"])
            if meds:
                s = stats(meds)
                out["core"][f"{voice}/{eng}"] = {"warm_core_s": s, "audio_s": sorted(audio), "n_chunks": sorted(chunks),
                                                 "rtf": s["median"] / min(audio), "x_realtime": min(audio) / s["median"],
                                                 **({"load_s": stats(load)} if load else {})}
        r = out["core"].get(f"{voice}/rust")
        for ref in ("ref-prod", "ref-f32"):
            x = out["core"].get(f"{voice}/{ref}")
            if r and x:
                out["core"][f"{voice}/ratio_{ref}_over_rust"] = x["warm_core_s"]["median"] / r["warm_core_s"]["median"]
    for tag in ("cold/ref", "cold/rust-text(py-bridge)", "cold/rust-phonemes(native)"):
        rows = [x for x in raw if x["tag"] == tag]
        if rows:
            e = {"wall_s": stats(x["wall_s"] for x in rows)}
            if tag == "cold/ref":
                child = [json.loads(x["stdout_tail"].strip().splitlines()[-1]) for x in rows]
                for k in ("import_s", "load_s", "synth_s", "first_audio_s", "audio_s"):
                    e[k] = stats(c[k] for c in child)
            else:
                import re
                loads = [float(re.search(r"loaded model \+ voice in ([0-9.]+)s", x["stderr_tail"]).group(1)) for x in rows]
                synth = [re.search(r"([0-9.]+)s audio in ([0-9.]+)s", x["stderr_tail"]) for x in rows]
                e["load_s"] = stats(loads)
                e["audio_s"] = stats(float(m.group(1)) for m in synth)
                e["synth_loop_s"] = stats(float(m.group(2)) for m in synth)
            out["cold"][tag] = e
    br = [x for x in raw if x["tag"] == "bridge"]
    if br:
        out["bridge"] = {k: stats(x[k] for x in br) for k in ("startup_s", "g2p_s", "total_s")}
        out["bridge"]["chunks"] = sorted({x["chunks"] for x in br})
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    for k, v in out["core"].items():
        if isinstance(v, dict):
            print(f"{k:28s} warm {v['warm_core_s']['median']:.3f}s (cv {v['warm_core_s']['cv_pct']:.1f}%, n={v['warm_core_s']['n']}) audio {v['audio_s']} chunks {v['n_chunks']} RTF {v['rtf']:.4f} ({v['x_realtime']:.0f}x RT)")
        else:
            print(f"{k:28s} {v:.2f}x")
    for k, v in out["cold"].items():
        print(f"{k:28s} " + " ".join(f"{m}={s['median']:.2f}" for m, s in v.items()))
    if out["bridge"]:
        print("bridge", {k: round(v["median"], 2) for k, v in out["bridge"].items() if isinstance(v, dict)}, out["bridge"]["chunks"])


if __name__ == "__main__":
    main()
