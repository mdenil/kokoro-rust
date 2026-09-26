"""Summarize a bench/system_compare.py evidence dir (raw.jsonl) into headline + attribution tables.
Usage: python bench/summarize_system.py <dir> [--json out.json]
Headline numbers come only from cold/warm runs; attribution only from attr runs (labelled).
"""
import json
import pathlib
import statistics
import sys


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = statistics.mean(xs)
    return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs),
            "cv_pct": 100 * statistics.stdev(xs) / m if len(xs) > 1 and m else 0.0, "values": xs}


def fmt(s, unit="s"):
    if not s:
        return "—"
    return f"{s['median']:.2f}{unit} (cv {s['cv_pct']:.1f}%, n={s['n']}; {', '.join(f'{v:.2f}' for v in s['values'])})"


def main():
    d = pathlib.Path(sys.argv[1])
    recs = [json.loads(l) for l in (d / "raw.jsonl").read_text().splitlines()]
    ident = json.loads((d / "identity.json").read_text())
    out = {"identity": {k: ident[k] for k in ("git", "corpus_sha256", "corpus_lines", "voice", "binaries", "started")}, "engines": {}}
    engines = sorted({r["engine"] for r in recs})
    lines = []
    for e in engines:
        cold = [r for r in recs if r["engine"] == e and r["run"].startswith("cold-")]
        warm = [r for r in recs if r["engine"] == e and r["run"].startswith("warm-")]
        attr = [r for r in recs if r["engine"] == e and r["run"].startswith("attr-")]
        bad = [r["run"] for r in cold + warm + attr if r["rc"] != 0]
        cov_ok = True
        audio = None
        for r in cold + warm + attr:
            cov = r.get("coverage") or r.get("record", {}).get("coverage") or []
            for c in cov:
                if c.get("wav_files") != ident["corpus_lines"] or c.get("complete", True) is not True:
                    cov_ok = False
        if e == "py":
            passes = [p for r in warm for p in r["record"]["passes"] if not p["warmup"]]
            audio = cold[0]["record"]["passes"][0]["audio_s"] if cold else None
        else:
            passes = [p for r in warm for p in r["record"]["passes"] if not p["warmup"]]
            audio = cold[0]["record"]["passes"][0]["audio_s"] if cold else None
        cold_wall = stats([r["wall_s"] for r in cold])
        warm_pass = stats([p["pass_wall_s"] if e != "py" else p["wall_s"] for p in passes])
        ent = {"cold_wall_s": cold_wall, "warm_pass_s": warm_pass, "audio_s": audio, "failed_runs": bad, "coverage_ok": cov_ok,
               "loadavg_1m": stats([r["host_before"]["loadavg"][0] for r in cold + warm])}
        if e == "py":
            ent["cold_parts"] = {k: stats([r["record"][k] for r in cold]) for k in ("import_s", "load_s")}
            ent["cold_parts"]["pass_s"] = stats([r["record"]["passes"][0]["wall_s"] for r in cold])
            if attr:
                ps = [p for p in attr[0]["record"]["passes"] if not p["warmup"]]
                ent["attribution_warm"] = {k: stats([p[k] for p in ps]) for k in ("wall_s", "g2p_s", "infer_s", "write_s")}
        else:
            ent["cold_parts"] = {
                "model_load_s": stats([r["record"]["load_s"] for r in cold]),
                "frontend_load_s": stats([r["record"]["frontend_load_s"] for r in cold]),
                "pass_s": stats([r["record"]["passes"][0]["pass_wall_s"] for r in cold]),
                "outside_main_s": stats([r["wall_s"] - r["record"]["exit_s"] for r in cold]),
            }
            src = attr[0]["record"]["passes"] if attr else []
            ps = [p for p in src if not p["warmup"]]
            if ps:
                def st(p, k):
                    return p["timing"]["stages"].get(k, {}).get("busy_s", 0.0)
                ent["attribution_warm"] = {
                    "pass_wall_s": stats([p["pass_wall_s"] for p in ps]),
                    "prepare.frontend_busy_s": stats([st(p, "prepare.frontend") for p in ps]),
                    "gpu.synth_busy_s": stats([st(p, "gpu.synth") for p in ps]),
                    "gpu.recv_wait_s (starved)": stats([st(p, "gpu.recv_wait") for p in ps]),
                    "gpu.send_wait_s (writer backpressure)": stats([st(p, "gpu.send_wait") for p in ps]),
                    "writer.write_busy_s": stats([st(p, "writer.write") for p in ps]),
                    "union_any_work_s": stats([p["timing"]["thread_work_union_s"]["any_pipeline_work"] for p in ps]),
                }
        out["engines"][e] = ent
        lines.append(f"## {e}  (audio {audio:.1f} s; coverage ok: {cov_ok}; failed runs: {bad or 'none'})" if audio else f"## {e}")
        lines.append(f"cold process wall: {fmt(cold_wall)}")
        lines.append(f"warm resident pass: {fmt(warm_pass)}")
        for k, v in ent["cold_parts"].items():
            lines.append(f"  cold part {k}: {fmt(v)}")
        for k, v in ent.get("attribution_warm", {}).items():
            lines.append(f"  [attribution run] {k}: {fmt(v)}")
    py = out["engines"].get("py")
    for e in engines:
        if e == "py" or not py:
            continue
        r = out["engines"][e]
        for k in ("cold_wall_s", "warm_pass_s"):
            if py[k] and r[k]:
                lines.append(f"RATIO py/{e} {k}: {py[k]['median'] / r[k]['median']:.2f}x (min/min {py[k]['min'] / r[k]['min']:.2f}x)")
    txt = "\n".join(lines)
    print(txt)
    (d / "SUMMARY.txt").write_text(txt + "\n")
    (d / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
