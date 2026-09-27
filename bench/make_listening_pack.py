"""PHASE 2 morning listening pack: RAW model-output WAVs (no normalization / trimming / resampling /
stitching), the same named public cases at every level, both voices, plus the unchanged production
Python reference. Built from a bench/phase2_quality.py run directory (which rendered the corpus at
every level). Worst-divergence cases (by spectral distance vs the f32 control) are added per level so
the pack is not only flattering examples.

Usage: python bench/make_listening_pack.py --quality DIR --out PACKDIR --cases name=line,... [--worst 2]
"""
import argparse
import hashlib
import json
import pathlib
import shutil


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", required=True, help="name=line,... (1-based lines of the quality corpus)")
    ap.add_argument("--worst", type=int, default=2)
    a = ap.parse_args()
    q = pathlib.Path(a.quality)
    man = json.loads((q / "manifest.json").read_text())
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = pathlib.Path(man["corpus"]).read_text(encoding="utf-8").split("\n")
    cases = dict((c.split("=")[0], int(c.split("=")[1])) for c in a.cases.split(","))
    stem = pathlib.Path(man["corpus"]).stem
    pack = {"status": "ALL non-f32 levels UNREVIEWED (experimental, phase 2). Raw model output.", "source_quality_run": str(q),
            "binary": man["binary"], "binary_sha256": man["binary_sha256"], "git": man["git"], "format": man["format"],
            "voice_speed": 1.0, "excitation_seed": "0 (per line: splitmix(seed, line<<16|chunk)); Python reference uses torch's RNG, so its noise differs",
            "cases": {}, "files": []}
    # worst cases: union over voices, so every voice has every case at every level
    worst = {}
    for voice, levels in man["results"].items():
        for level, rec in levels.items():
            if level in ("python-reference", "f32"):
                continue
            for _, line in rec["summary"].get("worst_lines_spec_vs_f32", [])[: a.worst]:
                worst.setdefault(line, []).append(f"{level}/{voice}")
    allc = list(cases.items()) + [(f"worst_line{line:03d}", line) for line in sorted(worst) if line not in cases.values()]
    pack["worst_case_origin"] = {f"worst_line{line:03d}": w for line, w in sorted(worst.items())}
    pack["metric_note"] = ("spec_db/rel vs f32 compare sample-aligned audio; on lines whose duration changed (predictor in reduced "
                           "precision: tf32, tf32all, fp16x, bf16x) the comparison is misaligned after the first changed phoneme, "
                           "so large values there mean 'different timing', not necessarily audible degradation")
    for voice, levels in man["results"].items():
        for name, line in allc:
            pack["cases"][name] = {"line": line, "text": lines[line - 1]}
            for level, rec in levels.items():
                if level == "python-reference":
                    src = q / rec["dir"] / f"line_{line:05d}.wav"
                    metrics = None
                else:
                    src = q / rec["dir"] / f"{stem}_{line:05d}.wav"
                    metrics = next((r for r in rec["lines"] if r["line"] == line), None)
                if not src.exists():
                    pack["files"].append({"case": name, "voice": voice, "level": level, "MISSING": str(src)})
                    continue
                dst = out / level / voice / f"{name}.wav"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                pack["files"].append({"case": name, "line": line, "voice": voice, "level": level, "file": str(dst.relative_to(out)), "sha256": sha(dst),
                                      "seconds": (metrics or {}).get("seconds"), "vs_f32": (metrics or {}).get("vs_f32"),
                                      "vs_reference": (metrics or {}).get("vs_reference"),
                                      "status": "reference (production Python)" if level == "python-reference" else ("f32 control" if level == "f32" else "UNREVIEWED")})
    (out / "manifest.json").write_text(json.dumps(pack, indent=1, ensure_ascii=False))
    missing = [f for f in pack["files"] if "MISSING" in f]
    print(f"pack: {len(pack['files']) - len(missing)} files, {len(missing)} missing, {len(pack['cases'])} cases -> {out}")


if __name__ == "__main__":
    main()
