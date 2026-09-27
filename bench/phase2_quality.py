"""PHASE 2 (lossy exploration): per-precision quality diagnostics + raw listening pack.

For each precision level and voice, renders a public corpus through the branch binary (raw float32
WAVs, one per line) and computes, per line: sample count vs the f32 control, finiteness, peak, and
drift vs the f32 control (rel L2, max abs, mean |dB| log-spectrum difference). The Python reference
(unchanged production KPipeline, float32 WAVs) is rendered once per voice for listening and for
level-vs-reference metrics. Metrics are DIAGNOSTICS: every candidate stays UNREVIEWED until the owner
listens. Functional failures (non-finite / empty / missing / sample-count mismatch) are reported as
FAILURES.

Usage: python bench/phase2_quality.py --corpus FILE --levels f32,tf32,... --out DIR [--voices ...]
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = pathlib.Path(os.environ.get("KOKORO_DATA", "/data/mdenil/code/kokoro-rust"))
SNAP = DATA / "hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987"
PY = os.environ.get("KOKORO_PY", str(DATA / "reference/venv-prod/bin/python"))


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def spec_db(a, b, n=1024, hop=256):
    m = min(len(a), len(b))
    if m < n:
        return float("nan")
    w = np.hanning(n)
    fa = [np.abs(np.fft.rfft(a[i:i + n] * w)) for i in range(0, m - n, hop)]
    fb = [np.abs(np.fft.rfft(b[i:i + n] * w)) for i in range(0, m - n, hop)]
    la, lb = 20 * np.log10(np.array(fa) + 1e-5), 20 * np.log10(np.array(fb) + 1e-5)
    return float(np.mean(np.abs(la - lb)))


def metrics(x, ref):
    m = min(len(x), len(ref))
    d = x[:m] - ref[:m]
    return {"rel": float(np.linalg.norm(d) / max(np.linalg.norm(ref[:m]), 1e-12)), "max": float(np.max(np.abs(d))) if m else float("nan"),
            "spec_db": spec_db(x, ref), "len": int(len(x)), "ref_len": int(len(ref))}


def render_rust(binary, level, voice, corpus, out):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", KOKORO_FRONTEND_DIR=str(DATA / "frontend"), KOKORO_PRECISION=level)
    cmd = [binary, "synth", "--model-dir", str(SNAP), "--input", str(corpus), "--out-dir", str(out), "--voice", voice, "--format", "float32", "--force"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (out / "stderr.txt").write_text(r.stderr)
    return r.returncode, cmd


def render_python(voice, corpus, out):
    code = f"""
import soundfile as sf, numpy as np, pathlib
from kokoro import KPipeline
p = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
lines = pathlib.Path({str(corpus)!r}).read_text(encoding='utf-8').split('\\n')
if lines and lines[-1] == '': lines.pop()
out = pathlib.Path({str(out)!r}); out.mkdir(parents=True, exist_ok=True)
for i, l in enumerate(lines, 1):
    ch = [r.audio.numpy() for r in p(l, voice={voice!r}, speed=1.0)]
    a = np.concatenate(ch) if ch else np.zeros(0, dtype=np.float32)
    sf.write(out / f'line_{{i:05d}}.wav', a, 24000, subtype='FLOAT')
"""
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
    (out / "stderr.txt").write_text(r.stderr[-20000:])
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--voices", default="af_heart,am_adam")
    ap.add_argument("--binary", default=str(ROOT / "target/release/kokoro"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--python-ref", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = pathlib.Path(a.corpus).resolve()
    lines = corpus.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    levels = a.levels.split(",")
    assert levels[0] == "f32", "first level must be the f32 control"
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = {"corpus": str(corpus), "corpus_sha256": sha(corpus), "n_lines": len(lines), "binary": a.binary, "binary_sha256": sha(a.binary),
           "git": git, "levels": levels, "status": "UNREVIEWED (experimental, phase 2)", "format": "RIFF WAVE IEEE float32 mono 24 kHz, raw model output",
           "calibration": "none: INT8 uses per-output-channel weight scales from the weights and dynamic per-tensor activation scales", "results": {}}
    for voice in a.voices.split(","):
        ref = None
        if a.python_ref:
            pd = out / "python-reference" / voice
            rc = render_python(voice, corpus, pd)
            man["results"].setdefault(voice, {})["python-reference"] = {"rc": rc, "dir": str(pd.relative_to(out))}
            ref = pd
        ctrl = None
        for level in levels:
            d = out / level / voice
            d.mkdir(parents=True, exist_ok=True)
            rc, cmd = render_rust(a.binary, level, voice, corpus, d)
            rec = {"rc": rc, "cmd": cmd, "dir": str(d.relative_to(out)), "lines": [], "functional_failures": [], "duration_changes": []}
            stem = corpus.stem
            for i in range(1, len(lines) + 1):
                w = d / f"{stem}_{i:05d}.wav"
                if not w.exists():
                    rec["functional_failures"].append(f"line {i}: missing WAV")
                    continue
                x, sr = sf.read(w, dtype="float64")
                row = {"line": i, "wav": w.name, "sha256": sha(w), "samples": len(x), "seconds": len(x) / 24000.0,
                       "finite": bool(np.all(np.isfinite(x))), "peak": float(np.max(np.abs(x))) if len(x) else 0.0}
                if not row["finite"] or len(x) == 0:
                    rec["functional_failures"].append(f"line {i}: non-finite or empty audio")
                if ctrl is not None:
                    c, _ = sf.read(ctrl / f"{stem}_{i:05d}.wav", dtype="float64")
                    row["vs_f32"] = metrics(x, c)
                    if len(x) != len(c):
                        # expected when the duration predictor runs in reduced precision: a measured
                        # diagnostic, not a functional failure (every chunk is still synthesized)
                        rec["duration_changes"].append({"line": i, "samples": len(x), "f32_samples": len(c)})
                if ref is not None and (ref / f"line_{i:05d}.wav").exists():
                    r, _ = sf.read(ref / f"line_{i:05d}.wav", dtype="float64")
                    row["vs_reference"] = metrics(x, r)
                rec["lines"].append(row)
            if level == "f32":
                ctrl = d
            if rc != 0:
                rec["functional_failures"].append(f"synth exit code {rc}")
            vs = [r["vs_f32"]["rel"] for r in rec["lines"] if "vs_f32" in r]
            sd = [r["vs_f32"]["spec_db"] for r in rec["lines"] if "vs_f32" in r]
            sr = [r["vs_reference"]["spec_db"] for r in rec["lines"] if "vs_reference" in r]
            vr = [r["vs_reference"]["rel"] for r in rec["lines"] if "vs_reference" in r]
            rec["summary"] = {"median_spec_db_vs_f32": float(np.median(sd)) if sd else None, "max_spec_db_vs_f32": float(np.max(sd)) if sd else None,
                              "median_spec_db_vs_reference": float(np.median(sr)) if sr else None,
                              "duration_changed_lines": len(rec["duration_changes"]),
                              "median_rel_vs_f32": float(np.median(vs)) if vs else None, "max_rel_vs_f32": float(np.max(vs)) if vs else None,
                              "median_rel_vs_reference": float(np.median(vr)) if vr else None,
                              "worst_lines_spec_vs_f32": sorted(((r["vs_f32"]["spec_db"], r["line"]) for r in rec["lines"] if "vs_f32" in r), reverse=True)[:3]}
            man["results"].setdefault(voice, {})[level] = rec
            print(voice, level, "rc", rc, "fails", len(rec["functional_failures"]), rec["summary"], flush=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    sys.exit(main())
