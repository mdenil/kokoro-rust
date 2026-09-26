"""Attainability evidence for the BINDING original E2E gates (rel<=0.019, max<=0.033,
corr>=0.9995 vs the frozen cpu-t1 fixture audio, identical injected noise).

Evaluates reference-produced waveforms only (no Rust subject involved):
  f64       pinned reference module graph in float64 (fixtures/f64/*, oracle/gen_f64.py)
  cpu-t2/4/8 pinned reference f32 on CPU at other thread counts (fixtures/f64/*)
  cuda-tf32 pinned reference on RTX 4090, production default (cuDNN TF32 on)  [run here]
  cuda-f32  pinned reference on RTX 4090 with TF32 disabled                   [run here]
Writes fixtures/attainability.json (does not modify any fixture).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

GATE_REL, GATE_MAX, GATE_CORR = 0.019, 0.033, 0.9995


def metrics(a, ref):
    import numpy as np
    a = a.astype(np.float64)
    ref = ref.astype(np.float64)
    d = a - ref
    return {"rel": float(np.sqrt((d ** 2).sum() / (ref ** 2).sum())), "max": float(np.abs(d).max()),
            "corr": float(np.corrcoef(a, ref)[0, 1])}


def verdict(m):
    return m["rel"] <= GATE_REL and m["max"] <= GATE_MAX and m["corr"] >= GATE_CORR


def main():
    common.assert_pins()
    import numpy as np
    import torch
    from safetensors.numpy import load_file
    torch.set_num_threads(1)
    model = common.load_model("cuda")
    root = common.FIXTURES / "cpu-t1"
    out = {"gates": {"rel": GATE_REL, "max": GATE_MAX, "corr": GATE_CORR}, "cases": {}, "runtime": common.runtime_meta()}
    for case in sorted(p.name for p in root.iterdir()):
        d = np.load(root / case / "fixture.npz")
        meta = json.loads((root / case / "meta.json").read_text())
        ref = d["audio"]
        row = {}
        f64 = load_file(str(common.FIXTURES / "f64" / case / "f64.safetensors"))
        row["f64"] = metrics(f64["audio_f64"], ref)
        for th in (2, 4, 8):
            row[f"cpu-t{th}"] = metrics(f64[f"audio_f32_t{th}"], ref)
        for tf32 in (True, False):
            torch.backends.cudnn.allow_tf32 = tf32
            noise = [torch.from_numpy(d[k]).clone() for k in ("noise.rand_ini", "noise.sine", "noise.uv")]
            with common.NoiseTap(mode="inject", tensors=noise):
                o = model(meta["phonemes"], torch.from_numpy(d["ref_s"]), meta["speed"], return_output=True)
            assert (o.pred_dur.numpy() == d["pred_dur"]).all(), (case, tf32)
            row["cuda-tf32" if tf32 else "cuda-f32"] = metrics(o.audio.numpy(), ref)
        torch.backends.cudnn.allow_tf32 = True
        for k in row:
            row[k]["pass"] = verdict(row[k])
        out["cases"][case] = row
        print(f"{case:32s} " + "  ".join(f"{k}:{'P' if v['pass'] else 'F'}(max {v['max']:.3f})" for k, v in row.items()), flush=True)
    totals = {k: sum(out["cases"][c][k]["pass"] for c in out["cases"]) for k in next(iter(out["cases"].values()))}
    out["passes_out_of_15"] = totals
    print("passes out of", len(out["cases"]), ":", totals)
    (common.FIXTURES / "attainability.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
