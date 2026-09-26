"""Per-case reference reorder floor (NONDET_FLOOR.md Revision 1).

For every frozen cpu-t1 fixture, re-run the pinned reference on CPU with the fixture's
injected noise at torch thread counts {2, 4, 8} and compare to the frozen t1 audio.
Output: $KOKORO_DATA/fixtures/floor_per_case.json
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402


def main():
    common.assert_pins()
    import numpy as np
    import torch
    model = common.load_model("cpu")
    root = common.FIXTURES / "cpu-t1"
    out = {"threads": [2, 4, 8], "cases": {}, "runtime": common.runtime_meta()}
    for case in sorted(p.name for p in root.iterdir()):
        d = np.load(root / case / "fixture.npz")
        meta = json.loads((root / case / "meta.json").read_text())
        want = d["audio"].astype(np.float64)
        per = {}
        for th in out["threads"]:
            torch.set_num_threads(th)
            noise = [torch.from_numpy(d[k]).clone() for k in ("noise.rand_ini", "noise.sine", "noise.uv")]
            with common.NoiseTap(mode="inject", tensors=noise):
                o = model(meta["phonemes"], torch.from_numpy(d["ref_s"]), meta["speed"], return_output=True)
            a = o.audio.numpy().astype(np.float64)
            assert a.shape == want.shape, (case, th, a.shape, want.shape)
            assert (o.pred_dur.numpy() == d["pred_dur"]).all(), (case, th, "durations differ")
            diff = a - want
            per[f"t{th}"] = {
                "rel_l2": float(np.sqrt((diff ** 2).sum() / (want ** 2).sum())),
                "max_abs": float(np.abs(diff).max()),
                "corr": float(np.corrcoef(a, want)[0, 1]),
            }
        per["floor_rel"] = max(v["rel_l2"] for k, v in per.items() if k.startswith("t"))
        per["floor_max"] = max(v["max_abs"] for k, v in per.items() if k.startswith("t"))
        out["cases"][case] = per
        print(f"{case:34s} floor_rel={per['floor_rel']:.3e} floor_max={per['floor_max']:.3e}", flush=True)
    (common.FIXTURES / "floor_per_case.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
