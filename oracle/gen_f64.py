"""Float64 'true-math' oracle: the pinned reference module graph executed in float64 on CPU,
with each fixture's recorded noise injected (cast to f64). Used to measure how far each f32
implementation (torch t1/t2/t4/t8, Rust subject) sits from exact arithmetic.

Output per case: fixtures/f64/<case>/f64.safetensors with audio, F0_pred, N_pred (float64)
and pred_dur; plus fixtures/f64/torch_f32_reorders.safetensors-style per-case reorder audio.
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
    from safetensors.numpy import save_file
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float64)
    m64 = common.load_model("cpu").double()
    m64.decoder.generator.stft.window = m64.decoder.generator.stft.window.double()
    torch.set_default_dtype(torch.float32)
    m32 = common.load_model("cpu")

    root = common.FIXTURES / "cpu-t1"
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for case in sorted(p.name for p in root.iterdir()):
        if only and only not in case:
            continue
        d = np.load(root / case / "fixture.npz")
        meta = json.loads((root / case / "meta.json").read_text())
        out = {}

        torch.set_default_dtype(torch.float64)
        noise = [torch.from_numpy(d[k]).double() for k in ("noise.rand_ini", "noise.sine", "noise.uv")]
        with common.NoiseTap(mode="inject", tensors=noise), common.SeamRecorder(m64, keep_dtype=True) as rec:
            o = m64(meta["phonemes"], torch.from_numpy(d["ref_s"]).double(), meta["speed"], return_output=True)
        torch.set_default_dtype(torch.float32)
        assert o.audio.dtype == torch.float64, o.audio.dtype
        out["audio_f64"] = o.audio.numpy().astype(np.float64)
        out["pred_dur_f64"] = o.pred_dur.numpy().astype(np.int64)
        out["F0_f64"] = rec.data["decoder.arg1"].astype(np.float64)
        out["N_f64"] = rec.data["decoder.arg2"].astype(np.float64)
        # f32 reorders of the same case (torch thread counts) for truth-distance comparison
        for th in (1, 2, 4, 8):
            torch.set_num_threads(th)
            noise = [torch.from_numpy(d[k]).clone() for k in ("noise.rand_ini", "noise.sine", "noise.uv")]
            with common.NoiseTap(mode="inject", tensors=noise):
                o32 = m32(meta["phonemes"], torch.from_numpy(d["ref_s"]), meta["speed"], return_output=True)
            out[f"audio_f32_t{th}"] = o32.audio.numpy()
        torch.set_num_threads(8)
        same_dur = bool((out["pred_dur_f64"] == d["pred_dur"]).all())
        outp = common.FIXTURES / "f64" / case
        outp.mkdir(parents=True, exist_ok=True)
        save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, str(outp / "f64.safetensors"))
        a64 = out["audio_f64"]

        def rel(x):
            return float(np.sqrt(((x.astype(np.float64) - a64) ** 2).sum() / (a64 ** 2).sum()))
        print(f"{case:34s} dur_same_as_f32={same_dur} torch-f32 rel-to-truth: "
              + " ".join(f"t{th}={rel(out[f'audio_f32_t{th}']):.3e}" for th in (1, 2, 4, 8)), flush=True)


if __name__ == "__main__":
    main()
