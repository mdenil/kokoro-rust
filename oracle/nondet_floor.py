"""Measure the pinned reference's OWN nondeterminism floor (Phase 0 gate).

Protocol (frozen BEFORE results are seen):
  case s02_fox / af_heart / speed 1.0.
  A. frozen-noise repeatability: inject identical noise tensors, 3 runs per config;
     configs: cpu-t1, cpu-t8, cuda. Metric: max|Δ| and RMS(Δ) on waveform + pred_dur equality.
  B. cross-config deltas (cpu-t1 vs cpu-t8 vs cuda) under identical frozen noise.
  C. free-running noise (2 runs, cpu-t1): magnitude of the stochastic path itself.
All tolerances for the Rust subject derive from A/B. C documents why noise must be frozen.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

TEXT = "The quick brown fox jumps over the lazy dog."
VOICE = "af_heart"


def run_once(model, ps, ref_s, device, tap):
    import torch
    with tap:
        audio = model(ps, ref_s.to(device), 1.0, return_output=True)
    return audio.audio.numpy(), audio.pred_dur.numpy()


def main():
    common.assert_pins()
    import numpy as np
    import torch
    from kokoro.pipeline import KPipeline

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
    _, tokens = pipe.g2p(TEXT)
    chunks = [(gs, ps) for gs, ps, _ in pipe.en_tokenize(tokens) if ps]
    ps = chunks[0][1]
    pack = common.load_voice(VOICE)
    ref_s = pack[len(ps) - 1]

    # Record one canonical noise set (cpu, seed 1234)
    torch.set_num_threads(1)
    model_cpu = common.load_model("cpu")
    torch.manual_seed(1234)
    tap = common.NoiseTap(mode="record")
    ref_audio, ref_dur = run_once(model_cpu, ps, ref_s, "cpu", tap)
    noise = tap.recorded

    results = {"case": TEXT, "voice": VOICE, "phoneme_len": len(ps),
               "samples": int(ref_audio.shape[-1]), "configs": {}}

    def stats(a, b):
        n = min(a.shape[-1], b.shape[-1])
        d = a[..., :n] - b[..., :n]
        return {"len_a": int(a.shape[-1]), "len_b": int(b.shape[-1]),
                "max_abs": float(np.abs(d).max()), "rms": float(np.sqrt((d ** 2).mean())),
                "ref_rms": float(np.sqrt((a[..., :n] ** 2).mean()))}

    runs = {}
    for cfg, device, threads in [("cpu-t1", "cpu", 1), ("cpu-t8", "cpu", 8), ("cuda", "cuda", 1)]:
        torch.set_num_threads(threads)
        model = model_cpu if device == "cpu" else common.load_model("cuda")
        outs = []
        for i in range(3):
            tap = common.NoiseTap(mode="inject", tensors=[t.clone() for t in noise])
            a, d = run_once(model, ps, ref_s, device, tap)
            outs.append((a, d))
        runs[cfg] = outs[0]
        results["configs"][cfg] = {
            "repeat_run0_vs_run1": stats(outs[0][0], outs[1][0]),
            "repeat_run0_vs_run2": stats(outs[0][0], outs[2][0]),
            "pred_dur_identical": bool(all((o[1] == outs[0][1]).all() for o in outs)),
        }

    results["cross"] = {
        "cpu_t1_vs_cpu_t8": stats(runs["cpu-t1"][0], runs["cpu-t8"][0]),
        "cpu_t1_vs_cuda": stats(runs["cpu-t1"][0], runs["cuda"][0]),
        "pred_dur_t1_vs_t8": bool((runs["cpu-t1"][1] == runs["cpu-t8"][1]).all()),
        "pred_dur_t1_vs_cuda": bool((runs["cpu-t1"][1] == runs["cuda"][1]).all()),
    }

    # free-running stochastic path magnitude
    torch.set_num_threads(1)
    torch.manual_seed(1)
    a1, _ = run_once(model_cpu, ps, ref_s, "cpu", common.NoiseTap(mode="record"))
    torch.manual_seed(2)
    a2, _ = run_once(model_cpu, ps, ref_s, "cpu", common.NoiseTap(mode="record"))
    results["free_noise_seed1_vs_seed2"] = stats(a1, a2)

    results["runtime"] = common.runtime_meta()
    out = common.FIXTURES / "nondet_floor.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
