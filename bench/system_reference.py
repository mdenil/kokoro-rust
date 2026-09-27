"""Whole-system timing of the ORIGINAL production Python usage: line file -> one WAV per
line, exactly as production uses kokoro 0.9.4 — KPipeline(lang_code='a') per line, concatenate the
chunk audio, soundfile.write. Unchanged: torch default threads, default device selection (CUDA),
no batching, no optimization.

--single-wav writes the whole input as ONE WAV, <input stem>.wav (the audio of all lines in order,
written once at the end of the pass, inside the timed pass), matching kokoro's default output.

Modes (each invocation is ONE process; the driver runs cold replicates as separate processes):
  --passes 0   cold: import + pipeline/model load + one pass over the file (writes WAVs)
  --passes N   warm resident: import + load + 1 untimed warm-up pass + N timed passes, each into a
               fresh directory
  --attribute  add per-call timers (g2p, inference incl. its .cpu() sync, WAV write) for stage
               attribution. Timers only; the calls and their order are unchanged. Clean headline
               runs omit this.
Prints one JSON record (raw per-pass times) on stdout. Output hashing/coverage is done OUTSIDE this
process by the driver (so it is not inside the timed process for either engine).
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import time

T0 = time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--passes", type=int, default=0)
    ap.add_argument("--attribute", action="store_true")
    ap.add_argument("--single-wav", action="store_true", help="write one WAV for the whole input instead of one per line")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KPipeline
    t_import = time.perf_counter()
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    pipe.load_voice(args.voice)
    t_load = time.perf_counter()

    timers = {"g2p_s": 0.0, "infer_s": 0.0, "write_s": 0.0, "g2p_calls": 0, "infer_calls": 0}
    if args.attribute:
        g2p = pipe.g2p
        infer = KPipeline.infer

        def timed_g2p(*a, **k):
            t = time.perf_counter()
            r = g2p(*a, **k)
            timers["g2p_s"] += time.perf_counter() - t
            timers["g2p_calls"] += 1
            return r

        def timed_infer(*a, **k):
            t = time.perf_counter()
            r = infer(*a, **k)
            timers["infer_s"] += time.perf_counter() - t
            timers["infer_calls"] += 1
            return r

        pipe.g2p = timed_g2p
        KPipeline.infer = staticmethod(timed_infer)

    lines = pathlib.Path(args.input).read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    base = pathlib.Path(args.out_dir)

    def one_pass(outdir):
        shutil.rmtree(outdir, ignore_errors=True)
        outdir.mkdir(parents=True)
        for k in ("g2p_s", "infer_s", "write_s", "g2p_calls", "infer_calls"):
            timers[k] = 0 if k.endswith("calls") else 0.0
        t = time.perf_counter()
        first = None
        samples = 0
        whole = []
        for i, line in enumerate(lines):
            chunks = [r.audio.numpy() for r in pipe(line, voice=args.voice, speed=args.speed)]
            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            tw = time.perf_counter()
            if args.single_wav:
                whole.append(audio)
            else:
                sf.write(outdir / f"{i + 1:05d}.wav", audio, 24000)
            if args.attribute:
                timers["write_s"] += time.perf_counter() - tw
            if first is None:
                first = time.perf_counter() - t
            samples += audio.shape[0]
        if args.single_wav:
            tw = time.perf_counter()
            sf.write(outdir / f"{pathlib.Path(args.input).stem}.wav", np.concatenate(whole) if whole else np.zeros(0, dtype=np.float32), 24000)
            if args.attribute:
                timers["write_s"] += time.perf_counter() - tw
        wall = time.perf_counter() - t
        return {"wall_s": wall, "first_line_s": first, "samples": samples, "audio_s": samples / 24000, **dict(timers)}

    passes = []
    if args.passes == 0:
        passes.append(dict(one_pass(base), pass_=0, warmup=False))
    else:
        for p in range(args.passes + 1):
            passes.append(dict(one_pass(base / f"pass{p}"), pass_=p, warmup=p == 0))
    t_end = time.perf_counter()
    print(json.dumps({
        "engine": "production python: kokoro 0.9.4 KPipeline(lang_code='a') per line + soundfile.write",
        "output": "one WAV for the whole input" if args.single_wav else "one WAV per line",
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "cuda": torch.cuda.is_available(),
        "device": str(pipe.model.device) if pipe.model is not None else None, "attribute": args.attribute,
        "voice": args.voice, "speed": args.speed, "input": args.input, "input_lines": len(lines),
        "pid": os.getpid(), "import_s": t_import - T0, "load_s": t_load - t_import, "process_body_s": t_end - T0,
        "passes": passes,
    }))


if __name__ == "__main__":
    sys.exit(main())
