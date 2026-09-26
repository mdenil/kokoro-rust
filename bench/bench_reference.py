"""Baseline timing of the pinned PRODUCTION reference (kokoro 0.9.4 / torch 2.12.1).

Scopes (reported separately, never mixed):
  frontend   misaki G2P + KPipeline chunking only (model=False), per corpus line
  inference  KModel forward per phoneme chunk (precomputed), device-synchronized, warm
  batch      resident KPipeline over the whole corpus (G2P + inference + concat), warm
  cold       fresh process: import + load + synth corpus + write WAVs (see --cold-child)

Usage:
  python bench/bench_reference.py --device cuda --threads 8 --reps 5 --out <dir>
Receipts: <out>/reference_<device>_t<threads>.json
"""
import argparse
import json
import os
import pathlib
import resource
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
import common  # noqa: E402

SR = 24000


def host_state():
    st = {"loadavg": os.getloadavg()}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        st["gpus"] = q.stdout.strip().splitlines()
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        st["gpu_apps"] = a.stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        st["gpus"] = f"unavailable: {e}"
    return st


def summarize(xs):
    xs = list(xs)
    m = statistics.mean(xs)
    return {"n": len(xs), "mean": m, "median": statistics.median(xs), "min": min(xs), "max": max(xs),
            "cv_pct": (100 * statistics.stdev(xs) / m) if len(xs) > 1 and m > 0 else 0.0}


def load_corpus(path):
    return [l for l in pathlib.Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def cold_child(args):
    """Runs in a fresh interpreter: the production-shaped end-to-end path."""
    t0 = time.perf_counter()
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KPipeline
    t_import = time.perf_counter()
    torch.set_num_threads(args.threads)
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=args.device)
    pipe.load_voice(args.voice)
    t_load = time.perf_counter()
    lines = load_corpus(args.corpus)
    outdir = pathlib.Path(args.wav_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    first_audio = None
    total_samples = 0
    for i, line in enumerate(lines):
        chunks = [r.audio.numpy() for r in pipe(line, voice=args.voice, speed=args.speed)]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if first_audio is None:
            first_audio = time.perf_counter()
        sf.write(outdir / f"{i:04d}.wav", audio, SR)
        total_samples += audio.shape[0]
    t_end = time.perf_counter()
    print(json.dumps({"import_s": t_import - t0, "load_s": t_load - t_import,
                      "first_audio_s": first_audio - t0, "synth_s": t_end - t_load,
                      "total_in_process_s": t_end - t0, "audio_s": total_samples / SR,
                      "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--corpus", default=str(HERE / "corpus_alice_ch1.txt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--scopes", default="frontend,inference,batch,cold")
    ap.add_argument("--cold-child", action="store_true")
    ap.add_argument("--wav-dir", default=None)
    args = ap.parse_args()
    if args.cold_child:
        return cold_child(args)

    common.assert_pins()
    import torch
    from kokoro.pipeline import KPipeline
    torch.set_num_threads(args.threads)
    scopes = args.scopes.split(",")
    out = pathlib.Path(args.out or (common.DATA / "evidence/baseline" / time.strftime("%Y%m%d-%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)
    lines = load_corpus(args.corpus)
    import hashlib
    rec = {"device": args.device, "threads": args.threads, "voice": args.voice, "speed": args.speed,
           "reps": args.reps, "corpus": args.corpus, "corpus_lines": len(lines),
           "corpus_sha256": hashlib.sha256(pathlib.Path(args.corpus).read_bytes()).hexdigest(),
           "runtime": common.runtime_meta(), "host_before": host_state(), "scopes": {}}

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    quiet = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)

    if "frontend" in scopes:
        per_rep = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            for line in lines:
                for _r in quiet(line):
                    pass
            per_rep.append(time.perf_counter() - t0)
        rec["scopes"]["frontend"] = {"total_s": summarize(per_rep)}

    # phoneme chunks exactly as production would produce them
    chunks = [r.phonemes for line in lines for r in quiet(line) if r.phonemes]
    rec["n_chunks"] = len(chunks)
    rec["chunk_phoneme_lens"] = [len(c) for c in chunks]

    model = common.load_model(args.device)
    pack = common.load_voice(args.voice).to(args.device)

    if "inference" in scopes:
        for c in chunks:  # warmup: one full uncounted pass (cuDNN heuristics + allocator see every shape)
            model(c, pack[len(c) - 1], args.speed)
        sync()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        per_chunk = [[] for _ in chunks]
        audio_s = [0.0] * len(chunks)
        totals = []
        for _ in range(args.reps):
            tt = 0.0
            for i, c in enumerate(chunks):
                sync()
                t0 = time.perf_counter()
                a = model(c, pack[len(c) - 1], args.speed)  # includes .cpu() of audio (as in KModel.forward)
                sync()
                dt = time.perf_counter() - t0
                per_chunk[i].append(dt)
                audio_s[i] = a.shape[-1] / SR
                tt += dt
            totals.append(tt)
        rec["scopes"]["inference"] = {
            "total_s": summarize(totals), "audio_s": sum(audio_s),
            "rtf_median": statistics.median(totals) / sum(audio_s),
            "per_chunk": [{"phonemes": len(chunks[i]), "audio_s": audio_s[i], **summarize(per_chunk[i])}
                          for i in range(len(chunks))],
            "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2**20) if args.device == "cuda" else None,
        }

    if "batch" in scopes:
        pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
        pipe.load_voice(args.voice)
        for r in pipe(lines[0], voice=args.voice, speed=args.speed):  # warmup
            pass
        totals, audio_total = [], 0.0
        for _ in range(args.reps):
            sync()
            t0 = time.perf_counter()
            n = 0
            for line in lines:
                for r in pipe(line, voice=args.voice, speed=args.speed):
                    n += r.audio.shape[-1]
            sync()
            totals.append(time.perf_counter() - t0)
            audio_total = n / SR
        rec["scopes"]["batch"] = {"total_s": summarize(totals), "audio_s": audio_total,
                                  "rtf_median": statistics.median(totals) / audio_total}

    if "cold" in scopes:
        runs = []
        env = dict(os.environ)
        if args.device == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
        for k in range(max(3, args.reps // 2)):
            wav_dir = out / f"cold_wavs_{args.device}_{k}"
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, __file__, "--cold-child", "--device", args.device,
                                "--threads", str(args.threads), "--voice", args.voice,
                                "--speed", str(args.speed), "--corpus", args.corpus,
                                "--wav-dir", str(wav_dir)], capture_output=True, text=True, env=env)
            wall = time.perf_counter() - t0
            if p.returncode != 0:
                raise RuntimeError(p.stderr[-2000:])
            child = json.loads(p.stdout.strip().splitlines()[-1])
            child["wall_s"] = wall
            runs.append(child)
        rec["scopes"]["cold"] = {"runs": runs, "wall_s": summarize(r["wall_s"] for r in runs),
                                 "audio_s": runs[0]["audio_s"]}

    rec["host_after"] = host_state()
    rec["peak_rss_mb_parent"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    path = out / f"reference_{args.device}_t{args.threads}_{args.voice}.json"
    path.write_text(json.dumps(rec, indent=1, default=str))
    s = rec["scopes"]
    print("receipt:", path)
    for k, v in s.items():
        tot = v.get("total_s") or v.get("wall_s")
        extra = f" RTF={v['rtf_median']:.4f}" if "rtf_median" in v else ""
        print(f"{k:10s} median {tot['median']:.3f}s cv {tot['cv_pct']:.1f}% audio {v.get('audio_s', 0):.1f}s{extra}")


if __name__ == "__main__":
    main()
