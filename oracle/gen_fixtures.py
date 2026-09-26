"""Generate frozen oracle fixtures: phonemes, ids, noise draws, seam tensors, audio.

Usage: python gen_fixtures.py [--device cpu|cuda] [--threads N] [--only CASE]
Fixtures land in $KOKORO_DATA/fixtures/<device>-t<threads>/<case>/.
The reference is EXECUTED, never re-derived. Noise draws are recorded so the subject
can consume identical stochastic inputs (equivalence: frozen-noise parity).
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

# (case_id, text, voices, speeds, full_internals)
V2 = ("af_heart", "am_adam")
CORPUS = [
    ("s01_hello", "Hello, world!", V2, (1.0,), True),
    ("s02_fox", "The quick brown fox jumps over the lazy dog.", V2, (0.8, 1.0, 1.25), False),
    ("s03_moon", "In 1969, astronauts landed on the Moon — a feat once thought impossible…",
     V2, (1.0,), False),
    ("s04_alice", "“Curiouser and curiouser!” cried Alice; she was quite surprised.",
     V2, (1.0,), False),
    ("s05_word", "Kokoro.", V2, (1.0,), False),
    # near-510-phoneme boundary case: assembled long sentence (public-domain Alice text)
    ("s06_long", "Alice was beginning to get very tired of sitting by her sister on the bank, "
     "and of having nothing to do: once or twice she had peeped into the book her sister was reading, "
     "but it had no pictures or conversations in it, and what is the use of a book, thought Alice, "
     "without pictures or conversations? So she was considering in her own mind, as well as she could, "
     "for the hot day made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain "
     "would be worth the trouble of getting up and picking the daisies.",
     ("af_heart",), (1.0,), False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    common.assert_pins()
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    from kokoro.pipeline import KPipeline
    model = common.load_model(args.device)
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)  # G2P only

    outroot = common.FIXTURES / f"{args.device}-t{args.threads}"
    outroot.mkdir(parents=True, exist_ok=True)

    for case_id, text, voices, speeds, full in CORPUS:
        if args.only and case_id != args.only:
            continue
        _, tokens = pipe.g2p(text)
        chunks = [(gs, ps) for gs, ps, _ in pipe.en_tokenize(tokens) if ps]
        # s06 intentionally exceeds 510 phonemes; its first waterfall chunk IS the
        # boundary-length case (exactly what production synthesizes for that text).
        assert len(chunks) == 1 or case_id == "s06_long", (case_id, len(chunks))
        gs, ps = chunks[0]
        assert len(ps) <= 510, (case_id, len(ps))

        for voice in voices:
            pack = common.load_voice(voice)
            ref_s = pack[len(ps) - 1]
            for speed in speeds:
                name = f"{case_id}__{voice}__s{speed}"
                outdir = outroot / name
                outdir.mkdir(exist_ok=True)
                torch.manual_seed(1234)  # fixed seed; draws additionally recorded verbatim
                with common.NoiseTap(mode="record") as tap, \
                     common.SeamRecorder(model, full_internals=full) as rec:
                    out = model(ps, ref_s, speed, return_output=True)
                audio = out.audio.numpy()
                pred_dur = out.pred_dur.numpy()

                arrs = {
                    "audio": audio,
                    "pred_dur": pred_dur.astype(np.int64),
                    "ref_s": ref_s.numpy(),
                    "input_ids": np.array(
                        [0] + [model.vocab[p] for p in ps if p in model.vocab] + [0],
                        dtype=np.int64),
                }
                assert len(tap.recorded) == 3, len(tap.recorded)
                arrs["noise.rand_ini"] = tap.recorded[0].numpy()
                arrs["noise.sine"] = tap.recorded[1].numpy()
                arrs["noise.uv"] = tap.recorded[2].numpy()
                for k, v in rec.data.items():
                    arrs[f"seam.{k}"] = v
                np.savez_compressed(outdir / "fixture.npz", **arrs)
                meta = {
                    "case": case_id, "text": text, "graphemes": gs, "phonemes": ps,
                    "voice": voice, "speed": speed, "device": args.device,
                    "threads": args.threads, "samples": int(audio.shape[-1]),
                    "sample_rate": 24000, "pred_dur_sum": int(pred_dur.sum()),
                    "seed": 1234,
                }
                common.save_meta(outdir / "meta.json", meta)
                print(f"{name}: T={len(ps)+2} frames={pred_dur.sum()} samples={audio.shape[-1]} "
                      f"seams={len(rec.data)}")


if __name__ == "__main__":
    main()
