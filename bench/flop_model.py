"""Analytic FLOP / minimum-traffic model of the Kokoro forward, from layer shapes (src/*.rs).

Per chunk: T = phoneme chars + 2 tokens, N = duration frames (= samples / 600).
FLOPs count 2 per multiply-add of conv / linear / attention / LSTM products (elementwise ops,
norms and activations excluded: they are bandwidth-bound and accounted as bytes).
Usage: python bench/flop_model.py <rust core receipt.json>
"""
import json
import sys


def conv(cin, cout, k, t_out):
    return 2 * cin * cout * k * t_out


def resblock_snake(c, k, t):   # AdaINResBlock1: 3 x (conv k dil + conv k)
    return 6 * conv(c, c, k, t)


def adain_resblk(din, dout, t, upsample=False):  # AdainResBlk1d
    t2 = 2 * t if upsample else t
    f = conv(din, dout, 3, t2) + conv(dout, dout, 3, t2)
    if din != dout:
        f += conv(din, dout, 1, t2)
    return f


def lstm(din, h, t):
    return 2 * (2 * t * 4 * h * din) + 2 * (2 * t * 4 * h * h)  # input proj + recurrence, 2 dirs


def chunk(T, N):
    f = {}
    # ALBERT: map_in + 12 x (qkv, dense, ffn in/out, QK^T, PV)
    f["albert"] = 2 * T * 128 * 768 + 12 * (2 * T * 768 * 768 * 4 + 2 * T * 768 * 2048 * 2 + 2 * 2 * T * T * 768)
    f["duration"] = 2 * T * 768 * 512 + 3 * lstm(640, 256, T) + 3 * 2 * 128 * 1024 + lstm(640, 256, T) + 2 * T * 512 * 50
    f["f0n"] = lstm(640, 256, N) + 2 * (adain_resblk(512, 512, N) + adain_resblk(512, 256, N, True) + adain_resblk(256, 256, 2 * N) + conv(256, 1, 1, 2 * N))
    f["text_encoder"] = 3 * conv(512, 512, 5, T) + lstm(512, 256, T)
    f["decoder_pre"] = adain_resblk(514, 1024, N) + conv(512, 64, 1, N) + 3 * adain_resblk(1090, 1024, N) + adain_resblk(1090, 512, N, True)
    t0, t1 = 20 * N, 120 * N + 1
    f["gen_stage0"] = conv(22, 256, 12, t0) + resblock_snake(256, 7, t0) + 2 * 512 * 256 * 20 * (2 * N) + sum(resblock_snake(256, k, t0) for k in (3, 7, 11))
    f["gen_stage1"] = conv(22, 128, 1, t1) + resblock_snake(128, 11, t1) + 2 * 256 * 128 * 12 * t0 + sum(resblock_snake(128, k, t1) for k in (3, 7, 11))
    f["gen_post"] = conv(128, 22, 7, t1)
    # minimum HBM traffic for the per-tap-GEMM conv design in the generator (A read + C read/write per tap, f32)
    taps1 = 6 * (3 + 7 + 11) + 6 * 11
    taps0 = 6 * (3 + 7 + 11) + 6 * 7
    per_tap_bytes = taps1 * t1 * 128 * 4 * 3 + taps0 * t0 * 256 * 4 * 3
    # traffic if each conv read its input once and wrote its output once (implicit-GEMM ideal)
    convs1, convs0 = 6 * 3 + 6, 6 * 3 + 6
    ideal_bytes = convs1 * t1 * 128 * 4 * 2 + convs0 * t0 * 256 * 4 * 2
    return f, per_tap_bytes, ideal_bytes


def main():
    rec = json.load(open(sys.argv[1]))
    tot, pt, ib = {}, 0, 0
    for c in rec["per_chunk"]:
        T = c["phonemes"] + 2
        N = round(c["audio_s"] * 40)
        f, a, b = chunk(T, N)
        for k, v in f.items():
            tot[k] = tot.get(k, 0) + v
        pt += a
        ib += b
    all_f = sum(tot.values())
    print(json.dumps({"tflop_per_pass": {k: v / 1e12 for k, v in tot.items()}, "tflop_total": all_f / 1e12,
                      "gen_per_tap_gemm_bytes_GB": pt / 1e9, "gen_implicit_conv_ideal_bytes_GB": ib / 1e9}, indent=1))


if __name__ == "__main__":
    main()
