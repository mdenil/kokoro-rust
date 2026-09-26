"""Export the loaded reference model's weights for the Rust subject.

Outputs into $KOKORO_DATA/models/:
  raw_state.safetensors        exact model.state_dict() after KModel load (weight_g/weight_v kept)
  materialized.safetensors     same but every weight-normed module resolved via torch._weight_norm
                               (bit-exactly what forward() consumes)
  voices_<v>.safetensors       voice packs [510,1,256]
  census.json                  key/shape/dtype census + load_state_dict missing/unexpected report
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402


def main():
    common.assert_pins()
    import torch
    from safetensors.torch import save_file

    model = common.load_model("cpu")
    sd = model.state_dict()
    common.MODELS.mkdir(parents=True, exist_ok=True)

    raw = {k: v.contiguous() for k, v in sd.items()}
    save_file(raw, common.MODELS / "raw_state.safetensors")

    # materialize weight norm exactly as torch's pre-forward hook does
    mat = {}
    gkeys = [k for k in sd if k.endswith("weight_g")]
    for k, v in sd.items():
        if k.endswith("weight_g"):
            base = k[: -len("weight_g")]
            g, vv = sd[base + "weight_g"], sd[base + "weight_v"]
            mat[base + "weight"] = torch._weight_norm(vv, g, 0).contiguous()
        elif k.endswith("weight_v"):
            continue
        else:
            mat[k] = v.contiguous()
    save_file(mat, common.MODELS / "materialized.safetensors")

    census = {
        "raw_keys": {k: [list(v.shape), str(v.dtype)] for k, v in sd.items()},
        "n_raw": len(sd), "n_weight_norm": len(gkeys),
        "n_params": sum(v.numel() for v in sd.values()),
    }
    (common.MODELS / "census.json").write_text(json.dumps(census, indent=1))

    for voice in ("af_heart", "am_adam"):
        pack = common.load_voice(voice)
        save_file({"pack": pack.contiguous()}, common.MODELS / f"voices_{voice}.safetensors")

    cfg = (common.SNAP / "config.json").read_text()
    (common.MODELS / "config.json").write_text(cfg)
    print(f"raw keys={len(sd)} weight_norm={len(gkeys)} params={census['n_params']:,}")


if __name__ == "__main__":
    main()
