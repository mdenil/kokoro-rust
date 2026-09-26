"""Shared oracle machinery: pin assertion, model load, noise capture/injection, hooks.

This module RUNS the pinned reference (kokoro==0.9.4); it never re-implements model math.
Everything captured here is frozen evidence for the Rust subject to be compared against.
"""
import importlib.metadata as md
import json
import os
import pathlib
import sys

DATA = pathlib.Path(os.environ.get("KOKORO_DATA", "/data/mdenil/kokoro-rust"))
SNAP = DATA / "hf/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987"
FIXTURES = DATA / "fixtures"
MODELS = DATA / "models"

PINS = {
    "torch": "2.12.1", "numpy": "2.4.6", "soundfile": "0.14.0", "transformers": "5.12.1",
    "espeakng-loader": "0.2.4", "spacy": "3.8.14", "huggingface-hub": "1.20.1",
    "kokoro": "0.9.4", "misaki": "0.9.4", "en_core_web_sm": "3.8.0",
}


def assert_pins():
    assert sys.version_info[:3] == (3, 12, 3), sys.version
    for pkg, want in PINS.items():
        got = md.version(pkg)
        assert got == want, f"pin mismatch: {pkg} got {got} want {want}"


def load_model(device="cpu"):
    import torch
    from kokoro.model import KModel
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(SNAP / "config.json"),
        model=str(SNAP / "kokoro-v1_0.pth"),
    ).to(device).eval()
    return model


def load_voice(name):
    import torch
    return torch.load(SNAP / "voices" / f"{name}.pt", weights_only=True)


class NoiseTap:
    """Intercept the exactly-three stochastic draws per KModel forward:
      1. torch.rand(B, 9)            -- SineGen._f02sine rand_ini (harmonic init phase)
      2. torch.randn_like(sine_waves)-- SineGen.forward additive noise [B, samples, 9]
      3. torch.randn_like(uv)        -- SourceModuleHnNSF noise branch [B, samples, 1] (unused downstream)
    mode='record': let reference RNG run, keep copies.
    mode='inject': replace draws with provided tensors (shape-asserted).
    """

    def __init__(self, mode="record", tensors=None):
        import torch
        self.torch = torch
        self.mode = mode
        self.tensors = list(tensors) if tensors is not None else None
        self.recorded = []
        self._orig_rand = None
        self._orig_randn_like = None

    def _next_inject(self, shape, device):
        t = self.tensors.pop(0)
        assert tuple(t.shape) == tuple(shape), (tuple(t.shape), tuple(shape))
        return t.to(device)

    def __enter__(self):
        torch = self.torch
        self._orig_rand = torch.rand
        self._orig_randn_like = torch.randn_like
        tap = self

        def rand(*args, **kw):
            if tap.mode == "inject":
                dev = kw.get("device", "cpu")
                out = tap._next_inject(tuple(args), dev)
            else:
                out = tap._orig_rand(*args, **kw)
            tap.recorded.append(out.detach().to("cpu").clone())
            return out

        def randn_like(t, **kw):
            if tap.mode == "inject":
                out = tap._next_inject(tuple(t.shape), t.device).to(t.dtype)
            else:
                out = tap._orig_randn_like(t, **kw)
            tap.recorded.append(out.detach().to("cpu").clone())
            return out

        torch.rand = rand
        torch.randn_like = randn_like
        return self

    def __exit__(self, *exc):
        self.torch.rand = self._orig_rand
        self.torch.randn_like = self._orig_randn_like
        return False


class SeamRecorder:
    """Forward hooks on named modules + method wraps for stft.transform/inverse.
    Captures cpu float32 numpy copies keyed by seam name."""

    def __init__(self, model, full_internals=False):
        self.model = model
        self.full = full_internals
        self.data = {}
        self.handles = []
        self._unwraps = []

    def _save(self, key, tensor):
        import torch
        if isinstance(tensor, torch.Tensor):
            self.data[key] = tensor.detach().to("cpu").float().numpy().copy()

    def _hook(self, key, io="out"):
        def fn(mod, args, output):
            if io == "out":
                out = output[0] if isinstance(output, tuple) else output
                self._save(key, out)
            else:
                for i, a in enumerate(args):
                    self._save(f"{key}.arg{i}", a)
        return fn

    def __enter__(self):
        m = self.model
        pairs = [
            ("bert", m.bert, "out"),
            ("bert_encoder", m.bert_encoder, "out"),
            ("dur_enc", m.predictor.text_encoder, "out"),
            ("pred_lstm", m.predictor.lstm, "out"),
            ("duration_proj", m.predictor.duration_proj, "out"),
            ("shared", m.predictor.shared, "in"),
            ("shared_out", m.predictor.shared, "out"),
            ("text_encoder", m.text_encoder, "out"),
            ("decoder", m.decoder, "in"),
            ("gen", m.decoder.generator, "in"),
            ("m_source", m.decoder.generator.m_source, "in"),
        ]
        if self.full:
            pairs += [
                ("F0_proj", m.predictor.F0_proj, "out"),
                ("N_proj", m.predictor.N_proj, "out"),
                ("F0_conv", m.decoder.F0_conv, "out"),
                ("N_conv", m.decoder.N_conv, "out"),
                ("asr_res", m.decoder.asr_res, "out"),
                ("dec_encode", m.decoder.encode, "out"),
                ("conv_post", m.decoder.generator.conv_post, "out"),
            ]
            for i, blk in enumerate(m.predictor.F0):
                pairs.append((f"F0_blk{i}", blk, "out"))
            for i, blk in enumerate(m.predictor.N):
                pairs.append((f"N_blk{i}", blk, "out"))
            for i, blk in enumerate(m.decoder.decode):
                pairs.append((f"dec_decode{i}", blk, "out"))
            for i, up in enumerate(m.decoder.generator.ups):
                pairs.append((f"gen_up{i}", up, "out"))
            for i, nc in enumerate(m.decoder.generator.noise_convs):
                pairs.append((f"gen_noise_conv{i}", nc, "out"))
            for i, nr in enumerate(m.decoder.generator.noise_res):
                pairs.append((f"gen_noise_res{i}", nr, "out"))
            for i, rb in enumerate(m.decoder.generator.resblocks):
                pairs.append((f"gen_resblk{i}", rb, "out"))
            # ALBERT internals: the single shared layer fires 12 times
            shared_layer = m.bert.encoder.albert_layer_groups[0].albert_layers[0]
            count = {"i": 0}
            rec = self

            def albert_hook(mod, args, output):
                out = output[0] if isinstance(output, tuple) else output
                rec._save(f"albert_layer{count['i']}", out)
                count["i"] += 1
            self.handles.append(shared_layer.register_forward_hook(albert_hook))
            self.handles.append(m.bert.embeddings.register_forward_hook(self._hook("albert_embeddings")))
            self.handles.append(m.bert.encoder.embedding_hidden_mapping_in.register_forward_hook(
                self._hook("albert_embed_proj")))

        for key, mod, io in pairs:
            if io == "in":
                self.handles.append(mod.register_forward_hook(
                    self._hook(key, io="in"), with_kwargs=False))
            else:
                self.handles.append(mod.register_forward_hook(self._hook(key)))

        # m_source outputs (tuple of 3) need a custom hook
        rec = self

        def msource_out(mod, args, output):
            rec._save("har_source", output[0])
            rec._save("uv", output[2])
        self.handles.append(m.decoder.generator.m_source.register_forward_hook(msource_out))

        # stft.transform / inverse are plain method calls; wrap them
        stft = m.decoder.generator.stft
        orig_transform, orig_inverse = stft.transform, stft.inverse

        def transform(x):
            spec, phase = orig_transform(x)
            rec._save("har_spec", spec)
            rec._save("har_phase", phase)
            return spec, phase

        def inverse(spec, phase):
            rec._save("gen_spec", spec)
            rec._save("gen_phase", phase)
            out = orig_inverse(spec, phase)
            rec._save("istft_out", out)
            return out

        stft.transform, stft.inverse = transform, inverse
        self._unwraps.append(lambda: (setattr(stft, "transform", orig_transform),
                                      setattr(stft, "inverse", orig_inverse)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        for u in self._unwraps:
            u()
        return False


def runtime_meta():
    import torch
    return {
        "python": sys.version,
        "pins": {p: md.version(p) for p in PINS},
        "torch_num_threads": torch.get_num_threads(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "gpu0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_meta(path, extra):
    meta = runtime_meta()
    meta.update(extra)
    pathlib.Path(path).write_text(json.dumps(meta, indent=1, default=str))
