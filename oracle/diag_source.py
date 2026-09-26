"""Diagnostic oracle seams inside SineGen and STFT, computed by the pinned torch ops.

For a fixture case, re-execute the reference SineGen/SourceModule/STFT on the fixture's own
f0 curve + recorded noise and dump: pre-sin phase [9,S], sines [S,9], har_source, and the raw
complex STFT (re, im). Verifies the re-execution reproduces the frozen har_source bit-exactly.
Usage: python diag_source.py <device-dir> <case>
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402


def main():
    common.assert_pins()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.numpy import save_file
    torch.set_num_threads(1)
    dev, case = sys.argv[1], sys.argv[2]
    fx = common.FIXTURES / dev / case
    d = np.load(fx / "fixture.npz")
    model = common.load_model("cpu")
    gen = model.decoder.generator
    sg = gen.m_source.l_sin_gen

    f0_curve = torch.from_numpy(d["seam.gen.arg2"])
    rand_ini = torch.from_numpy(d["noise.rand_ini"]).clone()
    sine_noise = torch.from_numpy(d["noise.sine"])

    with torch.no_grad():
        f0 = gen.f0_upsamp(f0_curve[:, None]).transpose(1, 2)
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, sg.harmonic_num + 2)]]))
        rad = (fn / sg.sampling_rate) % 1
        rand_ini[:, 0] = 0
        rad[:, 0, :] = rad[:, 0, :] + rand_ini
        down = F.interpolate(rad.transpose(1, 2), scale_factor=1 / sg.upsample_scale, mode="linear")
        cums = torch.cumsum(down.transpose(1, 2), dim=1) * 2 * torch.pi
        phase = F.interpolate(cums.transpose(1, 2) * sg.upsample_scale, scale_factor=sg.upsample_scale,
                              mode="linear")
        sines = torch.sin(phase.transpose(1, 2)) * sg.sine_amp
        uv = sg._f02uv(f0)
        noise_amp = uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3
        sw = sines * uv + noise_amp * sine_noise
        har = gen.m_source.l_tanh(gen.m_source.l_linear(sw))
        assert torch.equal(har, torch.from_numpy(d["seam.har_source"])), "re-execution != frozen har_source"
        x = har.transpose(1, 2).squeeze(1)
        spec = torch.stft(x, 20, 5, 20, window=gen.stft.window, return_complex=True)

    out = {
        "f0_up": f0.numpy().reshape(-1).copy(),
        "rad_down": down.numpy()[0].copy(),
        "phase_pre_up": (cums.transpose(1, 2) * sg.upsample_scale).numpy()[0].copy(),
        "phase": phase.numpy()[0].copy(),
        "sines": sines.numpy()[0].copy(),
        "har_source": har.numpy().reshape(-1).copy(),
        "stft_re": spec.real.numpy()[0].copy(),
        "stft_im": spec.imag.numpy()[0].copy(),
    }
    outp = common.FIXTURES / "diag" / dev / case
    outp.mkdir(parents=True, exist_ok=True)
    save_file(out, str(outp / "source_diag.safetensors"))
    print("wrote", outp, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
