"""Container conversion only: fixture.npz -> fixture.safetensors (+ meta untouched).

No arithmetic is performed; arrays are copied verbatim so the Rust test harness needs a
single hand-parsed container format.
"""
import pathlib
import sys

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402


def main():
    for npz in sorted(common.FIXTURES.glob("*/*/fixture.npz")):
        d = np.load(npz)
        arrs = {k: np.ascontiguousarray(d[k]) for k in d.files}
        save_file(arrs, str(npz.with_suffix(".safetensors")))
        print(npz.parent.parent.name, npz.parent.name, len(arrs))


if __name__ == "__main__":
    main()
