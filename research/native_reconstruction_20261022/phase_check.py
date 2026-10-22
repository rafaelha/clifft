"""Check the separately scoped residual phase core on native output only."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "global_reconstruction_20261022"))
from conditional_phase import PhaseOracle  # type: ignore[import-not-found]  # noqa: E402


def run():
    rng = np.random.default_rng(9001)
    result = []
    for name in ("d5", "d7"):
        for mode in ("detectors", "records"):
            a = dict(np.load(ROOT / f"{name}-{mode}.npz"))
            nh = int(np.sum((a["kind"] == 1) & (a["slot"] < 0)))
            for hidden in range(1 << nh):
                o = PhaseOracle(a, hidden)
                counts = set()
                for f in [np.zeros(a["sign"].shape[1] - 1, dtype=np.uint8)] + [
                    rng.integers(0, 2, a["sign"].shape[1] - 1, dtype=np.uint8) for _ in range(4)
                ]:
                    _, _, w, _ = o.reduce(f)
                    counts.add(int(bool(w)))
                row = dict(
                    name=name,
                    mode=mode,
                    hidden=hidden,
                    affine_width=o.width,
                    residual_T_counts=sorted(counts),
                    fault_sign_assignments=5,
                )
                result.append(row)
            print(name, mode, "done", flush=True)
    (ROOT / "phase_check.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    run()
