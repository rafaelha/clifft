"""Determine a lower bound on a shared fixed stabilizer frame for sampled faults."""

import json
from pathlib import Path

import numpy as np
from conditional_phase import PhaseOracle  # type: ignore[import-not-found]
from fault_topology import nullspace  # type: ignore[import-not-found]
from gf2 import Basis as _Basis  # type: ignore[import-not-found]
from gf2 import mask as _mask

ROOT = Path(__file__).parent


def intersection(left, right):
    combined = _Basis()
    for row in right:
        combined.insert(row, 0)
    out = []
    for i, row in enumerate(left):
        rel = combined.insert(row, 1 << i)
        if rel:
            v = 0
            for j, r in enumerate(left):
                if rel >> j & 1:
                    v ^= r
            out.append(v)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(912)
    out = []
    for name in ["d5", "d7"]:
        a = dict(np.load(ROOT / f"{name}-detectors-component0.npz"))
        oracle = PhaseOracle(a, 0)
        common = None
        widths = []
        for case in range(17):
            f = (
                np.zeros(a["sign"].shape[1] - 1, dtype=np.uint8)
                if case == 0
                else rng.integers(0, 2, a["sign"].shape[1] - 1, dtype=np.uint8)
            )
            sim, magic, witness, _ = oracle.reduce(f)
            stabs = sim.canonical_stabilizers()
            rows = [_mask(s.to_numpy()[0]) | (_mask(s.to_numpy()[1]) << oracle.n) for s in stabs]
            if witness:
                anti = sum(int(not s.commutes(magic)) << i for i, s in enumerate(stabs))
                keep = []
                for rel in nullspace([anti], len(rows)):
                    v = 0
                    for j, r in enumerate(rows):
                        if rel >> j & 1:
                            v ^= r
                    keep.append(v)
                rows = keep
            widths.append(oracle.n - len(rows))
            common = rows if common is None else intersection(common, rows)
        assert common is not None
        result = dict(
            circuit=name,
            cases=len(widths),
            fixed_fault_widths=sorted(set(widths)),
            common_stabilizer_dimension=len(common),
            virtual_qubits=oracle.n,
            shared_fixed_frame_width_lower_bound=oracle.n - len(common),
            scope=(
                "after all commuting rotations before output measurements; "
                "hidden outcomes fixed to zero; arbitrary error-basis assignments"
            ),
        )
        out.append(result)
        print(json.dumps(result), flush=True)
    (ROOT / "common_frame.json").write_text(json.dumps(out, indent=2) + "\n")
