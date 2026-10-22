"""Compare the entire reconstructed joint distribution with the Qiskit oracle."""

import itertools
import json
from pathlib import Path

import numpy as np
import tsim  # type: ignore[import-not-found]
from export_reference import export  # type: ignore[import-not-found]
from tsim.compile.pipeline import classify_direct  # type: ignore[import-not-found]
from tsim.core.graph import prepare_graph  # type: ignore[import-not-found]
from tsim.statevector.pbc import (  # type: ignore[import-not-found]
    ComponentLayout,
    _mask,
    read_pauli_ops,
)

import clifft

ROOT = Path(__file__).parent


def component_distribution(ops, faults):
    n = ops.num_lines
    labels = np.arange(1 << n, dtype=np.uint64)
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1
    branches = [(0, state)]
    for j in range(ops.num_ops):
        x, z = _mask(ops.x[j]), _mask(ops.z[j])
        sign = int(ops.sign[j].astype(np.uint8) @ np.r_[1, faults].astype(np.uint8) % 2)
        phase = (
            (1j) ** ((x & z).bit_count())
            * (1 - 2 * (np.bitwise_count(labels & np.uint64(z)) % 2).astype(int))  # type: ignore[attr-defined]
            * (-1) ** sign
        )

        def apply(v):
            out = np.zeros_like(v)
            out[labels ^ np.uint64(x)] = phase * v
            return out

        if ops.kind[j] == 0:
            branches = [
                (r, np.cos(ops.angle[j]) * v - 1j * np.sin(ops.angle[j]) * apply(v))
                for r, v in branches
            ]
        else:
            updated = []
            for r, v in branches:
                pv = apply(v)
                for bit in [0, 1]:
                    projected = (v + (-1) ** bit * pv) / 2
                    if np.vdot(projected, projected).real > 1e-25:
                        output = r | (bit << int(ops.slot[j])) if ops.slot[j] >= 0 else r
                        updated.append((output, projected))
            branches = updated
    dist: dict = {}
    for r, v in branches:
        dist[r] = dist.get(r, 0) + float(np.vdot(v, v).real)
    assert abs(sum(dist.values()) - 1) < 1e-12
    return dist


def exact(path):
    prepared = prepare_graph(tsim.Circuit.from_file(str(path)), sample_detectors=False)
    entries = []
    for component in prepared.graph.connected_components():
        outputs = [component.port(v) for v in component.outputs()]
        direct = classify_direct(component, prepared.layout)
        layout = ComponentLayout.of(component, outputs, prepared.layout)
        entries.append(
            (outputs, direct, layout, None if direct else read_pauli_ops(component, layout))
        )
    outcomes = []
    for probs in prepared.channel_probs:
        bits = len(probs).bit_length() - 1
        outcomes.append(
            [
                (np.array([(v >> i) & 1 for i in range(bits)], dtype=np.uint8), float(p))
                for v, p in enumerate(probs)
                if p
            ]
        )
    answer = np.zeros(1 << prepared.num_outputs)
    for choice in itertools.product(*outcomes):
        errors = np.concatenate([c[0] for c in choice]) if choice else np.zeros(0, dtype=np.uint8)
        probability = np.prod([c[1] for c in choice])
        faults = prepared.error_transform.astype(np.uint8) @ errors % 2
        distribution = {0: float(probability)}
        for outputs, direct, layout, ops in entries:
            if direct:
                f, flip = direct
                local = {int(faults[f] ^ flip): 1.0}
            else:
                local = component_distribution(ops, faults[list(layout.f_selection)])
            embedded = {
                sum(((r >> i) & 1) << o for i, o in enumerate(outputs)): p for r, p in local.items()
            }
            updated: dict = {}
            for r, p in distribution.items():
                for s, q in embedded.items():
                    updated[r | s] = updated.get(r | s, 0) + p * q
            distribution = updated
        for r, p in distribution.items():
            answer[r] += p
    return answer


if __name__ == "__main__":
    out = []
    for case in json.loads((ROOT / "exact_toy_expected.json").read_text()):
        path = ROOT / Path(case["path"]).name
        expected = np.array(case["probabilities"])
        actual = exact(path)
        error = float(np.max(np.abs(actual - expected)))
        assert error < 1e-12, (case["case"], error, actual, expected)
        export(f"toy{case['case']}", path, "records", compressed=True)
        p = clifft.compile((ROOT / f"toy{case['case']}-records-compressed.clifft").read_text())
        r = clifft.sample(p, shots=131072, seed=812 + case["case"], threads=1)
        values = r.detectors.astype(np.uint64) @ np.array([1, 2, 4, 8], dtype=np.uint64)
        freq = np.bincount(values.astype(int), minlength=16) / len(values)
        z = np.abs(freq - expected) / np.sqrt(
            np.maximum(expected * (1 - expected) / len(values), 1 / len(values) ** 2)
        )
        assert z.max() < 7, (case["case"], z.max())
        row = dict(
            case=case["case"],
            exact_max_absolute_error=error,
            bridge_samples=len(values),
            bridge_max_z=float(z.max()),
        )
        out.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "exact_reconstruction_check.json").write_text(json.dumps(out, indent=2) + "\n")
