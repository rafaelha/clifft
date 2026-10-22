"""Independent dense joint-law checks and unchanged-backend sampling checks."""

import itertools
import json
from pathlib import Path

import numpy as np
from export import export  # type: ignore[import-not-found]
from frontend import Dilation  # type: ignore[import-not-found]
from planning import hidden_measurements, schedule  # type: ignore[import-not-found]

import clifft

ROOT = Path(__file__).parent
OLD = ROOT.parent / "global_reconstruction_20261022"


def dense(n, ops, order, faults=0):
    labels = np.arange(1 << n, dtype=np.uint64)
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1
    branches = [(0, state)]
    for j in order:
        o = ops[j]
        sign = (o.sign & 1) ^ ((o.sign >> 1 & faults).bit_count() % 2)
        phase = (
            1j ** ((o.x & o.z).bit_count())
            * (1 - 2 * (np.bitwise_count(labels & np.uint64(o.z)) % 2).astype(int))  # type: ignore[attr-defined]
            * (-1) ** sign
        )

        def apply(v):
            out = np.empty_like(v)
            out[labels ^ np.uint64(o.x)] = phase * v
            return out

        if o.angle:
            branches = [
                (r, np.cos(o.angle) * v - 1j * np.sin(o.angle) * apply(v)) for r, v in branches
            ]
        else:
            out = []
            for r, v in branches:
                pv = apply(v)
                for b in (0, 1):
                    w = (v + (-1) ** b * pv) / 2
                    if np.vdot(w, w).real > 1e-25:
                        out.append((r | (b << o.slot) if o.slot >= 0 else r, w))
            branches = out
    count = max((o.slot for o in ops), default=-1) + 1
    result = np.zeros(1 << count)
    for r, v in branches:
        result[r] += np.vdot(v, v).real
    return result


def exact(text, mode="records", hidden=True):
    d = Dilation.parse(text)
    n, ops = d.reconstruct(mode)
    if hidden:
        ops = hidden_measurements(n, ops)
    order, _ = schedule(n, ops)
    answer = np.zeros(1 << (max((o.slot for o in ops), default=-1) + 1))
    for choice in itertools.product(*d.channels):
        prob = np.prod([p for p, f in choice])
        fault = 0
        for p, f in choice:
            fault ^= f
        if prob:
            answer += prob * dense(n, ops, order, fault)
    return answer


def compile_(text, mask=None):
    pm = clifft.default_hir_pass_manager()
    pm.add(clifft.ActiveWidthSchedulePass())
    return clifft.compile(text, hir_passes=pm, postselection_mask=[] if mask is None else mask)


def restore(result, meta):
    inv = np.argsort(meta["detector_permutation"])
    return np.concatenate([result.detectors[:, inv], result.observables], axis=1)


def toys():
    out = []
    for case in json.loads((OLD / "exact_toy_expected.json").read_text()):
        source = (OLD / f"toy{case['case']}.stim").read_text()
        expected = np.array(case["probabilities"])
        actual = exact(source)
        error = float(np.max(abs(actual - expected)))
        assert error < 1e-12, (case["case"], error, actual, expected)
        emitted, meta, _ = export(source, "records")
        p = compile_(emitted)
        r = clifft.sample(p, shots=65536, seed=9100 + case["case"], threads=1)
        records = restore(r, meta)
        values = records.astype(np.uint64) @ np.array([1, 2, 4, 8], dtype=np.uint64)
        frequency = np.bincount(values.astype(int), minlength=16) / len(values)
        z = abs(frequency - expected) / np.sqrt(
            np.maximum(expected * (1 - expected) / len(values), 1 / len(values) ** 2)
        )
        assert z.max() < 7, (case["case"], z)
        row = dict(
            case=case["case"],
            exact_max_error=error,
            backend_shots=len(values),
            backend_max_z=float(z.max()),
            width=p.peak_active_width,
        )
        print(json.dumps(row), flush=True)
        out.append(row)
    (ROOT / "exact_validation.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    toys()
