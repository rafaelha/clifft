"""Prepare a fault-parameterized quadratic phase sampler from reference graphs.

Reference extraction is deliberately outside the native timed execution. No
fault signatures or hidden-outcome families are enumerated into circuit plans.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import stim
import tsim  # type: ignore[import-not-found]
from tsim.compile.pipeline import classify_direct  # type: ignore[import-not-found]
from tsim.core.graph import prepare_graph  # type: ignore[import-not-found]
from tsim.statevector.pbc import ComponentLayout, read_pauli_ops  # type: ignore[import-not-found]

ROOT = Path(__file__).parent
STUDY = ROOT.parent / "global_reconstruction_20261022"
sys.path.insert(0, str(STUDY))
from conditional_phase import PhaseOracle, pauli  # type: ignore[import-not-found]  # noqa: E402
from gf2 import Basis, mask  # type: ignore[import-not-found]  # noqa: E402


def packed_pauli(p):
    x, z = map(mask, p.to_numpy())
    phase = ((x & z).bit_count() + (0 if p.sign == 1 else 2)) % 4
    assert p.sign in (1, -1)
    return [x, z, phase]


def graph_rows(sim, width, n):
    stabs = sim.canonical_stabilizers()
    basis = Basis()
    for i, s in enumerate(stabs):
        x, z = map(mask, s.to_numpy())
        basis.insert(x | (z << n), 1 << i)
    rows = []
    # Solve for each X direction modulo the fixed zero coordinates.
    xb = Basis()
    for i, s in enumerate(stabs):
        xb.insert(mask(s.to_numpy()[0]), 1 << i)
    for j in range(n):
        if j >= width:
            assert sim.peek_observable_expectation(pauli(0, 1 << j, n)) == 1
            rows.append([0, 1 << j, 0])
            continue
        combination = xb.solve(1 << j)
        assert combination is not None
        p = stim.PauliString(n)
        for i, s in enumerate(stabs):
            if combination >> i & 1:
                p *= s
        x, z = map(mask, p.to_numpy())
        p *= pauli(0, z & ~((1 << width) - 1), n)
        rows.append(packed_pauli(p))
        assert rows[-1][0] == 1 << j
    return rows


def output_axis(o, axis):
    p = o.frame.inverse()(axis)
    if (mask(p.to_numpy()[1]) & o.offset).bit_count() & 1:
        p *= -1
    return o.linear.inverse()(p)


def build(name, mode):
    started = time.perf_counter()
    prepared = prepare_graph(
        tsim.Circuit.from_file(str(STUDY / "inputs" / f"{name}.stim")),
        sample_detectors=mode == "detectors",
    )
    nf = len(prepared.error_transform)
    next_param = nf
    direct = []
    components = []
    for component in sorted(prepared.graph.connected_components(), key=lambda c: len(c.outputs())):
        outputs = [component.port(v) for v in component.outputs()]
        entry = classify_direct(component, prepared.layout)
        if entry is not None:
            f, flip = entry
            direct.append([outputs[0], 1 << f, int(flip)])
            continue
        layout = ComponentLayout.of(component, outputs, prepared.layout)
        ops = read_pauli_ops(component, layout)
        if not ops.num_ops:
            continue
        assert ops.num_lines <= 64
        a = vars(ops)
        if not np.any(a["kind"] == 0):
            # The record graph also contains isolated uniform classical bits.
            assert ops.num_lines == 1 and ops.num_ops == 2
            assert a["slot"].tolist() == [-1, 0] and not a["sign"].any()
            assert np.array_equal(a["x"][0], a["x"][1])
            assert np.array_equal(a["z"][0], a["z"][1]) and a["x"][0, 0]
            direct.append([outputs[0], 1 << next_param, 0])
            next_param += 1
            continue
        nh = int(sum((a["kind"] == 1) & (a["slot"] < 0)))
        hidden_ids = np.flatnonzero((a["kind"] == 1) & (a["slot"] < 0))
        check = stim.TableauSimulator(seed=73)
        check.set_num_qubits(ops.num_lines)
        for j in hidden_ids:
            axis = stim.PauliString.from_numpy(xs=a["x"][j], zs=a["z"][j])
            assert check.peek_observable_expectation(axis) == 0
            assert not a["sign"][j].any()
            check.postselect_observable(axis, desired_value=False)
        # These benchmark components have a rotation span covering all lines.
        base = PhaseOracle(a, 0)
        faults = np.zeros(a["sign"].shape[1] - 1, dtype=np.uint8)
        sim, _, witness, _ = base.reduce(faults)
        sim.do_tableau(base.frame.inverse(), list(range(base.n)))
        for i in range(base.n):
            if base.offset >> i & 1:
                sim.x(i)
        sim.do_tableau(base.linear.inverse(), list(range(base.n)))
        rows = graph_rows(sim, base.width, base.n)
        initial = graph_rows(base.shifted, base.width, base.n)
        init_flips = [0] * base.n
        rotations = []
        for k, j in enumerate(base.ids):
            fmask = sum(1 << layout.f_selection[i] for i in np.flatnonzero(a["sign"][j, 1:]))
            coefficient = int(base.coeffs[k]) * (-1 if a["sign"][j, 0] else 1)
            rotations.append([int(base.restricted[k]), (-coefficient) % 4, fmask])
        measured = []
        for j in base.output_ids:
            fmask = sum(1 << layout.f_selection[i] for i in np.flatnonzero(a["sign"][j, 1:]))
            p = output_axis(base, base.axes[j]) * (-1 if a["sign"][j, 0] else 1)
            measured.append([int(outputs[a["slot"][j]]), packed_pauli(p), fmask])
        for h in range(nh):
            other = PhaseOracle(a, 1 << h)
            assert other.width == base.width and other.restricted == base.restricted
            assert other.linear == base.linear and other.frame == base.frame
            other_initial = graph_rows(other.shifted, other.width, other.n)
            for i, (left, right) in enumerate(zip(initial, other_initial)):
                assert left[:2] == right[:2]
                assert (left[2] - right[2]) % 2 == 0
                if left[2] != right[2]:
                    init_flips[i] ^= 1 << (next_param + h)
            for k, (left, right) in enumerate(zip(base.coeffs, other.coeffs)):
                assert left == right or left == -right
                if left != right:
                    rotations[k][2] ^= 1 << (next_param + h)
            for k, j in enumerate(base.output_ids):
                left, right = output_axis(base, base.axes[j]), output_axis(other, other.axes[j])
                assert left == right or left == -right
                if left != right:
                    measured[k][2] ^= 1 << (next_param + h)
        components.append(
            dict(
                n=base.n,
                width=base.width,
                witness=int(witness),
                rows=rows,
                init_flips=init_flips,
                rotations=rotations,
                outputs=measured,
                hidden_start=next_param,
                hidden_count=nh,
            )
        )
        next_param += nh
    grouped: dict = {}
    start_bit = 0
    for probabilities in prepared.channel_probs:
        bits = len(probabilities).bit_length() - 1
        columns = [mask(c) for c in prepared.error_transform[:, start_bit : start_bit + bits].T]
        start_bit += bits
        distribution: dict = {}
        for value, probability in enumerate(probabilities):
            image = 0
            for b in range(bits):
                if value >> b & 1:
                    image ^= columns[b]
            distribution[image] = distribution.get(image, 0.0) + float(probability)
        key = tuple(sorted(distribution))
        if len(key) == 1:
            continue
        if key in grouped:
            merged: dict = {}
            for x, p in grouped[key].items():
                for y, q in distribution.items():
                    merged[x ^ y] = merged.get(x ^ y, 0.0) + p * q
            distribution = merged
        grouped[key] = distribution
    channels = [[[p, m] for m, p in sorted(d.items()) if m and p] for d in grouped.values()]
    plan = dict(
        name=name,
        mode=mode,
        nf=nf,
        params=next_param,
        outputs=prepared.num_outputs,
        detectors=prepared.num_detectors if mode == "detectors" else prepared.num_outputs,
        direct=direct,
        components=components,
        channels=channels,
        prepare_seconds=time.perf_counter() - started,
    )
    save(plan, ROOT / f"{name}-{mode}.plan")
    metadata = {k: v for k, v in plan.items() if k not in ("channels", "components", "direct")}
    metadata.update(
        channels=len(channels),
        component_lines=[c["n"] for c in components],
        hidden_bits=next_param - nf,
        residual_t=[int(bool(c["witness"])) for c in components],
        expected_nonzero_channels=sum(p for c in channels for p, _ in c),
    )
    (ROOT / f"{name}-{mode}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    return plan


def save(plan, path):
    words = (plan["params"] + 63) // 64
    tokens = []

    def add(*v):
        tokens.append(" ".join(map(str, v)))

    def bits(value):
        add(*[(value >> (64 * i)) & ((1 << 64) - 1) for i in range(words)])

    add(
        plan["nf"],
        plan["params"],
        plan["outputs"],
        plan["detectors"],
        len(plan["channels"]),
        len(plan["direct"]),
        len(plan["components"]),
    )
    for channel in plan["channels"]:
        add(len(channel))
        for probability, m in channel:
            add(format(probability, ".17g"))
            bits(m)
    for index, m, flip in plan["direct"]:
        add(index, flip)
        bits(m)
    for c in plan["components"]:
        add(c["n"], c["width"], c["witness"], len(c["rotations"]), len(c["outputs"]))
        for row, flip in zip(c["rows"], c["init_flips"]):
            add(*row)
            bits(flip)
        for m, coefficient, flip in c["rotations"]:
            add(m, coefficient)
            bits(flip)
        for index, p, flip in c["outputs"]:
            add(index, *p)
            bits(flip)
    path.write_text("\n".join(tokens) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=["d5", "d7"])
    parser.add_argument("--mode", default="detectors")
    args = parser.parse_args()
    for name in args.names:
        build(name, args.mode)
