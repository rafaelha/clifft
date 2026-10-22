"""Independent dense laws, fixed-fault phase oracles, and noisy output checks."""

import json
import os
import sys

import numpy as np
import stim
import tsim  # type: ignore[import-not-found]
from native import Native  # type: ignore[import-not-found]
from prepare import ROOT, STUDY, graph_rows, packed_pauli, save  # type: ignore[import-not-found]
from tsim.compile.pipeline import classify_direct  # type: ignore[import-not-found]
from tsim.core.graph import prepare_graph  # type: ignore[import-not-found]
from tsim.statevector.pbc import ComponentLayout, read_pauli_ops  # type: ignore[import-not-found]

sys.path.insert(0, str(STUDY))
from conditional_phase import PhaseOracle  # type: ignore[import-not-found]
from gf2 import mask  # type: ignore[import-not-found]
from study_io import read_circuit  # type: ignore[import-not-found]

import clifft


def words(value, count):
    return np.array([(value >> (64 * i)) & ((1 << 64) - 1) for i in range(count)], dtype=np.uint64)


def dense_tests():
    rng = np.random.default_rng(1801)
    maximum = 0.0
    cases = 0
    max_z = 0.0
    for case in range(24):
        n = 3 + case % 5
        w = n - (case % 3 == 0)
        witness = int(rng.integers(1, 1 << w)) if case % 2 else 0
        initial = stim.TableauSimulator()
        initial.set_num_qubits(n)
        initial.h(*range(w))
        for i in range(w):
            for _ in range(int(rng.integers(4))):
                initial.s(i)
            for j in range(i):
                if rng.integers(2):
                    initial.cz(i, j)
        rows = graph_rows(initial, w, n)
        rotations = [
            [int(rng.integers(1, 1 << w)), int(rng.choice([1, 2, 3])), int(rng.integers(1, 32))]
            for _ in range(9)
        ]
        # Use a seeded random Clifford measurement basis, independent of native algebra.
        circuit = stim.Circuit()
        for _ in range(5 * n):
            q = int(rng.integers(n))
            circuit.append("H" if rng.integers(2) else "S", [q])
            a, b = rng.choice(n, 2, replace=False)
            circuit.append("CX", [int(a), int(b)])
        tableau = circuit.to_tableau()
        axes = [tableau.z_output(i) for i in range(n)]
        flips = [int(rng.integers(32)) for _ in range(n)]
        component = dict(
            n=n,
            width=w,
            witness=witness,
            rows=rows,
            init_flips=[0] * n,
            rotations=rotations,
            outputs=[[i, packed_pauli(p), f] for i, (p, f) in enumerate(zip(axes, flips))],
        )
        plan = dict(
            nf=5, params=5, outputs=n, detectors=n, channels=[], direct=[], components=[component]
        )
        path = ROOT / "dense_validation.plan"
        save(plan, path)
        native = Native(path)
        labels = np.arange(1 << n, dtype=np.uint64)
        unitary = tableau.to_unitary_matrix(endian="little").astype(complex)
        for parameters in [0, int(rng.integers(32)), 31]:
            state = initial.state_vector(endian="little").astype(complex)
            phase = np.zeros(1 << n)
            for m, delta, f in rotations:
                if (parameters & f).bit_count() % 2:
                    # A flipped T sign contributes an S parity, represented by
                    # its original parity function, independently of quadratic expansion.
                    phase += delta * np.pi / 2 * (np.bitwise_count(labels & np.uint64(m)) % 2)  # type: ignore[attr-defined]
            phase += np.pi / 4 * (np.bitwise_count(labels & np.uint64(witness)) % 2)  # type: ignore[attr-defined]
            state *= np.exp(1j * phase)
            expected = abs(unitary.conj().T @ state) ** 2
            expected /= expected.sum()
            output_flip = sum(((parameters & f).bit_count() % 2) << i for i, f in enumerate(flips))
            expected = expected[np.arange(1 << n) ^ output_flip]
            all_bits = ((labels[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.uint8)
            actual = np.array([native.probability(words(parameters, 1), b) for b in all_bits])
            error = float(abs(expected - actual).max())
            maximum = max(maximum, error)
            assert error < 2e-7, (case, parameters, error)
            fixed = np.full((32768, 1), parameters, dtype=np.uint64)
            _, sampled, _ = native.sample(
                len(fixed), seed=500 + case, retain=True, parameters=fixed
            )
            values = sampled.astype(np.uint64) @ (1 << np.arange(n, dtype=np.uint64))
            frequencies = np.bincount(values.astype(int), minlength=1 << n) / len(values)
            z = abs(frequencies - expected) / np.sqrt(
                np.maximum(expected * (1 - expected) / len(values), 1 / len(values) ** 2)
            )
            max_z = max(max_z, float(z.max()))
            assert z.max() < 7
            cases += 1
    (ROOT / "dense_validation.plan").unlink()
    return dict(
        kind="dense_complete_joint_laws",
        cases=cases,
        max_absolute_error=maximum,
        max_sampling_z=max_z,
        reference="Stim float32 Clifford matrices followed by independent dense phase evolution",
    )


def oracle_tests(name, mode):
    native = Native(ROOT / f"{name}-{mode}.plan")
    prepared = prepare_graph(
        tsim.Circuit.from_file(str(STUDY / "inputs" / f"{name}.stim")),
        sample_detectors=mode == "detectors",
    )
    rng = np.random.default_rng(401)
    entries = []
    hidden_start = native.nf
    for c in sorted(prepared.graph.connected_components(), key=lambda c: len(c.outputs())):
        outputs = [c.port(v) for v in c.outputs()]
        direct = classify_direct(c, prepared.layout)
        if direct is not None:
            entries.append(("direct", outputs, direct))
            continue
        layout = ComponentLayout.of(c, outputs, prepared.layout)
        ops = read_pauli_ops(c, layout)
        if not np.any(ops.kind == 0):
            entries.append(("direct", outputs, (hidden_start, False)))
            hidden_start += 1
            continue
        nh = int(sum((ops.kind == 1) & (ops.slot < 0)))
        entries.append(("component", outputs, (vars(ops), layout, hidden_start, nh)))
        hidden_start += nh
    maximum = 0.0
    checked = 0
    nonzero = 0
    fractional = 0
    for case in range(32):
        if case == 0:
            f = np.zeros(native.nf, dtype=np.uint8)
        elif case < 12:
            f = prepared.error_transform[
                :, int(rng.integers(prepared.error_transform.shape[1]))
            ].astype(np.uint8)
        else:
            f = rng.integers(0, 2, native.nf, dtype=np.uint8)
        parameter = mask(f)
        states = []
        for kind, outputs, entry in entries:
            if kind == "direct":
                index, _ = entry
                if index >= native.nf and rng.integers(2):
                    parameter ^= 1 << index
                states.append((kind, outputs, entry))
                continue
            a, layout, hs, nh = entry
            hidden = int(rng.integers(1 << nh))
            parameter |= hidden << hs
            oracle = PhaseOracle(a, hidden)
            local = f[list(layout.f_selection)]
            sim, magic, witness, _ = oracle.reduce(local)
            axes = []
            for j in oracle.output_ids:
                sign = int(a["sign"][j].astype(np.uint8) @ np.r_[1, local].astype(np.uint8) % 2)
                axes.append(oracle.axes[j] * (-1 if sign else 1))
            states.append((kind, outputs, (sim, magic, witness, axes)))
        selections = []
        # Component marginals and parities exercise both direct and correlated outputs.
        for kind, outputs, entry in states:
            if kind == "component":
                for output in outputs:
                    s = np.zeros(native.outputs, dtype=np.uint8)
                    s[output] = 1
                    selections.append(s)
                for _ in range(96):
                    s = np.zeros(native.outputs, dtype=np.uint8)
                    chosen = rng.choice(
                        outputs, int(rng.integers(2, min(len(outputs), 9) + 1)), replace=False
                    )
                    s[chosen] = 1
                    selections.append(s)
        for selection in selections:
            expected = 1.0
            for kind, outputs, entry in states:
                if kind == "direct":
                    index, flip = entry
                    if selection[outputs[0]] and ((parameter >> index) & 1) ^ int(bool(flip)):
                        expected = -expected
                    continue
                sim, magic, witness, axes = entry
                p = stim.PauliString(len(magic))
                for output, axis in zip(outputs, axes):
                    if selection[output]:
                        p *= axis
                e = sim.peek_observable_expectation(p)
                if witness and not p.commutes(magic):
                    e = np.sqrt(0.5) * (e + sim.peek_observable_expectation(1j * magic * p))
                expected *= e
            actual = native.expectation(words(parameter, native.words), selection)
            error = abs(actual - expected)
            maximum = max(maximum, error)
            checked += 1
            nonzero += abs(expected) > 1e-9
            fractional += 1e-9 < abs(expected) < 1 - 1e-9
            assert error < 1e-12, (name, mode, case, actual, expected)
    return dict(
        kind="fixed_fault_oracle",
        circuit=name,
        mode=mode,
        cases=32,
        parities=checked,
        nonzero=int(nonzero),
        fractional=int(fractional),
        max_absolute_error=float(maximum),
    )


def features(bits, detectors, rng):
    values = [bits.astype(float), np.all(bits[:, :detectors] == 0, axis=1)[:, None]]
    for _ in range(128):
        chosen = rng.choice(bits.shape[1], int(rng.integers(2, 10)), replace=False)
        values.append((bits[:, chosen].sum(1) % 2)[:, None])
    return np.concatenate(values, axis=1)


def noisy_tests(name, mode):
    native = Native(ROOT / f"{name}-{mode}.plan")
    n = 65536 if name == "d5" else 8192
    _, bits, _ = native.sample(n, seed=7803, retain=True)
    base = features(bits, native.detectors, np.random.default_rng(91))
    result = []
    for label in ["original", "reconstructed"]:
        text = (
            (STUDY / "inputs" / f"{name}.stim").read_text()
            if label == "original"
            else read_circuit(STUDY / f"{name}-{mode}-compressed.clifft")
        )
        pm = clifft.default_hir_pass_manager()
        pm.add(clifft.ActiveWidthSchedulePass())
        program = clifft.compile(text, hir_passes=pm)
        r = clifft.sample(program, shots=n, threads=1, seed=8801 if label == "original" else 8809)
        if mode == "records":
            ref = r.measurements if label == "original" else r.detectors
        else:
            ref = np.concatenate([r.detectors, r.observables], axis=1)
        comparison = features(ref, native.detectors, np.random.default_rng(91))
        pool = (base.mean(0) + comparison.mean(0)) / 2
        z = abs(base.mean(0) - comparison.mean(0)) / np.sqrt(
            np.maximum(pool * (1 - pool) * 2 / n, 1 / n**2)
        )
        assert z.max() < 7, (name, mode, label, z.max())
        result.append(
            dict(
                reference=label,
                max_z=float(z.max()),
                acceptance=float(comparison[:, native.outputs].mean()),
            )
        )
    # Postselected count accumulation uses the same sampling law, with early rejection.
    if mode == "detectors":
        npost = 1048576
        passed, _, counts = native.sample(npost, seed=1081, postselect=True)
        pm = clifft.default_hir_pass_manager()
        pm.add(clifft.ActiveWidthSchedulePass())
        program = clifft.compile(
            read_circuit(STUDY / f"{name}-{mode}-compressed.clifft"),
            hir_passes=pm,
            postselection_mask=[1] * native.detectors,
        )
        ref = clifft.sample_survivors(
            program, shots=npost, threads=1, seed=9011, keep_records=False
        )
        for left, right in zip([passed, *counts], [ref.passed_shots, *ref.observable_ones]):
            pool = (left + right) / (2 * npost)
            z = abs(float(left) - float(right)) / np.sqrt(max(2 * npost * pool * (1 - pool), 1))
            assert z < 7, (name, "postselected", left, right, z)
        result.append(
            dict(
                reference="reconstructed_postselected",
                shots=npost,
                native_passed=passed,
                reference_passed=int(ref.passed_shots),
                native_observable_ones=counts.tolist(),
                reference_observable_ones=ref.observable_ones.tolist(),
            )
        )
    return dict(
        kind="noisy_joint_features",
        circuit=name,
        mode=mode,
        shots=n,
        features=base.shape[1],
        acceptance_native=float(base[:, native.outputs].mean()),
        comparisons=result,
    )


if __name__ == "__main__":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    results = json.loads((ROOT / "validation.json").read_text()) if "--resume" in sys.argv else []

    def record(row):
        results.append(row)
        print(json.dumps(row), flush=True)
        (ROOT / "validation.json").write_text(json.dumps(results, indent=2) + "\n")

    if not results:
        record(dense_tests())
    for mode in ["detectors", "records"]:
        for name in ["d5", "d7"]:
            if not any(
                r.get("circuit") == name
                and r.get("mode") == mode
                and r["kind"] == "fixed_fault_oracle"
                for r in results
            ):
                record(oracle_tests(name, mode))
            if not any(
                r.get("circuit") == name
                and r.get("mode") == mode
                and r["kind"] == "noisy_joint_features"
                for r in results
            ):
                record(noisy_tests(name, mode))
