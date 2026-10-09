"""Independent quantum-state and record checks for a reduced region's exit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import stim
from automatic_specialization import ANNOTATIONS, FaultModel, instruction_text
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from shared_phase_specialization import SharedPhase
from study_quadcycle_factory import pauli_channel_3
from validate_automatic_specialization import (
    all_histories,
    exact_record_probabilities,
    record_pauli,
    stim_to_qiskit_noiseless,
)
from validate_shared_phase_specialization import prefix_law, prefix_values

import clifft


def instrument(source: str, num_qubits: int = 0) -> np.ndarray:
    """Unnormalized output density matrix for each actual classical record.

    Deferred coherent records are dephased before further controls, so reset,
    feedback, and reuse of a measured physical wire have their instrument
    semantics. Keeping every output-qubit matrix element tests continuation.
    """
    from qiskit_aer.noise import pauli_error

    parsed = clifft.parse(source)
    n, m = max(num_qubits, parsed.num_qubits), parsed.num_measurements
    if n + m > 10:
        raise ValueError("Density-matrix validation exceeds its qubit budget")
    circuit = QuantumCircuit(n + m)
    records = 0
    for node in parsed.nodes:
        gate = node.gate.name
        targets = list(node.targets)
        if gate in ANNOTATIONS:
            continue
        if gate in {"M", "MX", "MY", "MPP", "MPAD"}:
            if gate == "MPAD":
                if targets[0].value:
                    circuit.x(n + records)
            else:
                terms = (
                    [(repr(t).lstrip("!")[0], t.value) for t in targets]
                    if gate == "MPP"
                    else [({"M": "Z", "MX": "X", "MY": "Y"}[gate], targets[0].value)]
                )
                record_pauli(
                    circuit,
                    terms,
                    n + records,
                    inverted=sum(repr(t).startswith("!") for t in targets) % 2 != 0,
                )
                circuit.append(pauli_error([("I", 0.5), ("Z", 0.5)]), [n + records])
            records += 1
        elif gate in {"R", "RX", "RY"}:
            for target in targets:
                circuit.reset(target.value)
                if gate != "R":
                    circuit.h(target.value)
                if gate == "RY":
                    circuit.s(target.value)
        elif gate == "I":
            continue
        elif any(t.is_rec for t in targets):
            if gate not in {"CX", "CY", "CZ"} or not targets[0].is_rec:
                raise ValueError("Unsupported reference feedback")
            getattr(circuit, gate.lower())(n + targets[0].value, targets[1].value)
        else:
            piece = stim_to_qiskit_noiseless(instruction_text(node, records))
            circuit.compose(piece, qubits=list(range(piece.num_qubits)), inplace=True)
    circuit.save_density_matrix()
    result = AerSimulator(method="density_matrix", max_parallel_threads=1).run(circuit).result()
    if not result.success:
        raise AssertionError("Independent instrument simulation failed")
    density = np.asarray(result.data()["density_matrix"])
    size = 1 << n
    return np.stack(
        [density[i * size : (i + 1) * size, i * size : (i + 1) * size] for i in range(1 << m)]
    )


def compare(region: str, tail: str, histories: list[Any]) -> dict[str, Any]:
    shared = SharedPhase(region, preserve_state=True)
    values = prefix_values(shared)
    maximum = 0.0
    for history in histories:
        prefix_law(shared, history)
        expected = instrument(shared.model.render(history) + tail, shared.model.num_qubits)
        observed = np.zeros_like(expected)
        for value in values:
            text = shared.render_state(shared.controls(history, value)) + tail
            actual = instrument(text, shared.model.num_qubits)
            observed += actual / len(values)
            program = clifft.lower(clifft.trace(clifft.parse(text)))
            probabilities = exact_record_probabilities(
                program, shared.model.num_records + clifft.parse(tail).num_measurements
            )
            np.testing.assert_allclose(
                probabilities, np.trace(actual, axis1=1, axis2=2).real, atol=1e-11, rtol=0
            )
        error = float(np.max(np.abs(expected - observed)))
        maximum = max(maximum, error)
        np.testing.assert_allclose(observed, expected, atol=1e-11, rtol=0)
    return {
        "histories": len(histories),
        "prefix_branches": len(values),
        "maximum_instrument_error": maximum,
        "source_sha256": hashlib.sha256((region + tail).encode()).hexdigest(),
        **shared.metadata(),
    }


def channel_control() -> dict[str, Any]:
    source = "H 0\nE(.2) X0 Y1 Z2\nELSE_CORRELATED_ERROR(.3) Z0 X1\nM 0 1 2\n"
    model = FaultModel(source)
    np.testing.assert_allclose(model.sites[0].probabilities, [0.56, 0.2, 0.24], atol=1e-15)
    expected = np.zeros(8)
    for history, weight in all_histories(model):
        expected += weight * np.trace(instrument(model.render(history)), axis1=1, axis2=2).real
    sample = stim.Circuit(source).compile_sampler(seed=17891).sample(65536)
    packed = sample.astype(int) @ (1 << np.arange(3))
    observed = np.bincount(packed, minlength=8) / len(sample)
    bound = 7 * np.sqrt(expected * (1 - expected) / len(sample)) + 2 / len(sample)
    if np.any(abs(observed - expected) > bound):
        raise AssertionError("Correlated categorical channel differs from Stim")
    rng = random.Random(17310)
    counts = np.zeros(3)
    for _ in range(len(sample)):
        history = model.draw(rng)
        counts[history[0][1] if history else 0] += 1
    weights = np.asarray(model.sites[0].probabilities)
    if np.any(abs(counts / len(sample) - weights) > 7 * np.sqrt(weights / len(sample))):
        raise AssertionError("Correlated host draws have wrong categorical weights")
    for text, probabilities in (
        ("E(0) X0\nELSE_CORRELATED_ERROR(1) Y0", [0, 0, 1]),
        ("E(1) X0\nELSE_CORRELATED_ERROR(1) Y0", [0, 1, 0]),
    ):
        np.testing.assert_array_equal(FaultModel(text).sites[0].probabilities, probabilities)
    return {"categorical_probabilities": weights.tolist(), "stim_and_host_draws": len(sample)}


def three_qubit_control() -> dict[str, Any]:
    channel = "\n".join(pauli_channel_3((0, 1, 2), 0.63)) + "\n"
    model = FaultModel(channel)
    expected = np.r_[0.37, np.full(63, 0.01)]
    np.testing.assert_allclose(model.sites[0].probabilities, expected, atol=1e-15)
    # Bell readout distinguishes every Pauli, including pure phase errors.
    preparation = "H 0 1 2\nCX 0 3 1 4 2 5\n"
    readout = "CX 0 3 1 4 2 5\nH 0 1 2\nM 0 1 2 3 4 5\n"
    law = np.zeros(64)
    for history, weight in all_histories(model):
        sample = (
            stim.Circuit(preparation + model.render(history) + readout)
            .compile_sampler()
            .sample(1)[0]
        )
        outcome = int(sample.astype(int) @ (1 << np.arange(6)))
        if law[outcome]:
            raise AssertionError("Distinct Pauli outcomes merged in the channel")
        law[outcome] = weight
    sampled = (
        stim.Circuit(preparation + channel + readout).compile_sampler(seed=92373).sample(65536)
    )
    counts = np.bincount(sampled.astype(int) @ (1 << np.arange(6)), minlength=64)
    if np.any(abs(counts / len(sampled) - law) > 7 * np.sqrt(law * (1 - law) / len(sampled))):
        raise AssertionError("Three-qubit depolarization differs from Stim")
    region = "H 0 1 2\n" + channel + "CCZ 0 1 2\n"
    result = compare(region, "H 0\nT 0\nCX 0 1\nMY 2\n", [h for h, _ in all_histories(model)])
    return {"uniform_nonidentity_outcomes": 63, "stim_shots": len(sampled), "continuation": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    # The first wire is measured in the prefix and reused by the continuation.
    controls = [
        ("H 0\nM 0\nH 1\nT 1\nI 2\n", "H 0\nCX 0 1\nT 0\nMY 1\n"),
        (
            "H 0 1\nCX 0 2\nMPP(.13) Y0*Z1\nR 2\nE(.17) X0 Y1 Z2\n"
            "ELSE_CORRELATED_ERROR(.23) Z0 X1\nT 0\nCX 0 1\nT_DAG 1\n",
            "H 0\nM 0\nCX rec[-1] 1\nT 1\nCX 1 2\n",
        ),
        (
            "H 0 1 2\nCCZ 0 1 2\nE(.1) X0 Y1 Z2\nELSE_CORRELATED_ERROR(.2) Z0 X1\nCX 0 3 1 4 2 5\n",
            "MX 0\nCZ rec[-1] 3\nMX 1\nCZ rec[-1] 4\nMX 2\nCZ rec[-1] 5\nH 3\nT 3\nCX 3 4\n",
        ),
    ]
    for region, tail in controls:
        histories = [h for h, _ in all_histories(FaultModel(region))]
        rows.append(compare(region, tail, histories))
        print("instrument control passed", flush=True)
    for seed in range(12):
        rng = random.Random(261003605 + seed)
        region_lines = ["H 0 1 2"]
        for _ in range(5):
            a, b = rng.sample(range(3), 2)
            region_lines += [f"{rng.choice(('H', 'S', 'S_DAG'))} {a}", f"CX {a} {b}"]
        region_lines += ["DEPOLARIZE2(.04) 0 2", "T 0"]
        for _ in range(4):
            a, b = rng.sample(range(3), 2)
            region_lines += [f"{rng.choice(('T', 'T_DAG', 'X', 'Y'))} {a}", f"CX {a} {b}"]
        source = "\n".join(region_lines) + "\n"
        rows.append(compare(source, "H 0\nT 0\nCX 0 2\nMY 1\n", [(), ((0, 6),), ((0, 15),)]))
    result = {
        "channel": channel_control(),
        "three_qubit_channel": three_qubit_control(),
        "cases": rows,
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("shared_phase_specialization.py"),
                Path(__file__).with_name("automatic_specialization.py"),
            )
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("state-preserving continuation validations passed", flush=True)


if __name__ == "__main__":
    main()
