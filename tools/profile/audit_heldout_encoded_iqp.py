"""Independent physical Aer checks for the reconstructed encoded IQP input."""

from __future__ import annotations

import argparse
import importlib.metadata
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from automatic_specialization import ANNOTATIONS, instruction_text
from complete_circuit_panel import packed, save
from continuation_trace_reuse import ContinuationPhase
from qiskit_aer import AerSimulator
from study_chan_boundary import aer_circuit
from study_heldout_encoded_iqp import circuit_text, load_results, verify
from validate_chan_algebraic import purify_resets

import clifft


def physical_preparation(source):
    """Reuse the physical gate converter, retaining terminal X readout separately."""
    parsed = clifft.parse(source)
    lines, readout, used = [], [], set()
    for node in parsed.nodes:
        gate = node.gate.name
        if gate in ANNOTATIONS:
            continue
        qubits = [t.value for t in node.targets]
        if gate == "R":
            if any(q in used for q in qubits):
                raise ValueError("Reference only removes resets of untouched zero inputs")
            gate = "I"
            lines.append("I " + " ".join(map(str, qubits)))
        elif gate == "MX":
            if len(qubits) != 1:
                raise ValueError("Expected individual terminal X measurements")
            readout.append((qubits[0], repr(node.targets[0]).startswith("!")))
        else:
            if readout:
                raise ValueError("Quantum operation follows terminal measurement")
            lines.append(instruction_text(node, 0))
        used.update(qubits)
    if [q for q, _ in readout] != list(range(parsed.num_qubits)):
        raise ValueError("Reference requires complete ordered physical readout")
    quantum = "\n".join(lines) + "\n"
    return aer_circuit(quantum, [], []), readout


def readout_basis(circuit, readout):
    for q, invert in readout:
        circuit.h(q)
        if invert:
            circuit.x(q)


def dense_states(source):
    circuit, readout = physical_preparation(source)
    circuit.save_statevector(label="live")
    readout_basis(circuit, readout)
    circuit.save_statevector(label="readout")
    result = AerSimulator(method="statevector", max_parallel_threads=1).run(circuit).result()
    if not result.success:
        raise RuntimeError(result.status)
    return np.asarray(result.data()["live"]), np.asarray(result.data()["readout"])


def mps_amplitudes(snapshot, records):
    tensors, bonds = snapshot
    amplitudes = []
    for record in records:
        vector = np.ones(1, dtype=complex)
        for q, pair in enumerate(tensors):
            vector = vector @ pair[int(record[q])]
            if q < len(bonds):
                vector *= bonds[q]
        amplitudes.append(vector[0])
    return np.asarray(amplitudes)


def small(binary):
    front = ContinuationPhase(circuit_text(1), binary)
    rng = random.Random(831610)
    histories = [()] + [front.model.draw(rng) for _ in range(8)]
    histories += [tuple((i, len(s.replacements) - 1) for i, s in enumerate(front.model.sites))]
    histories += [
        tuple((i, 1) for i, s in enumerate(front.model.sites) if s.gate == "READOUT_NOISE")
    ]
    rows = []
    for index, history in enumerate(histories):
        original = front.model.render(history)
        branch = front.rewrite(history, 921 + index)
        physical, distribution = dense_states(original)
        candidate, other_distribution = dense_states(branch.source)
        overlap = np.vdot(physical, candidate)
        phase = overlap / abs(overlap)
        live_error = float(np.max(np.abs(candidate - phase * physical)))
        probability_error = float(
            np.max(np.abs(abs(distribution) ** 2 - abs(other_distribution) ** 2))
        )
        if max(live_error, probability_error) > 1e-10:
            raise AssertionError((index, live_error, probability_error))
        row: dict[str, Any] = dict(
            index=index,
            history=history,
            live_state_error=live_error,
            full_distribution_error=probability_error,
        )
        if index == 0:
            circuit, readout = physical_preparation(original)
            readout_basis(circuit, readout)
            circuit.save_matrix_product_state()
            circuit.save_statevector()
            reference = (
                AerSimulator(
                    method="matrix_product_state",
                    matrix_product_state_truncation_threshold=0,
                    max_parallel_threads=1,
                )
                .run(circuit)
                .result()
            )
            vector = np.asarray(reference.data()["statevector"])
            row["mps_full_distribution_error"] = float(
                np.max(abs(abs(vector) ** 2 - abs(distribution) ** 2))
            )
            if row["mps_full_distribution_error"] > 1e-10:
                raise AssertionError(
                    "MPS amplitude reference differs from dense physical reference"
                )
            indices = [0, 3, 57047, 18573] + random.Random(725).sample(range(65536), 256)
            queries = [[(i >> q) & 1 for q in range(16)] for i in indices]
            amplitudes = mps_amplitudes(reference.data()["matrix_product_state"], queries)
            row["mps_contraction_queries"] = len(queries)
            row["mps_contraction_error"] = float(np.max(abs(amplitudes - vector[indices])))
            if row["mps_contraction_error"] > 1e-10:
                raise AssertionError("MPS contraction differs from physical statevector")
        rows.append(row)
        print("small", index, live_error, probability_error, flush=True)
    return dict(
        physical_qubits=16,
        logical_qubits=6,
        rows=rows,
        aer_version=importlib.metadata.version("qiskit-aer"),
        scope="full pre-readout state and exhaustive 65536-outcome probability distribution",
    )


def large(args):
    row, result = packed(args.input), load_results(args.results)
    verify(row)
    if result["input_sha256"] != row["sha256"]:
        raise ValueError("Results and physical source differ")
    front = ContinuationPhase(row["source"], args.bin / "phase_exporter")
    report = dict(
        input_sha256=row["sha256"],
        rows=[],
        method=(
            "original fixed-fault physical source versus phase export, "
            "native full-record probabilities"
        ),
        independence=(
            "independent of research phase reducer; shares Clifft compiler and probability engine"
        ),
        independent_large_aer=(
            "not performed by this native audit; see HELDOUT_IQP.md "
            "for the interrupted exploratory MPS reference"
        ),
    )
    save(args.output, report)
    for attempt in result["conditional_export"]["attempts"]:
        if attempt["status"] != "sampled":
            report["rows"].append(dict(index=attempt["index"], refusal=attempt.get("failure")))
            save(args.output, report)
            continue
        start = time.perf_counter()
        history = tuple(map(tuple, attempt["history"]))
        records = attempt["records"]
        queries = [
            records,
            result["fixed_fault"]["attempts"][attempt["index"]]["records"],
            [bit ^ (i == 0) for i, bit in enumerate(records)],
        ]
        physical = clifft.compile(purify_resets(front.model.render(history)))
        if physical.peak_active_width > 24:
            raise ValueError("Physical reference exceeds declared width budget")
        expected = clifft.record_probabilities(physical, np.asarray(queries, dtype=np.uint8))
        program = clifft.compile(purify_resets(attempt["conditional_source"]))
        actual = clifft.record_probabilities(program, np.asarray(queries, dtype=np.uint8))
        if not np.allclose(actual, expected, rtol=1e-9, atol=1e-28):
            raise AssertionError((attempt["index"], actual, expected))
        report["rows"].append(
            dict(
                index=attempt["index"],
                queries=queries,
                reference_probabilities=expected.tolist(),
                conditional_probabilities=actual.tolist(),
                reference_width=physical.peak_active_width,
                seconds=time.perf_counter() - start,
            )
        )
        save(args.output, report)
        print("large", attempt["index"], expected.tolist(), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("small", "large"))
    parser.add_argument("--bin", type=Path, default=Path("/tmp/clifft-complete-panel/bin"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, small(args.bin / "phase_exporter") if args.mode == "small" else large(args))


if __name__ == "__main__":
    main()
