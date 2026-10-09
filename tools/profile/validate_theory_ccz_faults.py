"""Exact bounded fault checks for the theory paper's depth-two tricycle circuit.

The reference expands the propagated physical Clifford error in Pauli Z
characters with integer coefficients. It does not use shared phase analysis,
Clifft optimization, or samples to calculate the syndrome law.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from automatic_specialization import History, analyze
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from shared_phase_specialization import SharedPhase
from study_quadcycle_factory import code_pair, load_css, pauli_channel_3
from validate_automatic_specialization import visible_probability

import clifft

Fault = tuple[int, tuple[tuple[int, str], ...]]


def propagated_error(
    gates: list[tuple[int, ...]], faults: list[Fault]
) -> tuple[int, set[tuple[int, ...]]]:
    """Return X offsets and a sign polynomial for U_faulty U_ideal dagger.

    Conjugating X through CCZ adds a quadratic sign. Multiple faults update
    the accumulated X offset before the next gate; no single-fault addition
    assumption is made. An overall phase is irrelevant to every query here.
    """
    x = 0
    terms: set[tuple[int, ...]] = set()
    by_boundary: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for boundary, paulis in faults:
        if not 0 <= boundary <= len(gates):
            raise ValueError("Fault lies outside the CCZ circuit")
        by_boundary[boundary].extend(paulis)
    for boundary in range(len(gates) + 1):
        if boundary:
            gate = gates[boundary - 1]
            for size in (1, 2):
                for subset in itertools.combinations(gate, size):
                    if all((x >> q) & 1 for q in gate if q not in subset):
                        terms.symmetric_difference_update([tuple(sorted(subset))])
        for q, axis in by_boundary[boundary]:
            if axis not in "XYZ":
                raise ValueError("Reference faults must be nonidentity Paulis")
            if axis in "YZ":
                terms.symmetric_difference_update([(q,)])
            if axis in "XY":
                x ^= 1 << q
    return x, terms


def fourier_coefficients(
    columns: list[int], terms: set[tuple[int, ...]]
) -> tuple[dict[int, int], int]:
    """Expand diagonal errors after restriction to a binary linear support.

    The character label records commutation with each generator row. Labels
    equal modulo Z stabilizers must be combined before squaring amplitudes.
    """
    coefficients = {0: 1}
    denominator = 1
    for term in sorted(terms):
        if len(term) == 1:
            coefficients = {key ^ columns[term[0]]: value for key, value in coefficients.items()}
        else:
            a, b = (columns[q] for q in term)
            updated: dict[int, int] = defaultdict(int)
            for key, value in coefficients.items():
                for shift, sign in ((0, 1), (a, 1), (b, 1), (a ^ b, -1)):
                    updated[key ^ shift] += sign * value
            coefficients = {key: value for key, value in updated.items() if value}
            denominator *= 2
            if len(coefficients) > 65536:
                raise ValueError("Physical-error expansion exceeds its bounded budget")
    if sum(value * value for value in coefficients.values()) != denominator * denominator:
        raise AssertionError("Physical-error expansion lost normalization")
    return coefficients, denominator


def oracle_control() -> dict[str, Any]:
    """Compare physical error propagation with direct independent statevectors."""
    simulator = AerSimulator(method="statevector", max_parallel_threads=1)
    maximum = 0.0
    for seed in range(32):
        rng = random.Random(71031 + seed)
        gates = [tuple(rng.sample(range(6), 3)) for _ in range(5)]
        faults: list[Fault] = [
            (rng.randrange(6), tuple((q, rng.choice("XYZ")) for q in rng.sample(range(6), 3)))
            for _ in range(1 + seed % 4)
        ]
        x, terms = propagated_error(gates, faults)
        preparation = QuantumCircuit(6)
        preparation.h(range(6))
        preparation.t(0)
        preparation.s(2)
        preparation.cx(0, 3)
        preparation.h(4)
        actual, expected = preparation.copy(), preparation.copy()
        for boundary in range(6):
            if boundary:
                actual.ccz(*gates[boundary - 1])
                expected.ccz(*gates[boundary - 1])
            for position, paulis in faults:
                if position == boundary:
                    for q, axis in paulis:
                        getattr(actual, axis.lower())(q)
        for term in terms:
            if len(term) == 1:
                expected.z(term[0])
            else:
                expected.cz(*term)
        for q in range(6):
            if (x >> q) & 1:
                expected.x(q)
        for circuit in (actual, expected):
            circuit.save_statevector()
        result = simulator.run([actual, expected]).result()
        vectors = [np.asarray(result.get_statevector(i)) for i in range(2)]
        error = abs(1 - abs(np.vdot(*vectors)) ** 2)
        maximum = max(maximum, float(error))
        if error > 1e-12:
            raise AssertionError("Physical error oracle differs from direct Aer evolution")
    return {"random_circuits": 32, "maximum_state_infidelity": maximum}


class Workload:
    def __init__(self, css: Any):
        tri, _, layered, _ = code_pair(css)
        self.tri = tri
        ordered = sorted(layered, key=lambda t: t[0] >= 27)
        self.gates = [tuple(27 * block + q % 27 for block, q in enumerate(t)) for t in ordered]
        self.preparation = []
        self.columns = []
        self.checks = []
        self.z_rows = css.f2_basis(tri.hz)
        self.z_checks = []
        for block in range(3):
            self.preparation += [f"H {27 * block + q}" for q in tri.plus_qubits()]
            self.preparation += [f"CX {a} {b}" for _, a, b in tri.encoder_ops(27 * block)]
            self.columns += [
                sum(int(bit) << (11 * block + row) for row, bit in enumerate(column))
                for column in tri.generator_matrix.T
            ]
            self.checks += [
                "MPP " + "*".join(f"X{27 * block + int(q)}" for q in np.flatnonzero(check))
                for check in tri.x_stabilizer_basis
            ]
            self.z_checks += [
                "MPP " + "*".join(f"Z{27 * block + int(q)}" for q in np.flatnonzero(check))
                for check in self.z_rows
            ]
        self.logical_mask = sum(7 << (11 * block) for block in range(3))
        self.syndrome_bits = [q for q in range(33) if not ((self.logical_mask >> q) & 1)]
        lines = self.preparation.copy()
        lines += [f"DEPOLARIZE1(.01) {q}" for q in range(81)]
        self.events: list[list[Fault | None]] = [
            [None, *((0, ((q, p),)) for p in "XYZ")] for q in range(81)
        ]
        for index, gate in enumerate(self.gates):
            lines.append("CCZ " + " ".join(map(str, gate)))
            lines += pauli_channel_3((gate[0], gate[1], gate[2]), 0.02)
            self.events.append(
                [None]
                + [
                    (index + 1, tuple((q, p) for q, p in zip(gate, paulis) if p != "I"))
                    for paulis in list(itertools.product("IXYZ", repeat=3))[1:]
                ]
            )
        self.source = "\n".join(lines) + "\n"
        self.shared = SharedPhase(self.source, preserve_state=True)
        if len(self.shared.model.sites) != len(self.events):
            raise AssertionError("Physical fault labels do not match categorical sites")

    def reference(self, history: History) -> dict[str, Any]:
        faults = [self.events[site][outcome] for site, outcome in history]
        x, terms = propagated_error(self.gates, [fault for fault in faults if fault is not None])
        coefficients, denominator = fourier_coefficients(self.columns, terms)
        law: dict[int, Fraction] = defaultdict(Fraction)
        for label, amplitude in coefficients.items():
            record = sum(((label >> q) & 1) << i for i, q in enumerate(self.syndrome_bits))
            law[record] += Fraction(amplitude * amplitude, denominator * denominator)
        joint_fidelity = Fraction(coefficients.get(0, 0) ** 2, denominator * denominator)
        z_record = [
            sum(int(bit) * ((x >> (27 * block + q)) & 1) for q, bit in enumerate(row)) % 2
            for block in range(3)
            for row in self.z_rows
        ]
        return {
            "x": x,
            "law": dict(law),
            "accept_x": law.get(0, Fraction()),
            "accept_xz": Fraction() if any(z_record) else law.get(0, Fraction()),
            "z_record": z_record,
            "joint_oracle_corrected_fidelity": joint_fidelity,
            "pauli_characters": len(coefficients),
        }

    def score_tail(self, x: int) -> str:
        # This correction knows the injected X error; it is a validation
        # observable, not a decoder permitted to see hidden fault locations.
        lines = [f"X {q}" for q in range(81) if (x >> q) & 1]
        lines += ["CCZ " + " ".join(map(str, gate)) for gate in reversed(self.gates)]
        lines += list(reversed(self.preparation))
        lines += [f"M {q}" for q in range(81)]
        return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-limit", type=int)
    args = parser.parse_args()
    if args.single_limit is not None and args.single_limit < 0:
        parser.error("Single-fault limit must be nonnegative")
    control = oracle_control()
    workload = Workload(load_css(args.merlin_checkout))
    singles = [
        ((site, outcome),)
        for site, events in enumerate(workload.events)
        for outcome in range(1, len(events))
    ]
    references = [workload.reference(history) for history in singles]
    histogram = Counter(str(reference["accept_x"]) for reference in references)
    expected = {"0": 3132, "1/16": 189, "1/4": 81, "1/64": 54, "1": 189}
    if dict(histogram) != expected:
        raise AssertionError(f"Theory fault acceptance table differs: {histogram}")
    logical_failures = sum(r["accept_x"] > r["joint_oracle_corrected_fidelity"] for r in references)
    if logical_failures != 54 or any(r["accept_xz"] for r in references):
        raise AssertionError("Theory logical-error or ideal Z rejection claim differs")
    print("Independent enumeration reproduces the paper's single-fault table", flush=True)
    rng = random.Random(261003605)
    multiple: list[History] = [
        tuple(
            (site, rng.randrange(1, len(workload.events[site])))
            for site in sorted(rng.sample(range(135), count))
        )
        for count in (2, 3, 5)
        for _ in range(12)
    ]
    first_q = workload.gates[0][0]
    cancel_z = ((first_q, 3), (81, 48))
    echo_x = ((first_q, 1), (81, 16))
    logical_row = min(workload.tri.logical_zs, key=np.count_nonzero)
    logical_z = tuple((int(q), 3) for q in np.flatnonzero(logical_row))
    if len(logical_z) != 3:
        raise AssertionError("Expected a weight-three logical Z control")
    second_q = workload.gates[0][1]
    logical_with_cancel = tuple(sorted((*logical_z, (second_q, 3), (81, 12))))
    targeted = [cancel_z, echo_x, logical_z, logical_with_cancel]
    target_expected = [
        (Fraction(1), Fraction(1)),
        (Fraction(1, 4), Fraction(1, 4)),
        (Fraction(1), Fraction()),
        (Fraction(1), Fraction()),
    ]
    for history, expected_pair in zip(targeted, target_expected):
        reference = workload.reference(history)
        if (reference["accept_x"], reference["joint_oracle_corrected_fidelity"]) != expected_pair:
            raise AssertionError("Accepted multiple-fault control differs")
    multiple += targeted
    histories = (
        [()] + (singles if args.single_limit is None else singles[: args.single_limit]) + multiple
    )
    maximum = 0.0
    max_width = 0
    probabilities_checked = scores_checked = 0
    scored_classes: set[tuple[str, bool, int]] = set()
    digest = hashlib.sha256()
    for index, history in enumerate(histories):
        reference = workload.reference(history)
        text = workload.shared.render_state(workload.shared.controls(history))
        measured = text + "\n".join(workload.checks + workload.z_checks) + "\n"
        hir, info = analyze(measured)
        max_width = max(max_width, info["peak_width"])
        if info["peak_width"] > 16:
            raise ValueError("Candidate exceeds the execution width budget")
        program = clifft.lower(hir)
        for record, expected_probability in reference["law"].items():
            actual = visible_probability(
                program, [(record >> q) & 1 for q in range(24)] + reference["z_record"]
            )
            difference = abs(actual - float(expected_probability))
            maximum = max(maximum, difference)
            if difference > 1e-11:
                raise AssertionError(
                    f"Complete syndrome law differs at {history}: "
                    f"{actual} versus {expected_probability}"
                )
            probabilities_checked += 1
        category = (
            str(reference["accept_x"]),
            reference["accept_x"] > reference["joint_oracle_corrected_fidelity"],
            0 if not history else 1 + (history[0][0] >= 81) + (history[0][0] >= 108),
        )
        if category not in scored_classes or len(history) > 1:
            scored_classes.add(category)
            scored, score_info = analyze(measured + workload.score_tail(reference["x"]))
            max_width = max(max_width, score_info["peak_width"])
            if score_info["peak_width"] > 16:
                raise ValueError("Score exceeds the execution width budget")
            actual = visible_probability(
                clifft.lower(scored), [0] * 24 + reference["z_record"] + [0] * 81
            )
            error = abs(actual - float(reference["joint_oracle_corrected_fidelity"]))
            maximum = max(maximum, error)
            if error > 1e-11:
                raise AssertionError("Oracle-corrected joint fidelity differs")
            scores_checked += 1
        digest.update(
            json.dumps([history, sorted((k, str(v)) for k, v in reference["law"].items())]).encode()
        )
        if index % 100 == 0:
            print(f"Checked {index + 1}/{len(histories)} complete syndrome laws", flush=True)
    result = {
        "paper": "https://arxiv.org/html/2610.03605v1",
        "oracle_aer_control": control,
        "single_fault_reference_cases": len(singles),
        "single_fault_acceptance_histogram": dict(histogram),
        "single_fault_logical_z_survivors": logical_failures,
        "single_fault_ideal_xz_survivors": 0,
        "candidate_histories": len(histories),
        "multiple_fault_histories": len(multiple),
        "targeted_accepted_histories": [
            {
                "history": history,
                "accept_x": str(values[0]),
                "joint_oracle_corrected_fidelity": str(values[1]),
            }
            for history, values in zip(targeted, target_expected)
        ],
        "complete_xz_syndrome_probabilities_checked": probabilities_checked,
        "joint_oracle_corrected_fidelity_queries": scores_checked,
        "maximum_probability_error": maximum,
        "maximum_active_width": max_width,
        "reference_laws_sha256": digest.hexdigest(),
        "scope": (
            "Ideal preparation and syndrome measurements; fixed faults, not noisy protocol rates. "
            "Fidelity queries use fault-aware X correction, not a physical decoder."
        ),
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("study_quadcycle_factory.py"),
                Path(__file__).with_name("automatic_specialization.py"),
                Path(__file__).with_name("shared_phase_specialization.py"),
            )
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
