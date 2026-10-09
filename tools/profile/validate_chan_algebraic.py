"""Independent small instruments and complete-circuit conditional record checks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import random
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import stim
from automatic_specialization import ANNOTATIONS
from chan_algebraic_continuation import ConditionalChecks, NeedDecision
from conditional_clifford_core import MonomialClifford, Refused, SmallState, StabilizerAmplitudes
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from study_automatic_specialization import sample_parities
from study_chan_boundary import aer_circuit

import clifft

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research/conditional_clifford"


def dense(state: SmallState) -> np.ndarray:
    return np.asarray(clifft.get_statevector(clifft.compile(state.source())))


def error(left: np.ndarray, right: np.ndarray) -> float:
    overlap = np.vdot(left, right)
    phase = overlap / abs(overlap) if abs(overlap) else 1
    return float(np.max(abs(left * phase - right)))


def matrix(p: stim.PauliString) -> np.ndarray:
    return np.asarray(p.sign * Pauli("".join("IXYZ"[a] for a in p)[::-1]).to_matrix())


def primitives() -> dict:
    rng = random.Random(2910495)
    maximum = 0.0
    measurements = accepted = refused = 0
    for n in range(1, 7):
        for _ in range(12):
            circuit = QuantumCircuit(n)
            circuit.h(n - 1)
            circuit.t(n - 1)
            state = SmallState(n)
            state.clifford(stim.Circuit(f"H {n - 1}"))
            p = stim.PauliString(n)
            p[n - 1] = "Z"
            state.rotate_t(p, 1)
            for _ in range(3 * n):
                q = rng.randrange(n)
                gate = rng.choice(["H", "S", "CX"] if n > 1 else ["H", "S"])
                qs = [q] if gate != "CX" else rng.sample(range(n), 2)
                getattr(circuit, gate.lower())(*qs)
                state.clifford(stim.Circuit(gate + " " + " ".join(map(str, qs))))
            original = np.asarray(Statevector.from_instruction(circuit))
            maximum = max(maximum, error(original, dense(state)))
            amplitudes = StabilizerAmplitudes(state.frame)
            v = np.array([amplitudes.amplitude(i) for i in range(1 << n)])
            reference = state.frame.to_state_vector(endian="little").astype(complex)
            reference /= np.linalg.norm(reference)
            maximum = max(maximum, error(reference, v))
            p = stim.PauliString("".join(rng.choice("IXYZ") for _ in range(n)))
            p.sign = rng.choice([-1, 1])
            signs = {q: rng.choice([-1, 1]) for q in range(n)}
            phase = np.exp(
                1j
                * np.pi
                / 4
                * np.array(
                    [sum(sign * ((i >> q) & 1) for q, sign in signs.items()) for i in range(1 << n)]
                )
            )
            clifford = phase.conj()[:, None] * matrix(p) * phase[None, :]
            for outcome in (0, 1):
                for instrument in (False, True):
                    candidate = SmallState(n)
                    candidate.frame, candidate.vector = state.frame.copy(), state.vector.copy()
                    expected = (
                        original
                        + (-1) ** outcome * (clifford if instrument else matrix(p)) @ original
                    ) / 2
                    expected_probability = float(np.vdot(expected, expected).real)
                    try:
                        probability = (
                            candidate.clifford_projector(MonomialClifford(p, signs), outcome)[0]
                            if instrument
                            else candidate.measure(p, outcome)
                        )
                    except Refused:
                        if not instrument:
                            raise
                        refused += 1
                        continue
                    maximum = max(maximum, abs(probability - expected_probability))
                    if probability > 1e-12:
                        maximum = max(
                            maximum, error(expected / np.sqrt(probability), dense(candidate))
                        )
                    if instrument:
                        accepted += 1
                    else:
                        measurements += 1
    # The supplied negative control cannot be certified by one-T gate count alone.
    state = SmallState(7)
    state.clifford(stim.Circuit("H 0 1 2 3 4 5 6"))
    state.rotate_t(stim.PauliString("Z______"), 1)
    try:
        state.clifford_projector(
            MonomialClifford(stim.PauliString("XXXXXXX"), dict.fromkeys(range(7), -1)), 0
        )
    except Refused:
        pass
    else:
        raise AssertionError("The incompatible one-T entry was accepted")
    if maximum > 1e-10:
        raise AssertionError(("Small independent instruments", maximum))
    return dict(
        pauli_branches=measurements,
        clifford_branches=accepted,
        bounded_refusals=refused,
        maximum_error=maximum,
        incompatible_entry_refused=True,
    )


def aer_trajectory(source: str, seed: int) -> tuple[list[int], np.ndarray, float]:
    parsed = clifft.parse(source)
    circuit = QuantumCircuit(parsed.num_qubits, parsed.num_measurements)
    flips: list[int] = []
    for node in parsed.nodes:
        gate, targets = node.gate.name, list(node.targets)
        qs = [t.value for t in targets]
        if gate in ANNOTATIONS:
            continue
        if gate in {"M", "MX", "MY", "MPP"}:
            axes = (
                [(repr(t).lstrip("!")[0], t.value) for t in targets]
                if gate == "MPP"
                else [({"M": "Z", "MX": "X", "MY": "Y"}[gate], qs[0])]
            )
            flip = sum(repr(t).startswith("!") for t in targets) % 2
            index = len(flips)
            circuit.save_expectation_value(
                Pauli("".join(a for a, _ in axes)[::-1]), [q for _, q in axes], label=f"p{index}"
            )
            for axis, q in axes:
                if axis == "Y":
                    circuit.sdg(q)
                if axis != "Z":
                    circuit.h(q)
            pivot = axes[-1][1]
            for _, q in axes[:-1]:
                circuit.cx(q, pivot)
            circuit.measure(pivot, index)
            for _, q in reversed(axes[:-1]):
                circuit.cx(q, pivot)
            for axis, q in reversed(axes):
                if axis != "Z":
                    circuit.h(q)
                if axis == "Y":
                    circuit.s(q)
            flips.append(flip)
        elif gate in {"R", "RX"}:
            circuit.reset(qs[0])
            if gate == "RX":
                circuit.h(qs[0])
        elif targets and targets[0].is_rec:
            with circuit.if_test((circuit.clbits[qs[0]], 1 ^ flips[qs[0]])):
                getattr(circuit, gate[1].lower())(qs[1])
        else:
            getattr(circuit, {"T_DAG": "tdg", "S_DAG": "sdg", "I": "id"}.get(gate, gate.lower()))(
                *qs
            )
    circuit.save_statevector()
    result = (
        AerSimulator(method="statevector", max_parallel_threads=1)
        .run(circuit, shots=1, seed_simulator=seed, memory=True)
        .result()
    )
    if not result.success:
        raise AssertionError("Aer trajectory failed")
    physical = [int(x) for x in result.get_memory()[0][::-1]]
    probability = float(
        np.prod([(1 + (-1) ** bit * result.data()[f"p{i}"]) / 2 for i, bit in enumerate(physical)])
    )
    return (
        [bit ^ flip for bit, flip in zip(physical, flips)],
        np.asarray(result.get_statevector()),
        probability,
    )


def purify_resets(source: str) -> str:
    """Use fresh environment wires so native queries marginalize hidden resets."""
    mapping: dict[int, int] = {}
    next_qubit = 0
    lines = []

    def allocate(q: int) -> int:
        nonlocal next_qubit
        mapping[q] = next_qubit
        next_qubit += 1
        return mapping[q]

    def target(match):
        q = int(match[0])
        return str(mapping[q] if q in mapping else allocate(q))

    for line in source.splitlines():
        gate, *rest = line.split(" ", 1)
        if gate.split("(")[0] in ANNOTATIONS:
            continue
        if gate in {"R", "RX"}:
            q = allocate(int(rest[0]))
            if gate == "RX":
                lines.append(f"H {q}")
        else:
            tokens = [
                t if t.startswith("rec[") else re.sub(r"\d+", target, t) for t in rest[0].split()
            ]
            lines.append(gate + " " + " ".join(tokens))
    return "\n".join(lines) + "\n"


def complete(cases: dict, trials: int) -> dict:
    report = {}
    for name, case in cases.items():
        if hashlib.sha256(case["source"].encode()).hexdigest() != case["sha256"]:
            raise AssertionError("Input hash changed")
        front = ConditionalChecks(case["source"])
        rng = random.Random(6495)
        sites = [site for segment in front.segments for site in segment.model.sites]
        histories = [()] + [front.draw(rng) for _ in range(trials - 1)]
        histories += [
            tuple(
                (j, rng.randrange(1, len(s.replacements)))
                for j, s in enumerate(sites)
                if rng.random() < 0.015
            )
            for _ in range(4)
        ]
        witness = None
        if name == "chan-four-round-d5":
            witness = tuple(
                map(
                    tuple,
                    json.loads((DATA / "chan-four-fault-history.json").read_text())["history"],
                )
            )
            histories.append(witness)
        rows = []
        for i, history in enumerate(histories):
            fixed = "".join(front.materialize(history))
            snapshots = []
            start = perf_counter()
            sampled = front.trajectory(
                history,
                i,
                checkpoint=lambda s, r, g, t: snapshots.append((s.source(), r.copy(), g.last_line)),
            )
            seconds = perf_counter() - start
            if sampled["hidden_random"]:
                raise AssertionError(
                    "These selected record queries need hidden-branch marginalization"
                )
            record = sampled["records"]
            exact = clifft.compile(purify_resets(fixed))
            queries = [record]
            ordinary = clifft.sample(
                clifft.compile(fixed), shots=2, seed=113 + i, threads=1
            ).measurements
            queries.extend(ordinary.astype(int).tolist())
            for index in rng.sample(range(len(record)), min(3, len(record))):
                mutated = record.copy()
                mutated[index] ^= 1
                queries.append(mutated)
            reference = clifft.record_probabilities(exact, np.asarray(queries, dtype=np.uint8))
            maximum_relative = 0.0
            for query, expected in zip(queries, reference):
                result = front.trajectory(history, i, forced_records=query)
                actual = result["probability"]
                if not np.isclose(actual, expected, rtol=1e-9, atol=1e-13):
                    raise AssertionError((name, i, actual, float(expected)))
                if expected:
                    maximum_relative = max(maximum_relative, abs(actual / expected - 1))
            emitted = clifft.sample(
                clifft.compile(sampled["final_source"]), shots=1, seed=17, threads=1
            )
            np.testing.assert_array_equal(emitted.measurements[0], record)
            sample_parities(case["source"], emitted)
            for preparation, _, _ in snapshots:
                if clifft.compile(preparation).peak_active_width > 1:
                    raise AssertionError("A boundary export lost the one-coordinate bound")
            row = dict(
                faults=len(history),
                history=history,
                seconds=seconds,
                reductions=sampled["reductions"],
                peak_carry_width=sampled["peak_carry_width"],
                exact_queries=len(queries),
                maximum_relative_error=maximum_relative,
                detector_events=int(emitted.detectors.sum()),
                witness=history == witness,
            )
            if name == "chan-v1-d3":
                bits, expected_state, expected_probability = aer_trajectory(fixed, 271 + i)
                actual = front.trajectory(history, i, forced_records=bits)
                preparation = actual["final_source"].split("MPAD")[0]
                size = len(clifft.parse(preparation).nodes)
                candidate = np.asarray(
                    AerSimulator(method="statevector", max_parallel_threads=1)
                    .run(aer_circuit(preparation, [size], []), shots=1)
                    .result()
                    .data()[f"boundary_{size}"]
                )
                discrepancy = error(expected_state, candidate)
                if discrepancy > 1e-10 or abs(actual["probability"] - expected_probability) > 1e-10:
                    raise AssertionError(("Aer complete instrument", i, discrepancy))
                row.update(
                    aer_state_error=discrepancy,
                    aer_probability_error=abs(actual["probability"] - expected_probability),
                )
            rows.append(row)
            print(name, i, len(history), "passed", flush=True)
        report[name] = dict(source_sha256=case["sha256"], fault_sites=front.fault_sites, rows=rows)
    return report


def controls(cases: dict) -> dict:
    results = []
    for name, case in cases.items():
        rng = random.Random(415)
        permutation = list(range(clifft.parse(case["source"]).num_qubits))
        rng.shuffle(permutation)
        lines = []
        for line in case["source"].splitlines():
            gate, rest = line.split(" ", 1)
            tokens = [
                t
                if t.startswith("rec[")
                else re.sub(r"\d+", lambda m: str(permutation[int(m[0])]), t)
                for t in rest.split()
            ]
            lines.append(gate + " " + " ".join(tokens))
        renamed = "\n".join(lines) + "\n"
        front = ConditionalChecks(renamed)
        history = front.draw(rng)
        sample = front.trajectory(history, 415)
        expected = float(
            clifft.record_probabilities(
                clifft.compile(purify_resets("".join(front.materialize(history)))),
                np.asarray([sample["records"]], dtype=np.uint8),
            )[0]
        )
        if not np.isclose(sample["probability"], expected, rtol=1e-10, atol=1e-13):
            raise AssertionError("Renaming changed the conditional law")
        results.append(dict(case=name, kind="permuted_qubits", probability=expected))

    case = cases["chan-v1-d3"]
    lines = case["source"].splitlines()
    lo, hi = case["envelope"]["first_line"] - 1, case["envelope"]["last_line"]
    root_changed = flag_changed = False
    for i in range(lo, hi):
        gate, *targets = lines[i].split()
        if gate == "MX" and not root_changed:
            lines[i] = "MX(0.25) " + targets[0]
            root_changed = True
        elif gate in {"M", "MZ"} and not flag_changed:
            lines[i] = "M(0.25) " + targets[0]
            flag_changed = True
    readout = "\n".join(lines) + "\n"
    front = ConditionalChecks(readout)
    readouts = [
        s.site_offset + i
        for s in front.segments
        for i, x in enumerate(s.model.sites)
        if x.gate == "READOUT_NOISE"
    ]
    if len(readouts) != 2:
        raise AssertionError("Readout controls changed")
    for mask in range(4):
        history = tuple((i, 1) for j, i in enumerate(readouts) if mask >> j & 1)
        sample = front.trajectory(history, 41)
        fixed = "".join(front.materialize(history))
        expected = float(
            clifft.record_probabilities(
                clifft.compile(purify_resets(fixed)),
                np.asarray([sample["records"]], dtype=np.uint8),
            )[0]
        )
        bits, state, probability = aer_trajectory(fixed, 412)
        actual = front.trajectory(history, 41, forced_records=bits)
        if (
            abs(sample["probability"] - expected) > 1e-10
            or abs(actual["probability"] - probability) > 1e-10
        ):
            raise AssertionError("Reported noise changed quantum collapse")
        results.append(dict(kind="reported_noise", fault_bits=mask, probability=expected))
    refused = []
    for source in ("H 0 1\nR 0 1\n", "T 0 1\n", "M 0 1\n"):
        try:
            ConditionalChecks(source)
        except Refused:
            refused.append("grouped_single_qubit_instruction")
        else:
            raise AssertionError("A grouped single-qubit instruction was not refused")
    for kind in ("inverse_network", "inverse_phase"):
        lines = case["source"].splitlines()
        if kind == "inverse_network":
            index = next(i for i in range(lo, hi) if lines[i].startswith("CX "))
            lines[index] = "I 0"
        else:
            lines[hi - 1] = lines[hi - 1].replace("T ", "T_DAG ")
        front = ConditionalChecks("\n".join(lines) + "\n")
        try:
            front.trajectory((), 4)
        except Refused:
            refused.append(kind)
        else:
            raise AssertionError("A malformed whole-check boundary was admitted")
    return dict(equivalence_controls=results, malformed_refused=refused)


def exhaustive_witness(case: dict) -> dict:
    front = ConditionalChecks(case["source"])
    history = tuple(
        map(tuple, json.loads((DATA / "chan-four-fault-history.json").read_text())["history"])
    )
    pending: list[tuple[int, ...]] = [()]
    rows = []
    acceptance = 0.0
    while pending:
        decisions = pending.pop()
        try:
            row = front.trajectory(history, 947, decisions=decisions)
        except NeedDecision:
            if len(decisions) >= 10:
                raise AssertionError("Witness enumeration exceeded its branch budget")
            pending.extend([decisions + (0,), decisions + (1,)])
            continue
        if row["hidden_random"]:
            raise AssertionError("Witness acquired unresolved reset branches")
        sample = clifft.sample(clifft.compile(row["final_source"]), shots=1, threads=1)
        if not sample.detectors.any():
            acceptance += row["probability"]
        rows.append(row)
    if abs(sum(r["probability"] for r in rows) - 1) > 1e-10:
        raise AssertionError("Witness branches lost probability")
    reference = clifft.record_probabilities(
        clifft.compile(purify_resets("".join(front.materialize(history)))),
        np.asarray([r["records"] for r in rows], dtype=np.uint8),
    )
    errors = [abs(float(p) - r["probability"]) for p, r in zip(reference, rows)]
    if max(errors) > 1e-10 or acceptance != 0:
        raise AssertionError(("Four-fault witness", max(errors), acceptance))
    return dict(
        branches=len(rows),
        unique_records=len({tuple(r["records"]) for r in rows}),
        probability_sum=sum(r["probability"] for r in rows),
        acceptance=acceptance,
        maximum_probability_error=max(errors),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.trials <= 64:
        parser.error("Use 1-64 trials")
    report: dict = dict(
        primitives=primitives(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("clifft", "stim", "qiskit", "qiskit-aer", "numpy")
        },
    )
    print("Independent bounded instruments passed", flush=True)
    cases = json.loads(gzip.decompress((DATA / "chan-boundary-inputs.json.gz").read_bytes()))[
        "cases"
    ]
    report["complete"] = complete(cases, args.trials)
    report["controls"] = controls(cases)
    report["four_fault_witness"] = exhaustive_witness(cases["chan-four-round-d5"])
    report["native_extension_sha256"] = hashlib.sha256(
        next(Path(clifft.__file__).parent.glob("_clifft_core*.so")).read_bytes()
    ).hexdigest()
    report["source_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path(__file__).with_name("conditional_clifford_core.py"),
            Path(__file__).with_name("chan_algebraic_continuation.py"),
            Path(__file__).with_name("parity_envelope_audit.py"),
            Path(__file__).with_name("study_chan_boundary.py"),
            Path(__file__).with_name("automatic_specialization.py"),
        ]
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
