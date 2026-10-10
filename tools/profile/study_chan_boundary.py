"""Offline state-boundary study of the retained flagged parity checks.

The supplied region boundaries are audit inputs, not a new automatic frontend.
Dense tomography is a bounded oracle, not a proposed simulation algorithm.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import stim
from automatic_specialization import ANNOTATIONS, FaultModel, History
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator

import clifft

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "research/conditional_clifford/chan-boundary-inputs.json.gz"


def pauli(x: int, z: int, n: int) -> stim.PauliString:
    return stim.PauliString("".join("IXZY"[((x >> q) & 1) + 2 * ((z >> q) & 1)] for q in range(n)))


def masks(p: stim.PauliString) -> tuple[int, int]:
    return (
        sum(1 << q for q, a in enumerate(p) if a in (1, 2)),
        sum(1 << q for q, a in enumerate(p) if a in (2, 3)),
    )


def apply_pauli(v: np.ndarray, p: stim.PauliString) -> np.ndarray:
    x, z = masks(p)
    indices = np.arange(len(v), dtype=np.uint64)
    signs = np.ones(len(v))
    for q in range(len(p)):
        if (z >> q) & 1:
            signs *= 1 - 2 * ((indices >> q) & 1).astype(float)
    result = np.empty_like(v)
    result[indices ^ x] = p.sign * (1j ** (x & z).bit_count()) * signs * v
    return result


def state_error(a: np.ndarray, b: np.ndarray) -> float:
    overlap = np.vdot(a, b)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-14 else 1
    return float(np.max(abs(a * phase - b)))


def independent(rows: list[stim.PauliString]) -> list[stim.PauliString]:
    basis: dict[int, int] = {}
    result = []
    for p in rows:
        x, z = masks(p)
        word = x | (z << len(p))
        while word:
            pivot = word.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = word
                result.append(p)
                break
            word ^= basis[pivot]
    return result


def classify(v: np.ndarray) -> tuple[dict, str | None]:
    """Exhaust every physical Pauli on at most seven data qubits."""
    n = len(v).bit_length() - 1
    if n > 7:
        raise ValueError("The exhaustive Pauli oracle is limited to seven qubits")
    rows = []
    indices = np.arange(len(v))
    for x in range(1 << n):
        correlations = v.conj() * v[indices ^ x]
        for k in range(n):
            blocks = correlations.reshape(-1, 2, 1 << k)
            a, b = blocks[:, 0, :].copy(), blocks[:, 1, :].copy()
            blocks[:, 0, :], blocks[:, 1, :] = a + b, a - b
        for raw_z in np.flatnonzero(abs(abs(correlations) - 1) < 1e-9):
            z = int(raw_z)
            value = ((-1j) ** (x & z).bit_count()) * correlations[z]
            p = pauli(x, z, n)
            p.sign = 1 if value.real > 0 else -1
            rows.append(p)
    generators = independent(rows)
    rank = len(generators)
    if len(rows) != 1 << rank:
        raise AssertionError("The enumerated stabilizers do not form a Pauli subgroup")
    result = dict(stabilizer_rank=rank, stabilizer_nullity=n - rank)
    if rank < n - 1:
        return result | {"class": "outside_one_t"}, None
    decoder = stim.Tableau.from_stabilizers(generators, allow_underconstrained=True)
    if rank == n:
        return result | {"class": "stabilizer"}, str(decoder.to_circuit())
    axes = [decoder.x_output(n - 1), decoder.y_output(n - 1), decoder.z_output(n - 1)]
    bloch = np.array([np.vdot(v, apply_pauli(v, p)).real for p in axes])
    result["decoded_bloch"] = bloch.tolist()
    target = np.array([1, 1, 0]) / np.sqrt(2)
    for frame in stim.Tableau.iter_all(1):
        rotated = np.array(
            [
                sum(
                    float(frame.inverse()(stim.PauliString(axis)).sign.real) * target[j]
                    for j in range(3)
                    if frame.inverse()(stim.PauliString(axis))[0] == j + 1
                )
                for axis in "XYZ"
            ]
        )
        if np.max(abs(rotated - bloch)) < 1e-9:
            local = remap(str(frame.to_circuit()), [n - 1])
            source = f"H {n - 1}\nT {n - 1}\n" + local + str(decoder.to_circuit())
            return result | {"class": "one_t"}, source
    return result | {"class": "one_qubit_other_angle"}, None


def remap(source: str, qubits: list[int]) -> str:
    result = []
    for line in source.splitlines():
        gate, *targets = line.split()
        result.append(gate + " " + " ".join(str(qubits[int(t)]) for t in targets))
    return "\n".join(result) + "\n"


def factor_data(v: np.ndarray, data: list[int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    n = len(v).bit_length() - 1
    environment = [q for q in range(n) if q not in data]
    matrix = v.reshape([2] * n).transpose([n - 1 - q for q in data[::-1] + environment[::-1]])
    matrix = matrix.reshape(1 << len(data), -1)
    column = int(np.argmax(np.sum(abs(matrix) ** 2, axis=0)))
    state = matrix[:, column] / np.linalg.norm(matrix[:, column])
    other = state.conj() @ matrix
    np.testing.assert_allclose(matrix, np.outer(state, other), rtol=0, atol=1e-10)
    return state, other, environment


def aer_circuit(source: str, checkpoints: list[int], central_nodes: list[int]) -> QuantumCircuit:
    parsed = clifft.parse(source)
    circuit = QuantumCircuit(parsed.num_qubits, parsed.num_measurements)
    records = 0
    for i, node in enumerate(parsed.nodes):
        if i in checkpoints:
            circuit.save_statevector(label=f"boundary_{i}")
        if i in central_nodes:
            circuit.save_expectation_value(
                Pauli("X"), [node.targets[0].value], label=f"central_{i}"
            )
        gate, targets = node.gate.name, list(node.targets)
        qubits = [t.value for t in targets]
        if gate in ANNOTATIONS:
            continue
        if gate in {"M", "MX"}:
            q = qubits[0]
            if gate == "MX":
                circuit.h(q)
            circuit.measure(q, records)
            if repr(targets[0]).startswith("!"):
                raise ValueError("This Aer adapter requires physical readout faults")
            if gate == "MX":
                circuit.h(q)
            records += 1
        elif gate in {"R", "RX"}:
            circuit.reset(qubits[0])
            if gate == "RX":
                circuit.h(qubits[0])
        elif gate in {"T", "T_DAG", "S", "S_DAG", "H", "X", "Y", "Z", "CX", "CZ", "I"}:
            if targets[0].is_rec:
                with circuit.if_test((circuit.clbits[qubits[0]], 1)):
                    getattr(circuit, gate[1].lower())(qubits[1])
            else:
                getattr(
                    circuit, {"T_DAG": "tdg", "S_DAG": "sdg", "I": "id"}.get(gate, gate.lower())
                )(*qubits)
        else:
            raise ValueError(f"Unsupported Aer boundary operation {gate}")
    if len(parsed.nodes) in checkpoints:
        circuit.save_statevector(label=f"boundary_{len(parsed.nodes)}")
    return circuit


def quantum_lines(source: str) -> list[str]:
    return [
        line
        for line in source.splitlines()
        if line and line.split()[0].split("(")[0] not in ANNOTATIONS
    ]


def classical_skeleton(case: dict) -> dict:
    """Recheck the actual inverse network and the product-state flag contract."""
    envelope = case["envelope"]
    certificate = envelope["certificate"]
    lines = case["source"].splitlines()
    source = "\n".join(
        quantum_lines("\n".join(lines[envelope["first_line"] - 1 : envelope["last_line"]]))
    )
    clean = quantum_lines(FaultModel(source).render(()))
    phases = certificate["phase_qubits"]
    n = clifft.parse(source).num_qubits
    middle = clean[len(phases) : -len(phases)]
    flags = {int(q): a for q, a in certificate["flags"].items()}
    root = certificate["central_qubit"]
    for q in flags:
        uses = [line.split()[0] for line in middle if str(q) in line.split()[1:]]
        if q in phases or not uses or uses[0] != "R" or uses[-1] != "M":
            raise AssertionError("A flag is not prepared first and measured last")
    root_measurements = [line for line in middle if line == f"MX {root}"]
    root_resets = [line for line in middle if line == f"RX {root}"]
    if len(root_measurements) != 1 or len(root_resets) != 1:
        raise AssertionError("Expected one central measurement and reset")
    for q in phases:
        first = [line.split()[0] for line in clean[: len(phases)] if int(line.split()[1]) == q]
        last = [line.split()[0] for line in clean[-len(phases) :] if int(line.split()[1]) == q]
        if sorted(first + last) != ["T", "T_DAG"]:
            raise AssertionError("The outer layers are not inverse single-T layers")
    before, after = stim.Circuit(f"I {n - 1}"), stim.Circuit(f"I {n - 1}")
    measured = False
    for line in middle:
        gate, *targets = line.split()
        qs = [int(q.lstrip("!")) for q in targets]
        if gate == "MX" and qs == [root]:
            measured = True
        elif gate == "RX" and qs == [root]:
            if not measured:
                raise AssertionError("Reset precedes its central measurement")
        elif gate == "CX":
            (after if measured else before).append("CX", qs)
        elif gate in {"R", "M"} and qs[0] in flags:
            if flags[qs[0]] != "Z":
                raise AssertionError("This retained check uses Z flags")
        else:
            raise ValueError(f"Unexpected middle operation {line}")
    if before.to_tableau().inverse() != after.to_tableau():
        raise AssertionError("The two Clifford networks are not inverses")
    x = stim.PauliString(n)
    x[root] = "X"
    parity = before.to_tableau().inverse()(x)
    expected = stim.PauliString(n)
    for q, axis in certificate["parity"].items():
        expected[int(q)] = axis
    if parity != expected:
        raise AssertionError("The audited pulled-back parity changed")
    for q in flags:
        if parity[q] not in (0, 3):
            raise AssertionError("The coherent branches distinguish a flag")
    for q in range(n):
        if q not in phases and q not in flags and parity[q]:
            if case["entry"]["spectator_axes"].get(str(q)) != ["X"] or parity[q] != 1:
                raise AssertionError("The relative parity acts on an unresolved spectator")
    return dict(source=source, parity=parity, flags=flags, root=root, physical_qubits=n)


def branch_frames(
    source: str, root: int, flags: dict, outcome: int
) -> tuple[stim.PauliString, stim.PauliString]:
    """Collect both coherent paths, including noise between measurement and reset."""
    n = clifft.parse(source).num_qubits
    frames = [stim.PauliString(n), stim.PauliString(n)]
    reset_bit = outcome
    between = False
    for line in quantum_lines(source):
        gate, *targets = line.split()
        if gate in {"T", "T_DAG"}:
            continue
        qs = [int(q.lstrip("!")) for q in targets]
        if gate == "CX":
            instruction = stim.Circuit(line)
            frames = [p.after(instruction) for p in frames]
        elif gate in {"X", "Y", "Z"}:
            p = stim.PauliString(n)
            p[qs[0]] = gate
            frames = [p * f for f in frames]
            if between and qs[0] == root and gate in {"Y", "Z"}:
                reset_bit ^= 1
        elif gate == "MX" and qs == [root]:
            p = stim.PauliString(n)
            p[root] = "X"
            frames[1] = p * frames[1]
            between = True
        elif gate == "RX" and qs == [root]:
            p = stim.PauliString(n)
            if reset_bit:
                p[root] = "Z"
            frames = [p * f for f in frames]
            between = False
        elif gate in {"R", "RX"} and qs[0] in flags:
            if flags[qs[0]] != ("X" if gate == "RX" else "Z"):
                raise ValueError("Flag reset basis differs from its certificate")
            for p in frames:
                p[qs[0]] = "I"
        elif gate in {"M", "MX"} and qs[0] in flags:
            if flags[qs[0]] != ("X" if gate == "MX" else "Z"):
                raise ValueError("Flag readout basis differs from its certificate")
        else:
            raise ValueError(f"Unexpected bound operation {line}")
    return frames[0], frames[1]


def checks_and_entry(case: dict, shots: int) -> tuple[list[stim.PauliString], dict]:
    data = case["envelope"]["certificate"]["phase_qubits"]
    mapping = {q: i for i, q in enumerate(data)}
    lines = case["source"].splitlines()
    last_t = max(i for i, line in enumerate(lines) if line.startswith(("T ", "T_DAG ")))
    checks, probes = [], []
    for line in lines[last_t + 1 :]:
        if not line.startswith("MPP "):
            continue
        p = stim.PauliString(len(data))
        for term in line.split()[1].split("*"):
            p[mapping[int(term[1:])]] = term[0]
        checks.append(p)
        probes.append(line.replace("MPP", "EXP_VAL"))
    probes += ["EXP_VAL " + "*".join(a + str(q) for q in data) for a in "XYZ"]
    start = case["envelope"]["first_line"] - 1
    end = case["envelope"]["last_line"]
    rows = {}
    for boundary, length in (("entry", start), ("exit", end)):
        prefix = "\n".join(lines[:length]) + "\n"
        for noisy in (False, True):
            source = prefix if noisy else FaultModel(prefix).render(())
            program = clifft.compile(source + "\n".join(probes) + "\n")
            samples = clifft.sample(
                program, shots=shots if noisy else 1, seed=495, threads=1, batch_size=1
            )
            values = samples.exp_vals
            defects = np.max(1 - abs(values[:, :-3]), axis=1)
            rows[f"{boundary}_{'noisy' if noisy else 'ideal'}"] = dict(
                shots=len(values),
                peak_active_width=program.peak_active_width,
                outside_original_signed_code=int(np.count_nonzero(defects > 1e-9)),
                distinct_logical_bloch=np.unique(np.round(values[:, -3:], 10), axis=0).tolist(),
            )
            if boundary == "entry":
                np.testing.assert_allclose(defects, 0, rtol=0, atol=1e-10)
                np.testing.assert_allclose(
                    np.sort(abs(values[:, -3:]), axis=1),
                    np.tile([0, 2**-0.5, 2**-0.5], (len(values), 1)),
                    rtol=0,
                    atol=1e-10,
                )
                if not noisy:
                    for p, value in zip(checks, values[0, :-3]):
                        p.sign = 1 if value > 0 else -1
    return checks, rows


def dense_phase(signs: dict, data: list[int]) -> np.ndarray:
    indices = np.arange(1 << len(data), dtype=np.uint64)
    powers = np.zeros(len(indices), dtype=np.int32)
    for i, q in enumerate(data):
        powers += int(signs.get(str(q), 0)) * ((indices >> i) & 1).astype(np.int32)
    result: np.ndarray = np.exp(1j * np.pi / 4 * powers)
    return result


def eigenstate_certificate(case: dict, checks: list, entry: dict, trials: int) -> dict:
    data = case["envelope"]["certificate"]["phase_qubits"]
    n = len(data)
    z = stim.PauliString("Z" * n)
    x = stim.PauliString("X" * n)
    zero = (
        stim.Tableau.from_stabilizers(checks + [z]).to_state_vector(endian="little").astype(complex)
    )
    bx, by, bz = entry["entry_ideal"]["distinct_logical_bloch"][0]
    if abs(bz) > 1e-9:
        raise ValueError("Expected the supplied equatorial encoded state")
    # All-data Y differs from the logical Y defined by all-data X and Z.
    bx = np.copysign(2**-0.5, bx)
    logical_y = np.copysign(2**-0.5, by) * (-1) ** ((n - 1) // 2)
    v = (zero + (bx + 1j * logical_y) * apply_pauli(zero, x)) / np.sqrt(2)
    v /= np.linalg.norm(v)
    phase = dense_phase(case["envelope"]["certificate"]["phase_signs"], data)

    def apply_c(state):
        return phase.conj() * apply_pauli(phase * state, x)

    cv = apply_c(v)
    eigenvalue = 1 if np.vdot(v, cv).real > 0 else -1
    eigen_error = float(np.max(abs(cv - eigenvalue * v)))
    if eigen_error > 1e-9:
        raise AssertionError("The ideal entry is not a parity-Clifford eigenstate")
    # Conjugating a Pauli by C is a Clifford tableau operation. Its global
    # phase is irrelevant to this conjugation, but the product E*E' phase is not.
    c_tableau = stim.Tableau(n)
    for i, q in enumerate(data):
        sign = int(case["envelope"]["certificate"]["phase_signs"][str(q)])
        t = np.diag([1, np.exp(1j * sign * np.pi / 4)])
        local = t.conj().T @ np.array([[0, 1], [1, 0]]) @ t
        c_tableau.append(stim.Tableau.from_unitary_matrix(local, endian="little"), [i])
    rng = random.Random(495)
    modes: Counter[str] = Counter()
    max_error = 0.0
    for _ in range(trials):
        e = stim.PauliString("".join(rng.choice("IXYZ") for _ in data))
        ep = c_tableau(e)
        q = e * ep
        mode = "pauli_projection" if q.sign in (1, -1) else "clifford_quarter_turn"
        modes[mode] += 1
        ev = apply_pauli(v, e)
        for outcome in (0, 1):
            s = (-1) ** outcome
            left = (ev + s * apply_c(ev)) / 2
            right = apply_pauli((v + s * eigenvalue * apply_pauli(v, q)) / 2, e)
            max_error = max(max_error, float(np.max(abs(left - right))))
    if max_error > 1e-9:
        raise AssertionError("The conditioned Clifford identity failed")
    return dict(
        data_qubits=n,
        ideal_parity_eigenvalue=eigenvalue,
        ideal_eigenstate_error=eigen_error,
        pauli_entry_trials=trials,
        outcome_identities=2 * trials,
        modes=dict(modes),
        maximum_identity_error=max_error,
    )


def physical_frame_checks(case: dict, trials: int) -> dict:
    skeleton = classical_skeleton(case)
    model = FaultModel(skeleton["source"])
    rng = random.Random(391495)
    errors = 0
    weights = []
    for i in range(trials):
        history = (
            model.draw(rng)
            if i < trials // 2
            else tuple(
                (j, rng.randrange(1, len(s.replacements)))
                for j, s in enumerate(model.sites)
                if rng.random() < 0.02
            )
        )
        weights.append(len(history))
        for outcome in (0, 1):
            a, b = branch_frames(
                model.render(history), skeleton["root"], skeleton["flags"], outcome
            )
            inverse = a.copy()
            inverse.sign = inverse.sign.conjugate()
            ratio = inverse * b
            if ratio not in (skeleton["parity"], -skeleton["parity"]):
                errors += 1
    if errors:
        raise AssertionError("The fixed-fault paths do not have the certified relative Clifford")
    return dict(
        histories=trials,
        conditioned_outcomes=2 * trials,
        minimum_faults=min(weights),
        maximum_faults=max(weights),
        failed_relative_paulis=errors,
    )


def trajectory_checks(case: dict, trials: int, repetitions: int) -> dict:
    lines = case["source"].splitlines()
    start, end = case["envelope"]["first_line"] - 1, case["envelope"]["last_line"]
    data = case["envelope"]["certificate"]["phase_qubits"]
    prefix, body = "\n".join(lines[:start]), "\n".join(lines[start:end])
    source = prefix + "\n" + (body + "\n") * repetitions
    model = FaultModel(source)
    rng = random.Random(572 + repetitions)
    classes: Counter[str] = Counter()
    central_outcomes: Counter[str] = Counter()
    maximum_error = 0.0
    maximum_path_error = maximum_probability_error = 0.0
    flag_records_checked = 0
    nonzero_flag_records = 0
    histories = []
    representative = None
    for i in range(trials):
        history: History
        if i == 0:
            history = ()
        elif i < trials * 3 // 4:
            history = model.draw(rng)
        else:
            history = tuple(
                (j, rng.randrange(1, len(s.replacements)))
                for j, s in enumerate(model.sites)
                if rng.random() < 0.02
            )
        histories.append(len(history))
        materialized = model.render(history)
        nodes = clifft.parse(materialized).nodes
        t_nodes = [j for j, node in enumerate(nodes) if node.gate.name in {"T", "T_DAG"}]
        prefix_t = sum(node.gate.name in {"T", "T_DAG"} for node in clifft.parse(prefix).nodes)
        blocks = [
            (
                t_nodes[prefix_t + 2 * len(data) * k],
                t_nodes[prefix_t + 2 * len(data) * (k + 1) - 1] + 1,
            )
            for k in range(repetitions)
        ]
        central_nodes = [
            j
            for a, b in blocks
            for j in range(a, b)
            if nodes[j].gate.name == "MX"
            and nodes[j].targets[0].value == case["envelope"]["certificate"]["central_qubit"]
        ]
        boundaries = sorted({p for block in blocks for p in block})
        circuit = aer_circuit(materialized, boundaries, central_nodes)
        result = (
            AerSimulator(method="statevector", max_parallel_threads=1)
            .run(circuit, shots=1, seed_simulator=870 + i, memory=True)
            .result()
        )
        if not result.success:
            raise AssertionError("Aer trajectory failed")
        v = np.asarray(result.data()[f"boundary_{len(nodes)}"])
        record = result.get_memory()[0][::-1]
        n = len(v).bit_length() - 1
        phase = dense_phase(case["envelope"]["certificate"]["phase_signs"], list(range(n)))
        flags = {int(q): a for q, a in case["envelope"]["certificate"]["flags"].items()}
        for (a, b), central in zip(blocks, central_nodes):
            incoming = np.array(result.data()[f"boundary_{a}"])
            for q in flags:
                z = stim.PauliString(n)
                z[q] = "Z"
                expectation = np.vdot(incoming, apply_pauli(incoming, z)).real
                if abs(abs(expectation) - 1) > 1e-9:
                    raise AssertionError("A reset flag is entangled at entry")
                if expectation < 0:
                    z[q] = "X"
                    incoming = apply_pauli(incoming, z)
            record_index = sum(node.gate.name in {"M", "MX"} for node in nodes[:central])
            outcome = int(record[record_index])
            bound_source = "\n".join(quantum_lines("\n".join(materialized.splitlines()[a:b])))
            f0, f1 = branch_frames(bound_source, nodes[central].targets[0].value, flags, outcome)
            f0 += stim.PauliString(n - len(f0))
            f1 += stim.PauliString(n - len(f1))
            predicted = (
                phase.conj()
                * (
                    apply_pauli(phase * incoming, f0)
                    + (-1) ** outcome * apply_pauli(phase * incoming, f1)
                )
                / 2
            )
            probability = float(np.vdot(predicted, predicted).real)
            expected_probability = (1 + (-1) ** outcome * result.data()[f"central_{central}"]) / 2
            maximum_probability_error = max(
                maximum_probability_error, abs(probability - expected_probability)
            )
            expected = np.asarray(result.data()[f"boundary_{b}"])
            maximum_path_error = max(
                maximum_path_error, state_error(predicted / np.sqrt(probability), expected)
            )
            if maximum_path_error > 1e-9 or maximum_probability_error > 1e-9:
                raise AssertionError(
                    "Bound coherent paths disagree with the physical Aer instrument"
                )
            for j in range(a, b):
                node = nodes[j]
                if node.gate.name != "M" or node.targets[0].value not in flags:
                    continue
                q = node.targets[0].value
                observable = stim.PauliString(n)
                observable[q] = "Z"
                expectation = (
                    np.vdot(predicted, apply_pauli(predicted, observable)).real / probability
                )
                if abs(abs(expectation) - 1) > 1e-9:
                    raise AssertionError("A flag record still depends on a coherent branch")
                later_flips = (
                    sum(
                        o.gate.name in {"X", "Y"} and o.targets[0].value == q
                        for o in nodes[j + 1 : b]
                    )
                    % 2
                )
                flag_record = sum(o.gate.name in {"M", "MX"} for o in nodes[:j])
                if int(expectation < 0) ^ later_flips != int(record[flag_record]):
                    raise AssertionError("The conditional flag record changed")
                flag_records_checked += 1
                nonzero_flag_records += int(record[flag_record])
        for j, node in enumerate(n for n in nodes if n.gate.name in {"M", "MX"}):
            if (
                node.gate.name == "MX"
                and node.targets[0].value == case["envelope"]["certificate"]["central_qubit"]
            ):
                central_outcomes[record[j]] += 1
        state, environment, environment_qubits = factor_data(v, data)
        classification, preparation = classify(state)
        classes[classification["class"]] += 1
        if preparation is None:
            raise AssertionError(f"A physical trajectory left the one-T class: {classification}")
        env = stim.Tableau.from_state_vector(environment, endian="little")
        rebuilt = remap(preparation, data) + remap(str(env.to_circuit()), environment_qubits)
        native_data = np.asarray(clifft.get_statevector(clifft.compile(preparation)))
        if state_error(state, native_data) > 1e-9:
            raise AssertionError("The native data preparation differs from Aer")
        compiled = clifft.compile(rebuilt)
        if compiled.peak_active_width > 1:
            raise AssertionError("The reconstructed state does not compile at width one")
        size = len(clifft.parse(rebuilt).nodes)
        reconstructed_result = (
            AerSimulator(method="statevector", max_parallel_threads=1)
            .run(aer_circuit(rebuilt, [size], []), shots=1)
            .result()
        )
        reconstructed = np.asarray(reconstructed_result.data()[f"boundary_{size}"])
        error = state_error(v, reconstructed)
        maximum_error = max(maximum_error, error)
        if error > 1e-9:
            raise AssertionError(
                "The emitted one-T preparation changed the complete physical state"
            )
        if representative is None and history:
            representative = dict(
                history=history, records=record, preparation=rebuilt, classification=classification
            )
    return dict(
        repetitions=repetitions,
        constructed_repeat_control=repetitions > 1,
        trajectories=trials,
        classes=dict(classes),
        central_outcomes=dict(central_outcomes),
        maximum_faults=max(histories),
        maximum_full_state_error=maximum_error,
        physical_instrument_checks=trials * repetitions,
        flag_records_checked=flag_records_checked,
        nonzero_flag_records=nonzero_flag_records,
        maximum_path_state_error=maximum_path_error,
        maximum_conditional_probability_error=maximum_probability_error,
        representative=representative,
    )


def negative_control(case: dict) -> dict:
    data = case["envelope"]["certificate"]["phase_qubits"]
    n = len(data)
    phase = dense_phase(case["envelope"]["certificate"]["phase_signs"], data)
    v = np.ones(1 << n, dtype=complex) / np.sqrt(1 << n)
    v[np.arange(len(v)) & 1 != 0] *= np.exp(1j * np.pi / 4)
    c = phase.conj() * apply_pauli(phase * v, stim.PauliString("X" * n))
    rows = []
    for outcome in (0, 1):
        state = (v + (-1) ** outcome * c) / 2
        probability = float(np.vdot(state, state).real)
        classification, _ = classify(state / np.sqrt(probability))
        rows.append(dict(outcome=outcome, probability=probability, **classification))
    if not any(r["class"] == "outside_one_t" for r in rows):
        raise AssertionError("The unsupported entry control no longer distinguishes the contract")
    return dict(
        entry="one T on a product of seven plus states; same data parity Clifford", outcomes=rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--probe-shots", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.trials <= 128 or not 1 <= args.probe_shots <= 4096:
        parser.error("Use 1-128 trials and 1-4096 probe shots for this bounded study")
    package = json.loads(gzip.decompress(args.inputs.read_bytes()))
    report: dict[str, Any] = dict(
        input_archive_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        package_versions={
            name: importlib.metadata.version(name)
            for name in ("clifft", "stim", "numpy", "qiskit", "qiskit-aer")
        },
        native_extension_sha256=hashlib.sha256(
            next(Path(clifft.__file__).parent.glob("_clifft_core*.so")).read_bytes()
        ).hexdigest(),
        provenance={k: v for k, v in package.items() if k != "cases"},
        cases={},
    )
    for name, case in package["cases"].items():
        if hashlib.sha256(case["source"].encode()).hexdigest() != case["sha256"]:
            raise ValueError("A retained source hash changed")
        checks, probes = checks_and_entry(case, args.probe_shots)
        row = dict(
            source_sha256=case["sha256"],
            probes=probes,
            eigenstate_certificate=eigenstate_certificate(case, checks, probes, args.trials),
            physical_frames=physical_frame_checks(case, args.trials),
        )
        if len(case["envelope"]["certificate"]["phase_qubits"]) == 7:
            row["physical_state_checks"] = [
                trajectory_checks(case, args.trials, repeat) for repeat in (1, 2, 3)
            ]
            row["unsupported_entry_control"] = negative_control(case)
        report["cases"][name] = row
        print(name, "boundary checks passed", flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
