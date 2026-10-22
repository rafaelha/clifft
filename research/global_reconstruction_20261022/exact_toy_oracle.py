"""Enumerate every trajectory of small circuits with Qiskit state vectors."""

import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

ROOT = Path(__file__).parent


def pauli(q, letter):
    s = ["I"] * 3
    s[2 - q] = letter
    return Pauli("".join(s))


def project(state, observable, bit):
    return Statevector((state.data + (-1) ** bit * state.evolve(observable).data) / 2)


def make(case):
    lines = []
    branches: list[tuple[tuple[int, ...], Statevector]] = [((), Statevector.from_int(0, 8))]

    def gate(gate, *qubits):
        nonlocal branches
        qc = QuantumCircuit(3)
        getattr(qc, gate.lower())(*qubits)
        lines.append(f"{gate.upper()} " + " ".join(map(str, qubits)))
        branches = [(r, s.evolve(qc)) for r, s in branches]

    def measure(q, letter="Z", p=0.0):
        nonlocal branches
        lines.append(f"{'M' if letter == 'Z' else 'M' + letter}({p}) {q}")
        out = []
        for r, s in branches:
            for bit in [0, 1]:
                state = project(s, pauli(q, letter), bit)
                for flip, prob in [(0, 1 - p), (1, p)]:
                    if prob:
                        out.append((r + (bit ^ flip,), Statevector(state.data * np.sqrt(prob))))
        branches = out

    def reset(q):
        nonlocal branches
        lines.append(f"R {q}")
        out = []
        for r, s in branches:
            for bit in [0, 1]:
                state = project(s, pauli(q, "Z"), bit)
                if bit:
                    state = state.evolve(pauli(q, "X"))
                out.append((r, state))
        branches = out

    def noise(q, p):
        nonlocal branches
        lines.append(f"DEPOLARIZE1({p}) {q}")
        branches = [
            (r, Statevector(s.evolve(pauli(q, letter)).data * np.sqrt(prob)))
            for r, s in branches
            for letter, prob in [("I", 1 - p), ("X", p / 3), ("Y", p / 3), ("Z", p / 3)]
        ]

    gate("h", 0)
    gate("cx", 0, 1)
    gate("t", 0)
    if case >= 8:
        gate("h", 2)
        gate("t", 2)
        gate("cz", 1, 2)
        lines.extend(["E(0.08) X0 X2", "ELSE_CORRELATED_ERROR(0.13) Z1"])
        correlated = pauli(0, "X").compose(pauli(2, "X"))
        branches = [
            (r, Statevector(s.evolve(operator).data * np.sqrt(p)))
            for r, s in branches
            for operator, p in [
                (Pauli("III"), 0.92 * 0.87),
                (correlated, 0.08),
                (pauli(1, "Z"), 0.92 * 0.13),
            ]
        ]
    if case & 1:
        noise(0, 0.13)
    gate("h", 1)
    measure(1, "Z", 0.07 if case & 2 else 0)
    lines.append("CX rec[-1] 2")
    branches = [(r, s.evolve(pauli(2, "X")) if r[-1] else s) for r, s in branches]
    gate("t", 2)
    gate("s", 0)
    gate("cx", 2, 0)
    if case & 4:
        reset(1)
        gate("h", 1)
        gate("cx", 1, 0)
    measure(0, "X")
    measure(1, "Y")
    measure(2, "Z")
    dist = np.zeros(16)
    for record, state in branches:
        index = sum(b << i for i, b in enumerate(record))
        dist[index] += np.vdot(state.data, state.data).real
    assert abs(dist.sum() - 1) < 1e-12
    path = ROOT / f"toy{case}.stim"
    path.write_text("\n".join(lines) + "\n")
    return dict(case=case, path=str(path), probabilities=dist.tolist())


def correlated_case():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.t(0)
    qc.h(0)
    branches: list[tuple[tuple[int, ...], Statevector]] = [
        ((), Statevector.from_int(0, 8).evolve(qc))
    ]
    lines = ["H 0", "T 0", "H 0"]
    for _ in range(2):
        lines.extend(["E(0.13) X0 X1", "ELSE_CORRELATED_ERROR(0.08) X1 X2"])
        choices = [(Pauli("III"), 0.87 * 0.92), (Pauli("IXX"), 0.13), (Pauli("XXI"), 0.87 * 0.08)]
        branches = [
            (r, Statevector(s.evolve(op).data * np.sqrt(p)))
            for r, s in branches
            for op, p in choices
        ]
    for q in [0, 1, 2, 0]:
        lines.append(f"M {q}")
        branches = [
            (r + (bit,), project(s, pauli(q, "Z"), bit)) for r, s in branches for bit in [0, 1]
        ]
    dist = np.zeros(16)
    for record, state in branches:
        dist[sum(b << i for i, b in enumerate(record))] += np.vdot(state.data, state.data).real
    path = ROOT / "toy12.stim"
    path.write_text("\n".join(lines) + "\n")
    return dict(case=12, path=str(path), probabilities=dist.tolist())


if __name__ == "__main__":
    cases = [make(i) for i in range(12)] + [correlated_case()]
    (ROOT / "exact_toy_expected.json").write_text(json.dumps(cases, indent=2) + "\n")
