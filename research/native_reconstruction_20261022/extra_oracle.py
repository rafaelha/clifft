"""Independent Qiskit source interpreter for bounded adversarial joint laws.

This file does not import the reconstruction parser or transformation code.
"""

import itertools
import json
import re
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

ROOT = Path(__file__).parent
CASES = {
    "noncommuting_rotations": "H 0\nT 0\nH 0\nT 0\nH 0\nM 0\n",
    "scalar_rotation": "T 0\nM 0\n",
    "reset_entangled": "H 0\nCX 0 1\nT 0\nR 0\nMX 0 1\n",
    "mr_feedback": "H 0\nT 0\nCX 0 1\nMR(0.13) !0\nCY rec[-1] 1\nT_DAG 1\nMY 0 1\n",
    "reset_xy": "H 0\nT 0\nCX 0 1\nRY 0\nRX 1\nT_DAG 0\nT 1\nMPP !Y0*X1 X0*Y1\n",
    "mpp_readout": "H 0 1\nT 0\nT_DAG 1\nMPP(0.09) !X0*Y1\nCZ rec[-1] 0\nT 0\nMX 0 1\n",
    "pauli_channel": "H 0 1\nT 0 1\nPAULI_CHANNEL_1(0.07,0.11,0.04) 0\nCX 0 1\nT_DAG 1\nMX 0 1\n",
    "two_qubit_category": (
        "H 0\nT 0\nCX 0 1\n"
        "PAULI_CHANNEL_2(0.01,0.02,0.03,0.04,0.05,0.06,"
        "0.01,0.02,0.03,0.04,0.01,0.02,0.03,0.01,0.02) 0 1\n"
        "T 1\nMX 0\nMY 1\n"
    ),
    "constant_feedback": "MPAD(0.17) 1\nCX rec[-1] 0\nH 0\nT 0\nMY !0\n",
    "clifford_stim": (
        "H 0\nCX 0 1\nDEPOLARIZE2(0.17) 0 1\nMRX(0.12) 0\nCY rec[-1] 1\nMRY 1\nMPP !X0*Y1\n"
    ),
    "two_independent_magic": "H 0 1\nT 0 1\nMX 0 1\n",
    "nonzero_reference": "X 0\nM 0\nH 1\nT 1\nMX 1\n",
}


def exact(text):
    n = 3
    branches: list[tuple[tuple[int, ...], Statevector]] = [((), Statevector.from_int(0, 1 << n))]

    def pauli(term):
        letters = ["I"] * n
        for q, c in term:
            letters[n - 1 - q] = c
        return Pauli("".join(letters))

    def evolve(qc):
        nonlocal branches
        branches = [(r, s.evolve(qc)) for r, s in branches]

    def noise(choices):
        nonlocal branches
        branches = [
            (r, Statevector(s.evolve(pauli(term)).data * np.sqrt(p)))
            for r, s in branches
            for p, term in choices
            if p
        ]

    def measure(term, flip=0, prob=0, keep=True, reset=None):
        nonlocal branches
        obs = pauli(term)
        out = []
        for r, s in branches:
            for b in (0, 1):
                v = Statevector((s.data + (-1) ** b * s.evolve(obs).data) / 2)
                if reset and b:
                    v = v.evolve(pauli([(term[0][0], reset)]))
                for e, p in [(0, 1 - prob), (1, prob)]:
                    if p and np.vdot(v.data, v.data).real > 1e-24:
                        out.append(
                            (r + (b ^ flip ^ e,) if keep else r, Statevector(v.data * np.sqrt(p)))
                        )
        branches = out

    for line in text.splitlines():
        match = re.fullmatch(r"([A-Z_0-9]+)(?:\(([^)]*)\))?\s*(.*)", line)
        assert match is not None
        name, arg, ts = match.groups()
        args = list(map(float, arg.split(","))) if arg else []
        ts = ts.split()
        if name in ("H", "S", "T", "T_DAG", "X", "Y", "Z", "CX", "CY", "CZ"):
            if ts[0].startswith("rec"):
                offset = int(ts[0][4:-1])
                op = pauli([(int(ts[1]), name[-1])])
                branches = [(r, s.evolve(op) if r[offset] else s) for r, s in branches]
            else:
                qc = QuantumCircuit(n)
                arity = 2 if name.startswith("C") else 1
                for k in range(0, len(ts), arity):
                    getattr(qc, "tdg" if name == "T_DAG" else name.lower())(
                        *map(int, ts[k : k + arity])
                    )
                evolve(qc)
        elif name in ("R", "RX", "RY", "M", "MX", "MY", "MR", "MRX", "MRY"):
            axis = name[-1] if name[-1] in "XY" else "Z"
            for t in ts:
                q = int(t.lstrip("!"))
                measure(
                    [(q, axis)],
                    int(t.startswith("!")),
                    args[0] if args else 0,
                    keep=name.startswith("M"),
                    reset=("Z" if axis == "X" else "X") if "R" in name else None,
                )
        elif name == "MPP":
            for group in ts:
                term = []
                flip = 0
                for t in group.split("*"):
                    flip ^= int(t.startswith("!"))
                    t = t.lstrip("!")
                    term.append((int(t[1:]), t[0]))
                measure(term, flip, args[0] if args else 0)
        elif name == "MPAD":
            for t in ts:
                branches = [
                    (r + (int(t) ^ e,), Statevector(s.data * np.sqrt(p)))
                    for r, s in branches
                    for e, p in [(0, 1 - args[0]), (1, args[0])]
                ]
        elif name.startswith("PAULI_CHANNEL") or name == "DEPOLARIZE2":
            arity = 2 if name.endswith("2") else 1
            probs = (
                [1 - sum(args), *args]
                if name.startswith("PAULI")
                else [1 - args[0]] + [args[0] / 15] * 15
            )
            choices = [
                (p, [(int(q), c) for q, c in zip(ts, letters) if c != "I"])
                for p, letters in zip(probs, itertools.product("IXYZ", repeat=arity))
            ]
            noise(choices)
        else:
            raise ValueError(name)
    count = len(branches[0][0])
    result = np.zeros(1 << count)
    for r, s in branches:
        result[sum(b << i for i, b in enumerate(r))] += np.vdot(s.data, s.data).real
    assert abs(result.sum() - 1) < 1e-12
    return result.tolist()


if __name__ == "__main__":
    cases = [dict(name=name, text=text, probabilities=exact(text)) for name, text in CASES.items()]
    (ROOT / "extra_expected.json").write_text(json.dumps(cases, indent=2) + "\n")
