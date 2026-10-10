"""Constructed large color-code checks, not a fault-tolerant cultivation schedule.

The retained geometry comes from the original cultivation repository. Encoding,
flag placement, gate noise, and repeated-check schedule here are research controls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import stim

DATA = Path(__file__).resolve().parents[2] / "research/conditional_clifford"


def binary_rank(rows: list[int]) -> int:
    basis: dict[int, int] = {}
    for word in rows:
        while word:
            pivot = word.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = word
                break
            word ^= basis[pivot]
    return len(basis)


def layout(distance: int) -> dict:
    code = json.loads((DATA / "color-code-layouts.json").read_text())["codes"][str(distance)]
    n = len(code["coordinates"])
    rows = [sum(1 << q for q in check) for check in code["checks"]]
    signs = [(-1) ** x for x, _ in code["coordinates"]]
    if n != (3 * distance * distance + 1) // 4 or binary_rank(rows) != (n - 1) // 2:
        raise ValueError("Unexpected color-code dimensions")
    if any((a & b).bit_count() % 2 for a in rows for b in rows):
        raise ValueError("The CSS checks are not self orthogonal")
    if any(sum(signs[q] for q in check) % 4 for check in code["checks"]):
        raise ValueError("The signed Clifford layer does not preserve the code")
    if sum(signs) != 1:
        raise ValueError("Unexpected logical magic-axis sign")
    return dict(code, n=n, rows=rows, signs=signs)


def encoder(code: dict) -> stim.Circuit:
    """Map independent computational seeds to CSS words with fixed logical X/Z."""
    n = code["n"]
    basis = [*code["rows"], (1 << n) - 1]
    for q in range(1, n):
        candidate = 1 | (1 << q)
        if binary_rank([*basis, candidate]) > len(basis):
            basis.append(candidate)
    if len(basis) != n:
        raise AssertionError("The CSS seeds do not form a complete basis")
    matrix = np.array([[(row >> q) & 1 for q in range(n)] for row in basis], dtype=np.uint8)
    inverse = np.eye(n, dtype=np.uint8)
    for q in range(n):
        pivot = next(i for i in range(q, n) if matrix[i, q])
        matrix[[q, pivot]] = matrix[[pivot, q]]
        inverse[[q, pivot]] = inverse[[pivot, q]]
        for i in range(n):
            if i != q and matrix[i, q]:
                matrix[i] ^= matrix[q]
                inverse[i] ^= inverse[q]
    xs = [
        stim.PauliString("".join("X" if row >> q & 1 else "_" for q in range(n))) for row in basis
    ]
    zs = [
        stim.PauliString("".join("Z" if inverse[q, i] else "_" for q in range(n))) for i in range(n)
    ]
    frame = stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)
    logical = len(code["rows"])
    if frame.x_output(logical) != stim.PauliString("X" * n) or frame.z_output(
        logical
    ) != stim.PauliString("Z" * n):
        raise AssertionError("Encoder changed the physical logical representatives")
    return frame.to_circuit()


def make_source(distance: int, probability: float, *, rounds: int = 2) -> tuple[str, dict]:
    if not 0 <= probability <= 1 or rounds < 1:
        raise ValueError("Invalid noise or round count")
    code = layout(distance)
    n = code["n"]
    data = [2 * q for q in range(n)]
    flags = [2 * q + 1 for q in range(n - 1)]
    root = 2 * n - 1
    lines: list[str] = []

    def gate(name: str, *qs: int, noisy: bool = True) -> None:
        lines.append(name + " " + " ".join(map(str, qs)))
        if noisy and probability:
            if name in {"R", "RX"}:
                lines.append(f"{'X' if name == 'R' else 'Z'}_ERROR({probability}) {qs[0]}")
            else:
                lines.append(f"DEPOLARIZE{len(qs)}({probability}) " + " ".join(map(str, qs)))

    def readout(name: str, q: int) -> None:
        lines.append(f"{name}({probability}) {q}")
        lines.append("DETECTOR rec[-1]")

    def phase_layer(inverse: bool = False, *, noisy: bool = True) -> None:
        for q, sign in enumerate(code["signs"]):
            gate("T" if sign * (-1 if inverse else 1) == 1 else "T_DAG", data[q], noisy=False)
        if noisy and probability:
            # Faults on distinct wires commute past the remaining T gates in the layer.
            for q in data:
                lines.append(f"DEPOLARIZE1({probability}) {q}")

    for q in range(root + 1):
        gate("R", q)
    for q in range(len(code["rows"]) + 1):
        gate("H", data[q])
    gate("T_DAG", data[len(code["rows"])])
    for op in encoder(code):
        targets = op.targets_copy()
        arity = 2 if op.name in {"CX", "CZ", "SWAP"} else 1
        for first in range(0, len(targets), arity):
            gate(op.name, *(data[t.value] for t in targets[first : first + arity]))

    for _ in range(rounds):
        gate("RX", root)
        phase_layer()
        for q in flags:
            gate("R", q)
        # Two taps per flag cancel in the pulled-back X parity while sensing faults.
        network = [(q, f) for a, b, f in zip(data, data[1:], flags) for q in (a, b)]
        network += [(root, q) for q in data]
        for a, b in network:
            gate("CX", a, b)
        readout("MX", root)
        gate("RX", root)
        for a, b in reversed(network):
            gate("CX", a, b)
        for q in flags:
            readout("M", q)
        phase_layer(inverse=True)
        # Noisy physical syndrome extraction tests continuation out of the check.
        for axis in "XZ":
            for check in code["checks"]:
                gate("RX" if axis == "X" else "R", root)
                for q in check:
                    gate("CX", *((root, data[q]) if axis == "X" else (data[q], root)))
                readout("MX" if axis == "X" else "M", root)

    phase_layer(noisy=False)
    lines.append("MPP " + "*".join(f"X{q}" for q in data))
    phase_layer(inverse=True, noisy=False)
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    return "\n".join(lines) + "\n", dict(
        distance=distance,
        data_qubits=n,
        physical_qubits=root + 1,
        flags=len(flags),
        check_rounds=rounds,
        physical_noise=probability,
        scope=(
            "constructed encoded preparation, flagged checks, syndrome extraction and readout; "
            "no growth or fault-distance claim"
        ),
    )
