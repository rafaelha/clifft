"""Offline structural audit of inverse-T parity instruments.

This certifies a local algebraic form, not a production sampler or an efficient
whole-circuit contraction. Noise is omitted only for the Clifford topology
identity; fixed Pauli faults change frames, signs and flag records.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import stim


@dataclass(frozen=True)
class Op:
    name: str
    qubits: tuple[int, ...]
    line: int


def operations(clifft: Any, text: str) -> list[Op]:
    result = []
    for node in clifft.parse(text).nodes:
        name = node.gate.name
        if name in {"DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "QUBIT_COORDS", "SHIFT_COORDS"}:
            continue
        if name in {
            "X_ERROR",
            "Y_ERROR",
            "Z_ERROR",
            "DEPOLARIZE1",
            "DEPOLARIZE2",
            "DEPOLARIZE3",
            "READOUT_NOISE",
        }:
            continue
        if any(t.is_rec for t in node.targets):
            result.append(Op("FEEDBACK", (), node.source_line))
        elif name in {"MPP", "R_PAULI"}:
            result.append(Op(name, (), node.source_line))
        else:
            size = 2 if name in {"CX", "CZ"} else 3 if name == "CCZ" else 1
            result.extend(
                Op(name, tuple(t.value for t in node.targets[j : j + size]), node.source_line)
                for j in range(0, len(node.targets), size)
            )
    return result


def layers(ops: list[Op]) -> list[tuple[int, int]]:
    result = []
    i = 0
    while i < len(ops):
        if ops[i].name not in {"T", "T_DAG"}:
            i += 1
            continue
        end = i + 1
        while end < len(ops) and ops[end].name in {"T", "T_DAG"}:
            end += 1
        result.append((i, end))
        i = end
    return result


def axes(pauli: Any) -> dict[int, str]:
    return {q: "IXYZ"[a] for q, a in enumerate(pauli) if a}


def certify(forward: list[Op], middle: list[Op], reverse: list[Op]) -> dict:
    signs = {o.qubits[0]: 1 if o.name == "T" else -1 for o in forward}
    opposite = {o.qubits[0]: -1 if o.name == "T" else 1 for o in reverse}
    if len(signs) != len(forward) or len(opposite) != len(reverse) or signs != opposite:
        raise ValueError("phase layers are not distinct inverses")
    roots = [
        (i, j, op.qubits[0])
        for i, op in enumerate(middle)
        if op.name == "MX"
        for j, reset in enumerate(middle[i + 1 :], i + 1)
        if reset.name == "RX" and op.qubits == reset.qubits
    ]
    if len(roots) != 1:
        raise ValueError("expected exactly one central X measurement followed by its reset")
    measured, reset, root = roots[0]
    if reset != measured + 1:
        raise ValueError("non-noise operation between central measurement and reset")
    flags = {}
    for i, op in enumerate(middle):
        if i == reset or op.name not in {"R", "RX"}:
            continue
        q = op.qubits[0]
        if q in signs or q == root or q in flags:
            raise ValueError("flag reset overlaps phase support or is repeated")
        if any(q in before.qubits for before in middle[:i]):
            raise ValueError("flag preparation is not its first use")
        flags[q] = "X" if op.name == "RX" else "Z"
    closed = set()
    width = 1 + max(q for op in forward + middle + reverse for q in op.qubits)
    before, after = stim.Circuit(), stim.Circuit()
    before.append("I", [width - 1])
    after.append("I", [width - 1])
    for i, op in enumerate(middle):
        if i in {measured, reset}:
            continue
        if op.name == "CX":
            (before if i < measured else after).append("CX", op.qubits)
        elif op.name in {"R", "RX"} and op.qubits[0] in flags:
            continue
        elif op.name in {"M", "MX"} and op.qubits[0] in flags:
            q = op.qubits[0]
            if flags[q] != ("X" if op.name == "MX" else "Z"):
                raise ValueError("flag preparation and readout bases differ")
            if q in closed or any(q in later.qubits for later in middle[i + 1 :]):
                raise ValueError("flag readout is not its last use")
            closed.add(q)
        else:
            raise ValueError(f"unsupported middle operation {op.name} at line {op.line}")
    if closed != set(flags):
        raise ValueError("unmeasured flag remains")
    inverse = before.to_tableau().inverse()
    if inverse != after.to_tableau():
        raise ValueError("Clifford networks are not inverses")
    x, z = stim.PauliString(width), stim.PauliString(width)
    x[root], z[root] = "X", "Z"
    parity, correction = inverse(x), inverse(z)
    if any(parity[q] not in {0, 1 if axis == "X" else 3} for q, axis in flags.items()):
        raise ValueError("coherent parity branch changes a flag readout")
    if any(parity[q] != 1 for q in signs):
        raise ValueError("pulled-back parity is not all X on the phase support")
    return dict(
        phase_qubits=sorted(signs),
        middle_qubits=sorted({q for op in middle for q in op.qubits}),
        phase_signs=signs,
        central_qubit=root,
        flags=flags,
        cnot_count=len(before) + len(after) - 2,
        clifford_networks_are_inverses=True,
        flags_have_branch_independent_readouts=True,
        parity=axes(parity),
        reset_correction=axes(correction),
        coherent_terms_per_conditioned_record=2,
        scope="local fixed-fault inverse-T instrument; no entry-code or contraction-cost proof",
    )


def audit(clifft: Any, text: str) -> dict:
    ops = operations(clifft, text)
    phases = layers(ops)
    results = []
    for (a, b), (c, d) in zip(phases, phases[1:]):
        if b - a < 2:
            continue
        middle = ops[b:c]
        row = dict(
            first_line=ops[a].line,
            last_line=ops[d - 1].line,
            phase_widths=[b - a, d - c],
            middle_gate_counts=dict(Counter(op.name for op in middle)),
        )
        try:
            row["certificate"] = certify(ops[a:b], middle, ops[c:d])
            row["accepted"] = True
        except ValueError as error:
            row.update(accepted=False, reason=str(error))
        results.append(row)
    return dict(
        phase_layer_count=len(phases),
        candidate_envelopes=results,
        accepted_envelopes=sum(row["accepted"] for row in results),
    )
