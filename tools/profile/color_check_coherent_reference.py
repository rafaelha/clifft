"""Physical Clifford-path reference without bounded-core state compression.

CH-form projection and overlap follow the earlier research reference at
b87c0da396b4bbde54fbee5535aeac1116170605. This implementation constructs each
whole physical Clifford path separately instead of using the new fault binder.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import cirq  # type: ignore[import-not-found]
import stim
from automatic_specialization import ANNOTATIONS, History
from chan_algebraic_continuation import ConditionalChecks


def ch_gate(state: Any, name: str, targets: list[int]) -> None:
    if name in {"S", "S_DAG"}:
        state.apply_z(targets[0], exponent=0.5 if name == "S" else -0.5)
    elif name != "I":
        getattr(state, "apply_" + name.lower())(*targets)


class Term:
    def __init__(self, n: int):
        self.coefficient = 1.0 + 0j
        self.ch = cirq.StabilizerStateChForm(n)
        self.tableau = stim.TableauSimulator()
        self.tableau.set_num_qubits(n)

    def copy(self) -> Term:
        other = object.__new__(Term)
        other.coefficient, other.ch, other.tableau = (
            self.coefficient,
            self.ch.copy(),
            self.tableau.copy(),
        )
        return other

    def gate(self, name: str, targets: list[int]) -> None:
        ch_gate(self.ch, name, targets)
        self.tableau.do(stim.CircuitInstruction(name, targets))

    def conjugated_pauli(self, p: stim.PauliString, signs: dict[int, int]) -> None:
        self.coefficient *= p.sign
        for q, axis in enumerate(p):
            if axis:
                self.gate("IXYZ"[axis], [q])
            if axis in (1, 2) and q in signs:
                self.gate("S_DAG" if signs[q] == 1 else "S", [q])
                self.coefficient *= cmath.exp(1j * signs[q] * math.pi / 4)

    def project(self, p: stim.PauliString, bit: int) -> bool:
        expectation = self.tableau.peek_observable_expectation(p)
        if expectation:
            return bool(expectation == 1 - 2 * bit)
        support = [q for q, axis in enumerate(p) if axis]
        pivot = support[0]
        gates = []
        for q in support:
            if p[q] == 2:
                gates.append(("S_DAG", [q]))
            if p[q] in (1, 2):
                gates.append(("H", [q]))
        gates += [("CX", [q, pivot]) for q in support[1:]]
        for name, targets in gates:
            self.gate(name, targets)
        self.ch.project_Z(pivot, bit)
        self.tableau.postselect_z(pivot, desired_value=bool(bit))
        self.coefficient *= 2**-0.5
        for name, targets in reversed(gates):
            self.gate("S" if name == "S_DAG" else name, targets)
        return True


def overlap(left: Term, right: Term) -> complex:
    if any(
        right.tableau.peek_observable_expectation(p) == -1
        for p in left.tableau.canonical_stabilizers()
    ):
        return 0j
    inverse = left.tableau.current_inverse_tableau().to_circuit()
    a, b = left.ch.copy(), right.ch.copy()
    for op in inverse:
        size = 2 if op.name in {"CX", "CZ", "SWAP"} else 1
        targets = [t.value for t in op.targets_copy()]
        for i in range(0, len(targets), size):
            ch_gate(a, op.name, targets[i : i + size])
            ch_gate(b, op.name, targets[i : i + size])
    anchor = a.inner_product_of_state_and_x(0)
    if abs(abs(anchor) - 1) > 1e-8:
        raise AssertionError("Independent CH and tableau states disagree")
    return complex(anchor.conjugate() * b.inner_product_of_state_and_x(0))


class CoherentState:
    def __init__(self, n: int):
        self.n = n
        self.terms = [Term(n)]
        self.peak_terms = 1

    def norm(self) -> float:
        diagonal = sum(abs(t.coefficient) ** 2 for t in self.terms)
        value = diagonal + sum(
            2 * (a.coefficient.conjugate() * b.coefficient * overlap(a, b)).real
            for i, a in enumerate(self.terms)
            for b in self.terms[i + 1 :]
        )
        if value < -1e-10 * diagonal:
            raise AssertionError("Negative coherent norm")
        return float(value) if value > 1e-12 * diagonal else 0.0

    def normalize(self) -> float:
        chance = self.norm()
        if chance:
            for t in self.terms:
                t.coefficient /= math.sqrt(chance)
        return chance

    def gate(self, name: str, targets: list[int]) -> None:
        if name in {"T", "T_DAG"}:
            phase = cmath.exp((1 if name == "T" else -1) * 1j * math.pi / 4)
            expanded = []
            for t in self.terms:
                other = t.copy()
                other.gate("Z", targets)
                other.coefficient *= (1 - phase) / 2
                t.coefficient *= (1 + phase) / 2
                expanded.extend([t, other])
            self.terms = expanded
            self.peak_terms = max(self.peak_terms, len(expanded))
        else:
            for t in self.terms:
                t.gate(name, targets)

    def project(self, p: stim.PauliString, bit: int) -> float:
        bit ^= int(p.sign == -1)
        p = p.copy()
        p.sign = 1
        if self.terms and all(
            t.tableau.peek_observable_expectation(p) == 1 - 2 * bit for t in self.terms
        ):
            return 1.0
        self.terms = [t for t in self.terms if t.project(p, bit)]
        return self.normalize()

    def expectation(self, p: stim.PauliString) -> float:
        saved = self.terms
        self.terms = [t.copy() for t in saved]
        chance = self.project(p, 0)
        self.terms = saved
        return 2 * chance - 1

    def reset(self, q: int, axis: str) -> None:
        # A shared definite local Pauli proves that discarding this wire is harmless.
        pure = False
        for a in "XYZ":
            p = stim.PauliString(self.n)
            p[q] = a
            values = [t.tableau.peek_observable_expectation(p) for t in self.terms]
            pure |= bool(values and values[0] and all(v == values[0] for v in values))
        if not pure:
            raise ValueError("Reference needs a product stabilizer reset")
        p = stim.PauliString(self.n)
        p[q] = axis
        saved = [t.copy() for t in self.terms]
        if not self.project(p, 0):
            self.terms = saved
            if not self.project(p, 1):
                raise AssertionError("Reset has no outcome")
            self.gate("X" if axis == "Z" else "Z", [q])


def path_pauli(lines: list[str], n: int, root: int, outcome: int, path: int) -> stim.PauliString:
    """Evaluate a full physical Clifford path and recover its exact Pauli phase."""
    circuit = stim.Circuit(f"I {n - 1}")
    reset_bit, between = outcome, False
    for line in lines:
        name, *targets = line.split()
        if name.split("(")[0] in ANNOTATIONS or name in {"T", "T_DAG", "R", "M"}:
            continue
        qs = [int(t.lstrip("!")) for t in targets]
        if name == "MX" and qs == [root]:
            if path:
                circuit.append("X", [root])
            between = True
        elif name == "RX" and qs == [root]:
            if reset_bit:
                circuit.append("Z", [root])
            between = False
        elif name in {"CX", "X", "Y", "Z"}:
            circuit.append(name, qs)
            if between and qs == [root] and name in {"Y", "Z"}:
                reset_bit ^= 1
        else:
            raise ValueError(f"Unsupported reference path operation {line}")
    action = circuit.to_tableau()
    p = stim.PauliString(n)
    for q in range(n):
        x, z = action.x_output(q), action.z_output(q)
        if x.weight != 1 or x[q] != 1 or z.weight != 1 or z[q] != 3:
            raise ValueError("The complete physical path is not Pauli")
        p[q] = (0, 1, 3, 2)[int(z.sign == -1) + 2 * int(x.sign == -1)]
    ch = cirq.StabilizerStateChForm(n)
    for op in circuit:
        qs = [t.value for t in op.targets_copy()]
        size = 2 if op.name == "CX" else 1
        for i in range(0, len(qs), size):
            ch_gate(ch, op.name, qs[i : i + size])
    index = sum(1 << (n - 1 - q) for q, a in enumerate(p) if a in (1, 2))
    phase = ch.inner_product_of_state_and_x(index) / 1j ** sum(a == 2 for a in p)
    p.sign = min((1, -1, 1j, -1j), key=lambda z: abs(phase - z))
    if abs(p.sign - phase) > 1e-8:
        raise AssertionError("Physical Clifford path has an invalid global phase")
    return p


def reference(
    front: ConditionalChecks,
    history: History,
    records: list[int],
    probes: list[stim.PauliString] | None = None,
) -> dict:
    state = CoherentState(front.n)
    log_probability = 0.0
    index = 0

    def measured(p: stim.PauliString, bit: int) -> bool:
        nonlocal log_probability
        chance = state.project(p, bit)
        if not chance:
            return False
        log_probability += math.log(chance)
        return True

    for segment, source in zip(front.segments, front.materialize(history)):
        lines = source.splitlines()
        cert = segment.certificate
        if cert is None:
            for line in lines:
                name, *targets = line.split()
                if name.split("(")[0] in ANNOTATIONS:
                    continue
                qs = [int(t.lstrip("!")) for t in targets]
                if name in {"R", "RX"}:
                    state.reset(qs[0], "X" if name == "RX" else "Z")
                elif name in {"M", "MX"}:
                    p = stim.PauliString(front.n)
                    p[qs[0]] = "X" if name == "MX" else "Z"
                    if not measured(p, records[index] ^ int(targets[0].startswith("!"))):
                        return dict(log_probability=-math.inf)
                    index += 1
                else:
                    state.gate(name, qs)
            continue
        signs = {int(q): sign for q, sign in cert["phase_signs"].items()}
        if cert.get("terminal"):
            line = next(line for line in lines if line.startswith("MPP "))
            p = stim.PauliString(front.n)
            for term in line.split()[1].split("*"):
                p[int(term[1:])] = term[0]
            paths = [(1, stim.PauliString(front.n)), ((-1) ** records[index], p)]
        else:
            for q, axis in cert["flags"].items():
                state.reset(int(q), axis)
            root = cert["central_qubit"]
            line = next(line for line in lines if line.startswith("MX "))
            outcome = records[index] ^ int(line.split()[1].startswith("!"))
            paths = [
                ((-1) ** (outcome * path), path_pauli(lines, front.n, root, outcome, path))
                for path in (0, 1)
            ]
        expanded = []
        for sign, p in paths:
            for old in state.terms:
                t = old.copy()
                t.coefficient *= sign / 2
                t.conjugated_pauli(p, signs)
                expanded.append(t)
        state.terms = expanded
        state.peak_terms = max(state.peak_terms, len(expanded))
        chance = state.normalize()
        if not chance:
            return dict(log_probability=-math.inf)
        log_probability += math.log(chance)
        index += 1
        if not cert.get("terminal"):
            for i, line in enumerate(lines):
                if line.startswith("M "):
                    target = line.split()[1]
                    q = int(target.lstrip("!"))
                    later = sum(s in {f"X {q}", f"Y {q}"} for s in lines[i + 1 :]) % 2
                    p = stim.PauliString(front.n)
                    p[q] = "Z"
                    if not measured(p, records[index] ^ int(target.startswith("!")) ^ later):
                        return dict(log_probability=-math.inf)
                    index += 1
    if index != len(records):
        raise AssertionError("Reference lost a visible record")
    return dict(
        log_probability=log_probability,
        peak_terms=state.peak_terms,
        probes=[state.expectation(p) for p in probes or []],
    )
