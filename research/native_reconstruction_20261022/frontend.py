"""ZX-independent reconstruction via a Clifford dilation and Pauli constraints.

The source is parsed directly. Stim is used only for compile-time Clifford
arithmetic; no Tsim/ZX graph, parser, schedule, or extracted map is consumed.
"""

import re
from dataclasses import dataclass

import numpy as np
import stim


def bits(value):
    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low


def mask(array):
    return int.from_bytes(np.packbits(array, bitorder="little").tobytes(), "little")


def unpack(value, n):
    return np.array([bool(value >> i & 1) for i in range(n)])


class Basis:
    def __init__(self):
        self.rows = {}

    def reduce(self, value, label=0):
        while value:
            pivot = value.bit_length() - 1
            if pivot not in self.rows:
                break
            row, combination = self.rows[pivot]
            value ^= row
            label ^= combination
        return value, label

    def insert(self, value, label):
        rest, combination = self.reduce(value, label)
        if rest:
            self.rows[rest.bit_length() - 1] = (rest, combination)
            return None
        return combination

    def solve(self, value):
        rest, label = self.reduce(value)
        return None if rest else label


def pauli(x, z, n, sign=0):
    p = stim.PauliString.from_numpy(xs=unpack(x, n), zs=unpack(z, n))
    return p * (-1 if sign else 1)


def source_instructions(text):
    if len(text) > 5_000_000:
        raise ValueError("research source budget exceeded")
    if re.search(r"(?m)^\s*HERALDED_", text):
        raise ValueError("unsupported heralded instrument")
    if re.search(r"(?m)^\s*(EXP_VAL|R_PAULI|LEAK|LOSS|QUBIT_LOSS|SPP)(?:\W|_)", text):
        raise ValueError("unsupported source operation or physical-state observation")
    # Stim expands the Clifford syntax; these two reserved marker forms carry
    # non-Clifford rotations through its repeat parser without changing operands.
    text = re.sub(r"(?m)^(\s*)T_DAG(?=\s)", r"\1HERALDED_PAULI_CHANNEL_1(0.125,0,0,0)", text)
    text = re.sub(r"(?m)^(\s*)T(?=\s)", r"\1HERALDED_PAULI_CHANNEL_1(0.25,0,0,0)", text)
    circuit = stim.Circuit(text)

    def expanded_count(circuit):
        total = 0
        for instruction in circuit:
            total += (
                instruction.repeat_count * expanded_count(instruction.body_copy())
                if isinstance(instruction, stim.CircuitRepeatBlock)
                else 1
            )
            if total > 50000:
                raise ValueError("research instruction budget exceeded")
        return total

    expanded_count(circuit)
    for instruction in circuit.flattened():
        if instruction.name == "HERALDED_PAULI_CHANNEL_1":
            args = instruction.gate_args_copy()
            if args not in ([0.125, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0]):
                raise ValueError("unsupported heralded instrument")
            yield ("T_DAG" if args[0] == 0.125 else "T"), [], instruction.targets_copy()
        else:
            yield instruction.name, instruction.gate_args_copy(), instruction.targets_copy()


@dataclass
class Operation:
    x: int
    z: int
    sign: int
    angle: float = 0.0
    slot: int = -1
    hidden: bool = False


class Dilation:
    def __init__(self):
        self.body = stim.Circuit()
        self.n = 0
        self.alias = {}
        self.fx = []
        self.fz = []
        self.rotations = []
        self.records = []
        self.detectors = []
        self.observables = {}
        self.channels = []
        self.error_bits = 0
        self.error_chain = None

    def fresh(self):
        if self.n >= 2048:
            raise ValueError("research dilation budget exceeded")
        q = self.n
        self.n += 1
        self.fx.append(0)
        self.fz.append(0)
        return q

    def qubit(self, source):
        if source not in self.alias:
            self.alias[source] = self.fresh()
        return self.alias[source]

    def error(self, probabilities, count):
        start = self.error_bits
        self.error_bits += count
        if self.error_bits > 65536:
            raise ValueError("research noise budget exceeded")
        self.channels.append([(float(p), int(v) << start) for v, p in probabilities if p])
        return [1 << (start + i + 1) for i in range(count)]

    def bit_error(self, probability):
        if not probability:
            return 0
        return self.error([(0, 1 - probability), (1, probability)], 1)[0]

    def apply_error(self, q, axis, bit):
        if axis in "XY":
            self.fx[q] ^= bit
        if axis in "ZY":
            self.fz[q] ^= bit

    def noise_sign(self, term):
        result = 0
        for q, axis in term:
            if axis in "ZY":
                result ^= self.fx[q]
            if axis in "XY":
                result ^= self.fz[q]
        return result

    def controlled(self, control, term):
        for q, axis in term:
            self.body.append("C" + axis, [control, q])

    def rotation(self, term, angle):
        a = self.fresh()
        self.rotations.append((a, angle, self.noise_sign(term)))
        self.body.append("H", [a])
        self.controlled(a, term)
        self.body.append("H", [a])

    def measure(self, term, flip=0, probability=0):
        a = self.fresh()
        self.body.append("H", [a])
        self.controlled(a, term)
        self.body.append("H", [a])
        self.records.append((a, self.noise_sign(term) ^ flip ^ self.bit_error(probability)))

    def clifford(self, name, targets):
        gate = stim.Tableau.from_named_gate(name)
        arity = len(gate)
        assert len(targets) % arity == 0
        for k in range(0, len(targets), arity):
            ts = targets[k : k + arity]
            if any(t.is_measurement_record_target for t in ts):
                if (
                    arity != 2
                    or not ts[0].is_measurement_record_target
                    or name not in ("CX", "CY", "CZ")
                ):
                    raise ValueError("unsupported feedback")
                r, offset = self.records[ts[0].value]
                q = self.qubit(ts[1].value)
                self.body.append(name, [r, q])
                self.apply_error(q, name[-1], offset)
                continue
            if any(not t.is_qubit_target for t in ts):
                raise ValueError("unsupported control")
            qs = [self.qubit(t.value) for t in ts]
            self.body.append(name, qs)
            old_x = [self.fx[q] for q in qs]
            old_z = [self.fz[q] for q in qs]
            new_x = [0] * arity
            new_z = [0] * arity
            for j in range(arity):
                for image, old in [(gate.x_output(j), old_x[j]), (gate.z_output(j), old_z[j])]:
                    xs, zs = image.to_numpy()
                    for i in range(arity):
                        if xs[i]:
                            new_x[i] ^= old
                        if zs[i]:
                            new_z[i] ^= old
            for q, x, z in zip(qs, new_x, new_z):
                self.fx[q], self.fz[q] = x, z

    def record_parity(self, targets):
        mask_ = 0
        offset = 0
        for t in targets:
            if not t.is_measurement_record_target:
                raise ValueError("non-record output")
            q, f = self.records[t.value]
            mask_ ^= 1 << q
            offset ^= f
        return mask_, offset

    @classmethod
    def parse(cls, text):
        self = cls()
        for name, args, targets in source_instructions(text):
            if name not in ("E", "ELSE_CORRELATED_ERROR"):
                self.error_chain = None
            if name in ("QUBIT_COORDS", "SHIFT_COORDS", "TICK"):
                continue
            if name == "DETECTOR":
                if len(self.detectors) >= 4096:
                    raise ValueError("research output budget exceeded")
                self.detectors.append(self.record_parity(targets))
                continue
            if name == "OBSERVABLE_INCLUDE":
                index = int(args[0])
                if index >= 4096:
                    raise ValueError("research output budget exceeded")
                x, f = self.record_parity(targets)
                a, b = self.observables.get(index, (0, 0))
                self.observables[index] = (a ^ x, b ^ f)
                continue
            if name in ("T", "T_DAG"):
                for t in targets:
                    self.rotation(
                        [(self.qubit(t.value), "Z")], np.pi / 8 * (1 if name == "T" else -1)
                    )
                continue
            if name in ("R", "RX", "RY", "MR", "MRX", "MRY", "M", "MX", "MY"):
                axis = "Z" if name in ("R", "MR", "M") else name[-1]
                for t in targets:
                    q = self.qubit(t.value)
                    if name.startswith("M"):
                        self.measure(
                            [(q, axis)], int(t.is_inverted_result_target), args[0] if args else 0
                        )
                    if "R" in name:
                        q = self.fresh()
                        self.alias[t.value] = q
                        if axis in "XY":
                            self.body.append("H", [q])
                        if axis == "Y":
                            self.body.append("S", [q])
                continue
            if name == "MPP":
                term = []
                flip = 0
                for i, t in enumerate(targets):
                    if t.is_combiner:
                        continue
                    axis = "X" if t.is_x_target else "Y" if t.is_y_target else "Z"
                    term.append((self.qubit(t.value), axis))
                    flip ^= int(t.is_inverted_result_target)
                    if i + 1 == len(targets) or not targets[i + 1].is_combiner:
                        self.measure(term, flip, args[0] if args else 0)
                        term = []
                        flip = 0
                continue
            if name == "MPAD":
                for t in targets:
                    self.records.append(
                        (self.fresh(), t.value ^ self.bit_error(args[0] if args else 0))
                    )
                continue
            if name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
                for t in targets:
                    self.apply_error(self.qubit(t.value), name[0], self.bit_error(args[0]))
                continue
            if name in ("DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2"):
                arity = 2 if name.endswith("2") else 1
                for start in range(0, len(targets), arity):
                    count = 1 << (2 * arity)
                    if name.startswith("DEPOLARIZE"):
                        probabilities = [1 - args[0]] + [args[0] / (count - 1)] * (count - 1)
                        # Binary labels are X and Z bits, independent of source spelling.
                        encoded = list(enumerate(probabilities))
                    else:
                        probabilities = [1 - sum(args), *args]
                        # Stim orders Paulis as I,X,Y,Z, with the last qubit varying fastest.
                        encoded = []
                        for value, p in enumerate(probabilities):
                            code = 0
                            for j in range(arity):
                                digit = (value >> (2 * (arity - 1 - j))) & 3
                                code |= ([0, 1, 3, 2][digit]) << (2 * j)
                            encoded.append((code, p))
                    fs = self.error(encoded, 2 * arity)
                    for j, t in enumerate(targets[start : start + arity]):
                        q = self.qubit(t.value)
                        self.fx[q] ^= fs[2 * j]
                        self.fz[q] ^= fs[2 * j + 1]
                continue
            if name in ("E", "ELSE_CORRELATED_ERROR"):
                if name == "E":
                    self.channels.append([(1.0, 0)])
                    self.error_chain = len(self.channels) - 1
                if self.error_chain is None:
                    raise ValueError("unpaired ELSE_CORRELATED_ERROR")
                channel = self.channels[self.error_chain]
                remaining = channel[0][0]
                p = remaining * args[0]
                channel[0] = (remaining - p, 0)
                bit = self.error_bits
                self.error_bits += 1
                if self.error_bits > 65536:
                    raise ValueError("research noise budget exceeded")
                channel.append((p, 1 << bit))
                for t in targets:
                    axis = "X" if t.is_x_target else "Y" if t.is_y_target else "Z"
                    self.apply_error(self.qubit(t.value), axis, 1 << (bit + 1))
                continue
            self.clifford(name, targets)
        return self

    def reconstruct(self, mode="detectors"):
        if mode not in ("detectors", "records"):
            raise ValueError("unsupported output contract")
        n = self.n
        if not n:
            outputs = (
                self.detectors
                + [
                    self.observables.get(i, (0, 0))
                    for i in range(max(self.observables, default=-1) + 1)
                ]
                if mode == "detectors"
                else self.records
            )
            return 0, [Operation(0, 0, f, slot=i) for i, (_, f) in enumerate(outputs)]
        self.body.append("I", [n - 1])
        backward = self.body.to_tableau().inverse()
        magic = {a for a, _, _ in self.rotations}
        initial = [pauli(1 << i, 0, n) if i in magic else pauli(0, 1 << i, n) for i in range(n)]
        constraints = [backward.z_output(a) for a, _, _ in self.rotations]
        patterns = [
            sum(int(not g.commutes(p)) << j for j, p in enumerate(constraints)) for g in initial
        ]
        # Place rotations backwards, using only stabilizers of as-yet-unrotated inputs.
        allowed = Basis()
        for i in range(n):
            if i not in magic:
                allowed.insert(patterns[i], 1 << i)
        rotations = []
        for a, angle, f in self.rotations[::-1]:
            axis = pauli(0, 1 << a, n)
            pattern = sum(int(not axis.commutes(p)) << j for j, p in enumerate(constraints))
            combination = allowed.solve(pattern)
            if combination is None:
                raise ValueError("no causal stabilizer dressing")
            for i in bits(combination):
                axis *= initial[i]
            assert all(axis.commutes(p) for p in constraints)
            rotations.append((axis, angle, f, -1))
            allowed.insert(patterns[a], 1 << a)
        rotations.reverse()
        independent = Basis()
        kernel = []
        for i, pattern in enumerate(patterns):
            dependent = independent.insert(pattern, 1 << i)
            if dependent is not None:
                kernel.append(dependent)
        assert len(independent.rows) == len(constraints), "nonuniform gadget postselection"
        generators = constraints[:]
        for combination in kernel:
            p = stim.PauliString(n)
            for i in bits(combination):
                p *= initial[i]
            generators.append(p)
        assert len(generators) == n
        frame = stim.Tableau.from_stabilizers(generators).inverse()
        outputs = (
            [(1 << q, f) for q, f in self.records]
            if mode == "records"
            else self.detectors
            + [
                self.observables.get(i, (0, 0))
                for i in range(max(self.observables, default=-1) + 1)
            ]
        )
        operations = rotations + [
            (backward(pauli(0, m, n)), 0.0, f, i) for i, (m, f) in enumerate(outputs)
        ]
        images = [frame(p) for p, _, _, _ in operations]
        # A linear coordinate change packs the X span. All remaining Z-only lines
        # are fixed at zero and can be removed without changing any output law.
        directions: list = []
        xb = Basis()
        for p in images:
            x = mask(p.to_numpy()[0])
            if xb.solve(x) is None:
                xb.insert(x, 1 << len(directions))
                directions.append(x)
        width = len(directions)
        for i in range(n):
            if xb.solve(1 << i) is None:
                xb.insert(1 << i, 1 << len(directions))
                directions.append(1 << i)
        inverse_columns = [xb.solve(1 << i) for i in range(n)]
        inverse_rows = [
            sum(((v >> i) & 1) << j for j, v in enumerate(inverse_columns)) for i in range(n)
        ]
        linear = stim.Tableau.from_conjugated_generators(
            xs=[pauli(x, 0, n) for x in directions], zs=[pauli(0, z, n) for z in inverse_rows]
        ).inverse()
        result = []
        for image, (_, angle, f, slot) in zip(images, operations):
            p = linear(image)
            x, z = map(mask, p.to_numpy())
            assert not x >> width
            result.append(Operation(x, z & ((1 << width) - 1), f ^ int(p.sign == -1), angle, slot))
        return width, result
