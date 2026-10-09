"""Offline conditional reduction of complete inverse-T parity instruments.

Candidates come from gate algebra, without code names or known code matrices.
Every shot retains its actual state and records; a refused boundary is explicit.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, replace

import stim
from automatic_specialization import ANNOTATIONS, FaultModel, History, record_count
from conditional_clifford_core import MonomialClifford, Refused, SmallState
from parity_envelope_audit import certify, layers, operations
from study_chan_boundary import branch_frames

import clifft


class NeedDecision(Exception):
    """Request both outcomes from the bounded exhaustive validation driver."""


@dataclass
class Segment:
    model: FaultModel
    certificate: dict | None
    first_line: int
    last_line: int
    site_offset: int


class ConditionalChecks:
    def __init__(self, source: str):
        parsed = clifft.parse(source)
        unary = {"R", "RX", "RY", "M", "MX", "MY", "MPAD", "T", "T_DAG"}
        unary_lines: set[int] = set()
        for node in parsed.nodes:
            if node.gate.name not in unary:
                continue
            if len(node.targets) != 1 or node.source_line in unary_lines:
                raise Refused(
                    "This research driver requires single-target reset, readout and T lines"
                )
            unary_lines.add(node.source_line)
        self.n = parsed.num_qubits
        self.num_records = parsed.num_measurements
        ops = operations(clifft, source)
        source_lines = source.splitlines()
        ts = layers(ops)
        regions: list[tuple[int, int, dict | None]] = []
        for (a, b), (c, d) in zip(ts, ts[1:]):
            if b - a < 2:
                continue
            forward, middle, reverse = ops[a:b], ops[b:c], ops[c:d]
            # The binder commutes faults through the complete outer T layers.
            # A fault part-way through either layer needs a different binding.
            if any(
                line and line.split()[0].split("(")[0] not in {"T", "T_DAG", *ANNOTATIONS}
                for layer in (forward, reverse)
                for line in source_lines[layer[0].line - 1 : layer[-1].line]
            ):
                continue
            try:
                certificate = certify(forward, middle, reverse)
            except ValueError:
                signs = {o.qubits[0]: 1 if o.name == "T" else -1 for o in forward}
                opposite = {o.qubits[0]: -1 if o.name == "T" else 1 for o in reverse}
                if not (
                    signs == opposite
                    and len(signs) == len(forward) == len(reverse)
                    and len(middle) == 1
                    and middle[0].name == "MPP"
                ):
                    continue
                certificate = dict(phase_signs=signs, terminal=True)
            if regions and ops[a].line <= regions[-1][1]:
                continue
            regions.append((ops[a].line, ops[d - 1].line, certificate))
        lines = source.splitlines()
        self.segments = []
        start, offset = 1, 0
        for first, last, selected in regions + [(len(lines) + 1, len(lines), None)]:
            for lo, hi, cert in ((start, first - 1, None), (first, last, selected)):
                if lo > hi:
                    continue
                prior = sum(record_count(node) for node in parsed.nodes if node.source_line < lo)
                model = FaultModel("MPAD 0\n" * prior + "\n".join(lines[lo - 1 : hi]) + "\n")
                model.lines = model.lines[prior:]
                model.sites = [replace(site, line=site.line - prior) for site in model.sites]
                model.num_records -= prior
                self.segments.append(Segment(model, cert, lo, hi, offset))
                offset += len(model.sites)
            start = last + 1
        self.fault_sites = offset

    def draw(self, rng: random.Random) -> History:
        return tuple(
            (i + segment.site_offset, outcome)
            for segment in self.segments
            for i, outcome in segment.model.draw(rng)
        )

    def materialize(self, history: History) -> list[str]:
        if len({i for i, _ in history}) != len(history) or any(
            not 0 <= i < self.fault_sites for i, _ in history
        ):
            raise ValueError("Invalid full-circuit fault history")
        return [
            segment.model.render(
                tuple(
                    (i - segment.site_offset, outcome)
                    for i, outcome in history
                    if segment.site_offset <= i < segment.site_offset + len(segment.model.sites)
                )
            )
            for segment in self.segments
        ]

    def trajectory(
        self,
        history: History,
        seed: int,
        *,
        forced_records: list[int] | None = None,
        decisions: tuple[int, ...] | None = None,
        checkpoint: Callable[[SmallState, list[int], Segment, str], None] | None = None,
    ) -> dict:
        if forced_records is not None and any(value not in (0, 1) for value in forced_records):
            raise ValueError("Forced records must contain bits")
        rng = random.Random(seed)
        state = SmallState(self.n)
        records: list[int] = []
        classical: list[str] = []
        reductions: list[dict] = []
        probability = 1.0
        hidden_random = 0
        product_resets = 0
        max_width = 0
        decision_index = 0

        def bit(p0: float, *, visible: bool, flip: int = 0) -> int:
            nonlocal probability, decision_index
            if forced_records is not None and visible:
                if len(records) >= len(forced_records):
                    raise ValueError("Too few forced records")
                result = forced_records[len(records)] ^ flip
            elif decisions is not None and 1e-10 < p0 < 1 - 1e-10:
                if decision_index == len(decisions):
                    raise NeedDecision()
                result = decisions[decision_index]
                if result not in (0, 1):
                    raise ValueError("An exhaustive decision must be a bit")
                decision_index += 1
            else:
                result = int(rng.random() >= p0)
            probability *= p0 if result == 0 else 1 - p0
            return result

        def record(value: int) -> None:
            records.append(value)
            classical.append(f"MPAD {value}")

        def observable(gate: str, targets: list[str]) -> stim.PauliString:
            p = stim.PauliString(self.n)
            if gate == "MPP":
                for term in targets[0].split("*"):
                    if term.startswith("!"):
                        p.sign *= -1
                        term = term[1:]
                    p[int(term[1:])] = term[0]
            else:
                p[int(targets[0].lstrip("!"))] = {"M": "Z", "MX": "X", "MY": "Y"}[gate]
            return p

        def reset(q: int, axis: str) -> None:
            nonlocal hidden_random, product_resets
            p = stim.PauliString(self.n)
            p[q] = axis
            p0 = (1 + state.expectation(p)) / 2
            random_reset = 1e-10 < p0 < 1 - 1e-10
            product_state = False
            for probe_axis in "XYZ":
                probe = stim.PauliString(self.n)
                probe[q] = probe_axis
                product_state |= abs(abs(state.expectation(probe)) - 1) < 1e-10
            if product_state:
                # A pure local stabilizer factor has no conditional effect on other wires.
                outcome = int(rng.random() >= p0)
                product_resets += int(random_reset)
            else:
                hidden_random += int(random_reset)
                outcome = bit(p0, visible=False)
            state.measure(p, outcome)
            if outcome:
                state.clifford(stim.Circuit(f"{'X' if axis == 'Z' else 'Z'} {q}"))

        for segment, source in zip(self.segments, self.materialize(history)):
            if segment.certificate is None:
                for line in source.splitlines():
                    gate, *targets = line.split()
                    bare = gate.split("(")[0]
                    if bare in ANNOTATIONS:
                        classical.append(line)
                    elif gate in {"R", "RX", "RY"}:
                        reset(int(targets[0]), {"R": "Z", "RX": "X", "RY": "Y"}[gate])
                    elif gate in {"M", "MX", "MY", "MPP"}:
                        flip = int(gate != "MPP" and targets[0].startswith("!"))
                        p = observable(gate, targets)
                        p0 = (1 + state.expectation(p)) / 2
                        outcome = bit(p0, visible=True, flip=flip)
                        if state.measure(p, outcome) < 1e-12:
                            return dict(probability=0.0, records=records, reductions=reductions)
                        record(outcome ^ flip)
                    elif gate == "MPAD":
                        value = int(targets[0])
                        if forced_records is not None and value != forced_records[len(records)]:
                            return dict(probability=0.0, records=records, reductions=reductions)
                        record(value)
                    elif gate in {"T", "T_DAG"}:
                        p = stim.PauliString(self.n)
                        p[int(targets[0])] = "Z"
                        state.rotate_t(p, 1 if gate == "T" else -1)
                    elif any(t.startswith("rec[") for t in targets):
                        if gate not in {"CX", "CY", "CZ"} or not targets[0].startswith("rec["):
                            raise Refused("Unsupported classical control")
                        if records[int(targets[0][4:-1])]:
                            state.clifford(stim.Circuit(f"{gate[1]} {targets[1]}"))
                    else:
                        state.clifford(stim.Circuit(line))
                    max_width = max(max_width, state.width)
                continue

            cert = segment.certificate
            signs = {int(q): sign for q, sign in cert["phase_signs"].items()}
            lines = source.splitlines()
            if cert.get("terminal"):
                # The terminal inverse-T sandwich has no internal faults or other work.
                middle = [line for line in lines if not line.startswith(("T ", "T_DAG "))]
                if len(middle) != 1 or not middle[0].startswith("MPP "):
                    raise Refused("The terminal sandwich has unsupported internal operations")
                p = observable("MPP", middle[0].split()[1:])
                operator = MonomialClifford(p, signs)
                p0, _ = state.clifford_projector(operator, None)
                outcome = bit(p0, visible=True)
                chance, metadata = state.clifford_projector(operator, outcome)
                if chance < 1e-12:
                    return dict(probability=0.0, records=records, reductions=reductions)
                record(outcome)
            else:
                root = cert["central_qubit"]
                flags = {int(q): axis for q, axis in cert["flags"].items()}
                for q, axis in flags.items():
                    reset(q, axis)
                quantum_source = (
                    "\n".join(
                        line for line in lines if line.split()[0].split("(")[0] not in ANNOTATIONS
                    )
                    + "\n"
                )
                a, b = branch_frames(quantum_source, root, flags, 0)
                a += stim.PauliString(self.n - len(a))
                b += stim.PauliString(self.n - len(b))
                inverse = a.copy()
                inverse.sign = inverse.sign.conjugate()
                relative = inverse * b
                operator = MonomialClifford(relative, signs)
                root_line = next(line for line in lines if line.startswith("MX "))
                if any(line.startswith("M ") for line in lines[: lines.index(root_line)]):
                    raise Refused("A flag is read before the central outcome")
                flip = int(root_line.split()[1].startswith("!"))
                p0, _ = state.clifford_projector(operator, None)
                outcome = bit(p0, visible=True, flip=flip)
                chance, metadata = state.clifford_projector(operator, outcome)
                if chance < 1e-12:
                    return dict(probability=0.0, records=records, reductions=reductions)
                a, _ = branch_frames(quantum_source, root, flags, outcome)
                a += stim.PauliString(self.n - len(a))
                state.frame = state.frame.then(MonomialClifford(a, signs).tableau())
                for index, line in enumerate(lines):
                    gate, *targets = line.split()
                    bare = gate.split("(")[0]
                    if gate == "MX":
                        record(outcome ^ int(targets[0].startswith("!")))
                    elif gate == "M":
                        q = int(targets[0].lstrip("!"))
                        p = stim.PauliString(self.n)
                        p[q] = "Z"
                        expectation = state.expectation(p)
                        if abs(abs(expectation) - 1) > 1e-8:
                            raise Refused("A closing flag distinguishes coherent paths")
                        later = (
                            sum(suffix in {f"X {q}", f"Y {q}"} for suffix in lines[index + 1 :]) % 2
                        )
                        flag_value = int(expectation < 0) ^ later ^ int(targets[0].startswith("!"))
                        if (
                            forced_records is not None
                            and flag_value != forced_records[len(records)]
                        ):
                            return dict(probability=0.0, records=records, reductions=reductions)
                        record(flag_value)
                    elif bare in ANNOTATIONS:
                        classical.append(line)
            metadata.update(first_line=segment.first_line, last_line=segment.last_line)
            reductions.append(metadata)
            max_width = max(max_width, state.width)
            if checkpoint is not None:
                checkpoint(state, records, segment, source)
        if len(records) != self.num_records:
            raise AssertionError("The complete record contract changed")
        if forced_records is not None and len(forced_records) != len(records):
            raise ValueError("Unused forced records")
        if decisions is not None and len(decisions) != decision_index:
            raise ValueError("Unused exhaustive decisions")
        return dict(
            probability=probability,
            records=records,
            reductions=reductions,
            hidden_random=hidden_random,
            product_resets=product_resets,
            peak_carry_width=max_width,
            final_source=state.source() + "\n" + "\n".join(classical) + "\n",
        )
