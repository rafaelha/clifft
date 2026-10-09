"""Check physical shared and separate Steane extraction after a reduced CCZ region.

Ideal encoded preparation and fixed Pauli faults isolate the continuation
interface. Actual ancilla records are retained; diagnostic data probes and
fault-aware corrections are validation tools, not decoder information.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import stim
from automatic_specialization import FaultModel, History, analyze
from study_quadcycle_factory import load_css
from validate_automatic_specialization import visible_probability
from validate_phase_continuation import compare
from validate_theory_ccz_faults import Workload, fourier_coefficients, propagated_error

import clifft


def mask(values: Any) -> int:
    return sum(int(bit) << q for q, bit in enumerate(values))


def bits(value: int, count: int) -> list[int]:
    return [(value >> q) & 1 for q in range(count)]


def parity(a: int, b: int) -> int:
    return (a & b).bit_count() % 2


class Extraction:
    def __init__(self, workload: Workload, protocol: str):
        self.workload = workload
        self.copies = {"D": 3, "E": 1}[protocol]
        self.n = 81 + 27 * self.copies
        self.m = self.n - 81
        self.generator = [mask(row) for row in workload.tri.generator_matrix]
        self.z_rows = [mask(row) for row in workload.z_rows]
        self.preparation = []
        for copy in range(self.copies):
            offset = 81 + 27 * copy
            self.preparation += [f"H {q}" for q in workload.tri.plus_qubits(offset)]
            self.preparation += [f"CX {a} {b}" for _, a, b in workload.tri.encoder_ops(offset)]
        self.gates = [
            (27 * block + q, 81 + 27 * (block if protocol == "D" else 0) + q)
            for block in range(3)
            for q in range(27)
        ]
        source = self.preparation + [f"DEPOLARIZE1(.01) {q}" for q in range(self.n)]
        for a, b in self.gates:
            source += [f"CX {a} {b}", f"DEPOLARIZE2(.01) {a} {b}"]
        source += [f"M(.01) {q}" for q in range(81, self.n)]
        self.model = FaultModel("\n".join(source) + "\n")
        expected = ["DEPOLARIZE1"] * self.n + ["DEPOLARIZE2"] * 81 + ["READOUT_NOISE"] * self.m
        if [site.gate for site in self.model.sites] != expected:
            raise AssertionError("Extraction fault labels differ from parsed locations")
        self.paulis = list(itertools.product(range(4), repeat=2))

    def certify_ideal(self) -> dict[str, Any]:
        # Bell partners make this an arbitrary-logical-input check. The
        # unchanged joint state also certifies ancilla decoupling before M.
        lines = []
        for block in range(3):
            for logical in range(3):
                q = 27 * block + logical
                lines += [f"H {q}", f"CX {q} {self.n + 3 * block + logical}"]
            lines += [f"H {27 * block + q}" for q in range(3, 11)]
            lines += [f"CX {a} {b}" for _, a, b in self.workload.tri.encoder_ops(27 * block)]
        simulator = stim.TableauSimulator()
        simulator.do(stim.Circuit("\n".join(lines + self.preparation)))
        before = simulator.canonical_stabilizers()
        simulator.do(stim.Circuit("\n".join(f"CX {a} {b}" for a, b in self.gates)))
        if before != simulator.canonical_stabilizers():
            raise AssertionError(
                "Ideal extraction changes the encoded data or entangles its ancilla"
            )
        return {
            "encoded_bell_pairs": 9,
            "ancilla_blocks": self.copies,
            "exact_joint_stabilizers": len(before),
        }

    def reference(self, region: History, history: History) -> dict[str, Any]:
        events = [self.workload.events[site][outcome] for site, outcome in region]
        x, terms = propagated_error(self.workload.gates, [e for e in events if e is not None])
        frame = stim.PauliString(self.n)
        for q in range(81):
            frame[q] = (x >> q) & 1
        faults = dict(history)
        for q in range(self.n):
            frame[q] ^= faults.get(q, 0)
        # Original diagonal data errors commute with every extraction CNOT.
        # Only the Pauli frame needs Clifford propagation through this tail.
        for index, (a, b) in enumerate(self.gates):
            frame = frame.after(stim.Circuit(f"CX {a} {b}"))
            pauli = self.paulis[faults.get(self.n + index, 0)]
            frame[a] ^= pauli[0]
            frame[b] ^= pauli[1]
        for q in range(81):
            if frame[q] in (2, 3):
                terms.symmetric_difference_update([(q,)])
        x = sum((frame[q] in (1, 2)) << q for q in range(81))
        coefficients, denominator = fourier_coefficients(self.workload.columns, terms)
        law: dict[int, Fraction] = defaultdict(Fraction)
        for label, amplitude in coefficients.items():
            syndrome = mask((label >> q) & 1 for q in self.workload.syndrome_bits)
            law[syndrome] += Fraction(amplitude * amplitude, denominator * denominator)
        offsets = [
            mask(
                (frame[81 + 27 * copy + q] in (1, 2))
                ^ bool(faults.get(self.n + 81 + 27 * copy + q, 0))
                for q in range(27)
            )
            for copy in range(self.copies)
        ]
        anc_syndrome = [parity(row, offset) for offset in offsets for row in self.z_rows]
        data_z = [parity(row, x >> (27 * block)) for block in range(3) for row in self.z_rows]
        return {
            "x": x,
            "coefficients": coefficients,
            "denominator": denominator,
            "law": dict(law),
            "offsets": offsets,
            "anc_syndrome": anc_syndrome,
            "data_z": data_z,
            "acceptance": Fraction() if any(anc_syndrome) else law.get(0, Fraction()),
        }

    def coarse_source(self, history: History) -> str:
        lines = self.model.render(history).splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith("M "))
        measurements = lines[first:]
        if len(measurements) != self.m:
            raise AssertionError("Unexpected extraction record count")
        output = lines[:first]
        for copy in range(self.copies):
            flips = mask("!" in line for line in measurements[27 * copy : 27 * (copy + 1)])
            for row in self.z_rows:
                product = "*".join(f"Z{81 + 27 * copy + q}" for q in range(27) if (row >> q) & 1)
                output.append("MPP " + ("!" if parity(row, flips) else "") + product)
        return "\n".join(output) + "\n"

    def record_probability(self, record: list[int], reference: dict[str, Any]) -> Fraction:
        for copy, offset in enumerate(reference["offsets"]):
            word = mask(record[27 * copy : 27 * (copy + 1)]) ^ offset
            if any(parity(row, word) for row in self.z_rows):
                return Fraction()
        if record[self.m + 24 : self.m + 72] != reference["data_z"]:
            return Fraction()
        law: dict[int, Fraction] = reference["law"]
        return law.get(mask(record[self.m : self.m + 24]), Fraction()) / (1 << (11 * self.copies))

    def draw_record(self, reference: dict[str, Any], rng: random.Random) -> list[int]:
        record = []
        for offset in reference["offsets"]:
            word = offset
            for row in self.generator:
                if rng.getrandbits(1):
                    word ^= row
            record += bits(word, 27)
        law = reference["law"]
        syndrome = rng.choices(list(law), weights=[float(v) for v in law.values()])[0]
        data_z: list[int] = reference["data_z"]
        return record + bits(syndrome, 24) + data_z

    def score_source(self, x: int, hadamard: bool) -> str:
        lines = self.workload.score_tail(x).splitlines()[:-81]
        if hadamard:
            lines += [f"H {27 * block + q}" for block in range(3) for q in range(3)]
        lines += [f"M {q}" for q in range(81)]
        return "\n".join(lines) + "\n"

    def score_weights(self, syndrome: int, reference: dict[str, Any], hadamard: bool) -> list[int]:
        vector = [0] * 512
        for label, value in reference["coefficients"].items():
            if mask((label >> q) & 1 for q in self.workload.syndrome_bits) == syndrome:
                logical = sum(((label >> (11 * block)) & 7) << (3 * block) for block in range(3))
                vector[logical] = value
        if hadamard:
            for q in range(9):
                for i in range(512):
                    if not (i >> q) & 1:
                        a, b = vector[i], vector[i ^ (1 << q)]
                        vector[i], vector[i ^ (1 << q)] = a + b, a - b
        return [value * value for value in vector]

    def score_probability(
        self, record: list[int], reference: dict[str, Any], hadamard: bool
    ) -> Fraction:
        initial = record[: self.m + 72]
        if not self.record_probability(initial, reference):
            return Fraction()
        decoded = record[self.m + 72 :]
        if len(decoded) != 81 or any(decoded[27 * b + q] for b in range(3) for q in range(11, 27)):
            return Fraction()
        label = sum(mask(decoded[27 * b : 27 * b + 11]) << (11 * b) for b in range(3))
        syndrome = mask(initial[self.m : self.m + 24])
        if mask((label >> q) & 1 for q in self.workload.syndrome_bits) != syndrome:
            return Fraction()
        logical = sum(((label >> (11 * b)) & 7) << (3 * b) for b in range(3))
        numerator = self.score_weights(syndrome, reference, hadamard)[logical]
        denominator = reference["denominator"] ** 2 * (1 << (11 * self.copies + 9 * hadamard))
        return Fraction(numerator, denominator)

    def draw_score(
        self, reference: dict[str, Any], rng: random.Random, hadamard: bool
    ) -> list[int]:
        record = self.draw_record(reference, rng)
        syndrome = mask(record[self.m : self.m + 24])
        weights = self.score_weights(syndrome, reference, hadamard)
        logical = rng.choices(range(512), weights=weights)[0]
        decoded = [0] * 81
        for b in range(3):
            decoded[27 * b : 27 * b + 3] = bits(logical >> (3 * b), 3)
            decoded[27 * b + 3 : 27 * b + 11] = bits(syndrome >> (8 * b), 8)
        return record + decoded


def small_instruments() -> list[dict[str, Any]]:
    rows = []
    region = "H 0 2 4\nCX 0 1 2 3 4 5\nCCZ 0 2 4\nE(.01) X0 Y2 Z4\n"
    for fault in ("", "Z 6\n", "Y 0\nZ 6\n"):
        tail = "H 6\nCX 6 7\nCX 0 6 1 7\n" + fault + "CX 2 6 3 7 4 6 5 7\nM 6 7\nH 0\nT 0\n"
        rows.append(compare(region, tail, [(), ((0, 1),)]))
    region = "H 0 1 2\nCCZ 0 1 2\nE(.01) X0 Y1 Z2\n"
    for fault in ("", "Y 3\n", "X 0\nZ 4\n"):
        tail = "H 3 4 5\nCX 0 3\n" + fault + "CX 1 4 2 5\nM 3 4 5\nH 1\nT 1\n"
        rows.append(compare(region, tail, [(), ((0, 1),)]))
    return rows


def compile_bounded(source: str) -> tuple[Any, int]:
    hir, info = analyze(source)
    if info["peak_width"] > 16:
        raise ValueError("Extraction validation exceeds the active-width budget")
    return clifft.lower(hir), int(info["peak_width"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-limit", type=int)
    args = parser.parse_args()
    if args.tail_limit is not None and args.tail_limit < 0:
        parser.error("Tail limit must be nonnegative")
    instruments = small_instruments()
    print("Six independent small instruments passed", flush=True)
    workload = Workload(load_css(args.merlin_checkout))
    rows = []
    for protocol in ("D", "E"):
        extraction = Extraction(workload, protocol)
        certificate = extraction.certify_ideal()
        singles = [
            ((site, outcome),)
            for site in range(81, len(extraction.model.sites))
            for outcome in range(1, len(extraction.model.sites[site].replacements))
        ]
        cases: list[tuple[str, History, History, bool]] = [("ideal", (), (), True)]
        selected = singles if args.tail_limit is None else singles[: args.tail_limit]
        cases += [("extraction_single", (), history, False) for history in selected]
        cases += [
            ("matching_data_x", (), ((0, 1), (27, 1)), True),
            ("ancilla_z_before", (), ((81, 3),), True),
            ("ancilla_z_after_first_coupling", (), ((extraction.n, 3),), True),
            ("readout_flip", (), ((extraction.n + 81, 1),), True),
        ]
        region_rng = random.Random(3605)
        tail_rng = random.Random(3606)
        for count in (1, 2, 3):
            for _ in range(6):
                region = tuple(
                    (site, region_rng.randrange(1, len(workload.events[site])))
                    for site in sorted(region_rng.sample(range(135), count))
                )
                tail = tuple(
                    (site, tail_rng.randrange(1, len(extraction.model.sites[site].replacements)))
                    for site in sorted(tail_rng.sample(range(len(extraction.model.sites)), count))
                )
                cases.append(("mixed_faults", region, tail, True))
        rng = random.Random(3610)
        maximum = relative = 0.0
        probability_checks = state_checks = coarse_checks = raw_checks = 0
        max_width = 0
        witnesses = []
        digest = hashlib.sha256()

        def check(program: Any, record: list[int], expected: Fraction) -> None:
            nonlocal maximum, relative, probability_checks
            actual = visible_probability(program, record)
            error = abs(actual - float(expected))
            maximum = max(maximum, error)
            if expected:
                relative = max(relative, error / float(expected))
            if not math.isclose(actual, float(expected), rel_tol=1e-9, abs_tol=1e-300):
                raise AssertionError(
                    f"Physical extraction probability differs: {actual} versus {expected}"
                )
            probability_checks += 1

        data_probes = "\n".join(workload.checks + workload.z_checks) + "\n"
        for index, (name, region, tail, score) in enumerate(cases):
            reference = extraction.reference(region, tail)
            state = workload.shared.render_state(workload.shared.controls(region))
            raw_source = state + extraction.model.render(tail) + data_probes
            raw, width = compile_bounded(raw_source)
            coarse, other_width = compile_bounded(
                state + extraction.coarse_source(tail) + data_probes
            )
            max_width = max(max_width, width, other_width)
            for syndrome, expected in reference["law"].items():
                record = reference["anc_syndrome"] + bits(syndrome, 24) + reference["data_z"]
                check(coarse, record, expected)
                coarse_checks += 1
            sampled = (
                clifft.sample(raw, shots=2, seed=7301, threads=1).measurements.astype(int).tolist()
            )
            records = sampled + [extraction.draw_record(reference, rng) for _ in range(2)]
            invalid = records[0].copy()
            invalid[0] ^= 1
            records.append(invalid)
            for record in records:
                check(raw, record, extraction.record_probability(record, reference))
                raw_checks += 1
            if score:
                for hadamard in (False, True):
                    scored, width = compile_bounded(
                        raw_source + extraction.score_source(reference["x"], hadamard)
                    )
                    max_width = max(max_width, width)
                    sample = (
                        clifft.sample(scored, shots=2, seed=7302, threads=1)
                        .measurements.astype(int)
                        .tolist()
                    )
                    sample += [extraction.draw_score(reference, rng, hadamard) for _ in range(2)]
                    for record in sample:
                        check(
                            scored,
                            record,
                            extraction.score_probability(record, reference, hadamard),
                        )
                        state_checks += 1
            if name not in ("extraction_single", "mixed_faults"):
                witnesses.append(
                    {
                        "name": name,
                        "acceptance": str(reference["acceptance"]),
                        "ancilla_syndrome": reference["anc_syndrome"],
                        "data_x_syndrome": {str(k): str(v) for k, v in reference["law"].items()},
                    }
                )
            digest.update(
                json.dumps(
                    [
                        region,
                        tail,
                        reference["anc_syndrome"],
                        reference["data_z"],
                        sorted((k, str(v)) for k, v in reference["law"].items()),
                    ]
                ).encode()
            )
            if index % 100 == 0:
                print(f"{protocol}: checked {index + 1}/{len(cases)} fault histories", flush=True)
        expected_acceptance = {
            "ideal": "1",
            "matching_data_x": "0" if protocol == "D" else "1",
            "ancilla_z_before": "0",
            "ancilla_z_after_first_coupling": "1" if protocol == "D" else "0",
            "readout_flip": "0",
        }
        if {w["name"]: w["acceptance"] for w in witnesses} != expected_acceptance:
            raise AssertionError("Extraction cancellation/backpropagation witnesses differ")
        rows.append(
            {
                "protocol": protocol,
                "physical_qubits": extraction.n,
                "physical_ancilla_records": extraction.m,
                "ideal_channel_certificate": certificate,
                "fault_histories": len(cases),
                "single_tail_faults": len(selected),
                "record_probabilities_checked": probability_checks,
                "coarse_syndrome_probabilities_checked": coarse_checks,
                "physical_record_probabilities_checked": raw_checks,
                "state_probe_probabilities_checked": state_checks,
                "maximum_absolute_probability_error": maximum,
                "maximum_relative_probability_error": relative,
                "maximum_active_width": max_width,
                "witnesses": witnesses,
                "reference_laws_sha256": digest.hexdigest(),
            }
        )
    result = {
        "cases": rows,
        "small_quantum_instruments": instruments,
        "scope": (
            "Ideal preparation, fixed faults; complete coarse syndrome laws and selected "
            "physical-record/state-probe probabilities. No decoder or noisy protocol rates."
        ),
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("validate_theory_ccz_faults.py"),
                Path(__file__).with_name("study_quadcycle_factory.py"),
                Path(__file__).with_name("shared_phase_specialization.py"),
                Path(__file__).with_name("automatic_specialization.py"),
                Path(__file__).with_name("validate_phase_continuation.py"),
            )
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
