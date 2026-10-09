"""Bounded structural study of the coupled tricycle-to-quadcycle escape circuit.

The code matrices, fan-out, and physical CCZ tensor follow the papers. The
syndrome-extraction schedule and linear preparation recovery are diagnostic
choices, not a reproduction of the authors' optimized decoder or noise model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import stim
from automatic_specialization import analyze
from profile_external_feedback import MERLIN_REVISION
from shared_phase_specialization import SharedPhase
from validate_shared_phase_specialization import exact_law

import clifft


def pauli_channel_3(targets: tuple[int, int, int], probability: float) -> list[str]:
    if not 0 <= probability <= 1 or len(set(targets)) != 3:
        raise ValueError("Invalid three-qubit depolarizing channel")
    lines = []
    for index, product in enumerate(list(itertools.product("IXYZ", repeat=3))[1:]):
        conditional = probability / (63 - index * probability)
        gate = "E" if index == 0 else "ELSE_CORRELATED_ERROR"
        pauli = " ".join(f"{axis}{q}" for axis, q in zip(product, targets) if axis != "I")
        lines.append(f"{gate}({conditional:.17g}) {pauli}")
    return lines


def load_css(checkout: Path) -> Any:
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "diff", "HEAD", "--", "benchmarks/protocols/css.py"], text=True
    )
    if revision != MERLIN_REVISION or dirty:
        raise ValueError("Expected the pinned, unchanged CSS helpers")
    sys.path.insert(0, str(checkout.resolve()))
    return importlib.import_module("benchmarks.protocols.css")


def code_pair(css: Any) -> tuple[Any, Any, list[tuple[int, ...]], dict[str, Any]]:
    def polynomial(terms: list[tuple[int, int]]) -> np.ndarray:
        matrix = np.zeros((9, 9), dtype=np.int64)
        for x, y in itertools.product(range(3), repeat=2):
            for dx, dy in terms:
                matrix[3 * ((x + dx) % 3) + (y + dy) % 3, 3 * x + y] ^= 1
        return matrix

    a, b, c, d = [
        polynomial(terms)
        for terms in ([(0, 0), (1, 0)], [(0, 0), (0, 1)], [(1, 0), (0, 1)], [(0, 0), (1, 1)])
    ]
    zero = np.zeros_like(a)
    tri_x = np.hstack([a.T, b.T, c.T])
    tri_z = np.block([[b, a, zero], [c, zero, a], [zero, c, b]])
    quad_x = np.block(
        [
            [c.T, zero, b.T, d.T, zero, zero],
            [zero, c.T, a.T, zero, d.T, zero],
            [a.T, b.T, zero, zero, zero, d.T],
            [zero, zero, zero, a.T, b.T, c.T],
        ]
    )
    quad_z = np.block(
        [
            [b, a, c, zero, zero, zero],
            [d, zero, zero, c, zero, a],
            [zero, d, zero, zero, c, b],
            [zero, zero, d, b, a, zero],
        ]
    )
    tri = css.CSSCode.from_checks("tricycle27", tri_x, tri_z)
    embedded = np.hstack([np.zeros_like(tri.x_logicals), tri.x_logicals])
    quad = css.CSSCode.from_checks("quadcycle54", quad_x, quad_z, preferred_x_logicals=embedded)
    if (tri.n, tri.k, quad.n, quad.k) != (27, 3, 54, 6):
        raise AssertionError("Paper code dimensions differ")
    for row in tri.hx:
        if not css.f2_in_row_span(np.r_[np.zeros(27, dtype=int), row], quad.hx):
            raise AssertionError("Escape does not preserve an X stabilizer")
    for row in quad.hz:
        if not css.f2_in_row_span(row[27:], tri.hz):
            raise AssertionError("Escape does not preserve a Z stabilizer")
    logical_map = embedded @ quad.logical_zs.T % 2
    np.testing.assert_array_equal(
        logical_map, np.c_[np.eye(3, dtype=int), np.zeros((3, 3), dtype=int)]
    )
    incoming, outgoing = [(0, 0), (0, 0), (1, 0)], [(1, 0), (0, 1), (0, 1)]
    triples = []
    for i, j, k in itertools.permutations(range(3)):
        for g in itertools.product(range(3), repeat=2):
            shifts = [
                tuple((g[t] - outgoing[j][t] - outgoing[k][t]) % 3 for t in range(2)),
                tuple((g[t] - incoming[i][t] - outgoing[k][t]) % 3 for t in range(2)),
                tuple((g[t] - incoming[i][t] - incoming[j][t]) % 3 for t in range(2)),
            ]
            triples.append(
                tuple(9 * sector + 3 * x + y for sector, (x, y) in zip((i, j, k), shifts))
            )
    generator = tri.generator_matrix
    tensor = np.zeros((len(generator),) * 3, dtype=np.int64)
    for u, v, w in triples:
        tensor ^= np.einsum("i,j,k->ijk", generator[:, u], generator[:, v], generator[:, w])
    residual = tensor.copy()
    residual[:3, :3, :3] = 0
    if residual.any():
        raise AssertionError("CCZ does not preserve the tricycle code space")
    # Color the disjoint triples into two physical layers before fan-out.
    first = {(0, 1, 2), (1, 2, 0), (2, 0, 1)}
    layered = [
        tuple(q + (0 if tuple(t // 9 for t in triple) in first else 27) for q in triple)
        for triple in triples
    ]
    for leg in range(3):
        if sorted(t[leg] for t in layered) != list(range(54)):
            raise AssertionError("Fan-out CCZ layer reuses a physical qubit")
    certificate = {
        "tricycle": [tri.n, tri.k],
        "quadcycle": [quad.n, quad.k],
        "check_ranks": [
            css.f2_rank(tri.hx),
            css.f2_rank(tri.hz),
            css.f2_rank(quad.hx),
            css.f2_rank(quad.hz),
        ],
        "escape_logical_rank": css.f2_rank(logical_map),
        "physical_ccz_gates": len(layered),
        "logical_ccz_tensor": tensor[:3, :3, :3].tolist(),
        "ccz_check_dependence": int(residual.sum()),
        "matrix_sha256": hashlib.sha256(
            b"".join(m.tobytes() for m in (tri_x, tri_z, quad_x, quad_z))
        ).hexdigest(),
    }
    return tri, quad, layered, certificate


class Builder:
    def __init__(self, p: float):
        self.p = p
        self.lines: list[str] = []
        self.records = 0

    def reset(self, qubits: list[int]) -> None:
        self.lines.append("R " + " ".join(map(str, qubits)))
        if self.p:
            self.lines.append(f"X_ERROR({self.p}) " + " ".join(map(str, qubits)))

    def cnot(self, a: int, b: int) -> None:
        self.lines.append(f"CX {a} {b}")
        if self.p:
            self.lines.append(f"DEPOLARIZE2({self.p}) {a} {b}")

    def prepare(self, css: Any, checks: np.ndarray, axis: str, data: int, ancilla: int) -> None:
        ancillas = list(range(ancilla, ancilla + len(checks)))
        self.reset(ancillas)
        if axis == "X":
            self.lines.append("H " + " ".join(map(str, ancillas)))
            if self.p:
                self.lines.append(f"DEPOLARIZE1({self.p}) " + " ".join(map(str, ancillas)))
        for row, check in enumerate(checks):
            for q in np.flatnonzero(check):
                a, b = (
                    (data + int(q), ancilla + row)
                    if axis == "Z"
                    else (ancilla + row, data + int(q))
                )
                self.cnot(a, b)
        first_record = self.records
        for ancilla_qubit in ancillas:
            self.lines.append(f"{'M' if axis == 'Z' else 'MX'}({self.p}) {ancilla_qubit}")
            self.records += 1
        basis = css.f2_basis(checks)
        indices = [next(i for i, row in enumerate(checks) if np.array_equal(row, v)) for v in basis]
        for i, record in enumerate(indices):
            correction = css.f2_solve(basis, np.eye(len(basis), dtype=int)[i])
            for q in np.flatnonzero(correction):
                self.lines.append(
                    f"{'CX' if axis == 'Z' else 'CZ'} "
                    f"rec[{first_record + record - self.records}] {data + int(q)}"
                )
        self.reset(ancillas)


def factory(css: Any, p: float) -> tuple[str, str, dict[str, Any]]:
    tri, quad, triples, certificate = code_pair(css)
    builder = Builder(p)
    for block in range(3):
        start = 108 * block
        builder.reset(list(range(start, start + 27)))
        builder.lines.append("H " + " ".join(map(str, range(start, start + 27))))
        if p:
            builder.lines.append(
                f"DEPOLARIZE1({p}) " + " ".join(map(str, range(start, start + 27)))
            )
        builder.prepare(css, tri.hz, "Z", start, start + 27)
        for q in range(27):
            builder.cnot(start + q, start + 27 + q)
        builder.reset(list(range(start + 54, start + 108)))
        builder.prepare(css, quad.hx, "X", start + 54, 324 + block * 36)
    for triple in triples:
        ccz_targets = (triple[0], 108 + triple[1], 216 + triple[2])
        builder.lines.append("CCZ " + " ".join(map(str, ccz_targets)))
        if p:
            builder.lines += pauli_channel_3(ccz_targets, 2 * p)
    region = "\n".join(builder.lines) + "\n"
    prefix_records = builder.records
    builder.lines = []
    for block in range(3):
        for q in range(27):
            builder.cnot(108 * block + q, 108 * block + 81 + q)
    for block in range(3):
        start = 108 * block
        first = builder.records
        for q in range(54):
            builder.lines.append(f"MX({p}) {start + q}")
            builder.records += 1
        for check in tri.hx:
            support = [int(q) for q in np.flatnonzero(check)]
            targets = [
                f"rec[{first + q + copy * 27 - builder.records}]"
                for copy in range(2)
                for q in support
            ]
            builder.lines.append("DETECTOR " + " ".join(targets))
        for logical, x in enumerate(tri.x_logicals):
            for copy in range(2):
                for measured_qubit in np.flatnonzero(x):
                    for target in np.flatnonzero(quad.logical_zs[logical]):
                        builder.lines.append(
                            f"CZ rec[{first + copy * 27 + int(measured_qubit) - builder.records}] "
                            f"{start + 54 + int(target)}"
                        )
    # Diagnostic terminal readout after the physical escape and actual feedback.
    for block in range(3):
        for _, a, b in quad.decoder_ops(108 * block + 54):
            builder.lines.append(f"CX {a} {b}")
        for logical in range(6):
            builder.lines.append(f"MX {108 * block + 54 + logical}")
    certificate.update(
        {
            "physical_qubits": 432,
            "prefix_records": prefix_records,
            "escape_records": 162,
            "logical_readouts": 18,
            "scope": (
                "Structural escape circuit; diagnostic preparation recovery and schedule, "
                "no idle noise or postselection"
            ),
        }
    )
    return region, "\n".join(builder.lines) + "\n", certificate


def certify_escape(css: Any) -> dict[str, Any]:
    """Check the ideal quantum channel by teleporting three encoded Bell halves."""
    tri, quad, _, _ = code_pair(css)
    lines = ["H 0 1 2", "CX 0 108 1 109 2 110"]
    lines += [f"H {q}" for q in range(tri.k, tri.n_encoder_rows)]
    lines += [f"CX {a} {b}" for _, a, b in tri.encoder_ops()]
    lines += [f"CX {q} {q + 27}" for q in range(27)]
    lines += [f"H {q + 54}" for q in range(quad.k, quad.n_encoder_rows)]
    lines += [f"CX {a} {b}" for _, a, b in quad.encoder_ops(54)]
    lines += [f"CX {q} {q + 81}" for q in range(27)]
    lines += [f"MX {q}" for q in range(54)]
    for logical, x in enumerate(tri.x_logicals):
        for copy in range(2):
            for q in np.flatnonzero(x):
                for target in np.flatnonzero(quad.logical_zs[logical]):
                    lines.append(f"CZ rec[{copy * 27 + int(q) - 54}] {54 + int(target)}")
    probes = []
    for logical in range(3):
        for axis, matrix in (("X", quad.x_logicals), ("Z", quad.logical_zs)):
            probes.append(
                f"{axis}{108 + logical}*"
                + "*".join(f"{axis}{54 + int(q)}" for q in np.flatnonzero(matrix[logical]))
            )
    for axis, checks in (("X", quad.hx), ("Z", quad.hz), ("Z", quad.logical_zs[3:])):
        probes += ["*".join(f"{axis}{54 + int(q)}" for q in np.flatnonzero(row)) for row in checks]
    circuit = stim.Circuit("\n".join(lines + ["MPP " + product for product in probes]))
    law = exact_law(circuit, list(range(circuit.num_measurements)), circuit.num_measurements)
    if any((1 << i) not in law for i in range(54, circuit.num_measurements)):
        raise AssertionError("Escape failed a complete encoded Bell-pair channel check")
    return {
        "reference_bell_pairs": 3,
        "deterministic_stabilizer_probes": len(probes),
        "all_measurement_branches": True,
        "law_sha256": hashlib.sha256(str(law).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=12)
    args = parser.parse_args()
    if not 0 <= args.max_width <= 16:
        parser.error("Execution width must be between zero and sixteen")
    css = load_css(args.merlin_checkout)
    escape_certificate = certify_escape(css)
    rows = []
    for p in (0.0, 0.001):
        region, tail, certificate = factory(css, p)
        _, ordinary = analyze(region + tail)
        start = perf_counter()
        shared = SharedPhase(region, preserve_state=True)
        setup = perf_counter() - start
        rng = random.Random(261010731)
        results = []
        histories = [()] + [shared.model.draw(rng) for _ in range(4 if p else 0)]
        if p:
            ccz_sites = [
                i for i, site in enumerate(shared.model.sites) if site.gate == "CORRELATED_ERROR"
            ]
            histories += [((ccz_sites[0], outcome),) for outcome in (1, 21, 63)]
            histories += [tuple((site, 63) for site in ccz_sites[:5])]
        for history in histories:
            text = shared.render_state(shared.controls(history)) + tail
            raw = clifft.trace(clifft.parse(text))
            optimized, info = analyze(text)
            row = {
                "faults": len(history),
                "region_history": history,
                "raw_t": raw.num_t_gates,
                "raw_width": clifft.active_width_trace(raw)["peak"],
                "optimized": info,
            }
            if info["peak_width"] <= args.max_width:
                program = clifft.lower(optimized)
                samples = clifft.sample(program, shots=16, seed=81731, threads=1)
                row["sampled_records"] = list(samples.measurements.shape)
            results.append(row)
        rows.append(
            {
                "p": p,
                "ordinary": ordinary,
                "certificate": certificate,
                "shared": shared.metadata(),
                "setup_seconds": setup,
                "variants": results,
                "source_sha256": hashlib.sha256((region + tail).encode()).hexdigest(),
            }
        )
        result = {
            "cases": rows,
            "escape_certificate": escape_certificate,
            "merlin_helper_revision": MERLIN_REVISION,
            "source_hashes": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("automatic_specialization.py"),
                    Path(__file__).with_name("shared_phase_specialization.py"),
                )
            },
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
