"""Complete supplied SOFT/RP2 instruments under conditional Clifford reduction."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import random
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import stim
from automatic_specialization import FaultModel, History, instruction_text, record_count
from chan_algebraic_continuation import ConditionalChecks
from conditional_clifford_core import Refused, SmallState
from study_automatic_specialization import sample_parities
from validate_chan_algebraic import aer_trajectory, error, purify_resets

import clifft

DATA = Path(__file__).resolve().parents[2] / "research/conditional_clifford"


def canonical_source(source: str) -> str:
    """Expand grouped operands without moving gates, channels, or records."""
    lines: list[str] = []
    records = 0
    for node in clifft.parse(source).nodes:
        if node.gate.name == "READOUT_NOISE":
            if node.targets[0].value != records - 1:
                raise ValueError("Readout noise is not adjacent to its record")
            gate, targets = lines[-1].split(" ", 1)
            lines[-1] = f"{gate}({node.arg:.17g}) {targets}"
        else:
            lines.append(instruction_text(node, records))
            records += record_count(node)
    result = "\n".join(lines) + "\n"
    before, after = FaultModel(source), FaultModel(result)
    if before.lines != after.lines or before.sites != after.sites:
        raise AssertionError("Canonicalization changed gates or categorical noise")
    return result


def pauli_line(p: stim.PauliString) -> str:
    targets = "*".join(f"{'IXYZ'[a]}{q}" for q, a in enumerate(p) if a)
    return "MPP " + ("!" if p.sign == -1 else "") + targets


def compare(actual: float, expected: float) -> float:
    if expected == 0:
        if actual != 0:
            raise AssertionError(("Impossible record admitted", actual))
        return 0.0
    discrepancy = abs(actual / expected - 1)
    if discrepancy > 1e-9:
        raise AssertionError(("Relative physical probability", actual, expected, discrepancy))
    return discrepancy


def state_queries(fixed: str, result: dict, state: SmallState) -> dict:
    """Characterize the entire conditional state with its dormant and core Paulis."""
    restore = result["output_layer"]
    undo = "\n".join(
        ("T_DAG" if line.split()[0] == "T" else "T") + " " + line.split()[1]
        for line in reversed(restore.splitlines())
    )
    if undo:
        undo += "\n"
    # Purification relabels wires, so append physical probes before that transformation.
    dormant = [state.frame.z_output(q) for q in range(state.n - state.width)]
    checks = "\n".join(map(pauli_line, dormant)) + "\n"
    base = result["records"] + [0] * len(dormant)
    axes = "XYZ" if state.width else "I"
    maximum = 0.0
    for axis in axes:
        suffix = checks
        queries = [base]
        expected = [result["probability"]]
        if axis != "I":
            local = stim.PauliString(state.n)
            local[-1] = axis
            p = state.frame(local)
            suffix += pauli_line(p) + "\n"
            queries = [base + [0], base + [1]]
            mean = state.expectation(p)
            expected = [result["probability"] * (1 + sign * mean) / 2 for sign in (1, -1)]
        source = purify_resets(fixed + undo + suffix)
        actual = clifft.record_probabilities(
            clifft.compile(source), np.asarray(queries, dtype=np.uint8)
        )
        for a, b in zip(actual, expected):
            if abs(b) < 1e-12 * result["probability"]:
                if abs(a) > 1e-12 * result["probability"]:
                    raise AssertionError("Conditional-state zero probability changed")
            else:
                maximum = max(maximum, compare(float(a), b))
    return dict(
        dormant_stabilizers=len(dormant),
        logical_axes=3 if state.width else 0,
        maximum_relative_error=maximum,
        physical_output_layer_restored=bool(restore),
    )


def aer_check(front: ConditionalChecks, history: tuple, seed: int) -> dict:
    fixed = "".join(front.materialize(history))
    bits, reference, probability = aer_trajectory(fixed, seed)
    actual = front.trajectory(history, seed, forced_records=bits)
    if actual["hidden_random"]:
        raise AssertionError("Aer comparison needs hidden-reset marginalization")
    preparation = actual["final_source"].split("MPAD")[0]
    # Adding a deterministic record lets the same physical adapter handle a preparation.
    _, candidate, _ = aer_trajectory(preparation + "MPP Z" + str(front.n) + "\n", seed)
    candidate = candidate[: 1 << front.n]
    discrepancy = error(reference, candidate)
    relative = compare(actual["probability"], probability)
    if discrepancy > 1e-10:
        raise AssertionError(("Aer physical output state", discrepancy))
    return dict(state_error=discrepancy, relative_probability_error=relative)


def run(name: str, case: dict, trials: int) -> dict:
    raw = case["source"]
    if hashlib.sha256(raw.encode()).hexdigest() != case["sha256"]:
        raise AssertionError("Retained input changed")
    source = canonical_source(raw)
    start = perf_counter()
    front = ConditionalChecks(source)
    setup = perf_counter() - start
    raw_model = FaultModel(raw)
    ordinary = clifft.compile(raw)
    rng = random.Random(617231)
    sites = [s for segment in front.segments for s in segment.model.sites]
    histories: list[tuple[str, History]] = [("ideal", ())]
    histories += [("original_noise", front.draw(rng)) for _ in range(trials)]
    histories += [
        (
            "stress",
            tuple(
                (j, rng.randrange(1, len(s.replacements)))
                for j, s in enumerate(sites)
                if rng.random() < 0.03
            ),
        )
        for _ in range(4)
    ]
    rows = []
    for index, (kind, history) in enumerate(histories):
        final = SmallState(front.n)
        boundaries = []

        def checkpoint(state, records, segment, fixed):
            final.frame, final.vector = state.frame.copy(), state.vector.copy()
            boundaries.append(dict(line=segment.last_line, width=state.width, records=len(records)))

        start = perf_counter()
        result = front.trajectory(history, 812 + index, checkpoint=checkpoint)
        program = clifft.compile(result["final_source"])
        emitted = clifft.sample(program, shots=1, seed=73, threads=1)
        seconds = perf_counter() - start
        np.testing.assert_array_equal(emitted.measurements[0], result["records"])
        sample_parities(raw, emitted)
        if result["hidden_random"]:
            raise AssertionError("Visible probability requires unresolved reset marginalization")
        fixed = raw_model.render(history)
        if fixed != "".join(front.materialize(history)):
            raise AssertionError("Segmentation changed physical gates or noise-site ordering")
        native = clifft.compile(fixed)
        other = clifft.sample(native, shots=1, seed=267 + index, threads=1).measurements[0].tolist()
        mutated = result["records"].copy()
        mutated[len(mutated) // 2] ^= 1
        queries = [result["records"], other, mutated]
        exact = clifft.record_probabilities(
            clifft.compile(purify_resets(fixed)), np.asarray(queries, dtype=np.uint8)
        )
        maximum = 0.0
        for query, expected in zip(queries, exact):
            actual = front.trajectory(history, 812 + index, forced_records=query)
            maximum = max(maximum, compare(actual["probability"], float(expected)))
        row = dict(
            kind=kind,
            history=history,
            faults=len(history),
            records=result["records"],
            probability=result["probability"],
            seconds=seconds,
            boundaries=boundaries,
            reductions=result["reductions"],
            peak_carry_width=result["peak_carry_width"],
            export_active_width=program.peak_active_width,
            deferred_output_t_count=len(result["output_layer"].splitlines()),
            native_queries=len(queries),
            maximum_relative_probability_error=maximum,
            detector_events=int(emitted.detectors.sum()),
            observable_bits=emitted.observables[0].tolist(),
        )
        if index in (0, len(histories) - 1):
            row["state_validation"] = state_queries(fixed, result, final)
        if name == "soft-d3":
            row["aer"] = aer_check(front, history, 723 + index)
        rows.append(row)
        print(name, index, kind, len(history), "passed", flush=True)
    return dict(
        source_sha256=case["sha256"],
        canonical_sha256=hashlib.sha256(source.encode()).hexdigest(),
        qubit_index_extent=front.n,
        records=front.num_records,
        fault_sites=front.fault_sites,
        expected_faults=sum(1 - s.probabilities[0] for s in sites),
        ordinary_active_width=ordinary.peak_active_width,
        setup_seconds=setup,
        median_seconds=statistics.median(
            r["seconds"] for r in rows if r["kind"] == "original_noise"
        ),
        rows=rows,
    )


def controls() -> dict:
    source = """H 0
T 0
CX 0 1
CX 0 2
RX 3
T 0
T 1
T 2
DEPOLARIZE1(0.1) 0 1 2
RX 4
Z_ERROR(0.1) 4
CX 4 0
CX 3 0
CX 3 1
CX 3 2
MX(0.1) 3
DEPOLARIZE1(0.1) 3
RX 3
Z_ERROR(0.1) 3
CX 3 2
CX 3 1
CX 3 0
CX 4 0
MX(0.1) 4
DEPOLARIZE1(0.1) 4
T_DAG 0
T_DAG 1
T_DAG 2
"""
    front = ConditionalChecks(source)
    sites = [s for segment in front.segments for s in segment.model.sites]
    histories = [()] + [
        ((i, j),) for i, s in enumerate(sites) for j in range(1, len(s.replacements))
    ]
    results = [aer_check(front, h, 517 + i) for i, h in enumerate(histories)]
    # A terminal open phase must retain its physical action in the exported state.
    open_front = ConditionalChecks("H 0\nH 1\nT 0\nT 1\nMPP Z0*Z1\n")
    open_results = [aer_check(open_front, (), i) for i in range(4)]
    rotations = []
    for sign in (1, -1):
        f = ConditionalChecks(f"H 0\nT 0\nR_PAULI({sign * 0.25}) Z0\nMPP X0\n")
        r = f.trajectory((), 17)
        p = clifft.record_probabilities(
            clifft.compile("".join(f.materialize(()))), np.asarray([r["records"]], dtype=np.uint8)
        )[0]
        rotations.append(compare(r["probability"], float(p)))
    refused = []
    for source in (
        "H 0\nH 1\nT 0\nT 1\n",
        "H 0\nT 0\nR_PAULI(0.125) Z0\n",
        "H 0\nH 1\nT 0\nT 1\nMPP Z0*Z1\nH 0\n",
    ):
        try:
            ConditionalChecks(source).trajectory((), 0)
        except Refused as ex:
            refused.append(str(ex))
        else:
            raise AssertionError("Unsupported extension was silently admitted")
    return dict(
        x_flag_physical_aer=results,
        open_output_aer=open_results,
        logical_rotations=rotations,
        refused=refused,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.trials <= 32:
        parser.error("Use 1-32 original-noise histories")
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    inputs = json.loads(gzip.decompress((DATA / "family-transfer-inputs.json.gz").read_bytes()))
    report: dict = dict(controls=controls(), cpu=cpu, cases={})
    for name, case in inputs["cases"].items():
        if args.cases is None or name in args.cases:
            report["cases"][name] = run(name, case, args.trials)
    report["versions"] = {
        name: importlib.metadata.version(name) for name in ("clifft", "stim", "numpy", "qiskit-aer")
    }
    report["native_extension_sha256"] = hashlib.sha256(
        next(Path(clifft.__file__).parent.glob("_clifft_core*.so")).read_bytes()
    ).hexdigest()
    report["source_hashes"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            Path(__file__).name,
            "chan_algebraic_continuation.py",
            "conditional_clifford_core.py",
            "study_chan_boundary.py",
            "parity_envelope_audit.py",
            "validate_chan_algebraic.py",
        )
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
