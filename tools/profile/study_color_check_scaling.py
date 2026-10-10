"""Capability and probability checks on constructed large color-code circuits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import stim
from automatic_specialization import History
from chan_algebraic_continuation import ConditionalChecks
from color_check_coherent_reference import reference
from conditional_clifford_core import SmallState
from constructed_color_checks import DATA, make_source
from study_automatic_specialization import sample_parities
from validate_chan_algebraic import aer_trajectory, error, purify_resets

import clifft


def state_probes(state: SmallState) -> tuple[list[stim.PauliString], list[float]]:
    probes = [state.frame.z_output(q) for q in range(state.n - state.width)]
    expected = [1.0] * len(probes)
    if state.width:
        for axis in "XYZ":
            p = stim.PauliString(state.n)
            p[-1] = axis
            physical = state.frame(p)
            probes.append(physical)
            expected.append(state.expectation(physical))
    return probes, expected


def run(distance: int, trials: int) -> dict:
    source, metadata = make_source(distance, 0.001)
    begin = perf_counter()
    front = ConditionalChecks(source)
    setup_seconds = perf_counter() - begin
    begin = perf_counter()
    ordinary = clifft.compile(source)
    compile_seconds = perf_counter() - begin
    width = ordinary.peak_active_width
    coefficient_bytes = 16 * (1 << width)
    baseline = []
    if width <= 20:
        for i in range(3):
            begin = perf_counter()
            clifft.sample(ordinary, shots=4, seed=184 + i, threads=1)
            baseline.append((perf_counter() - begin) / 4)
    rng = random.Random(60813)
    sites = [site for segment in front.segments for site in segment.model.sites]
    rows = []
    for i in range(trials + 4):
        start = perf_counter()
        history: History
        if i == 0:
            history = ()
            kind = "ideal"
        elif i < trials:
            history = front.draw(rng)
            kind = "original_noise"
        else:
            history = tuple(
                (j, rng.randrange(1, len(s.replacements)))
                for j, s in enumerate(sites)
                if rng.random() < 0.03
            )
            kind = "stress"
        final = SmallState(front.n)

        def checkpoint(state, records, segment, fixed):
            final.frame, final.vector = state.frame.copy(), state.vector.copy()

        sampled = front.trajectory(history, 319 + i, checkpoint=checkpoint)
        program = clifft.compile(sampled["final_source"])
        emitted = clifft.sample(program, shots=1, seed=73, threads=1)
        seconds = perf_counter() - start
        if sampled["hidden_random"] or program.peak_active_width > 1:
            raise AssertionError("The complete continuation lost its bounded record contract")
        np.testing.assert_array_equal(emitted.measurements[0], sampled["records"])
        sample_parities(source, emitted)
        if i == 0 and (emitted.detectors.any() or emitted.observables.any()):
            raise AssertionError("The constructed ideal encoded operation is wrong")
        probes, expected_probes = state_probes(final) if i in (0, trials + 3) else ([], [])
        begin = perf_counter()
        independent = reference(front, history, sampled["records"], probes)
        oracle_seconds = perf_counter() - begin
        log_error = abs(independent["log_probability"] - math.log(sampled["probability"]))
        probe_error = max(
            (abs(x - y) for x, y in zip(independent["probes"], expected_probes)), default=0.0
        )
        if log_error > 1e-9 or probe_error > 1e-9:
            raise AssertionError(
                ("Independent coherent reference", distance, i, log_error, probe_error)
            )
        row = dict(
            kind=kind,
            history=history,
            faults=len(history),
            probability=sampled["probability"],
            end_to_end_seconds=seconds,
            reductions=sampled["reductions"],
            peak_carry_width=sampled["peak_carry_width"],
            export_active_width=program.peak_active_width,
            records=sampled["records"],
            detector_events=int(emitted.detectors.sum()),
            logical_bit=int(emitted.observables[0, 0]),
            coherent_log_probability_error=log_error,
            coherent_state_probes=len(probes),
            coherent_state_probe_error=probe_error,
            coherent_peak_terms=independent["peak_terms"],
            coherent_reference_seconds=oracle_seconds,
        )
        fixed = "".join(front.materialize(history))
        if distance <= 5 or (distance == 7 and i in (0, 1, trials + 3)):
            native = clifft.compile(fixed)
            other = (
                clifft.sample(native, shots=1, seed=157 + i, threads=1)
                .measurements[0]
                .astype(int)
                .tolist()
            )
            mutated = sampled["records"].copy()
            mutated[0] ^= 1
            queries = [sampled["records"], other, mutated]
            exact = clifft.record_probabilities(
                clifft.compile(purify_resets(fixed)), np.asarray(queries, dtype=np.uint8)
            )
            max_relative = 0.0
            for query, p in zip(queries, exact):
                candidate = front.trajectory(history, 319 + i, forced_records=query)["probability"]
                if p == 0:
                    if candidate != 0:
                        raise AssertionError("An impossible physical record was accepted")
                else:
                    relative = abs(candidate / float(p) - 1)
                    max_relative = max(max_relative, relative)
                    if relative > 1e-9:
                        raise AssertionError(("Native physical probability", distance, i, relative))
            row.update(native_queries=len(queries), native_max_relative_error=max_relative)
        if distance == 3:
            bits, vector, probability = aer_trajectory(fixed, 841 + i)
            result = front.trajectory(history, 319 + i, forced_records=bits)
            # Aer states use all physical wires; native state-vector export has a small width cap.
            from qiskit_aer import AerSimulator
            from study_chan_boundary import aer_circuit

            preparation = result["final_source"].split("MPAD")[0]
            count = len(clifft.parse(preparation).nodes)
            reconstructed = np.asarray(
                AerSimulator(method="statevector", max_parallel_threads=1)
                .run(aer_circuit(preparation, [count], []), shots=1)
                .result()
                .data()[f"boundary_{count}"]
            )
            discrepancy = error(vector, reconstructed)
            relative = abs(result["probability"] / probability - 1)
            if discrepancy > 1e-10 or relative > 1e-9:
                raise AssertionError(("Physical Aer state", discrepancy, relative))
            row.update(aer_state_error=discrepancy, aer_relative_probability_error=relative)
        rows.append(row)
        print(distance, i, kind, len(history), "passed", flush=True)
    return dict(
        **metadata,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_lines=len(source.splitlines()),
        fault_sites=front.fault_sites,
        records=front.num_records,
        conditional_setup_seconds=setup_seconds,
        ordinary_compile_seconds=compile_seconds,
        ordinary_active_width=width,
        ordinary_coefficient_bytes=coefficient_bytes,
        ordinary_seconds_per_shot_trials=baseline,
        ordinary_median_seconds_per_shot=statistics.median(baseline) if baseline else None,
        ordinary_sampling_status="measured"
        if baseline
        else "not allocated: active width exceeds research limit 20",
        conditional_median_seconds_per_shot=statistics.median(
            r["end_to_end_seconds"] for r in rows if r["kind"] == "original_noise"
        ),
        rows=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distances", nargs="+", type=int, choices=(3, 5, 7, 9, 11), default=[3, 5, 7, 9, 11]
    )
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.trials <= 32:
        parser.error("Use 2-32 trials")
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    report = dict(
        cpu_affinity=cpu,
        machine=platform.machine(),
        native_extension_sha256=hashlib.sha256(
            next(Path(clifft.__file__).parent.glob("_clifft_core*.so")).read_bytes()
        ).hexdigest(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("clifft", "stim", "numpy", "cirq-core", "qiskit", "qiskit-aer")
        },
        scope=(
            "constructed scaling controls; not author-supplied Chan cultivation "
            "or a fault-tolerance benchmark"
        ),
        cases={str(d): run(d, args.trials) for d in args.distances},
    )
    report["source_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path(__file__).with_name("constructed_color_checks.py"),
            Path(__file__).with_name("color_check_coherent_reference.py"),
            Path(__file__).with_name("chan_algebraic_continuation.py"),
            Path(__file__).with_name("conditional_clifford_core.py"),
            DATA / "color-code-layouts.json",
        ]
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
