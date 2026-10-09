"""Check selected full-factory record probabilities against direct compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import stim
from automatic_specialization import FaultModel, analyze
from shared_phase_specialization import SharedPhase
from study_quadcycle_factory import certify_escape, factory, load_css
from validate_automatic_specialization import visible_probability
from validate_shared_phase_specialization import normalized_lines, prefix_law

import clifft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    css = load_css(args.merlin_checkout)
    region, tail, _ = factory(css, 0.001)
    shared = SharedPhase(region, preserve_state=True)
    tail_model = FaultModel(tail)
    ccz = [i for i, s in enumerate(shared.model.sites) if s.gate == "CORRELATED_ERROR"]
    readout = next(i for i, s in enumerate(tail_model.sites) if s.gate == "READOUT_NOISE")
    histories = [(), ((0, 1),), ((ccz[0], 63),), ((0, 1), (ccz[0], 21), (ccz[-1], 63))]
    tails = [(), ((0, 15),), ((readout, 1),), tail_model.draw(random.Random(81397))]
    rows = []
    for history, tail_history in zip(histories, tails):
        prefix_law(shared, history)
        lines = normalized_lines(shared, history)
        prefix = stim.Circuit("\n".join(lines[: shared.first]))
        for seed in (17, 311):
            simulator = stim.TableauSimulator(seed=seed)
            simulator.set_num_qubits(shared.model.num_qubits)
            simulator.do(prefix)
            records = list(map(int, simulator.current_measurement_record()))
            signs = [simulator.peek_observable_expectation(p) for p in shared.probe_rows]
            if any(sign not in (-1, 1) for sign in signs):
                raise AssertionError("Conditional preparation has an unsupported stabilizer span")
            values = records + [int(sign == -1) for sign in signs]
            faults = 0
            for site, outcome in history:
                faults ^= shared.outcome_controls[site][outcome]
            controls = (
                1
                | (sum(bit << q for q, bit in enumerate(values)) << 1)
                | (faults << shared.fault_base)
            )
            fixed_tail = tail_model.render(tail_history)
            candidate = shared.render_state(controls) + fixed_tail
            reference = (
                str(simulator.current_inverse_tableau().inverse().to_circuit())
                + "\n"
                + "\n".join(f"MPAD {b}" for b in records)
                + "\n"
                + "\n".join(lines[shared.first :])
                + "\n"
                + fixed_tail
            )
            programs = []
            widths = []
            for source in (candidate, reference):
                hir, info = analyze(source)
                if info["peak_width"] > 12:
                    raise ValueError("Conditional reference exceeds the execution budget")
                programs.append(clifft.lower(hir))
                widths.append(info["peak_width"])
            maximum = 0.0
            checked = 0
            for index, program in enumerate(programs):
                samples = clifft.sample(program, shots=8, seed=9271 + index, threads=1).measurements
                for sample in samples:
                    probabilities = [
                        visible_probability(p, list(map(int, sample))) for p in programs
                    ]
                    np.testing.assert_allclose(
                        probabilities[0], probabilities[1], rtol=1e-9, atol=0
                    )
                    maximum = max(maximum, abs(probabilities[0] / probabilities[1] - 1))
                    checked += 1
            rows.append(
                {
                    "region_history": history,
                    "tail_history": tail_history,
                    "prefix_seed": seed,
                    "widths_candidate_reference": widths,
                    "record_probabilities_checked": checked,
                    "maximum_relative_error": maximum,
                }
            )
            print(f"conditional record check {len(rows)} passed", flush=True)
    result = {
        "cases": rows,
        "escape_certificate": certify_escape(css),
        "scope": (
            "Selected full-record probabilities, not equality of the complete noisy law; "
            "reference uses the existing optimizer"
        ),
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("study_quadcycle_factory.py"),
                Path(__file__).with_name("automatic_specialization.py"),
                Path(__file__).with_name("shared_phase_specialization.py"),
            )
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
