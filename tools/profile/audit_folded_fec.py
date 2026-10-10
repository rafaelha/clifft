"""Independent physical histories and live-state checks for the bounded FEC consumer."""

from __future__ import annotations

import argparse
import importlib
import math
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from complete_circuit_panel import packed, save
from folded_fec_consumer import Recovery, fec_circuit, physical_recovery


def probes(sample, distance, width):
    import stim

    Surface = importlib.import_module("folded_msc_family").Surface

    surface = Surface(distance)
    result = []
    supports = (
        ("X", "logical_x", [surface.index[0, y] for y in range(0, 2 * distance - 1, 2)]),
        ("Z", "logical_z", [surface.index[x, 0] for x in range(0, 2 * distance - 1, 2)]),
    )
    for axis, key, support in supports:
        expected = "".join(str(int(q in support)) for q in range(len(surface.points)))
        if sample["recovery"][key] != expected:
            raise ValueError("native logical axes differ from the physical surface-code axes")
        p = stim.PauliString(width)
        for q in support:
            p[q] = axis
        result.append(p)
    result.insert(1, 1j * result[0] * result[1])
    for axis in ("X", "Z"):
        for check in surface.checks(axis):
            p = stim.PauliString(width)
            for q in check:
                p[q] = axis
            result.append(p)
    return result


def dense_aer(circuit, records, observables):
    """Replay the selected physical branch with dense Aer unitary segments.

    Each physical measurement is projected explicitly, including reset outcomes;
    no stabilizer/code reduction or native logical update is used here.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, Statevector
    from qiskit_aer import AerSimulator

    width = circuit.manifest()["num_qubits"]
    if width > 20:
        raise ValueError("dense validation is only for the small instance")
    simulator = AerSimulator(method="statevector", max_parallel_threads=1)
    state = np.zeros(1 << width, dtype=np.complex128)
    state[0] = 1
    qc = QuantumCircuit(width)
    indices = np.arange(len(state), dtype=np.uint64)
    m, r, log_probability = 0, len(circuit.measurements), 0.0
    states = {}

    def flush():
        nonlocal state, qc
        segment = QuantumCircuit(width)
        segment.set_statevector(state)
        segment.compose(qc, inplace=True)
        segment.save_statevector()
        state = np.asarray(simulator.run(segment, shots=1).result().get_statevector()).copy()
        qc = QuantumCircuit(width)

    def expectations():
        sv = Statevector(state)
        return [
            float(sv.expectation_value(Pauli(str(p)[1:][::-1].replace("_", "I"))).real)
            for p in observables
        ]

    for op in circuit.operations:
        if op.stage in {"fec_recovery", "final_logical"} and op.stage not in states:
            flush()
            states[op.stage] = expectations()
        if op.name in {"M", "R"}:
            flush()
            slot = m if op.name == "M" else r
            bit = int(records[slot])
            keep = ((indices >> op.qubits[0]) & 1) == bit
            probability = float(np.vdot(state[keep], state[keep]).real)
            if probability < 1e-15:
                raise AssertionError("Aer found an impossible retained history")
            log_probability += math.log(probability)
            state[~keep] = 0
            state /= math.sqrt(probability)
            if op.name == "R":
                if bit:
                    state = state[indices ^ (1 << op.qubits[0])]
                r += 1
            else:
                m += 1
        else:
            getattr(qc, {"T_DAG": "tdg"}.get(op.name, op.name.lower()))(*op.qubits)
    if "fec_recovery" not in states:
        states["fec_recovery"] = states["final_logical"]
    return dict(
        log_probability=log_probability,
        before=states["fec_recovery"],
        after=states["final_logical"],
    )


def coherent_physical(circuit, records, observables, last_syndrome, recovery_ops):
    folded_msc_oracle = importlib.import_module("folded_msc_oracle")
    Atom = importlib.import_module("clifford_gadget").Atom
    CoherentState = importlib.import_module("clifford_gadget").CoherentState

    captured = []

    def observe_projection(state, atom, outcome):
        CoherentState.project(state, atom, outcome)
        if atom.record == last_syndrome:
            captured.append(state.copy())

    observed_type = type("ObservedState", (CoherentState,), {"project": observe_projection})
    with patch.object(folded_msc_oracle, "CoherentState", observed_type):
        evaluated = folded_msc_oracle.evaluate(circuit, records)
    if len(captured) != 1:
        raise AssertionError("expected one physical state before recovery")
    state = captured[0]
    state.merge()
    norm = state.norm()
    if not norm:
        raise AssertionError("independent reference rejected the live history")

    def expectations(current):
        values = []
        for p in observables:
            signs = [term.tableau.peek_observable_expectation(p) for term in current.terms]
            if signs[0] and all(value == signs[0] for value in signs):
                values.append(float(signs[0]))
            else:
                projected = current.copy()
                projected.project(Atom("MPP", (), 0, pauli=p), 0)
                values.append(2 * projected.norm() / norm - 1)
        return values

    before = expectations(state)
    for op in recovery_ops:
        state.gate(Atom(op.name, op.qubits))
    return before, expectations(state), evaluated


def physical_input(circuit, metadata, sample):
    bind_faults = importlib.import_module("validate_folded_msc").bind_faults

    faults = {}
    for key, table in (
        ("prefix_faults", "prefix_sites"),
        ("suffix_faults", "sites"),
        ("earlier_faults", "earlier_sites"),
        ("growth_faults", "growth_sites"),
    ):
        for i, choice in sample.get(key, []):
            site = metadata[table][i]
            faults[site["operation"]] = site["labels"][choice - 1]
    return bind_faults(circuit, faults), faults


def validate(data, reference):
    native_replay = importlib.import_module("validate_folded_msc").native_replay

    distance, metadata = data["distance"], data["circuit_metadata"]
    circuit = fec_circuit(distance, data["probability"])
    if circuit.text() != data["source"]:
        raise ValueError("changed physical input")
    decoder = Recovery(circuit, distance)
    if decoder.version != data["decoder_policy"]["version"]:
        raise ValueError("decoder version differs from the retained policy")
    rows = []
    for sample in data["attempts"]:
        recovery = sample["recovery"]
        decoded = decoder.decode(sample["records"])
        if any(decoded[key] != recovery[key] for key in decoded):
            raise ValueError("retained correction differs from the recorded-syndrome policy")
        physical, faults = physical_input(circuit, metadata, sample)
        complete = physical_recovery(physical, recovery["x"], recovery["z"])
        width = physical.manifest()["num_qubits"]
        observables = probes(sample, distance, width)
        cut = next(i for i, op in enumerate(physical.operations) if op.stage == "final_logical")
        prefix_ops = physical.operations[:cut]
        visible = sum(op.name == "M" for op in prefix_ops)
        if prefix_ops[-1].name != "M" or prefix_ops[-1].stage != "final_syndrome":
            raise ValueError("recovery must immediately follow the last physical syndrome check")
        recovery_ops = [op for op in complete.operations if op.stage == "fec_recovery"]
        before, after, full = coherent_physical(
            complete, sample["records"], observables, visible - 1, recovery_ops
        )
        start = metadata.get(
            "prefix_stage", f"d{distance}_logical_check_1" if distance > 3 else "logical_check_1"
        )
        prefix_cut = next(i for i, op in enumerate(physical.operations) if op.stage == start)
        native_prefix = replace(
            physical,
            operations=physical.operations[:prefix_cut],
            measurements=physical.measurements[: metadata["prefix_visible"]],
        )
        native_records = (
            sample["records"][: metadata["prefix_visible"]]
            + sample["records"][
                metadata["visible"] : metadata["visible"] + metadata["prefix_hidden"]
            ]
        )
        prior = native_replay(native_prefix, native_records, reference)
        error = abs(
            full["log_probability"]
            - prior["log_probability"]
            - sample["conditional_log_probability"]
        )
        targets = [1 - 2 * int(bit) for bit in recovery["x_syndrome"] + recovery["z_syndrome"]]
        state_error = max(
            max(abs(np.array(before[:3]) - recovery["bloch_before"])),
            max(abs(np.array(after[:3]) - recovery["bloch_after"])),
            max(abs(np.array(before[3:]) - targets)),
            max(abs(np.array(after[3:]) - 1)),
        )
        dense = dense_aer(complete, sample["records"], observables) if distance == 3 else None
        aer_error = (
            None
            if dense is None
            else max(
                abs(dense["log_probability"] - full["log_probability"]),
                max(abs(np.array(dense["before"]) - before)),
                max(abs(np.array(dense["after"]) - after)),
            )
        )
        detectors = "".join(
            str(
                int(sample["records"][i])
                ^ (int(sample["records"][j]) if j < metadata["visible"] else 0)
            )
            for i, j in zip(metadata["postselect"], metadata["detector_partners"], strict=True)
        )
        if detectors != sample["detectors"] or ("1" not in detectors) != sample["accepted"]:
            raise AssertionError("incorrect complete-attempt acceptance")
        if max(error, state_error, aer_error or 0) > 1e-8:
            raise AssertionError(
                f"independent FEC disagreement: {error}, {state_error}, {aer_error}"
            )
        row = dict(
            index=sample["index"],
            faults=faults,
            accepted=sample["accepted"],
            full_log_probability=full["log_probability"],
            log_error=error,
            state_error=state_error,
            aer_error=aer_error,
            before=before,
            after=after,
            peak_oracle_terms=full["peak_terms"],
        )
        rows.append(row)
        print(
            dict(
                distance=distance,
                index=sample["index"],
                log_error=error,
                state_error=state_error,
                aer_error=aer_error,
            ),
            flush=True,
        )
    return dict(status="passed", distance=distance, attempts=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, validate(packed(args.input), args.reference))


if __name__ == "__main__":
    main()
