"""Audit retained complete attempts and existing live-state exports separately from timing."""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from chan_algebraic_continuation import ConditionalChecks
from complete_circuit_panel import DATA, digest, packed, save, select
from conditional_clifford_core import SmallState
from continuation_trace_reuse import ContinuationPhase, ContinuationWorker
from study_conditional_family_transfer import canonical_source, compare, state_queries
from validate_chan_algebraic import purify_resets

import clifft


def check_records(front: ConditionalChecks, history: Any, result: dict[str, Any]) -> dict:
    fixed = "".join(front.materialize(history))
    if result["hidden_random"]:
        return dict(refusal="hidden reset branch requires marginalization")
    program = clifft.compile(purify_resets(fixed))
    if program.peak_active_width > 20:
        return dict(refusal="physical probability reference exceeds width 20")
    expected = clifft.record_probabilities(
        program, np.asarray([result["records"]], dtype=np.uint8)
    )[0]
    return dict(relative_error=compare(result["probability"], float(expected)))


def check_live(front: ConditionalChecks, history: Any, result: dict[str, Any]) -> dict:
    snapshots: list[tuple[SmallState, list[int], str, int]] = []
    prefix: list[str] = []

    def checkpoint(state: SmallState, records: list[int], segment: Any, fixed: str) -> None:
        prefix.append(fixed)
        if state.width and segment.certificate is not None:
            copy = SmallState(state.n)
            copy.frame, copy.vector = state.frame.copy(), state.vector.copy()
            snapshots.append((copy, records.copy(), "".join(prefix), segment.last_line))

    front.trajectory(
        history, result["seed"], forced_records=result["records"], checkpoint=checkpoint
    )
    if not snapshots:
        return dict(refusal="no live quantum check boundary in this history")
    state, records, physical, line = snapshots[-1]
    probability = float(
        clifft.record_probabilities(
            clifft.compile(purify_resets(physical)), np.asarray([records], dtype=np.uint8)
        )[0]
    )
    original = state_queries(
        physical, dict(records=records, probability=probability, output_layer=""), state
    )
    exported = state_queries(
        state.source() + "\n", dict(records=[], probability=1.0, output_layer=""), state
    )
    return dict(
        boundary_line=line,
        records_before_readout=len(records),
        core_width=state.width,
        physical_prefix=original,
        exported_preparation=exported,
        export_width=clifft.compile(state.source()).peak_active_width,
    )


def measured(row: dict[str, Any], result: dict[str, Any]) -> dict:
    front = ConditionalChecks(canonical_source(row["source"]))
    checks = []
    for attempt in result["attempts"]:
        if attempt["status"] != "sampled":
            checks.append(dict(index=attempt["index"], refusal=attempt.get("failure")))
            continue
        history = tuple(map(tuple, attempt["history"]))
        checks.append(dict(index=attempt["index"], **check_records(front, history, attempt)))
    live = []
    for attempt in (result["attempts"][0], result["attempts"][-1]):
        if attempt["status"] == "sampled":
            live.append(check_live(front, tuple(map(tuple, attempt["history"])), attempt))
    sites = [s for segment in front.segments for s in segment.model.sites]
    selected = [(), tuple((i, len(s.replacements) - 1) for i, s in enumerate(sites))]
    rng = random.Random(981421)
    selected.append(
        tuple(
            (i, rng.randrange(1, len(s.replacements)))
            for i, s in enumerate(sites)
            if rng.random() < 0.03
        )
    )
    stress = []
    for i, history in enumerate(selected):
        attempt = dict(history=history, seed=1700 + i)
        try:
            sample = front.trajectory(history, attempt["seed"])
            attempt.update(sample)
            attempt["reference"] = check_records(front, history, sample)
            attempt["status"] = "sampled"
        except (ValueError, RuntimeError) as error:
            attempt.update(status="certificate_failure", failure=str(error))
        stress.append(attempt)
    return dict(complete_record_checks=checks, live_exports=live, selected_stress=stress)


def phase(row: dict[str, Any], result: dict[str, Any], args: argparse.Namespace) -> dict:
    front = ContinuationPhase(row["source"], args.bin / "phase_exporter")
    attempts = [result["attempts"][0], result["attempts"][-1]]
    sites = front.model.sites
    attempts += [
        dict(
            history=tuple((i, len(s.replacements) - 1) for i, s in enumerate(sites)),
            prefix_seed=721,
            seed=172,
        )
    ]
    checks = []
    with ContinuationWorker(
        args.bin / "phase_worker",
        front.optimized_prefix,
        max_width=16,
        phase=front.phase_required,
        continuation=front.template_source,
    ) as native:
        for attempt in attempts:
            history = tuple(map(tuple, attempt["history"]))
            branch = front.rewrite(history, attempt["prefix_seed"])
            actual = native.instantiate(
                front.payload(history, attempt["prefix_seed"]),
                attempt["seed"],
                mode="coordinate-identity",
                reference=branch.source[len(front.optimized_prefix) :],
            )
            if not all(actual[k] for k in ("checked", "optimized_checked", "plan_checked")):
                raise AssertionError("Native conditional continuation comparison failed")
            if "records" in attempt and actual["measurements"] != "".join(
                map(str, attempt["records"])
            ):
                raise AssertionError("Retained attempt is not reproducible")
            checks.append(dict(history=history, native=actual))
    return dict(
        selected_native_checks=checks,
        limitation=(
            "native coordinate/preparation equivalence, "
            "not an independent full physical BT81 oracle"
        ),
    )


def folded(row: dict[str, Any], result: dict[str, Any], args: argparse.Namespace) -> dict:
    validate_trace = importlib.import_module("audit_folded_protocol").validate_trace
    make_circuit = importlib.import_module("folded_msc_family").make_circuit

    spec = row["annotated_model"]
    circuit = make_circuit(spec["probability"], distance=spec["distance"])
    if circuit.text() != row["source"]:
        raise ValueError("Frozen source and oracle model differ")
    checks = []
    for attempt in result["attempts"]:
        check = validate_trace(circuit, result["metadata"], attempt, args.bin / "replay")
        checks.append(check)
        print("folded", len(checks), check, flush=True)
    # A retained annotation cannot authorize a kernel for different physical gates.
    try:
        select(row | dict(source=row["source"] + "H 0\n"), args)
    except ValueError as error:
        refusal = str(error)
        if refusal != "Annotated model does not cover the exact complete source":
            raise
    else:
        raise AssertionError("A changed physical source retained its folded certificate")
    return dict(
        complete_record_checks=checks,
        altered_source_certificate_refused=refusal,
        quantum_output_certificate="missing; no new handoff added",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    args = parser.parse_args()
    sys.path.append(str(args.reference / "tools/profile"))
    panel = packed(DATA / "complete-panel-inputs.json.gz")
    results = packed(DATA / "complete-panel-results.json.gz")
    audits = {}
    for name in [args.case] if args.case else panel:
        row, result = panel[name], results[name]["conditional"]
        if digest(row["source"].encode()) != result["source_sha256"]:
            raise ValueError("Results use a different circuit")
        route = result["route"]
        if route == "measured_check":
            audit = measured(row, result)
        elif route == "phase":
            audit = phase(row, result, args)
        elif route == "folded":
            audit = folded(row, result, args)
        else:
            raise ValueError("No certified route")
        audits[name] = audit
        save(args.output, audits)
        print(name, "audited", flush=True)


if __name__ == "__main__":
    main()
