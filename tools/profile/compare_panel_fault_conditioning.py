"""Replay frozen panel faults through the original physical circuits."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from automatic_specialization import FaultModel
from complete_circuit_panel import BASELINE, DATA, digest, packed, sample_row, save, supervise

PARTITIONS = (
    ("prefix_faults", "prefix_sites"),
    ("suffix_faults", "sites"),
    ("earlier_faults", "earlier_sites"),
    ("growth_faults", "growth_sites"),
)


class RetainedBinder:
    """Adapt saved fault identifiers, without specializing the quantum state."""

    def __init__(self, row, previous, reference):
        self.previous = previous
        self.model = None
        if "annotated_model" not in row:
            self.model = FaultModel(row["source"])
            self.sites = len(self.model.sites)
            return
        sys.path.append(str(reference / "tools/profile"))
        make_circuit = importlib.import_module("folded_msc_family").make_circuit
        validation = importlib.import_module("validate_folded_msc")
        spec = row["annotated_model"]
        self.circuit = make_circuit(spec["probability"], distance=spec["distance"])
        if self.circuit.text() != row["source"]:
            raise ValueError("Folded fault labels do not cover the frozen physical source")
        self.bind = validation.bind_faults
        self.metadata = previous["metadata"]
        covered = []
        for _, key in PARTITIONS:
            for site in self.metadata[key]:
                operation = self.circuit.operations[site["operation"]]
                if list(validation.noise_labels(operation)) != site["labels"]:
                    raise ValueError("Retained categorical outcomes changed")
                covered.append(site["operation"])
        expected = [i for i, op in enumerate(self.circuit.operations) if op.probability is not None]
        if sorted(covered) != expected:
            raise ValueError("Fault partitions must cover each original noise operation once")
        self.sites = len(expected)

    def render(self, attempt):
        if self.model is not None:
            history = tuple(map(tuple, attempt["history"]))
            return self.model.render(history), dict(history=history)
        faults = {}
        partitions = {}
        for saved, key in PARTITIONS:
            partitions[saved] = attempt[saved]
            for site_index, outcome in attempt[saved]:
                site = self.metadata[key][site_index]
                if not 0 < outcome <= len(site["labels"]):
                    raise ValueError("Invalid retained categorical outcome")
                if site["operation"] in faults:
                    raise ValueError("Repeated retained fault site")
                faults[site["operation"]] = site["labels"][outcome - 1]
        return self.bind(self.circuit, faults).text(), dict(
            **partitions, physical_faults=sorted(faults.items())
        )


def inputs(args):
    manifest = json.loads((DATA / "complete-panel-manifest.json").read_text())
    for name in ("complete-panel-inputs.json.gz", "complete-panel-results.json.gz"):
        if digest((DATA / name).read_bytes()) != manifest["artifact_sha256"][name]:
            raise ValueError("Frozen panel archive changed: " + name)
    panel = packed(DATA / "complete-panel-inputs.json.gz")
    previous = packed(DATA / "complete-panel-results.json.gz")
    row = panel[args.case]
    if digest(row["source"].encode()) != manifest["inputs"][args.case]["sha256"]:
        raise ValueError("Frozen input changed")
    if "annotated_model" in row:
        for name, expected in manifest["folded_source_hashes"].items():
            if digest((args.reference / name).read_bytes()) != expected:
                raise ValueError("Pinned fault-binder reference changed: " + name)
    import clifft._clifft_core as core

    if digest(Path(core.__file__).read_bytes()) != manifest["build"]["python_native_sha256"]:
        raise ValueError("Native Clifft build differs from the pinned panel baseline")
    return row, previous[args.case]["conditional"], manifest


def worker(args):
    from study_automatic_specialization import sample_parities

    import clifft
    import clifft._clifft_core as core

    row, previous, manifest = inputs(args)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    start = time.perf_counter()
    binder = RetainedBinder(row, previous, args.reference)
    setup = time.perf_counter() - start
    rng = random.Random(previous["seed"])
    attempts: list[dict[str, Any]] = []
    for index, retained in enumerate(previous["attempts"]):
        start = time.perf_counter()
        if binder.model is not None:
            draw = binder.model.draw(rng)
            if list(map(list, draw)) != retained["history"]:
                raise AssertionError("Fault draw differs from the saved history")
        drawn = time.perf_counter()
        source, history = binder.render(retained)
        rendered = time.perf_counter()
        attempt = dict(
            index=index,
            **history,
            seed=retained.get("seed"),
            source_sha256=digest(source.encode()),
            retained_status=retained["status"],
            retained_accepted=retained["accepted"],
            fault_draw_seconds=drawn - start if binder.model is not None else None,
            binding_seconds=rendered - drawn,
        )
        try:
            hir = clifft.trace(clifft.parse(source))
            core.default_hir_pass_manager().run(hir)
            width = clifft.active_width_trace(hir)["peak"]
            attempt.update(
                active_width=width,
                residual_t=hir.num_t_gates,
                coefficient_bytes=16 * (1 << width),
                trace_optimize_seconds=time.perf_counter() - rendered,
            )
            if width > args.max_width:
                attempt.update(
                    status="width_budget",
                    failure="Original conditional program exceeds allocation budget",
                )
            else:
                program = clifft.lower(hir)
                prepared = time.perf_counter()
                if attempt["seed"] is None:
                    raise ValueError("Retained native stream has no individual quantum seed")
                sample = clifft.sample(program, shots=1, seed=attempt["seed"], threads=1)
                sampled = time.perf_counter()
                attempt.update(
                    sample_row(sample, row.get("expected_detectors")),
                    status="sampled",
                    prepare_seconds=prepared - rendered,
                    sample_seconds=sampled - prepared,
                )
                sample_parities(row["source"], sample)
        except (ValueError, RuntimeError) as error:
            attempt.update(status="execution_refusal", failure=str(error))
        attempt["seconds"] = time.perf_counter() - start
        attempts.append(attempt)
        print(args.case, index, attempt["status"], attempt.get("active_width"), flush=True)
    completed = [a for a in attempts if a["status"] == "sampled"]
    return dict(
        input_sha256=row["sha256"],
        baseline_revision=BASELINE,
        native_sha256=manifest["build"]["python_native_sha256"],
        source_results_sha256=manifest["artifact_sha256"]["complete-panel-results.json.gz"],
        noise_sites=binder.sites,
        setup_seconds=setup,
        allocation_width_limit=args.max_width,
        policy=(
            "same retained faults; original physical circuit; default compiler; no early rejection"
        ),
        fault_draw_policy=(
            "reproduce and verify original draws, charging draw cost"
            if binder.model
            else (
                "replay saved native categorical choices; "
                "native draw cost unavailable; no execution claimed"
            )
        ),
        seconds_per_attempt=(
            statistics.mean(a["seconds"] for a in completed)
            if len(completed) == len(attempts)
            else None
        ),
        mean_seconds_to_result=statistics.mean(a["seconds"] for a in attempts),
        state_access=(
            "physical input plus faults, program hash, seed and visible records; "
            "no compact state export"
        ),
        process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        attempts=attempts,
    )


def audit(args):
    import numpy as np
    from chan_algebraic_continuation import ConditionalChecks
    from study_conditional_family_transfer import canonical_source, compare
    from validate_chan_algebraic import purify_resets

    import clifft

    row, previous, _ = inputs(args)
    binder = RetainedBinder(row, previous, args.reference)
    measured = (
        ConditionalChecks(canonical_source(row["source"]))
        if previous["route"] == "measured_check"
        else None
    )
    results = packed(args.results)[args.case]
    checks = []
    for old, new in zip(previous["attempts"], results["attempts"], strict=True):
        source, _ = binder.render(old)
        if digest(source.encode()) != new["source_sha256"]:
            raise AssertionError("Executed source differs from the retained fault binding")
        check = dict(index=new["index"], source_sha256_verified=True)
        if measured is not None:
            segmented = "".join(measured.materialize(tuple(map(tuple, old["history"]))))
            if segmented != source:
                raise AssertionError(
                    "Full-circuit fault binding differs from the existing segmented binder"
                )
            if old["hidden_random"]:
                raise ValueError("Retained reference requires hidden-reset marginalization")
            program = clifft.compile(purify_resets(source))
            if program.peak_active_width > args.max_width:
                raise ValueError("Physical probability reference exceeds allocation budget")
            queries = [old["records"], new["records"]]
            probabilities = clifft.record_probabilities(
                program, np.asarray(queries, dtype=np.uint8)
            )
            error = compare(old["probability"], float(probabilities[0]))
            if probabilities[1] <= 0:
                raise AssertionError("Fault-only sampler emitted an impossible record")
            check.update(
                binding_matches_existing=True,
                retained_probability=old["probability"],
                physical_probabilities=probabilities.tolist(),
                relative_error=error,
            )
        else:
            check["probability_refusal"] = (
                "unreduced full circuit exceeds declared allocation budget"
            )
        checks.append(check)
    return dict(
        checks=checks,
        independence="existing physical-source probability engine; no new independent simulator",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker", "audit"))
    parser.add_argument("--case")
    parser.add_argument("--reference", type=Path, default=Path("/tmp/clifft-495-scaling-reference"))
    parser.add_argument("--work", type=Path, default=Path("/tmp/clifft-panel-fault-only"))
    parser.add_argument("--max-width", type=int, default=24)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode != "run":
        save(args.output, worker(args) if args.mode == "worker" else audit(args))
        return
    panel = packed(DATA / "complete-panel-inputs.json.gz")
    args.work.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in [args.case] if args.case else panel:
        output = args.work / (name + ".json.gz")
        output.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--case",
            name,
            "--reference",
            str(args.reference),
            "--max-width",
            str(args.max_width),
            "--output",
            str(output),
        ]
        memory = supervise(command, output, timeout=600)
        result[name] = (
            packed(output) if output.exists() else dict(status="worker_failure")
        ) | dict(memory=memory)
        save(args.output, result)
        print(name, memory, flush=True)


if __name__ == "__main__":
    main()
