"""Compare existing conditional programs on a frozen complete-input panel.

This is an assessment adapter, not a simulator or a production dispatcher.
Names select retained inputs only. Reducers must supply their own certificates.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import random
import resource
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[2] / "research/conditional_clifford"
BASELINE = "847757e2bc8c4f7439b9e92011550fd61719eb85"
FOLDED = "b87c0da396b4bbde54fbee5535aeac1116170605"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def packed(path: Path) -> dict[str, Any]:
    return dict(json.loads(gzip.decompress(path.read_bytes())))


def save(path: Path, data: Any) -> None:
    raw = (json.dumps(data, indent=2) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def verify_reference(reference: Path) -> dict:
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", FOLDED, "tools/profile"], text=True
    ).splitlines()
    hashes = {}
    for name in names:
        expected = subprocess.check_output(["git", "show", FOLDED + ":" + name])
        actual = (reference / name).read_bytes()
        if actual != expected:
            raise ValueError("Changed folded reference: " + name)
        hashes[name] = digest(actual)
    return hashes


def freeze(args: argparse.Namespace) -> None:
    make_circuit = importlib.import_module("folded_msc_family").make_circuit

    references = verify_reference(args.reference)
    panel = {}
    selections = [
        ("bt81-direct-x", "capability-survey-bt81_noisy.json.gz", "bt81_direct_x"),
        ("steane-e", "capability-survey-factory.json.gz", "steane-E.stim"),
        ("chan-four-round-d5", "chan-boundary-inputs.json.gz", "chan-four-round-d5"),
        ("rp2-d5", "family-transfer-inputs.json.gz", "rp2-d5"),
    ]
    for name, archive, key in selections:
        original = packed(DATA / archive)
        row = original.get("cases", original)[key]
        source = row["source"]
        if digest(source.encode()) != row.get("sha256", row.get("source_sha256")):
            raise ValueError("Changed retained source")
        panel[name] = dict(
            source=source,
            sha256=digest(source.encode()),
            archive=archive,
            archive_sha256=digest((DATA / archive).read_bytes()),
            archive_key=key,
        )
    circuit = make_circuit(0.001, distance=7)
    source = circuit.text()
    if (
        digest(source.encode())
        != "fff72e8f48a0509a00d52133d8cb334a1743fc7c74ce1f40eeeccc9785909545"
    ):
        raise ValueError("Folded reconstruction changed")
    panel["folded-f7"] = dict(
        source=source,
        sha256=digest(source.encode()),
        annotated_model=dict(revision=FOLDED, distance=7, probability=0.001),
    )
    # Expected ideal detector signs belong to the input contract, not the reducer.
    rp2 = packed(DATA / "family-transfer-inputs.json.gz")["cases"]["rp2-d5"]
    panel["rp2-d5"]["expected_detectors"] = rp2["export_metadata"]["expected_detectors"]
    save(DATA / "complete-panel-inputs.json.gz", panel)
    save(
        DATA / "complete-panel-manifest.json",
        dict(
            baseline_revision=BASELINE,
            previous_native_baseline="ed2a125ad7daf75deef7e21d66af2a4af4a366bf",
            folded_reference_revision=FOLDED,
            folded_source_hashes=references,
            inputs={
                name: {k: v for k, v in row.items() if k != "source"} for name, row in panel.items()
            },
            scope="See COMPLETE_CIRCUIT_CHECKPOINT.md for provenance, omissions and decision.",
        ),
    )


def sample_row(sample: Any, expected: list[int] | None = None) -> dict:
    measurements = sample.measurements[0].tolist()
    detectors = sample.detectors[0].tolist()
    events = [a ^ b for a, b in zip(detectors, expected, strict=True)] if expected else detectors
    return dict(
        records=measurements,
        detectors=detectors,
        detector_events=events,
        observables=sample.observables[0].tolist(),
        accepted=not any(events) if events else None,
    )


def baseline(row: dict[str, Any], args: argparse.Namespace) -> dict:
    import clifft
    import clifft._clifft_core as core

    start = time.perf_counter()
    hir = clifft.trace(clifft.parse(row["source"]))
    core.default_hir_pass_manager().run(hir)
    width = clifft.active_width_trace(hir)["peak"]
    result = dict(
        setup_seconds=time.perf_counter() - start,
        ordinary_width=width,
        coefficient_bytes=16 * (1 << width),
        allocation_width_limit=args.max_width,
        policy="current default compile; serial complete attempts; no rejection shortcut",
    )
    if width > args.max_width:
        return result | dict(status="width_budget", attempts=[], seconds_per_attempt=None)
    program = clifft.lower(hir)
    result["setup_seconds"] = time.perf_counter() - start
    attempts = []
    for i in range(args.shots):
        start = time.perf_counter()
        sample = clifft.sample(program, shots=1, seed=args.seed + i, threads=1)
        attempt = sample_row(sample, row.get("expected_detectors"))
        attempt["seconds"] = time.perf_counter() - start
        attempts.append(attempt)
    return result | dict(
        status="sampled",
        attempts=attempts,
        seconds_per_attempt=statistics.mean(a["seconds"] for a in attempts),
    )


def select(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, Any, list[dict]]:
    from chan_algebraic_continuation import ConditionalChecks
    from continuation_trace_reuse import ContinuationPhase
    from study_conditional_family_transfer import canonical_source

    certificates: list[dict] = []
    front: Any
    try:
        front = ConditionalChecks(canonical_source(row["source"]))
        checks = [s for s in front.segments if s.certificate is not None]
        if checks:
            certificates.append(dict(reducer="measured_check", candidates=len(checks)))
            return "measured_check", front, certificates
        certificates.append(dict(reducer="measured_check", refusal="no certified check envelope"))
    except (ValueError, RuntimeError) as error:
        certificates.append(dict(reducer="measured_check", refusal=str(error)))
    try:
        front = ContinuationPhase(row["source"], args.bin / "phase_exporter")
        if front.continuation_info["eligible"]:
            certificates.append(
                dict(
                    reducer="phase",
                    first=front.first.metadata(),
                    continuation=front.continuation_info,
                )
            )
            return "phase", front, certificates
        certificates.append(dict(reducer="phase", refusal=front.continuation_info))
    except (ValueError, RuntimeError) as error:
        certificates.append(dict(reducer="phase", refusal=str(error)))
    if "annotated_model" in row:
        compile_protocol = importlib.import_module("compile_folded_protocol").compile_protocol
        make_circuit = importlib.import_module("folded_msc_family").make_circuit

        spec = row["annotated_model"]
        circuit = make_circuit(spec["probability"], distance=spec["distance"])
        if circuit.text() != row["source"]:
            raise ValueError("Annotated model does not cover the exact complete source")
        directory = args.work / "bundle"
        metadata = compile_protocol(circuit, spec["distance"], directory, native_growth=True)
        certificates.append(
            dict(
                reducer="folded",
                source_coverage=True,
                checks="monomial branches, signed CSS, prefix probes, growth operation equality",
                limitation="candidate discovery requires existing annotated folded model",
            )
        )
        return "folded", (circuit, metadata, directory), certificates
    raise ValueError(json.dumps(certificates))


def measured_attempt(front: Any, history: Any, seed: int, row: dict[str, Any]) -> dict:
    from study_automatic_specialization import sample_parities

    import clifft

    boundaries: list[dict] = []

    def checkpoint(state: Any, records: list[int], segment: Any, fixed: str) -> None:
        boundaries.append(
            dict(
                line=segment.last_line,
                width=state.width,
                records=len(records),
                # This existing exporter is exercised while quantum information remains.
                preparation=state.source(),
            )
        )

    result = front.trajectory(history, seed, checkpoint=checkpoint)
    program = clifft.compile(result["final_source"])
    sample = clifft.sample(program, shots=1, seed=seed, threads=1)
    sample_parities(row["source"], sample)
    if sample.measurements[0].tolist() != result["records"]:
        raise AssertionError("Output export changed complete records")
    return dict(
        **sample_row(sample, row.get("expected_detectors")),
        probability=result["probability"],
        hidden_random=result["hidden_random"],
        reductions=result["reductions"],
        boundaries=boundaries,
        peak_carry_width=result["peak_carry_width"],
        export_width=program.peak_active_width,
        final_source=result["final_source"],
        output_layer=result["output_layer"],
        state_access="explicit_preparation",
    )


def conditional(row: dict[str, Any], args: argparse.Namespace) -> dict:
    from continuation_trace_reuse import ContinuationWorker
    from study_automatic_specialization import sample_parities
    from survey_conditional_capability import native_sample

    started = time.perf_counter()
    route, front, certificates = select(row, args)
    if route == "folded":
        circuit, metadata, directory = front
        compile_seconds = time.perf_counter() - started
        start = time.perf_counter()
        raw = subprocess.check_output(
            [str(args.bin / "folded_retained"), str(directory), str(args.shots), str(args.seed)],
            text=True,
        )
        elapsed = time.perf_counter() - start
        lines = [json.loads(line) for line in raw.splitlines()]
        setup, attempts, end = lines[0], lines[1:-1], lines[-1]
        if len(attempts) != args.shots:
            raise AssertionError("Folded attempt count changed")
        for attempt in attempts:
            attempt.update(
                status="sampled",
                state_access="physical_source_and_complete_visible_hidden_record_only",
                state_certificate_failure=(
                    "no exported post-readout quantum state or generic suffix handoff"
                ),
            )
        return dict(
            route=route,
            certificates=certificates,
            setup_seconds=compile_seconds + setup["seconds"],
            seconds_per_attempt=statistics.mean(a["seconds"] for a in attempts),
            process_wall_seconds=elapsed,
            compile_seconds=compile_seconds,
            native_setup=setup,
            native_end=end,
            metadata=metadata,
            attempts=attempts,
            noise_sites=sum(
                len(metadata.get(k, []))
                for k in ("prefix_sites", "sites", "earlier_sites", "growth_sites")
            ),
            residual_output=(
                "exact physical circuit conditioned on retained faults and all records; "
                "compact export uncertified"
            ),
        )
    native = None
    if route == "phase":
        native = ContinuationWorker(
            args.bin / "phase_worker",
            front.optimized_prefix,
            max_width=min(args.max_width, 16),
            phase=front.phase_required,
            continuation=front.template_source,
        )
        native.__enter__()
    setup_seconds = time.perf_counter() - started
    faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x7821)
    attempts = []
    try:
        for i in range(args.shots):
            start = time.perf_counter()
            history = front.model.draw(faults) if route == "phase" else front.draw(faults)
            seed, prefix_seed = quantum.getrandbits(64), quantum.getrandbits(64)
            attempt = dict(index=i, history=history, seed=seed, prefix_seed=prefix_seed)
            try:
                if native is not None:
                    payload = front.payload(history, prefix_seed)
                    branch = front.rewrite(history, prefix_seed)
                    result = native.instantiate(payload, seed, mode="coordinate-identity")
                    attempt.update(
                        native=result,
                        export_width=result["width"],
                        conditional_source=branch.source,
                        state_access="executable_conditional_circuit_and_records",
                        residual_t=result["t_count"],
                        branch_stop=branch.stop_reason,
                        state_certificate_failure=(
                            "native worker does not expose post-shot quantum state"
                        ),
                    )
                    if result["width"] > min(args.max_width, 16):
                        raise ValueError("Conditional allocation width limit exceeded")
                    sample = native_sample(result)
                    sample_parities(row["source"], sample)
                    attempt.update(sample_row(sample, row.get("expected_detectors")))
                else:
                    attempt.update(measured_attempt(front, history, seed, row))
                attempt["status"] = "sampled"
            except (ValueError, RuntimeError) as error:
                attempt.update(status="certificate_failure", failure=str(error))
            attempt["seconds"] = time.perf_counter() - start
            attempts.append(attempt)
            print(i, attempt["status"], file=sys.stderr, flush=True)
    finally:
        if native is not None:
            native.__exit__(None, None, None)
    return dict(
        route=route,
        certificates=certificates,
        setup_seconds=setup_seconds,
        seconds_per_attempt=statistics.mean(a["seconds"] for a in attempts),
        attempts=attempts,
    )


def memory_tree(pid: int) -> tuple[int, int]:
    rss = pss = 0
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            pending.extend(
                map(int, Path(f"/proc/{current}/task/{current}/children").read_text().split())
            )
            for line in Path(f"/proc/{current}/smaps_rollup").read_text().splitlines():
                if line.startswith("Rss:"):
                    rss += int(line.split()[1]) * 1024
                if line.startswith("Pss:"):
                    pss += int(line.split()[1]) * 1024
        except (FileNotFoundError, ProcessLookupError):
            pass
    return rss, pss


def supervise(command: list[str], output: Path, timeout: float = 1800) -> dict:
    start = time.perf_counter()
    rss = pss = samples = 0
    with output.with_suffix(".log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        while child.poll() is None:
            a, b = memory_tree(child.pid)
            rss, pss, samples = max(rss, a), max(pss, b), samples + 1
            if time.perf_counter() - start > timeout:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                break
            time.sleep(0.02)
    return dict(
        exit_code=child.returncode,
        wall_seconds=time.perf_counter() - start,
        peak_process_tree_rss_bytes=rss,
        peak_process_tree_pss_bytes=pss,
        samples=samples,
        nominal_interval_seconds=0.02,
        scope="imports, setup, complete attempts and serialization; excludes supervising parent",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "baseline", "conditional", "run"])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7102026)
    parser.add_argument("--max-width", type=int, default=20)
    args = parser.parse_args()
    sys.path.append(str(args.reference / "tools/profile"))
    args.work.mkdir(parents=True, exist_ok=True)
    if args.mode == "freeze":
        freeze(args)
        return
    panel = packed(DATA / "complete-panel-inputs.json.gz")
    if args.mode == "run":
        results: dict[str, Any] = {}
        for name in [args.case] if args.case else panel:
            results[name] = {}
            for mode in ("baseline", "conditional"):
                output = args.work / (name + "-" + mode + ".json.gz")
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    mode,
                    "--reference",
                    str(args.reference),
                    "--bin",
                    str(args.bin),
                    "--work",
                    str(args.work / name),
                    "--output",
                    str(output),
                    "--case",
                    name,
                    "--shots",
                    str(args.shots),
                    "--seed",
                    str(args.seed),
                    "--max-width",
                    str(args.max_width),
                ]
                memory = supervise(command, output)
                result = (
                    packed(output) if memory["exit_code"] == 0 else dict(status="worker_failure")
                )
                results[name][mode] = result | dict(memory=memory)
                save(args.output, results)
                print(name, mode, memory, flush=True)
        return
    import clifft
    import clifft._clifft_core as core

    manifest = json.loads((DATA / "complete-panel-manifest.json").read_text())
    build = manifest["build"]
    if build["source_revision"] != BASELINE:
        raise ValueError("Baseline source revision differs from the checkpoint")
    native_hash = digest(Path(core.__file__).read_bytes())
    if native_hash != build["python_native_sha256"]:
        raise ValueError(
            "Native build differs; verify and record the pinned build before measuring"
        )
    binaries = {Path(path).name: value for path, value in build["files"].items()}
    for name in ("phase_exporter", "phase_worker", "folded_retained", "replay"):
        if digest((args.bin / name).read_bytes()) != binaries[name]:
            raise ValueError("Native helper differs from the recorded build: " + name)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    row = panel[args.case]
    if (
        digest(row["source"].encode()) != row["sha256"]
        or row["sha256"] != manifest["inputs"][args.case]["sha256"]
    ):
        raise ValueError("Frozen complete input changed")
    result = (baseline if args.mode == "baseline" else conditional)(row, args)
    result.update(
        source_sha256=row["sha256"],
        baseline_revision=BASELINE,
        python_package=clifft.__file__,
        native_sha256=native_hash,
        process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        seed=args.seed,
    )
    save(args.output, result)


if __name__ == "__main__":
    main()
