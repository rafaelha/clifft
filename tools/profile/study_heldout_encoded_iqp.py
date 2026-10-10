"""Frozen literature-motivated encoded IQP test of the unchanged phase reducer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from complete_circuit_panel import baseline, conditional, packed, sample_row, save, supervise

DATA = Path(__file__).resolve().parents[2] / "research/conditional_clifford"


def circuit_text(dimensions=2, probability=0.001):
    """Canonical affine encoder and the paper's transversal gate vocabulary.

    Vertex v holds g XOR a*v0 XOR b*v1 XOR c*v2. The encoder schedule and
    deterministic gate-layer choices are explicit reconstruction choices.
    """
    blocks = 1 << dimensions
    lines = ["# Complete reconstructed encoded IQP sampling instance"]

    def gate(name, *targets):
        lines.append(name + " " + " ".join(map(str, targets)))
        lines.append(f"DEPOLARIZE{len(targets)}({probability}) " + " ".join(map(str, targets)))

    for block in range(blocks):
        base = 8 * block
        for q in range(base, base + 8):
            lines.extend((f"R {q}", f"X_ERROR({probability}) {q}"))
        for q in (0, 1, 2, 4):
            gate("H", base + q)
        for source in (1, 2, 4):
            for target in range(1, 8):
                if target != source and target & source:
                    gate("CX", base + source, base + target)
        for target in range(1, 8):
            gate("CX", base, base + target)
        # Logical Z_a Z_c prepares the paper's minus, plus, minus input signs.
        gate("Z", base + 1)
        gate("Z", base + 4)
    for layer in range(dimensions + 1):
        lines.append("TICK")
        for block in range(blocks):
            for vertex in range(8):
                name = "T" if layer % 2 and vertex < 4 else "T_DAG"
                gate(name, 8 * block + vertex)
        if layer < dimensions:
            for block in range(blocks):
                if not block & (1 << layer):
                    for vertex in range(8):
                        gate("CX", 8 * block + vertex, 8 * (block ^ (1 << layer)) + vertex)
    for q in range(8 * blocks):
        lines.append(f"MX({probability}) {q}")
    for block in range(blocks):
        records = [8 * block + v - 8 * blocks for v in range(8)]
        lines.append("DETECTOR " + " ".join(f"rec[{r}]" for r in records))
        for axis in range(3):
            lines.append(
                f"OBSERVABLE_INCLUDE({3 * block + axis}) "
                + " ".join(f"rec[{records[v]}]" for v in range(8) if v & (1 << axis))
            )
    return "\n".join(lines) + "\n"


def freeze(path, dimensions):
    source = circuit_text(dimensions)
    blocks = 1 << dimensions
    profile = Path(__file__).resolve().parent
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    hashes = {
        str(p.relative_to(profile)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(profile.iterdir())
        if p.suffix in {".py", ".cpp", ".h"} and p.name != Path(__file__).name
    }
    save(
        path,
        dict(
            source=source,
            sha256=hashlib.sha256(source.encode()).hexdigest(),
            frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            reducer_revision=revision,
            reducer_source_hashes=hashes,
            literature="https://doi.org/10.1038/s41586-023-06927-3",
            status="independent complete reconstruction; not an author-supplied circuit",
            specification=dict(
                physical_qubits=8 * blocks,
                logical_qubits=3 * blocks,
                blocks=blocks,
                hypercube_dimensions=dimensions,
                phase_layers=dimensions + 1,
                probability=0.001,
                initialization="canonical affine CSS encoder; logical minus, plus, minus",
                gates="global T_DAG alternating with T on vertices 0..3 and T_DAG on 4..7",
                coupling="transversal CNOT along successive hypercube dimensions, low bit to high",
                readout=(
                    "all physical X records; global X syndrome and "
                    "three logical X parities per block"
                ),
            ),
            qualifications=[
                "encoder and layer schedule are reconstruction choices",
                "synthetic local stochastic Pauli noise, not experimental calibrated noise",
                "no atom transport, leakage, loss, coherent calibration error or idle noise",
                "single-copy sampling; no two-copy purity experiment or decoded error-rate claim",
            ],
            prespecified_test=dict(
                shots=32,
                seed=1010202601,
                ordinary_width_budget=20,
                conditional_width_budget=16,
                reducer="existing complete panel phase workflow",
                alterations="none",
                selection="structural match, before observing reducer results",
            ),
        ),
    )
    print(json.dumps(dict(frozen=str(path), sha256=hashlib.sha256(source.encode()).hexdigest())))


def verify(row):
    if hashlib.sha256(row["source"].encode()).hexdigest() != row["sha256"]:
        raise ValueError("held-out source changed")
    for name, expected in row["reducer_source_hashes"].items():
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError("reducer changed after input freeze: " + name)


def load_results(path):
    result = packed(path)
    for route in ("conditional_export", "fixed_fault"):
        if route in result and "artifact" in result[route]:
            reference = result[route]
            artifact = path.parent / reference["artifact"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != reference["sha256"]:
                raise ValueError("Changed result artifact: " + str(artifact))
            result[route] = packed(artifact)
    return result


def save_results(path, result):
    # Retain every program and history while respecting the repository file-size limit.
    index = result.copy()
    for route in ("conditional_export", "fixed_fault"):
        if route in result:
            artifact = path.with_name(path.name.removesuffix(".json.gz") + "-" + route + ".json.gz")
            save(artifact, result[route])
            index[route] = dict(
                artifact=artifact.name, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()
            )
    save(path, index)


def export_execution(row, args):
    """Execute the existing exact export beyond the native research worker's guard.

    This adds only assessment plumbing for the public compiler and executor;
    no phase reduction, coordinate conversion, or sampling algorithm is added.
    """
    from continuation_trace_reuse import ContinuationPhase
    from study_automatic_specialization import sample_parities

    import clifft
    import clifft._clifft_core as core

    retained = packed(args.retained)["conditional"]["attempts"]
    start = time.perf_counter()
    from automatic_specialization import FaultModel

    plain = args.mode == "fixed_fault"
    front = None if plain else ContinuationPhase(row["source"], args.bin / "phase_exporter")
    if plain:
        model = FaultModel(row["source"])
    else:
        assert front is not None
        model = front.model
    setup = time.perf_counter() - start
    faults, quantum = random.Random(args.seed), random.Random(args.seed ^ 0x7821)
    attempts: list[dict[str, Any]] = []
    for index, previous in enumerate(retained):
        start = time.perf_counter()
        history = model.draw(faults)
        seed, prefix_seed = quantum.getrandbits(64), quantum.getrandbits(64)
        if (list(map(list, history)), seed, prefix_seed) != (
            previous["history"],
            previous["seed"],
            previous["prefix_seed"],
        ):
            raise AssertionError("Replay changed the retained draw or quantum seeds")
        attempt = dict(index=index, history=history, seed=seed, prefix_seed=prefix_seed)
        try:
            if plain:
                source = model.render(history)
            else:
                assert front is not None
                source = front.rewrite(history, prefix_seed).source
            if not plain and source != previous["conditional_source"]:
                raise AssertionError("Conditional export changed after native refusal")
            hir = clifft.trace(clifft.parse(source))
            core.default_hir_pass_manager().run(hir)
            width = clifft.active_width_trace(hir)["peak"]
            attempt.update(
                conditional_source=source,
                export_width=width,
                residual_t=hir.num_t_gates,
                coefficient_bytes=16 * (1 << width),
            )
            if width > args.max_width:
                raise ValueError("Conditional export exceeds declared allocation budget")
            program = clifft.lower(hir)
            prepared = time.perf_counter()
            sample = clifft.sample(program, shots=1, seed=seed, threads=1)
            attempt.update(
                sample_row(sample),
                status="sampled",
                compile_seconds=prepared - start,
                sample_seconds=time.perf_counter() - prepared,
            )
            sample_parities(row["source"], sample)
        except (ValueError, RuntimeError) as error:
            attempt.update(status="certificate_failure", failure=str(error))
        attempt["seconds"] = time.perf_counter() - start
        attempts.append(attempt)
        print(index, attempt["status"], attempt.get("export_width"), flush=True)
    return dict(
        setup_seconds=setup,
        allocation_width_limit=args.max_width,
        policy=(
            "same retained draws; original physical circuit with fixed Pauli faults"
            if plain
            else (
                "same retained draws; unchanged exact export; public compile and sample per attempt"
            )
        ),
        seconds_per_attempt=statistics.mean(a["seconds"] for a in attempts),
        attempts=attempts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "export", "ablate", "worker"))
    parser.add_argument("--input", type=Path, default=DATA / "heldout-iqp-input.json.gz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work", type=Path, default=Path("/tmp/clifft-heldout-iqp"))
    parser.add_argument("--bin", type=Path, default=Path("/tmp/clifft-complete-panel/bin"))
    parser.add_argument(
        "--mode", choices=("ordinary", "conditional", "conditional_export", "fixed_fault")
    )
    parser.add_argument("--retained", type=Path)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1010202601)
    parser.add_argument("--max-width", type=int, default=20)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.input, args.dimensions)
        return
    row = packed(args.input)
    verify(row)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.command == "worker":
        if args.mode in {"conditional_export", "fixed_fault"}:
            result = export_execution(row, args)
        else:
            result = (baseline if args.mode == "ordinary" else conditional)(row, args)
        save(args.output, result)
        return
    import sys

    result = (
        load_results(args.output)
        if args.command in {"export", "ablate"}
        else dict(input_sha256=row["sha256"], reducer_unchanged=True)
    )
    modes = {"export": ("conditional_export",), "ablate": ("fixed_fault",)}
    for mode in modes.get(args.command, ("ordinary", "conditional")):
        output = args.work / (mode + ".json.gz")
        output.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--input",
            str(args.input),
            "--mode",
            mode,
            "--work",
            str(args.work / mode),
            "--bin",
            str(args.bin),
            "--shots",
            str(args.shots),
            "--seed",
            str(args.seed),
            "--output",
            str(output),
            "--max-width",
            str(args.max_width),
        ]
        if mode in {"conditional_export", "fixed_fault"}:
            command += ["--retained", str(args.output)]
        memory = supervise(command, output)
        result[mode] = (
            packed(output) if output.exists() else dict(status="worker_failure")
        ) | dict(memory=memory)
        print(mode, memory, flush=True)
        save_results(args.output, result)


if __name__ == "__main__":
    main()
