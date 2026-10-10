"""Bounded FEC consumer of the existing folded state before logical readout.

This offline experiment adds no general output interface or production mode.
The complete input is the reconstructed physical schedule plus the pinned
decoder policy, including its recovery before the final logical measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from complete_circuit_panel import BASELINE, FOLDED, save, verify_reference


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    verify_reference(args.reference)
    revision = subprocess.check_output(
        ["git", "-C", str(args.baseline_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != BASELINE:
        raise ValueError("baseline source revision differs from the pinned comparison")
    subprocess.run(
        ["git", "-C", str(args.baseline_source), "diff", "--exit-code", BASELINE, "--", "src"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    source = args.reference / "tools/profile/sample_folded_protocol.cpp"
    text = source.read_text()
    insertions = {
        "class FoldedBlock {": '#include "folded_fec_boundary.h"\n\nclass FoldedBlock {',
        "        bind(final_, boundary);": (
            "        recover_folded_fec(boundary, records, x_rows_, z_rows_, "
            "logical_x_, logical_z_, n_);\n        bind(final_, boundary);"
        ),
    }
    args.work.mkdir(parents=True, exist_ok=True)
    for anchor, replacement in insertions.items():
        if text.count(anchor) != 1:
            raise ValueError("pinned boundary insertion no longer unique")
        text = text.replace(anchor, replacement)
    overlay = args.work / "sample_folded_protocol.cpp"
    overlay.write_text(text)
    local = Path(__file__).resolve().parent
    observer = args.work / "retain.cpp"
    observer.write_text((local / "retain_folded_attempts.cpp").read_text())
    command = [
        "c++",
        "-std=c++20",
        "-O3",
        "-march=native",
        "-fopenmp",
        "-I" + str(args.baseline_source / "src"),
        "-I" + str(args.baseline_build / "generated"),
        "-I" + str(args.work),
        "-I" + str(local),
        "-I" + str(args.reference / "tools/profile"),
        str(observer),
        str(args.baseline_build / "src/clifft/libclifft_core.a"),
        "-o",
        str(args.work / "folded_fec"),
    ]
    subprocess.run(command, check=True)
    save(
        args.work / "build.json",
        dict(
            baseline=BASELINE,
            folded=FOLDED,
            command=command,
            original_sha256=sha(source),
            overlay_sha256=sha(overlay),
            boundary_sha256=sha(local / "folded_fec_boundary.h"),
            observer_sha256=sha(observer),
            executable_sha256=sha(args.work / "folded_fec"),
            core_sha256=sha(args.baseline_build / "src/clifft/libclifft_core.a"),
        ),
    )


def fec_circuit(distance, probability):
    make_circuit = importlib.import_module("folded_msc_family").make_circuit

    circuit = make_circuit(probability, distance=distance)
    # Fictitious checks supply decoder input; only earlier protocol checks reject.
    circuit.measurements = [
        dict(m, postselect=False) if m["stage"] == "final_syndrome" else m.copy()
        for m in circuit.measurements
    ]
    return circuit


class Recovery:
    def __init__(self, circuit, distance):
        import numpy as np

        pymatching = importlib.import_module("pymatching")
        Surface = importlib.import_module("folded_msc_family").Surface

        self.np = np
        self.version = pymatching.__version__
        self.slots = [m["index"] for m in circuit.measurements if m["stage"] == "final_syndrome"]
        surface = Surface(distance)
        self.matrices, self.matchers = {}, {}
        for axis in ("X", "Z"):
            matrix = np.zeros((len(surface.checks(axis)), len(surface.points)), dtype=np.uint8)
            for row, support in enumerate(surface.checks(axis)):
                matrix[row, list(support)] = 1
            self.matrices[axis] = matrix
            self.matchers[axis] = pymatching.Matching.from_check_matrix(matrix, weights=1.0)

    def decode(self, records):
        syndrome = self.np.array([int(records[i]) for i in self.slots], dtype=self.np.uint8)
        sx, sz = self.np.split(syndrome, 2)
        z = self.matchers["X"].decode(sx)
        x = self.matchers["Z"].decode(sz)
        if not self.np.array_equal(self.matrices["X"] @ z % 2, sx) or not self.np.array_equal(
            self.matrices["Z"] @ x % 2, sz
        ):
            raise ValueError("decoder returned an invalid syndrome recovery")

        def bits(a):
            return "".join(map(str, map(int, a)))

        return dict(x=bits(x), z=bits(z), x_syndrome=bits(sx), z_syndrome=bits(sz))


def physical_recovery(circuit, x, z):
    Operation = importlib.import_module("folded_msc").Operation

    cut = next(i for i, op in enumerate(circuit.operations) if op.stage == "final_logical")
    operations = [
        Operation(axis, (q,), "fec_recovery", "retained decoder result")
        for axis, bits in (("X", x), ("Z", z))
        for q, bit in enumerate(bits)
        if bit == "1"
    ]
    return replace(
        circuit, operations=circuit.operations[:cut] + operations + circuit.operations[cut:]
    )


def run(args):
    compile_protocol = importlib.import_module("compile_folded_protocol").compile_protocol

    args.work.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    circuit = fec_circuit(args.distance, args.probability)
    decoder = Recovery(circuit, args.distance)
    metadata = compile_protocol(
        circuit, args.distance, args.work / "bundle", native_growth=args.distance == 7
    )
    setup = time.perf_counter() - begun
    policy = dict(
        package="pymatching",
        version=decoder.version,
        weights="unit weight per data qubit",
        decoding="separate CSS minimum-weight matching on final noiseless syndrome only",
        order="physical X recovery then Z recovery before final H_XY",
        syndrome_slots=decoder.slots,
        x_checks=decoder.matrices["X"].tolist(),
        z_checks=decoder.matrices["Z"].tolist(),
        acceptance="original noisy-protocol predicate; final syndrome does not reject",
    )
    result = dict(
        status="running",
        distance=args.distance,
        probability=args.probability,
        seed=args.seed,
        baseline=BASELINE,
        folded=FOLDED,
        source=circuit.text(),
        source_sha256=hashlib.sha256(circuit.text().encode()).hexdigest(),
        decoder_policy=policy,
        circuit_metadata=metadata,
        setup_seconds=setup,
        executable_sha256=sha(args.executable),
        attempts=[],
        certificate_failures=[],
    )
    # Preserve an interrupted attempt rather than silently replacing it.
    journal = args.work / "events.jsonl"
    command = [str(args.executable), str(args.work / "bundle"), str(args.shots), str(args.seed)]
    with (
        journal.open("w") as log,
        subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        ) as child,
    ):
        assert child.stdin is not None and child.stdout is not None
        pending = None
        try:
            for line in child.stdout:
                event = json.loads(line)
                log.write(line)
                log.flush()
                if event["kind"] == "recovery":
                    started = time.perf_counter()
                    recovery = decoder.decode(event["records"])
                    for key in ("x_syndrome", "z_syndrome"):
                        if recovery[key] != event[key]:
                            raise ValueError(
                                "live state differs from actual final syndrome records"
                            )
                    pending = event | recovery
                    pending["decode_seconds"] = time.perf_counter() - started
                    child.stdin.write(recovery["x"] + " " + recovery["z"] + "\n")
                    child.stdin.flush()
                elif event["kind"] == "recovered":
                    if pending is None:
                        raise ValueError("recovery response without a request")
                    pending.update(event)
                elif event["kind"] == "attempt":
                    if pending is None:
                        raise ValueError("logical readout occurred without live recovery")
                    result["attempts"].append(event | dict(recovery=pending))
                    pending = None
                else:
                    result[event["kind"]] = event
            child.wait()
            if child.returncode or len(result["attempts"]) != args.shots:
                raise ValueError("native consumer failed or lost attempts")
            result["status"] = "complete"
        except Exception as error:
            result["status"] = "certificate_failure"
            result["certificate_failures"].append(str(error))
            result["unfinished_recovery"] = pending
            child.kill()
            raise
        finally:
            result["wall_seconds"] = time.perf_counter() - begun
            save(args.output, result)
    print(
        json.dumps(
            dict(distance=args.distance, status=result["status"], attempts=len(result["attempts"]))
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    for name in ("reference", "work", "baseline-source", "baseline-build"):
        build_parser.add_argument("--" + name, type=Path, required=True)
    run_parser = commands.add_parser("run")
    for name in ("work", "executable", "output"):
        run_parser.add_argument("--" + name, type=Path, required=True)
    run_parser.add_argument("--distance", type=int, choices=(3, 7), required=True)
    run_parser.add_argument("--probability", type=float, default=0.001)
    run_parser.add_argument("--shots", type=int, default=16)
    run_parser.add_argument("--seed", type=int, default=10102026)
    args = parser.parse_args()
    (build if args.command == "build" else run)(args)


if __name__ == "__main__":
    main()
