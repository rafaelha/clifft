"""Measure the research bridge with an unchanged Clifft native backend."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from study_io import read_circuit  # type: ignore[import-not-found]

import clifft

ROOT = Path(__file__).parent


def compile_source(text, postselect, detectors):
    pm = clifft.default_hir_pass_manager()
    pm.add(clifft.ActiveWidthSchedulePass())
    return clifft.compile(text, hir_passes=pm, postselection_mask=[int(postselect)] * detectors)


def run(name, shots_cap=2000, suffix="bridge"):
    meta = json.loads((ROOT / f"{name}-detectors-bridge.json").read_text())
    source_path = ROOT / "inputs" / f"{name}.stim"
    if not source_path.exists():
        source_path = Path(meta["source"])
    sources = {
        "original": source_path.read_text(),
        "bridge": read_circuit(ROOT / f"{name}-detectors-{suffix}.clifft"),
    }
    for postselect in [False, True]:
        programs = {}
        compiles = {}
        for label, source in sources.items():
            start = time.perf_counter()
            programs[label] = compile_source(source, postselect, meta["num_original_detectors"])
            compiles[label] = time.perf_counter() - start
        for batch in [1, "auto"]:

            def sample(p, n, seed):
                return clifft.sample_survivors(
                    p, shots=n, seed=seed, threads=1, batch_size=batch, keep_records=False
                )

            pilot = {}
            for label, program in programs.items():
                start = time.perf_counter()
                sample(program, 3, 1)
                pilot[label] = (time.perf_counter() - start) / 3
            shots = min(shots_cap, max(3, int(1.0 / max(pilot.values()))))
            times: dict = {label: [] for label in programs}
            survivors = {}
            for repeat in range(3):
                for label, program in programs.items():
                    start = time.perf_counter()
                    result = sample(program, shots, 400 + repeat)
                    times[label].append(time.perf_counter() - start)
                    survivors[label] = result.passed_shots
            result = dict(
                circuit=name,
                postselection=postselect,
                batch=batch,
                shots=shots,
                version=clifft.__version__,
                compile_seconds=compiles,
                widths={label: p.peak_active_width for label, p in programs.items()},
                seconds=times,
                median_seconds={k: statistics.median(v) for k, v in times.items()},
                survivor_counts_last_repeat=survivors,
            )
            result["speedup"] = (
                result["median_seconds"]["original"] / result["median_seconds"]["bridge"]
            )
            with (ROOT / f"{suffix}_benchmark.jsonl").open("a") as f:
                f.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names", nargs="*", default=["d5", "d7", "coherent_d5", "d3", "coherent_d3", "distillation"]
    )
    parser.add_argument("--suffix", default="bridge")
    args = parser.parse_args()
    for name in args.names:
        run(name, suffix=args.suffix)
