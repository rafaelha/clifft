"""Interleaved single-core timings including noise, sampling and result creation."""

import json
import os
import statistics
import sys
import time
from pathlib import Path

from early_bridge import early  # type: ignore[import-not-found]
from native import Native  # type: ignore[import-not-found]

ROOT = Path(__file__).parent
STUDY = ROOT.parent / "global_reconstruction_20261022"
sys.path.insert(0, str(STUDY))
from study_io import read_circuit  # type: ignore[import-not-found]  # noqa: E402

import clifft  # noqa: E402


def run(name, mode, postselect=False, retain=False):
    path = ROOT / f"{name}-{mode}.plan"
    start = time.perf_counter()
    native = Native(path)
    load = time.perf_counter() - start
    variants = {}
    compiles = {}
    for label in ["original", "reconstructed"] + (["reconstructed_early"] if postselect else []):
        text = (
            (STUDY / "inputs" / f"{name}.stim").read_text()
            if label == "original"
            else early(name)[0]
            if label == "reconstructed_early"
            else read_circuit(STUDY / f"{name}-{mode}-compressed.clifft")
        )
        pm = clifft.default_hir_pass_manager()
        pm.add(clifft.ActiveWidthSchedulePass())
        start = time.perf_counter()
        variants[label] = clifft.compile(
            text,
            hir_passes=pm,
            postselection_mask=[int(postselect)] * native.detectors
            if mode == "detectors"
            else None,
        )
        compiles[label] = time.perf_counter() - start

    def sample(label, n, seed):
        if label == "algebra":
            return native.sample(n, seed=seed, postselect=postselect, retain=retain)
        if label == "noise_only":
            return native.sample(n, seed=seed, noise_only=True)
        if retain:
            return clifft.sample(variants[label], shots=n, seed=seed, threads=1)
        return clifft.sample_survivors(
            variants[label], shots=n, seed=seed, threads=1, keep_records=False
        )

    labels = [*variants, "algebra", "noise_only"]
    shots = {
        label: (1024 if label == "original" and name == "d7" and not postselect else 16384)
        for label in labels
    }
    times: dict = {label: [] for label in labels}
    for label in labels:
        sample(label, 32, 1)
    for repeat in range(5):
        order = labels if repeat % 2 == 0 else labels[::-1]
        for label in order:
            start = time.perf_counter()
            sample(label, shots[label], 350 + repeat)
            times[label].append((time.perf_counter() - start) * 1e6 / shots[label])
    medians = {label: statistics.median(t) for label, t in times.items()}
    return dict(
        circuit=name,
        mode=mode,
        postselection=postselect,
        retain=retain,
        shots=shots,
        microseconds_per_shot=times,
        median_microseconds=medians,
        algebra_speedup_over_reconstructed=medians["reconstructed"] / medians["algebra"],
        algebra_speedup_over_original=medians["original"] / medians["algebra"],
        compile_seconds=compiles,
        algebra_load_seconds=load,
        widths={label: p.peak_active_width for label, p in variants.items()},
        version=clifft.__version__,
    )


if __name__ == "__main__":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    results = []
    for name in ["d5", "d7"]:
        for mode, postselect, retain in [
            ("detectors", False, False),
            ("detectors", True, False),
            ("detectors", False, True),
            ("records", False, True),
        ]:
            row = run(name, mode, postselect, retain)
            results.append(row)
            print(json.dumps(row), flush=True)
            (ROOT / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
