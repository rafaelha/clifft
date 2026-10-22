"""Validate the earlier-check control and time unrestricted parameter draws."""

import json
import os
import statistics
import time

import numpy as np
from early_bridge import ROOT, early  # type: ignore[import-not-found]
from native import Native  # type: ignore[import-not-found]
from validate import features  # type: ignore[import-not-found]

import clifft

if __name__ == "__main__":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    results = []
    for name in ["d5", "d7"]:
        native = Native(ROOT / f"{name}-detectors.plan")
        text, mapping = early(name)
        (ROOT / f"{name}-early.clifft").write_text(text)
        (ROOT / f"{name}-early-map.json").write_text(json.dumps(mapping) + "\n")
        pm = clifft.default_hir_pass_manager()
        pm.add(clifft.ActiveWidthSchedulePass())
        p = clifft.compile(text, hir_passes=pm)
        n = 65536
        r = clifft.sample(p, shots=n, seed=3019, threads=1)
        restored = np.empty_like(r.detectors)
        restored[:, mapping] = r.detectors
        ref = np.concatenate([restored, r.observables], axis=1)
        _, actual, _ = native.sample(n, seed=3119, retain=True)
        a = features(actual, native.detectors, np.random.default_rng(73))
        b = features(ref, native.detectors, np.random.default_rng(73))
        pool = (a.mean(0) + b.mean(0)) / 2
        z = abs(a.mean(0) - b.mean(0)) / np.sqrt(np.maximum(pool * (1 - pool) * 2 / n, 1 / n**2))
        assert z.max() < 7
        p = clifft.compile(text, hir_passes=pm, postselection_mask=[1] * native.detectors)
        post_n = 1048576
        r = clifft.sample_survivors(p, shots=post_n, seed=8893, threads=1, keep_records=False)
        passed, _, counts = native.sample(post_n, seed=9901, postselect=True)
        post_z = []
        for x, y in zip([passed, *counts], [r.passed_shots, *r.observable_ones]):
            pool = (float(x) + float(y)) / (2 * post_n)
            post_z.append(
                abs(float(x) - float(y)) / np.sqrt(max(2 * post_n * pool * (1 - pool), 1))
            )
        assert max(post_z) < 7
        # These are unrestricted algebraic assignments, not a physical noise model.
        rng = np.random.default_rng(71)
        parameters = rng.integers(0, 2**64 - 1, size=(32768, native.words), dtype=np.uint64)
        elapsed = []
        for i in range(5):
            start = time.perf_counter()
            native.sample(len(parameters), seed=450 + i, parameters=parameters, retain=True)
            elapsed.append((time.perf_counter() - start) * 1e6 / len(parameters))
        row = dict(
            circuit=name,
            control="early_reconstructed_checks",
            shots=n,
            joint_max_z=float(z.max()),
            postselected_shots=post_n,
            postselected_z=post_z,
            native_passed=passed,
            reference_passed=int(r.passed_shots),
            native_observable_ones=counts.tolist(),
            reference_observable_ones=r.observable_ones.tolist(),
            unrestricted_parameter_microseconds=elapsed,
            unrestricted_parameter_median_microseconds=statistics.median(elapsed),
        )
        results.append(row)
        print(json.dumps(row), flush=True)
        (ROOT / "controls.json").write_text(json.dumps(results, indent=2) + "\n")
