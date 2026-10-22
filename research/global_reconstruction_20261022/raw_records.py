"""Check raw-record specialization and measure its actual retained-output cost."""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from study_io import read_circuit  # type: ignore[import-not-found]
from validate_sampling import features  # type: ignore[import-not-found]

import clifft

ROOT = Path(__file__).parent


if __name__ == "__main__":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    output = []
    for name, shots in [("d5", 32768), ("d7", 4096)]:
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        meta = json.loads((ROOT / f"{name}-records-compressed.json").read_text())
        programs = {}
        observed = []
        source_path = ROOT / "inputs" / f"{name}.stim"
        if not source_path.exists():
            source_path = Path(meta["source"])
        for label, source in [
            ("original", source_path.read_text()),
            ("compressed", read_circuit(ROOT / f"{name}-records-compressed.clifft")),
        ]:
            pm = clifft.default_hir_pass_manager()
            pm.add(clifft.ActiveWidthSchedulePass())
            program = clifft.compile(source, hir_passes=pm)
            programs[label] = program
            r = clifft.sample(
                program, shots=shots, seed=890 if label == "original" else 934, threads=1
            )
            bits = r.measurements if label == "original" else r.detectors
            observed.append(features(bits, bits.shape[1], np.random.default_rng(42)))
        a, b = observed
        pooled = (a.mean(axis=0) + b.mean(axis=0)) / 2
        z = np.abs(a.mean(axis=0) - b.mean(axis=0)) / np.sqrt(
            np.maximum(pooled * (1 - pooled) * 2 / shots, 1 / shots**2)
        )
        assert z.max() < 7, (name, z.max())
        timings: dict = {}
        for label, p in programs.items():
            timings[label] = []
            for repeat in range(3):
                start = time.perf_counter()
                clifft.sample(p, shots=512, seed=50 + repeat, threads=1)
                timings[label].append(time.perf_counter() - start)
        row = dict(
            circuit=name,
            validation_shots=shots,
            max_z=float(z.max()),
            features=len(z),
            widths={k: p.peak_active_width for k, p in programs.items()},
            timed_shots=512,
            seconds=timings,
            speedup=statistics.median(timings["original"])
            / statistics.median(timings["compressed"]),
        )
        output.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "raw_records.json").write_text(json.dumps(output, indent=2) + "\n")
