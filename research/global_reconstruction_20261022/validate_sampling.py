"""Statistical checks on the unchanged native backend and exported benchmarks."""

import json
import os
from pathlib import Path

import numpy as np
from study_io import read_circuit  # type: ignore[import-not-found]

import clifft

ROOT = Path(__file__).parent


def features(bits, detectors, rng):
    values = [bits.astype(np.float64), np.all(bits[:, :detectors] == 0, axis=1)[:, None]]
    for _ in range(64):
        chosen = rng.choice(
            bits.shape[1], min(int(rng.integers(2, 9)), bits.shape[1]), replace=False
        )
        values.append((bits[:, chosen].sum(axis=1) % 2)[:, None])
    return np.concatenate(values, axis=1)


if __name__ == "__main__":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    out = []
    for name, n in [
        ("d3", 65536),
        ("d5", 65536),
        ("d7", 8192),
        ("coherent_d5", 16384),
        ("distillation", 65536),
    ]:
        meta = json.loads((ROOT / f"{name}-detectors-bridge.json").read_text())
        observed = []
        source_path = ROOT / "inputs" / f"{name}.stim"
        if not source_path.exists():
            source_path = Path(meta["source"])
        for label, text in [
            ("original", source_path.read_text()),
            ("compressed", read_circuit(ROOT / f"{name}-detectors-compressed.clifft")),
        ]:
            pm = clifft.default_hir_pass_manager()
            pm.add(clifft.ActiveWidthSchedulePass())
            program = clifft.compile(text, hir_passes=pm)
            r = clifft.sample(
                program, shots=n, seed=4701 if label == "original" else 7709, threads=1
            )
            bits = np.concatenate([r.detectors, r.observables], axis=1)
            observed.append(
                features(bits, meta["num_original_detectors"], np.random.default_rng(42))
            )
        left, right = observed
        pooled = (left.mean(axis=0) + right.mean(axis=0)) / 2
        stderr = np.sqrt(np.maximum(pooled * (1 - pooled) * 2 / n, 1 / n**2))
        z = np.abs(left.mean(axis=0) - right.mean(axis=0)) / stderr
        result = dict(
            circuit=name,
            shots_per_sampler=n,
            features=len(z),
            max_z=float(z.max()),
            acceptance_original=float(left[:, meta["outputs"]].mean()),
            acceptance_compressed=float(right[:, meta["outputs"]].mean()),
            passed=bool(z.max() < 7),
        )
        out.append(result)
        print(json.dumps(result), flush=True)
        assert result["passed"], result
    (ROOT / "sampling_validation.json").write_text(json.dumps(out, indent=2) + "\n")
