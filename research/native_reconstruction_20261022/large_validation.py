"""Focused original-source versus reconstructed sampling checks."""

import gzip
import json
import os

import numpy as np
from validate import OLD, ROOT, compile_, restore  # type: ignore[import-not-found]

import clifft


def features(bits, detectors):
    rng = np.random.default_rng(550)
    values = [bits.astype(float), np.all(bits[:, :detectors] == 0, axis=1)[:, None]]
    for _ in range(128):
        chosen = rng.choice(
            bits.shape[1], min(int(rng.integers(2, 13)), bits.shape[1]), replace=False
        )
        values.append((bits[:, chosen].sum(axis=1) % 2)[:, None])
    return np.concatenate(values, axis=1)


def run():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    out = []
    for name, shots in [("d5", 32768), ("d7", 8192)]:
        original = compile_((OLD / "inputs" / f"{name}.stim").read_text())
        source = clifft.sample(original, shots=shots, seed=7701, threads=1)
        for mode in ["detectors", "records"]:
            meta = json.loads((ROOT / f"{name}-{mode}.json").read_text())
            p = compile_(gzip.open(ROOT / f"{name}-{mode}.clifft.gz", "rt").read())
            r = clifft.sample(p, shots=shots, seed=7709, threads=1)
            a = (
                source.measurements
                if mode == "records"
                else np.concatenate([source.detectors, source.observables], axis=1)
            )
            b = restore(r, meta)
            count = a.shape[1] if mode == "records" else source.detectors.shape[1]
            left, right = features(a, count), features(b, count)
            pooled = (left.mean(0) + right.mean(0)) / 2
            z = abs(left.mean(0) - right.mean(0)) / np.sqrt(
                np.maximum(pooled * (1 - pooled) * 2 / shots, 1 / shots**2)
            )
            row = dict(
                name=name,
                mode=mode,
                shots=shots,
                width_original=original.peak_active_width,
                width_native=p.peak_active_width,
                max_z=float(z.max()),
                features=len(z),
                all_zero_original=float(left[:, a.shape[1]].mean()),
                all_zero_native=float(right[:, b.shape[1]].mean()),
            )
            print(json.dumps(row), flush=True)
            out.append(row)
            assert z.max() < 7, row
            (ROOT / "large_validation.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    run()
