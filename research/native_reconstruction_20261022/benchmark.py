"""Matched current-main CPU timings, including noise and output production."""

import gzip
import json
import os
import resource
import statistics
import time

from validate import OLD, ROOT, compile_, restore  # type: ignore[import-not-found]

import clifft


def late_outputs(text):
    # Rebase every output declaration at the final record count.
    import re

    body = []
    outputs = []
    count = 0
    for line in text.splitlines():
        gate = line.split()[0]
        if gate == "DETECTOR" or gate.startswith("OBSERVABLE_INCLUDE("):
            outputs.append((gate, [count + int(v) for v in re.findall(r"rec\[(-?\d+)\]", line)]))
        else:
            body.append(line)
            if gate in ("M", "MPP", "MPAD"):
                count += len(line.split()) - 1
    return (
        "\n".join(
            body
            + [gate + " " + " ".join(f"rec[{v - count}]" for v in deps) for gate, deps in outputs]
        )
        + "\n"
    )


def run(name, mode, post, retain):
    meta = json.loads((ROOT / f"{name}-{mode}.json").read_text())
    early = gzip.open(ROOT / f"{name}-{mode}.clifft.gz", "rt").read()
    sources = {
        "original": (OLD / "inputs" / f"{name}.stim").read_text(),
        "native_late": late_outputs(early),
        "native_early": early,
    }
    if mode == "records":
        sources["zx_reference"] = gzip.open(
            OLD / f"{name}-{mode}-compressed.clifft.gz", "rt"
        ).read()
    variants = {}
    compiles = {}
    for label, text in sources.items():
        start = time.perf_counter()
        count = (
            meta["source_detectors"]
            if label == "original"
            else meta["outputs"] - 1
            if mode == "detectors"
            else meta["outputs"]
        )
        variants[label] = compile_(text, [int(post)] * count)
        compiles[label] = time.perf_counter() - start

    def sample(label, shots, seed):
        r = clifft.sample_survivors(
            variants[label], shots=shots, seed=seed, threads=1, keep_records=retain
        )
        if retain and label.startswith("native"):
            restore(r, meta)
        return r

    shots = {
        label: (1024 if name == "d7" and label == "original" and not post else 16384)
        for label in variants
    }
    for label in variants:
        sample(label, 32, 1)
    times: dict = {label: [] for label in variants}
    counts: dict = {label: [] for label in variants}
    labels = list(variants)
    for rep in range(5):
        for label in labels if rep % 2 == 0 else labels[::-1]:
            t = time.perf_counter()
            r = sample(label, shots[label], 870 + rep)
            elapsed = time.perf_counter() - t
            times[label].append(elapsed * 1e6 / shots[label])
            counts[label].append(r.passed_shots)
    return dict(
        name=name,
        mode=mode,
        postselect=post,
        retain=retain,
        shots=shots,
        microseconds_per_attempt=times,
        median_microseconds={k: statistics.median(v) for k, v in times.items()},
        compile_seconds=compiles,
        widths={k: p.peak_active_width for k, p in variants.items()},
        survivors=counts,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


if __name__ == "__main__":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    rows = []
    for name in ("d5", "d7"):
        for mode, post, retain in [
            ("detectors", False, False),
            ("detectors", True, False),
            ("records", False, True),
        ]:
            row = run(name, mode, post, retain)
            rows.append(row)
            print(json.dumps(row), flush=True)
            (ROOT / "benchmark.json").write_text(json.dumps(rows, indent=2) + "\n")
