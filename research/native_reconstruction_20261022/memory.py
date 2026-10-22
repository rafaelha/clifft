"""Fresh-process setup and execution peak RSS, including imports and outputs."""

import argparse
import json
import os
import resource
import time

from export import export  # type: ignore[import-not-found]
from validate import OLD, compile_, restore  # type: ignore[import-not-found]

import clifft

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("mode")
    parser.add_argument("variant")
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    source = (OLD / "inputs" / f"{args.name}.stim").read_text()
    t = time.perf_counter()
    frontend = 0.0
    if args.variant == "native":
        source, meta, _ = export(source, args.mode)
        frontend = time.perf_counter() - t
    t = time.perf_counter()
    p = compile_(source)
    compilation = time.perf_counter() - t
    setup_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r = clifft.sample_survivors(
        p, shots=4096, seed=10, threads=1, keep_records=args.mode == "records"
    )
    if args.mode == "records" and args.variant == "native":
        restore(r, meta)
    print(
        json.dumps(
            dict(
                name=args.name,
                mode=args.mode,
                variant=args.variant,
                import_peak_kib=start_rss,
                setup_peak_kib=setup_rss,
                sampling_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                frontend_seconds=frontend,
                backend_compile_seconds=compilation,
                shots=4096,
                width=p.peak_active_width,
            )
        )
    )
