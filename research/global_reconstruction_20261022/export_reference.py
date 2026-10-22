"""Export a closed tsim sampling representation into ordinary Clifft gates.

This is a research bridge, not a dependency-free reconstruction implementation.
The error channels are sampled before the virtual quantum component. Requested
outputs are exposed as detectors, with an explicit map in the metadata.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tsim  # type: ignore[import-not-found]
from tsim.compile.pipeline import classify_direct  # type: ignore[import-not-found]
from tsim.core.graph import prepare_graph  # type: ignore[import-not-found]
from tsim.statevector.pbc import ComponentLayout, read_pauli_ops  # type: ignore[import-not-found]
from tsim.statevector.schedule import schedule  # type: ignore[import-not-found]

ROOT = Path(__file__).parent
PATHS = {
    "d5": Path("/home/exedev/tsim-study/circuits/clifft-bench/msc_d5_inject_cultivate_p1e-3.stim"),
    "d7": Path("/home/exedev/tsim-study/circuits/symft/MSC_circuit_d7_p0.0005.stim"),
    "d3": Path("/home/exedev/tsim-study/circuits/clifft-bench/msc_d3_inject_cultivate_p1e-3.stim"),
    "coherent_d3": Path("/home/exedev/tsim-study/circuits/clifft-bench/coherent_d3_r3.stim"),
    "coherent_d5": Path("/home/exedev/tsim-study/circuits/clifft-bench/coherent_d5_r5.stim"),
    "distillation": Path("/home/exedev/tsim-study/circuits/clifft-bench/distillation.stim"),
}
for name in PATHS:
    if (ROOT / "inputs" / f"{name}.stim").exists():
        PATHS[name] = ROOT / "inputs" / f"{name}.stim"


class Writer:
    def __init__(self):
        self.lines = []
        self.records = 0

    def add(self, line):
        self.lines.append(line)

    def ref(self, i):
        assert i < self.records
        return f"rec[{i - self.records}]"

    def measure(self, line, count=1):
        self.add(line)
        start = self.records
        self.records += count
        return start


def export(name, path, mode, compressed=False):
    start = time.perf_counter()
    circuit = tsim.Circuit.from_file(str(path))
    prepared = prepare_graph(circuit, sample_detectors=mode == "detectors")
    components = []
    direct = []
    max_lines = 1
    for component in sorted(prepared.graph.connected_components(), key=lambda c: len(c.outputs())):
        outputs = [component.port(v) for v in component.outputs()]
        entry = classify_direct(component, prepared.layout)
        if entry is not None:
            direct.append((outputs[0], *entry))
            continue
        layout = ComponentLayout.of(component, outputs, prepared.layout)
        ops = read_pauli_ops(component, layout)
        if ops.num_ops:
            scheduled = schedule(ops)
            components.append((ops, layout, scheduled))
            max_lines = max(max_lines, ops.num_lines)
    graph_seconds = time.perf_counter() - start
    writer = Writer()
    error_records: list = []
    q = max_lines
    channels = prepared.channel_probs
    if compressed:
        # Independent channels with identical image subgroups can be convolved.
        # This preserves ordinary sampling but not fixed-fault-count sampling.
        from tsim.statevector.pbc import _mask

        grouped = {}
        start_bit = 0
        for probs in channels:
            bits = len(probs).bit_length() - 1
            columns = [
                _mask(c) for c in prepared.error_transform[:, start_bit : start_bit + bits].T
            ]
            start_bit += bits
            distribution: dict = {}
            for value, probability in enumerate(probs):
                mask = 0
                for b in range(bits):
                    if value >> b & 1:
                        mask ^= columns[b]
                distribution[mask] = distribution.get(mask, 0) + float(probability)
            key = tuple(sorted(distribution))
            if len(key) == 1:
                continue
            if key not in grouped:
                grouped[key] = distribution
            else:
                merged: dict = {}
                for x, p in grouped[key].items():
                    for y, r in distribution.items():
                        merged[x ^ y] = merged.get(x ^ y, 0) + p * r
                grouped[key] = merged
        if len(prepared.error_transform):
            writer.add("R " + " ".join(str(q + i) for i in range(len(prepared.error_transform))))
        for distribution in grouped.values():
            used = 0.0
            first = True
            for mask, probability in sorted(distribution.items()):
                if not mask or not probability:
                    continue
                p = probability / (1 - used)
                used += probability
                targets = " ".join(f"X{q + i}" for i in range(mask.bit_length()) if mask >> i & 1)
                writer.add(f"{'E' if first else 'ELSE_CORRELATED_ERROR'}({p:.17g}) {targets}")
                first = False
        start_rec = writer.records
        if len(prepared.error_transform):
            writer.measure(
                "M " + " ".join(str(q + i) for i in range(len(prepared.error_transform))),
                len(prepared.error_transform),
            )
        f_records = list(range(start_rec, writer.records))
        channels = []
    for probs in channels:
        bits = len(probs).bit_length() - 1
        writer.add("R " + " ".join(str(q + i) for i in range(bits)))
        used = 0.0
        first = True
        for value in range(1, len(probs)):
            if probs[value] == 0:
                continue
            p = float(probs[value]) / (1 - used)
            used += float(probs[value])
            targets = " ".join(f"X{q + i}" for i in range(bits) if value >> i & 1)
            writer.add(f"{'E' if first else 'ELSE_CORRELATED_ERROR'}({p:.17g}) {targets}")
            first = False
        rec = writer.measure("M " + " ".join(str(q + i) for i in range(bits)), bits)
        error_records.extend(range(rec, rec + bits))
    if not compressed:
        f_records = []
    for row in [] if compressed else prepared.error_transform:
        writer.add(f"R {q}")
        for i in np.flatnonzero(row):
            writer.add(f"CX {writer.ref(error_records[i])} {q}")
        f_records.append(writer.measure(f"M {q}"))
    writer.add(f"R {q}")
    writer.add(f"X {q}")
    one = writer.measure(f"M {q}")
    output_parities = [None] * prepared.num_outputs
    for output, f, flip in direct:
        output_parities[output] = [f_records[f]] + ([one] if flip else [])
    component_meta = []
    for k, (ops, layout, scheduled) in enumerate(components):
        if ops.num_lines:
            writer.add("R " + " ".join(map(str, range(ops.num_lines))))
        for j in scheduled.order:
            letters = ["I", "X", "Z", "Y"]
            axes = [
                (i, letters[int(x) + 2 * int(z)])
                for i, (x, z) in enumerate(zip(ops.x[j], ops.z[j]))
                if x or z
            ]
            term = "*".join(f"{p}{i}" for i, p in axes)
            signs = [f_records[layout.f_selection[t]] for t in np.flatnonzero(ops.sign[j, 1:])]
            if ops.kind[j] == 0:
                pivot, pauli = axes[0]
                feedback = "CZ" if pauli in ("X", "Y") else "CX"
                for rec in signs:
                    writer.add(f"{feedback} {writer.ref(rec)} {pivot}")
                angle = ops.angle[j] * 2 / np.pi * (-1 if ops.sign[j, 0] else 1)
                writer.add(f"R_PAULI({angle:.17g}) {term}")
                for rec in signs:
                    writer.add(f"{feedback} {writer.ref(rec)} {pivot}")
            else:
                rec = writer.measure("MPP " + term) if axes else writer.measure("MPAD 0")
                if ops.slot[j] >= 0:
                    output_parities[layout.output_indices[ops.slot[j]]] = (
                        [rec] + signs + ([one] if ops.sign[j, 0] else [])
                    )
        np.savez(
            ROOT / f"{name}-{mode}-component{k}.npz",
            **vars(ops),
            error_transform=prepared.error_transform[np.array(layout.f_selection, dtype=int)],
            output_indices=np.array(layout.output_indices),
            order=scheduled.order,
        )
        component_meta.append(
            dict(
                lines=ops.num_lines,
                width=scheduled.width,
                rotations=int((ops.kind == 0).sum()),
                outputs=ops.num_outputs,
            )
        )
    assert all(p is not None for p in output_parities)
    for output_index, parity in enumerate(output_parities):
        assert parity is not None
        gate = "DETECTOR"
        if mode == "detectors" and output_index >= prepared.num_detectors:
            gate = f"OBSERVABLE_INCLUDE({output_index - prepared.num_detectors})"
        writer.add(gate + " " + " ".join(writer.ref(i) for i in parity))
    suffix = "compressed" if compressed else "bridge"
    output = ROOT / f"{name}-{mode}-{suffix}.clifft"
    output.write_text("\n".join(writer.lines) + "\n")
    metadata = dict(
        name=name,
        mode=mode,
        source=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        outputs=prepared.num_outputs,
        num_original_detectors=prepared.num_detectors,
        error_channels=len(prepared.channel_probs),
        error_bits=prepared.error_transform.shape[1],
        f_bits=len(f_records),
        bridge_records=writer.records,
        components=component_meta,
        prepare_seconds=graph_seconds,
        total_seconds=time.perf_counter() - start,
    )
    if compressed:
        metadata["compressed_channels"] = len(grouped)
        metadata["contract"] = (
            "ordinary sampling only; fault identities and counts are not preserved"
        )
    (ROOT / f"{name}-{mode}-{suffix}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names", nargs="*", default=["d5", "d7", "d3", "coherent_d3", "coherent_d5", "distillation"]
    )
    parser.add_argument("--modes", nargs="+", default=["detectors", "records"])
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        for mode in args.modes:
            export(name, PATHS[name], mode, args.compressed)
