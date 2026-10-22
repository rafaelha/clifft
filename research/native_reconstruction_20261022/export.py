"""Lower independently reconstructed operations into unchanged Clifft gates."""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from frontend import Basis, Dilation, bits, unpack  # type: ignore[import-not-found]
from planning import hidden_measurements, schedule  # type: ignore[import-not-found]

ROOT = Path(__file__).parent
INPUTS = ROOT.parent / "global_reconstruction_20261022" / "inputs"


def fault_basis(operations, error_bits):
    basis = Basis()
    rows: list = []
    for op in sorted(operations, key=lambda o: o.slot < 0):
        f = op.sign >> 1
        if basis.solve(f) is None:
            basis.insert(f, 1 << len(rows))
            rows.append(f)
    return basis, rows


def channel_images(channels, rows, compress=True):
    grouped: dict = {}
    columns = {}
    for channel in channels:
        d: dict = {}
        for p, value in channel:
            image = 0
            for bit in bits(value):
                if bit not in columns:
                    columns[bit] = sum(((r >> bit) & 1) << i for i, r in enumerate(rows))
                image ^= columns[bit]
            d[image] = d.get(image, 0.0) + p
        if len(d) == 1 and 0 in d:
            continue
        key = tuple(sorted(d)) if compress else len(grouped)
        if key in grouped:
            if len(grouped[key]) * len(d) > 1_000_000:
                raise ValueError("research channel convolution budget exceeded")
            merged: dict = {}
            for x, p in grouped[key].items():
                for y, q in d.items():
                    merged[x ^ y] = merged.get(x ^ y, 0.0) + p * q
            d = merged
        if len(d) > 4096:
            raise ValueError("research channel support budget exceeded")
        grouped[key] = d
    return list(grouped.values())


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


def export(text, mode="detectors", fixed_fault=None, compress=True, hidden=True, early=True):
    start = time.perf_counter()
    d = Dilation.parse(text)
    parse_seconds = time.perf_counter() - start
    n, ops = d.reconstruct(mode)
    if hidden:
        ops = hidden_measurements(n, ops)
    order, width = schedule(n, ops)
    basis, rows = fault_basis(ops, d.error_bits)
    channels = channel_images(d.channels, rows, compress) if fixed_fault is None else []
    reconstruction_seconds = time.perf_counter() - start
    w = Writer()
    q = n
    if rows:
        w.add("R " + " ".join(str(q + i) for i in range(len(rows))))
    if fixed_fault is None:
        for distribution in channels:
            used = 0.0
            first = True
            for value, p in sorted(distribution.items()):
                if not value or not p:
                    continue
                conditional = p / (1 - used)
                used += p
                targets = " ".join(f"X{q + i}" for i in bits(value))
                w.add(f"{'E' if first else 'ELSE_CORRELATED_ERROR'}({conditional:.17g}) {targets}")
                first = False
    else:
        fixed_targets = [
            str(q + i) for i, r in enumerate(rows) if (r & fixed_fault).bit_count() % 2
        ]
        if fixed_targets:
            w.add("X " + " ".join(fixed_targets))
    base = w.records
    if rows:
        w.measure("M " + " ".join(str(q + i) for i in range(len(rows))), len(rows))
    f_records = list(range(base, w.records))
    w.add(f"R {q}")
    w.add(f"X {q}")
    one = w.measure(f"M {q}")
    if n:
        w.add("R " + " ".join(map(str, range(n))))
    output_count = (
        len(d.records)
        if mode == "records"
        else len(d.detectors) + max(d.observables, default=-1) + 1
    )
    output_parities: list[list[int] | None] = [None] * output_count
    for j in order:
        op = ops[j]
        axis = "*".join(
            f"{'IXZY'[((op.x >> i) & 1) + 2 * ((op.z >> i) & 1)]}{i}" for i in bits(op.x | op.z)
        )
        f = basis.solve(op.sign >> 1)
        signs = [f_records[i] for i in bits(f)]
        if op.angle:
            if not axis:
                # A scalar Pauli rotation contributes only a global phase.
                continue
            pivot = next(bits(op.x | op.z))
            feedback = "CZ" if op.x >> pivot & 1 else "CX"
            for rec in signs:
                w.add(f"{feedback} {w.ref(rec)} {pivot}")
            w.add(f"R_PAULI({op.angle * 2 / np.pi * (-1 if op.sign & 1 else 1):.17g}) {axis}")
            for rec in signs:
                w.add(f"{feedback} {w.ref(rec)} {pivot}")
        else:
            if not axis:
                parity = signs + ([one] if op.sign & 1 else [])
            else:
                rec = w.measure("MPP " + axis)
                parity = [rec] + signs + ([one] if op.sign & 1 else [])
            if op.slot >= 0:
                output_parities[op.slot] = parity
    # Declare outputs in source order before the optional early-check permutation.
    for i, output_parity in enumerate(output_parities):
        assert output_parity is not None
        gate = (
            f"OBSERVABLE_INCLUDE({i - len(d.detectors)})"
            if mode == "detectors" and i >= len(d.detectors)
            else "DETECTOR"
        )
        w.add(gate + " " + " ".join(w.ref(r) for r in output_parity))
    emitted = "\n".join(w.lines) + "\n"
    # A separate event-ordering control may permute detector declarations; the
    # stored map restores their public indices and the postselection mask.
    permutation = list(range(len(d.detectors) if mode == "detectors" else output_count))
    if early:
        emitted, permutation = early_outputs(emitted)
    meta = dict(
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        mode=mode,
        lines=n,
        estimated_width=width,
        hidden=sum(o.hidden for o in ops),
        rotations=sum(bool(o.angle) for o in ops),
        dilation_lines=d.n,
        source_records=len(d.records),
        outputs=output_count,
        source_detectors=len(d.detectors),
        error_bits=d.error_bits,
        error_channels=len(d.channels),
        f_bits=len(rows),
        compressed_channels=len(channels),
        parse_seconds=parse_seconds,
        reconstruction_seconds=reconstruction_seconds,
        total_seconds=time.perf_counter() - start,
        detector_permutation=permutation,
        imports_zx=False,
    )
    return emitted, meta, (d, n, ops, order, rows)


def early_outputs(text):
    operations = []
    outputs = []
    records = 0
    index = 0
    import re

    for line in text.splitlines():
        gate = line.split()[0]
        if gate == "DETECTOR" or gate.startswith("OBSERVABLE_INCLUDE("):
            deps = [records + int(i) for i in re.findall(r"rec\[(-?\d+)\]", line)]
            outputs.append((gate, deps, index if gate == "DETECTOR" else -1))
            index += gate == "DETECTOR"
        else:
            count = len(line.split()) - 1 if gate in ("M", "MPP", "MPAD") else 0
            operations.append((line, count))
            records += count
    result = []
    permutation = []
    available = 0

    def flush():
        nonlocal outputs
        pending = []
        for gate, deps, slot in outputs:
            if all(i < available for i in deps):
                result.append(gate + " " + " ".join(f"rec[{i - available}]" for i in deps))
                if slot >= 0:
                    permutation.append(slot)
            else:
                pending.append((gate, deps, slot))
        outputs = pending

    flush()
    for line, count in operations:
        result.append(line)
        available += count
        if count:
            flush()
    assert not outputs
    return "\n".join(result) + "\n", permutation


def save(name, mode):
    text, meta, (d, n, ops, order, rows) = export((INPUTS / f"{name}.stim").read_text(), mode)
    with gzip.open(ROOT / f"{name}-{mode}.clifft.gz", "wt") as f:
        f.write(text)
    (ROOT / f"{name}-{mode}.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.savez_compressed(
        ROOT / f"{name}-{mode}.npz",
        x=np.array([unpack(o.x, n) for o in ops]),
        z=np.array([unpack(o.z, n) for o in ops]),
        kind=np.array([0 if o.angle else 1 for o in ops]),
        angle=np.array([o.angle for o in ops]),
        slot=np.array([o.slot for o in ops]),
        sign=np.array([unpack(o.sign, 1 + d.error_bits) for o in ops]),
        order=np.array(order),
    )
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=["d5", "d7"])
    parser.add_argument("--modes", nargs="+", default=["detectors", "records"])
    args = parser.parse_args()
    for name in args.names:
        for mode in args.modes:
            save(name, mode)
