"""Materialize original physical faults before either compilation path.

This is a validation-only source replay. It does not establish a public replay
contract for the noise-compressed reconstructed program.
"""

import json
import os

import numpy as np
import stim
from export import export  # type: ignore[import-not-found]
from frontend import Dilation, source_instructions  # type: ignore[import-not-found]
from large_validation import features  # type: ignore[import-not-found]
from validate import OLD, ROOT, compile_, restore  # type: ignore[import-not-found]

import clifft


def materialize(text, fault):
    out = []
    cursor = 0

    def bit(p):
        nonlocal cursor
        if not p:
            return 0
        v = (fault >> cursor) & 1
        cursor += 1
        return v

    def append(name, args, ts):
        out.append(str(stim.CircuitInstruction(name, ts, args)))

    for name, args, ts in source_instructions(text):
        if name in ("T", "T_DAG"):
            out.append(name + " " + " ".join(str(t.value) for t in ts))
        elif name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            for t in ts:
                if bit(args[0]):
                    append(name[0], [], [t])
        elif name in ("DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2"):
            for t in ts:
                x = (fault >> cursor) & 1
                z = (fault >> (cursor + 1)) & 1
                cursor += 2
                if x or z:
                    append("IXZY"[x + 2 * z], [], [t])
        elif name in ("E", "ELSE_CORRELATED_ERROR"):
            selected = (fault >> cursor) & 1
            cursor += 1
            if selected:
                for t in ts:
                    append("X" if t.is_x_target else "Y" if t.is_y_target else "Z", [], [t.value])
        elif name in ("M", "MX", "MY", "MR", "MRX", "MRY"):
            for t in ts:
                inverted = bool(t.is_inverted_result_target) ^ bool(bit(args[0] if args else 0))
                append(name, [], [stim.target_inv(t.value) if inverted else t.value])
        elif name == "MPP":
            group = []
            for i, t in enumerate(ts):
                group.append(t)
                if not t.is_combiner and (i + 1 == len(ts) or not ts[i + 1].is_combiner):
                    if bit(args[0] if args else 0):
                        t = group[0]
                        fn = (
                            stim.target_x
                            if t.is_x_target
                            else stim.target_y
                            if t.is_y_target
                            else stim.target_z
                        )
                        group[0] = fn(t.value, invert=not t.is_inverted_result_target)
                    append(name, [], group)
                    group = []
        elif name == "MPAD":
            append(name, [], [t.value ^ bit(args[0] if args else 0) for t in ts])
        else:
            append(name, args, ts)
    return "\n".join(out) + "\n", cursor


def run():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    rows = []
    for name in ("d5", "d7"):
        text = (OLD / "inputs" / f"{name}.stim").read_text()
        d = Dilation.parse(text)
        selected = [int(len(d.channels) * v) for v in (0.03, 0.18, 0.37, 0.63, 0.83, 0.97)]
        cases = [("none", 0)] + [
            (f"channel_{j}", next(f for p, f in d.channels[j] if p and f)) for j in selected
        ]
        cases.append(("combined", sum(f for _, f in cases[1:])))
        for label, fault in cases:
            source, count = materialize(text, fault)
            assert count == d.error_bits
            # Reconstruct the original source with its original error-bit map.
            native, meta, _ = export(text, "records", fixed_fault=fault)
            p = compile_(source)
            q = compile_(native)
            shots = 4096
            a = clifft.sample(p, shots=shots, seed=882, threads=1).measurements
            b = restore(clifft.sample(q, shots=shots, seed=998, threads=1), meta)
            left, right = features(a, a.shape[1]), features(b, b.shape[1])
            pool = (left.mean(0) + right.mean(0)) / 2
            z = abs(left.mean(0) - right.mean(0)) / np.sqrt(
                np.maximum(pool * (1 - pool) * 2 / shots, 1 / shots**2)
            )
            row = dict(
                name=name,
                case=label,
                fault_hex=hex(fault),
                shots=shots,
                features=len(z),
                max_z=float(z.max()),
                original_width=p.peak_active_width,
                native_width=q.peak_active_width,
            )
            print(json.dumps({k: v for k, v in row.items() if k != "fault_hex"}), flush=True)
            rows.append(row)
            assert z.max() < 7, row
            (ROOT / "fixed_fault.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    run()
