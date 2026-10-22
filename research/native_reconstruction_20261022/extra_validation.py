"""Adversarial exact laws, correlated postselection, and fallback controls."""

import json

import numpy as np
import stim
from export import export  # type: ignore[import-not-found]
from validate import ROOT, exact, restore  # type: ignore[import-not-found]

import clifft


def run():
    out = []
    for case in json.loads((ROOT / "extra_expected.json").read_text()):
        text = case["text"]
        expected = np.array(case["probabilities"])
        n = int(np.log2(len(expected)))
        actual = exact(text)
        error = float(np.max(abs(actual - expected)))
        assert error < 1e-12, (case["name"], error)
        # Include overlapping parities and an observable to test their joint law.
        declarations = "".join(f"DETECTOR rec[{-n + i}]\n" for i in range(n))
        declarations += f"OBSERVABLE_INCLUDE(0) rec[{-n}] rec[-1]\n"
        source = text + declarations
        joint = np.zeros(1 << (n + 1))
        for v, p in enumerate(expected):
            joint[v | ((((v & 1) ^ ((v >> (n - 1)) & 1)) if n > 1 else 0) << n)] += p
        det = exact(source, "detectors")
        assert max(abs(det - joint)) < 1e-12
        emitted, meta, _ = export(source)
        permutation = meta["detector_permutation"]
        rows = []
        for selected in ([], list(range(0, n, 2)), list(range(n))):
            # Nonzero explicit reference: maps both the retained outputs and selection.
            reference = np.array([(i + 1) % 2 for i in range(n)], dtype=np.uint8)
            obsref = np.array([1], dtype=np.uint8)
            mask = [int(i in selected) for i in permutation]
            pm = clifft.default_hir_pass_manager()
            pm.add(clifft.ActiveWidthSchedulePass())
            p = clifft.compile(
                emitted,
                hir_passes=pm,
                postselection_mask=mask,
                expected_detectors=reference[permutation].tolist(),
                expected_observables=obsref.tolist(),
            )
            shots = 65536
            r = clifft.sample_survivors(p, shots=shots, seed=700, threads=1, keep_records=True)
            values = restore(r, meta).astype(np.uint64) @ np.array(
                [1 << i for i in range(n + 1)], dtype=np.uint64
            )
            # Count accepted outputs per attempted shot, preserving rejection mass.
            target = np.zeros_like(joint)
            flip = sum(int(b) << i for i, b in enumerate(reference)) | (1 << n)
            for v, prob in enumerate(joint):
                normalized = v ^ flip
                if all(not (normalized >> i & 1) for i in selected):
                    target[normalized] += prob
            frequency = np.bincount(values.astype(int), minlength=len(joint)) / shots
            z = abs(frequency - target) / np.sqrt(
                np.maximum(target * (1 - target) / shots, 1 / shots**2)
            )
            assert z.max() < 7, (case["name"], selected, z.max(), frequency, target)
            rows.append(
                dict(
                    mask=selected,
                    acceptance_exact=float(target.sum()),
                    acceptance_sample=r.passed_shots / shots,
                    max_z=float(z.max()),
                )
            )
        if case["name"] == "clifford_stim":
            records = stim.Circuit(text).compile_sampler(seed=119).sample(131072)
            values = records.astype(np.uint64) @ np.array(
                [1 << i for i in range(n)], dtype=np.uint64
            )
            freq = np.bincount(values.astype(int), minlength=len(expected)) / len(values)
            z = abs(freq - expected) / np.sqrt(
                np.maximum(expected * (1 - expected) / len(values), 1 / len(values) ** 2)
            )
            assert z.max() < 7
        row = dict(name=case["name"], exact_max_error=error, postselection=rows)
        print(case["name"], error, flush=True)
        out.append(row)
    negative = {}
    for text in (
        "H 0\nT 0\nEXP_VAL X0",
        "HERALDED_PAULI_CHANNEL_1(0.25,0,0,0) 0",
        "R_PAULI(0.2) X0",
        "CX sweep[0] 1",
    ):
        try:
            export(text)
        except ValueError as e:
            negative[text] = str(e)
        else:
            raise AssertionError(text)
    (ROOT / "extra_validation.json").write_text(
        json.dumps(dict(cases=out, rejections=negative), indent=2) + "\n"
    )


if __name__ == "__main__":
    run()
