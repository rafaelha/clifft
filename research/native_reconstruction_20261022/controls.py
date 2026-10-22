"""Reject false universal phase-core claims and verify normalization handoff."""

import gzip
import importlib.abc
import json
import sys

import numpy as np
from export import export  # type: ignore[import-not-found]
from frontend import unpack  # type: ignore[import-not-found]
from validate import ROOT, restore  # type: ignore[import-not-found]

import clifft


class RejectReferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in ("tsim", "zxparam", "pyzx"):
            raise AssertionError("Reference dependency entered native path: " + fullname)
        return None


def run():
    assert not any(name.split(".")[0] in ("tsim", "zxparam", "pyzx") for name in sys.modules)
    sys.meta_path.insert(0, RejectReferenceImports())
    for name in ("d5", "d7"):
        source = (
            ROOT.parent / "global_reconstruction_20261022" / "inputs" / f"{name}.stim"
        ).read_text()
        emitted, meta, _ = export(source)
        assert emitted == gzip.open(ROOT / f"{name}-detectors.clifft.gz", "rt").read()
        assert meta["estimated_width"] == (6 if name == "d5" else 8)
    # The normalizer is chosen by the original compile pipeline, then handed to
    # the reconstruction with the same output-index mapping as its results.
    source = (
        "X 0\nM 0\nDETECTOR rec[-1]\nH 1\nT 1\nMX 1\n"
        "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    _, ed, eo = clifft._prepare_hir_for_lowering(
        source, normalize_syndromes=True, postselection_mask=[1, 0]
    )
    emitted, meta, _ = export(source)
    permutation = meta["detector_permutation"]
    p = clifft.compile(source, normalize_syndromes=True, postselection_mask=[1, 0])
    q = clifft.compile(
        emitted,
        expected_detectors=[ed[i] for i in permutation],
        expected_observables=eo,
        postselection_mask=[int(i == 0) for i in permutation],
    )
    a = clifft.sample_survivors(p, shots=65536, seed=827, threads=1, keep_records=True)
    b = clifft.sample_survivors(q, shots=65536, seed=829, threads=1, keep_records=True)
    av = np.concatenate([a.detectors, a.observables], axis=1).astype(np.uint64) @ np.array(
        [1, 2, 4], dtype=np.uint64
    )
    bv = restore(b, meta).astype(np.uint64) @ np.array([1, 2, 4], dtype=np.uint64)
    af = np.bincount(av.astype(int), minlength=8) / 65536
    bf = np.bincount(bv.astype(int), minlength=8) / 65536
    pool = (af + bf) / 2
    z = abs(af - bf) / np.sqrt(np.maximum(pool * (1 - pool) * 2 / 65536, 1 / 65536**2))
    assert max(z) < 7 and a.passed_shots == b.passed_shots == 65536
    sys.path.insert(0, str(ROOT.parent / "global_reconstruction_20261022"))
    from conditional_phase import PhaseOracle  # type: ignore[import-not-found]

    _, _, (d, n, ops, order, rows) = export("H 0 1\nT 0 1\nMX 0 1\n", "records")
    arrays = dict(
        x=np.array([unpack(o.x, n) for o in ops]),
        z=np.array([unpack(o.z, n) for o in ops]),
        kind=np.array([0 if o.angle else 1 for o in ops]),
        angle=np.array([o.angle for o in ops]),
        slot=np.array([o.slot for o in ops]),
        sign=np.array([unpack(o.sign, d.error_bits + 1) for o in ops]),
    )
    try:
        PhaseOracle(arrays).reduce(np.zeros(d.error_bits, dtype=np.uint8))
    except AssertionError:
        rejected = True
    else:
        raise AssertionError("Two independent magic states falsely reduced to one T")
    result = dict(
        reference_imports_blocked=True,
        native_source_exports=["d5", "d7"],
        original_expected_detectors=ed,
        original_expected_observables=eo,
        normalization_max_z=float(max(z)),
        two_magic_zero_one_core_rejected=rejected,
    )
    print(json.dumps(result))
    (ROOT / "controls.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    run()
