"""Measure whether the reduced Clifford frame is static across fault choices."""

import json
from pathlib import Path

import numpy as np
from gf2 import Basis as _Basis  # type: ignore[import-not-found]
from gf2 import mask as _mask


def independent(rows):
    basis = _Basis()
    kept: list = []
    for row in rows:
        if basis.solve(int(row)) is None:
            basis.insert(int(row), 1 << len(kept))
            kept.append(int(row))
    return kept


def nullspace(rows, width):
    basis = _Basis()
    for row in rows:
        basis.insert(int(row), 0)
    out = []
    for i in range(width):
        if i in basis.rows:
            continue
        v = 1 << i
        for p, (row, _) in sorted(basis.rows.items()):
            if (row & v).bit_count() & 1:
                v ^= 1 << p
        out.append(v)
    assert all((row & v).bit_count() % 2 == 0 for row in rows for v in out)
    return out


def support(a):
    n = a["x"].shape[1]
    ids = np.flatnonzero(a["kind"] == 0)
    rows = [_mask(a["x"][j]) | (_mask(a["z"][j]) << n) for j in ids]
    gens = independent(rows)
    basis = _Basis()
    for i, g in enumerate(gens):
        basis.insert(g, 1 << i)
    masks = [basis.solve(r) for r in rows]
    stabilizers = [1 << (n + i) for i in range(n)]
    for j in np.flatnonzero((a["kind"] == 1) & (a["slot"] < 0)):
        x, z = _mask(a["x"][j]), _mask(a["z"][j])
        anti = [
            i
            for i, s in enumerate(stabilizers)
            if ((s & z).bit_count() + ((s >> n) & x).bit_count()) % 2
        ]
        if anti:
            first = anti[0]
            for i in anti[1:]:
                stabilizers[i] ^= stabilizers[first]
            stabilizers[first] = x | (z << n)
    combined = _Basis()
    for row in stabilizers:
        combined.insert(row, 0)
    constraints = []
    for i, g in enumerate(gens):
        if rel := combined.insert(g, 1 << i):
            constraints.append(rel)
    free = nullspace(constraints, len(gens))
    restricted = [sum(((r & v).bit_count() & 1) << i for i, v in enumerate(free)) for r in masks]
    return ids, masks, free, restricted


def analyze(path):
    a = dict(np.load(path))
    units = a["angle"][a["kind"] == 0] / (np.pi / 8)
    if not len(units) or not np.allclose(units, np.round(units)):
        return None
    ids, masks, free, restricted = support(a)
    width = len(free)
    odd = np.round(units).astype(int) % 2 != 0
    patterns = np.array([[(r >> i) & 1 for i in range(width)] for r in restricted], dtype=np.uint8)
    pairs = [(i, j) for i in range(width) for j in range(i, width)]
    outer = np.array([patterns[:, i] * patterns[:, j] for i, j in pairs], dtype=np.uint8)
    # A T sign change adds an S parity. Its symplectic action is r r^T.
    dynamic = (outer[:, odd] @ a["sign"][ids[odd], 1:].astype(np.uint8)) % 2
    fault_generators = [_mask(col) for col in dynamic.T]
    changes = independent(fault_generators)
    offset_signs = np.array(
        [[(r >> i) & 1 for i in range(len(masks) and max(masks).bit_length())] for r in masks],
        dtype=np.uint8,
    )
    offset_dynamic = (outer[:, odd] @ offset_signs[odd]) % 2
    return dict(
        circuit=path.stem,
        support_width=width,
        fault_bits=a["sign"].shape[1] - 1,
        fault_dependent_symplectic_rank=len(changes),
        faults_changing_symplectic_form=sum(bool(c) for c in fault_generators),
        arbitrary_affine_offset_symplectic_rank=len(
            independent([_mask(c) for c in offset_dynamic.T])
        ),
        symmetric_form_entries=len(pairs),
    )


if __name__ == "__main__":
    out = []
    for path in sorted(Path(__file__).parent.glob("*-component*.npz")):
        result = analyze(path)
        if result is not None:
            out.append(result)
            print(json.dumps(result), flush=True)
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
