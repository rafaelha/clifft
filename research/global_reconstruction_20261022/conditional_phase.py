"""Independent fixed-fault phase oracle using Stim and exact mod-eight algebra.

All tableau and polynomial work here is research-time work. This is expressly
not an implementation permitted in Clifft's ordinary hot dispatch.
"""

import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import stim
from fault_topology import independent  # type: ignore[import-not-found]
from gf2 import Basis as _Basis  # type: ignore[import-not-found]
from gf2 import mask as _mask
from gf2 import unmask as _unmask

ROOT = Path(__file__).parent


def polynomial(masks, coeffs):
    p: dict[int, int] = defaultdict(int)
    for mask, c in zip(masks, coeffs):
        indices = [1 << i for i in range(mask.bit_length()) if mask >> i & 1]
        for degree, multiplier in [(1, 1), (2, -2), (3, 4)]:
            for group in itertools.combinations(indices, degree):
                p[sum(group)] = (p[sum(group)] + int(c) * multiplier) % 8
    return dict(p)


def pauli(x, z, n):
    return stim.PauliString.from_numpy(xs=_unmask(x, n), zs=_unmask(z, n))


class PhaseOracle:
    def __init__(self, a, hidden_bits=0):
        self.a = a
        self.n = n = a["x"].shape[1]
        self.ids = np.flatnonzero(a["kind"] == 0)
        self.axes = [stim.PauliString.from_numpy(xs=x, zs=z) for x, z in zip(a["x"], a["z"])]
        basis = _Basis()
        generators: list = []
        self.masks = []
        signs = []
        for j in self.ids:
            mask = _mask(a["x"][j]) | (_mask(a["z"][j]) << n)
            if basis.solve(mask) is None:
                basis.insert(mask, 1 << len(generators))
                generators.append(self.axes[j])
            coeff = basis.solve(mask)
            product = stim.PauliString(n)
            for k, g in enumerate(generators):
                if coeff >> k & 1:
                    product *= g
            assert product == self.axes[j] or product == -self.axes[j]
            self.masks.append(coeff)
            signs.append(1 if product == self.axes[j] else -1)
        self.frame = stim.Tableau.from_stabilizers(generators, allow_underconstrained=True)
        initial = stim.TableauSimulator(seed=0)
        initial.set_num_qubits(n)
        hidden_ids = np.flatnonzero((a["kind"] == 1) & (a["slot"] < 0))
        for i, j in enumerate(hidden_ids):
            initial.postselect_observable(self.axes[j], desired_value=bool(hidden_bits >> i & 1))
        initial.do_tableau(self.frame.inverse(), list(range(n)))
        self.initial = initial.copy()
        # X supports of stabilizers span every translation of the affine support.
        directions = independent([_mask(s.to_numpy()[0]) for s in initial.canonical_stabilizers()])
        self.width = len(directions)
        completed = independent(directions + [1 << i for i in range(n)])
        assert len(completed) == n
        coords = _Basis()
        for i, r in enumerate(completed):
            coords.insert(r, 1 << i)
        inverse_columns = [coords.solve(1 << i) for i in range(n)]
        inverse_rows = [
            sum(((v >> i) & 1) << j for j, v in enumerate(inverse_columns)) for i in range(n)
        ]
        self.linear = stim.Tableau.from_conjugated_generators(
            xs=[pauli(r, 0, n) for r in completed], zs=[pauli(0, r, n) for r in inverse_rows]
        )
        sampled = initial.copy()
        offset = _mask(np.array(sampled.measure_many(*range(n)), dtype=bool))
        self.offset = offset
        shifted = initial.copy()
        for i in range(n):
            if offset >> i & 1:
                shifted.x(i)
        shifted.do_tableau(self.linear.inverse(), list(range(n)))
        self.shifted = shifted
        self.restricted = [
            sum(((m & r).bit_count() & 1) << i for i, r in enumerate(directions))
            for m in self.masks
        ]
        units = a["angle"][self.ids] / (np.pi / 8)
        assert np.allclose(units, np.round(units))
        self.coeffs = (
            np.round(units).astype(int)
            * np.array(signs)
            * np.array([(-1) ** ((m & offset).bit_count() & 1) for m in self.masks])
        )
        self.output_ids = np.flatnonzero(a["slot"] >= 0)

    def reduce(self, faults):
        signs = (self.a["sign"][self.ids].astype(np.uint8) @ np.r_[1, faults].astype(np.uint8)) % 2
        coeffs = self.coeffs * (1 - 2 * signs.astype(int))
        p = polynomial(self.restricted, coeffs)
        witness = sum(m for m, c in p.items() if m.bit_count() == 1 and c & 1)
        for m, c in polynomial([witness], [1]).items():
            p[m] = (p.get(m, 0) - c) % 8
        assert all(
            c % (2 if m.bit_count() == 1 else 4 if m.bit_count() == 2 else 8) == 0
            for m, c in p.items()
        )
        sim = self.shifted.copy()
        for mask, c in p.items():
            bits = [i for i in range(self.width) if mask >> i & 1]
            if len(bits) == 1:
                for _ in range(c // 2):
                    sim.s(bits[0])
            elif len(bits) == 2 and c:
                sim.cz(*bits)
        sim.do_tableau(self.linear, list(range(self.n)))
        for i in range(self.n):
            if self.offset >> i & 1:
                sim.x(i)
        sim.do_tableau(self.frame, list(range(self.n)))
        magic = self.linear(pauli(0, witness, self.n))
        if (_mask(magic.to_numpy()[1]) & self.offset).bit_count() & 1:
            magic *= -1
        magic = self.frame(magic)
        return sim, magic, witness, p

    def dense_check(self, faults):
        sim, magic, witness, _ = self.reduce(faults)
        labels = np.arange(1 << self.n, dtype=np.uint64)
        target = self.initial.state_vector(endian="little").astype(np.complex128)
        phases = np.zeros(1 << self.n)
        for j, m in zip(self.ids, self.masks):
            image = self.frame.inverse()(self.axes[j])
            coeff = (
                self.a["angle"][j]
                * image.sign.real
                * (-1)
                ** int(self.a["sign"][j].astype(np.uint8) @ np.r_[1, faults].astype(np.uint8) % 2)
            )
            parity = np.bitwise_count(labels & np.uint64(m)) % 2  # type: ignore[attr-defined]
            phases += coeff * (2 * parity.astype(float) - 1)
        target *= np.exp(1j * phases)
        # Compare in the diagonal frame. For d5 the residual is identically zero.
        assert witness == 0
        sim.do_tableau(self.frame.inverse(), list(range(self.n)))
        actual = sim.state_vector(endian="little").astype(np.complex128)
        fidelity = abs(np.vdot(actual, target)) ** 2 / (
            np.vdot(actual, actual).real * np.vdot(target, target).real
        )
        assert fidelity > 1 - 1e-12
        return float(fidelity)

    def output_randomness(self, sim):
        stabs = sim.canonical_stabilizers()
        pairings = [
            sum(int(not s.commutes(self.axes[j])) << i for i, s in enumerate(stabs))
            for j in self.output_ids
        ]
        return len(independent(pairings))


if __name__ == "__main__":
    out = []
    rng = np.random.default_rng(907)
    for name in ["d5", "d7"]:
        a = dict(np.load(ROOT / f"{name}-detectors-component0.npz"))
        for hidden in [0, 1]:
            start = time.perf_counter()
            oracle = PhaseOracle(a, hidden)
            prep = time.perf_counter() - start
            cases = [("no_fault", np.zeros(a["sign"].shape[1] - 1, dtype=np.uint8))]
            seen = set()
            for i, f in enumerate(a["error_transform"].T):
                key = f.tobytes()
                if f.any() and key not in seen:
                    seen.add(key)
                    cases.append((f"error_bit_{i}", f.astype(np.uint8)))
                if len(cases) == 25:
                    break
            cases += [
                (f"random_{i}", rng.integers(0, 2, a["sign"].shape[1] - 1, dtype=np.uint8))
                for i in range(8)
            ]
            for label, f in cases:
                start = time.perf_counter()
                sim, magic, witness, p = oracle.reduce(f)
                seconds = time.perf_counter() - start
                result = dict(
                    circuit=name,
                    hidden=hidden,
                    case=label,
                    compile_once_seconds=prep,
                    fault_specialization_seconds=seconds,
                    residual_T_count=int(bool(witness)),
                )
                if not witness:
                    result["random_output_bits"] = oracle.output_randomness(sim)
                    if label == "no_fault" or label.startswith("random"):
                        result["dense_fidelity"] = oracle.dense_check(f)
                out.append(result)
                print(json.dumps(result), flush=True)
    (ROOT / "conditional_phase.json").write_text(json.dumps(out, indent=2) + "\n")
