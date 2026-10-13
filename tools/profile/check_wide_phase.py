"""Bounded static phase probe against a pinned independent research reducer.

Requires the research checkout, Qiskit Aer, Stim, and the native probe executable.
Writes one comparison JSON plus reproducibility inputs and raw native diagnostics.
"""

import argparse
import gzip
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--research-checkout", type=Path, required=True)
parser.add_argument("--probe", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=5)
options = parser.parse_args()
root = options.output.resolve()
root.mkdir(parents=True, exist_ok=True)
research = options.research_checkout.resolve()
probe = options.probe.resolve()
sys.path.insert(0, str(research / "tools/profile"))
from shared_phase_specialization import (  # type: ignore[import-not-found]  # noqa: E402
    SharedPhase,
    add_parity,
    bits,
    inverse_rows,
    pauli_masks,
    pullback,
)

archive = research / "research/conditional_clifford/capability-survey-bt81_ideal.json.gz"
row = json.loads(gzip.decompress(archive.read_bytes()))["bt81_direct_x_ideal"]
source = row["source"]
assert (
    hashlib.sha256(source.encode()).hexdigest()
    == row["source_sha256"]
    == "2db837e4ab23eb30288da1b7983b929c25a84c6bca5c9ea810c71f61b2c6262a"
)
(root / "input.stim").write_text(source)
lines = source.splitlines()
first = next(i for i, line in enumerate(lines) if line.split()[0] in {"T", "T_DAG"})
(root / "prefix.stim").write_text("\n".join(lines[:first]) + "\n")
shared = SharedPhase(source)
xrows = [pauli_masks(p)[0] for p in shared.probe_rows[: shared.variables]]
coordinates = [
    sum(1 << i for i, x in enumerate(xrows) if x >> q & 1) for q in range(shared.model.num_qubits)
]
outputs = coordinates.copy()
for line in shared.lines[shared.first :]:
    words = line.split()
    if words and words[0] == "CX" and not words[1].startswith("rec["):
        outputs[int(words[2])] ^= outputs[int(words[1])]
selected = sorted(
    (q for q, i in shared.quantum_map.items() if i < shared.variables), key=shared.quantum_map.get
)
inverse = inverse_rows([outputs[q] for q in selected])
coordinates = [pullback(c, inverse) for c in coordinates]
prefix = stim.Circuit((root / "prefix.stim").read_text())
sim = stim.TableauSimulator(seed=71)
sim.set_num_qubits(shared.model.num_qubits)
sim.do(prefix)
data = [p for p in sim.canonical_stabilizers() if all(p[q] == 0 for q in range(243, len(p)))]
assert len(data) == 243
flows = [stim.Flow(output=p) for p in data]
assert (
    stim.Circuit("R " + " ".join(map(str, range(shared.model.num_qubits)))) + prefix
).has_all_flows(flows, unsigned=False)
(root / "probes.txt").write_text("\n".join(map(str, data)) + "\n")
report = dict(
    source_sha256=row["source_sha256"],
    variables=shared.variables,
    polynomial={str(k): v for k, v in shared.nominal.items()},
    entry_coordinates=list(map(str, coordinates)),
    stabilizers_checked=len(data),
    nonclifford=shared.nonclifford,
)
(root / "reference.json").write_text(json.dumps(report, indent=2) + "\n")

args = [str(probe), str(root / "input.stim"), str(root / "prefix.stim"), str(root / "probes.txt")]
trials: dict[str, list[dict[str, Any]]] = {
    mode: [] for mode in ["baseline", "wide", "output", "raw-output"]
}
for repetition in range(options.repeats + 1):
    modes = list(trials) if repetition % 2 == 0 else list(reversed(trials))
    for mode in modes:
        result = subprocess.run(
            [
                "/usr/bin/time",
                "-f",
                "%M",
                "-o",
                str(root / "rss.txt"),
                "taskset",
                "-c",
                "0",
                *args,
                mode,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        row = json.loads(result.stdout)
        row["rss_kib"] = int((root / "rss.txt").read_text())
        (root / f"probe-{mode}.json").write_text(result.stdout)
        if repetition:
            trials[mode].append(row)
summary = {
    mode: {
        k: statistics.median(r[k] for r in records)
        for k in [
            "final_t",
            "preparation_width",
            "final_width",
            "hir_seconds",
            "compile_seconds",
            "rss_kib",
        ]
    }
    for mode, records in trials.items()
}
timings = summary.copy()
for name, value in [
    ("CLIFFT_PROBE_VARIABLES", "64"),
    ("CLIFFT_PROBE_TERMS", "100"),
    ("CLIFFT_PROBE_UPDATES", "1000"),
]:
    row = json.loads(
        subprocess.check_output([*args, "output"], env={**os.environ, name: value}, text=True)
    )
    assert row["fallback"] and row["final_t"] == 648 and row["final_width"] == 56, row
    (root / f"fallback-{name}.json").write_text(json.dumps(row, indent=2) + "\n")

polynomial_summary = {}
for mode in ("output", "raw-output"):
    a = json.loads((root / f"probe-{mode}.json").read_text())
    b = json.loads((root / "reference.json").read_text())
    coordinates = list(map(int, b["entry_coordinates"]))
    transforms = []
    for axis in a["generators"]:
        assert set(axis[1:]) <= set("IZ"), set(axis)
        mask = 0
        for q, c in enumerate(axis[1:]):
            if c == "Z":
                mask ^= coordinates[q]
        transforms.append((mask, axis[0] == "-"))
    out: dict[int, int] = {}
    for (low, high), coefficient in a["terms"]:
        mask = offset = 0
        for i in bits(low | (high << 64)):
            p, s = transforms[i]
            mask ^= p
            offset ^= s
        add_parity(out, mask, -coefficient if offset else coefficient)
    expected = {int(k): v for k, v in b["polynomial"].items()}
    delta = {
        m: (out.get(m, 0) - expected.get(m, 0)) % 8
        for m in out.keys() | expected.keys()
        if out.get(m, 0) != expected.get(m, 0)
    }
    clifford = all(
        (m.bit_count() == 1 and c % 2 == 0) or (m.bit_count() == 2 and c == 4)
        for m, c in delta.items()
    )
    assert a["fixed_probes"] == a["probes"] == 243
    assert clifford
    if mode == "raw-output":
        assert out == expected
    polynomial_summary[mode] = dict(
        fixed_constraints=a["fixed_probes"],
        support_variables=a["variables"],
        equal_complete_polynomial=out == expected,
        clifford_difference_terms=len(delta),
        nonclifford_terms=len(expected),
    )
subprocess.run(
    [*args, "output"],
    env={**os.environ, "CLIFFT_PROBE_RECORDS": str(root / "records.bin")},
    stdout=subprocess.DEVNULL,
    check=True,
)

r = json.loads((root / "reference.json").read_text())
active = sorted({i for m in r["polynomial"] for i in range(87) if int(m) >> i & 1})
c = QuantumCircuit(9)
c.h(range(9))
for m in r["polynomial"]:
    c.ccz(*[active.index(i) for i in active if int(m) >> i & 1])
c.h(range(9))
c.save_probabilities()
probabilities = np.array(
    AerSimulator(method="statevector").run(c).result().data(0)["probabilities"]
)
samples = np.fromfile(root / "records.bin", dtype=np.uint8).reshape(-1, 405)
idx = samples[:, -9:].astype(np.int64) @ (1 << np.arange(9))
observed = np.bincount(idx, minlength=512) / len(samples)
tvd = float(np.abs(probabilities - observed).sum() / 2)
se = np.sqrt(np.maximum(probabilities * (1 - probabilities) / len(samples), 1e-20))
max_z = float(np.max(np.abs(probabilities - observed) / se))
assert max_z < 6 and tvd < 0.045, (max_z, tvd)
# Independent stabilizer samples define the affine record support. With 4096
# samples its rank is stable; additionally check rank on each half separately.
prefix = stim.Circuit((root / "prefix.stim").read_text())
reference = prefix.compile_sampler(seed=919).sample(4096)


def integers(a):
    return [
        int.from_bytes(row.tobytes(), "little") for row in np.packbits(a, axis=1, bitorder="little")
    ]


def span(a):
    rows: dict[int, int] = {}
    for row in a:
        while row:
            p = row.bit_length() - 1
            if p in rows:
                row ^= rows[p]
            else:
                rows[p] = row
                break
    return rows


ref = integers(reference)
origin = ref[0]
rows = span([x ^ origin for x in ref])
assert (
    len(rows)
    == len(span([x ^ origin for x in ref[:2048]]))
    == len(span([x ^ origin for x in ref[2048:]]))
)
for row in integers(samples[:, :162]):
    row ^= origin
    for p in sorted(rows, reverse=True):
        if row >> p & 1:
            row ^= rows[p]
    assert row == 0
# Check record/readout correlations against independence established by the
# record-free data stabilizer flows, including random logical parity probes.
rng = np.random.default_rng(13)
worst = 0
for _ in range(256):
    mask = int(rng.integers(1, 512))
    record = int(rng.choice(list(rows)))
    logical = np.array([(j & mask).bit_count() & 1 for j in range(512)], dtype=float)
    mu = float(probabilities @ logical)
    ones = logical[idx].astype(bool)
    for record_value in (0, 1):
        selection = samples[:, record] == record_value
        n = int(selection.sum())
        z = abs(float(ones[selection].mean()) - mu) / max((mu * (1 - mu) / n) ** 0.5, 1e-12)
        worst = max(worst, z)
assert worst < 6, worst
sample_summary = dict(
    shots=len(samples),
    records=samples.shape[1],
    aer_logical_tvd=tvd,
    aer_max_bin_z=max_z,
    stim_prefix_rank=len(rows),
    max_joint_z=worst,
)

comparison = dict(
    source_sha256=report["source_sha256"],
    polynomials=polynomial_summary,
    timings=timings,
    sampling=sample_summary,
)
(root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps(comparison, indent=2))
