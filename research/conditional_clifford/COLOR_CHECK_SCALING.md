# Conditional-state scaling on constructed color-code checks

This study takes the [validated Chan update](CHAN_ALGEBRAIC_CONTINUATION.md)
to larger physical circuits. The inputs are explicitly constructed controls,
not author-supplied larger-distance Chan cultivation protocols. They test the
memory cost of sampling complete noisy records and carrying a quantum output.
Production Clifft behavior is unchanged.

All 60 retained trajectories carry one quantum coordinate, temporarily using
two inside a check. At d11 the original circuit has ordinary active width 46,
which implies a 1 PiB complex-double coefficient buffer. The conditional
Python path takes a measured median 5.94 seconds per complete attempt on this
control. This establishes a substantial memory gap on a constructed circuit;
it does not establish a complete d11 cultivation protocol or a speedup ratio.

## Why the inputs are constructed

The supplied Chan corpus contains complete d3/d5 circuits. An inventory of
[the current author repository](https://github.com/timchan0/cliffordea/tree/27385c688ed40d595eab4e2378dccb5c4d32dd9a)
at `27385c688ed40d595eab4e2378dccb5c4d32dd9a` found 52 Stim files and no d7/d9
paths. The [retained inventory](chan-author-inventory.json) records that
limited search; it does not establish absence elsewhere. The paper explicitly
leaves higher-distance behavior open in
[its conclusion](https://arxiv.org/html/2609.17706v3#S7).

The code layouts come from `make_color_code` in
[the original cultivation repository](https://github.com/Strilanc/magic-state-cultivation/blob/871e68ff6df2f75190b1bfd6351459d1b5a037e3/src/cultiv/_construction/_color_code.py).
The [retained geometry](color-code-layouts.json) contains coordinates, CSS
supports, the upstream revision, and its source hash. The new circuit
generator uses these facts; it does not copy a larger cultivation schedule.

Each input contains:

1. Preparation of one T-dagger state and a Clifford CSS encoder, starting all
   physical wires in zero. The encoder has fixed all-data logical X and Z.
2. Two complete flagged inverse-T parity checks. Each has a central X
   measurement and reset, an inverse CNOT network, and one Z flag for each
   adjacent pair in the chosen data ordering.
3. Physical noisy X- and Z-syndrome extraction after each check, using a reset
   ancilla and CNOTs. Every central, flag, and syndrome record is retained.
4. A noiseless terminal logical magic-axis measurement and its output quantum
   state. Detectors and the logical bit are emitted through ordinary Clifft.

The gate noise is independent probability-p uniform nonidentity Pauli noise
after each one- or two-qubit unitary, and probability-p reset/readout flips.
The check T gates also have noise: faults are placed after the complete T
layer, which is equivalent because faults on one wire commute with the other
wires' T gates. No fault moves across a CNOT. There is no idle-noise schedule.
The final logical measurement is noiseless. Sampling does not discard failed
checks or truncate the sampled physical fault weight.

These inputs include preparation and continuation but **exclude code growth,
escape, and decoding**. The encoder and flag network have unrestricted
connectivity. No fault-distance or logical-error-rate claim is made. In
particular, code distance in the table is not a demonstrated fault distance
of these constructed circuits.

## What is certified

The geometry checks establish independent self-orthogonal CSS generators of
rank `(n-1)/2`. The signed phase pattern has signed weight zero modulo four
on every generator and total signed weight one. Consequently the conjugated
all-X parity has logical action `T-dagger X T`, with the encoded T-dagger state
as its positive eigenstate. The encoder independently pins both logical
Pauli representatives. Noiseless detector and logical results check this
contract through the complete generated circuit.

The flag network has two taps per flag. Their contributions cancel in the
pulled-back all-X parity; the ordinary structural audit verifies this on the
actual gates. The same algebraic continuation driver used for the supplied
Chan circuit then checks each actual incoming conditional state. The generator
does not supply it with ideal code matrices or tell it to assume code entry.

The independent reference retains coherent stabilizer terms using Cirq CH
forms and Stim tableaux. For each whole-check branch it builds the entire
physical Clifford circuit, recovers its Pauli action and global phase, and
applies the surrounding T conjugation. It does not call the new signed
intersection, bounded-core compression, physical-amplitude helper, or Pauli
fault-propagation binder. It shares the region selection and materialized
fault history, so independent Aer/native physical-circuit comparisons are
also required at smaller sizes.

## Evidence and cost

The [retained run](color-check-scaling.json) records each circuit hash, complete
physical fault history, full measurement record, per-boundary core dimensions,
source/native hashes, package versions, and timings. There are ideal cases,
fresh draws from the original p=0.001 channels, and denser stress histories.
Stress histories are validation inputs, not samples used to estimate p=0.001
logical error rates.

| Code distance | Physical qubits | Ordinary active width | Ordinary coefficients | Conditional seconds per attempt | Ordinary seconds per attempt |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 14 | 4 | 256 B | 0.0321 | 0.00000862 |
| 5 | 38 | 10 | 16 KiB | 0.160 | 0.0000225 |
| 7 | 74 | 19 | 8 MiB | 0.639 | 0.0204 |
| 9 | 122 | 31 | 32 GiB | 2.06 | Not allocated |
| 11 | 182 | 46 | 1 PiB | 5.94 | Not allocated |

Every sampled whole check, including terminal readout, returns to a one-qubit
core; the maximum temporary width is two. Every final exported preparation
also compiles at width one. This core dimension excludes the polynomial-size
Clifford encoding and Python/reference overhead; it is not a total-memory
measurement. No history was refused or redrawn in this run.

Validation includes:

- 60 complete trajectories: at each distance, one ideal, seven original-noise,
  and four stress histories. The largest stress history has 127 physical faults.
- 60 conditional record probabilities checked by the coherent reference,
  whose maximum retained term count is 16. The largest log-probability error
  is `3.56e-14`.
- Ten full conditional-state checks across all five distances, comprising
  880 stabilizer and logical-coordinate expectations. The maximum discrepancy
  is `2.82e-14`.
- 81 native physical-circuit record queries at d3/d5/d7, including independently
  sampled and altered records. The maximum nonzero relative error is
  `3.11e-15`; impossible records are checked separately.
- Twelve independent Aer full-state and probability checks at d3. The maximum
  statevector discrepancy up to global phase is `3.51e-16`.

The original Chan implementation is unchanged. Its earlier validation source
hashes still match, and this run uses the same native extension binary.

The ordinary baseline is the same PR 552 native library used by the earlier
study. It compiles each original noisy circuit once. Sampling uses one thread
and three trials of four shots for cases with active width at most 20. Larger
cases are planned but never allocate the exponential coefficient buffer.
The memory figures are `16 * 2^k` coefficient bytes, not measured process RSS
or a lower bound on all possible Clifft optimizations.

Conditional timings include drawing the fault history, the full Python
trajectory, exporting the final state and records, compiling that export,
and a native shot producing the detector/logical contract. Setup is recorded
separately. Validation time is excluded. The process is pinned to one CPU;
thread settings and reproduction instructions are below. Small sample counts
make these timings diagnostic, not a mature performance benchmark.
At d7 the Python path is about 31 times slower than ordinary execution in
this run. The evidence supports a capability mode for large active width;
the ordinary path remains preferable where its state is affordable. There
is no measured ordinary-versus-conditional speedup at d9/d11. The coherent
oracle timings are fixed-record queries, not complete sampling throughput.

Final-state checks measure all independent dormant stabilizers and the three
logical Bloch coordinates in the independent coherent reference. These
expectations characterize the full conditional pure state, including flags
and ancillas. Complete record probabilities, conditional on each fixed fault
history, are compared relatively, so very small probabilities cannot pass
solely on an absolute tolerance. The physical
Aer checks use full d3 statevectors. Native Clifft checks preserve the full
original gates and marginalize reset branches using fresh environment wires.

## Consequence for the research direction

The useful resource is the size of the conditional quantum core and the work
needed to update its Clifford encoding. A large ordinary active width can
coexist with a tiny core on these structured instruments. The carried state
and complete records matter: reducing just an acceptance query would not
establish this continuation capability.

This supports an optional capability path with a checked algebraic boundary.
The remaining scientific gap is a complete, independently validated larger
cultivation protocol including growth and its noise schedule. The constructed
flag network does not fill that gap. More repetitions of this same control
would not add an independent circuit family.

Before a production implementation, the host's per-shot tableau work needs a
deliberate compiler/continuation design. The current executor forbids runtime
topology planning. This study performs all such work offline and proposes no
change to that invariant. No competitive claim against published bespoke
samplers follows from these controls.

## Reproduction

Use the PR 552 Python build with NumPy, Stim, Qiskit Aer, and
`cirq-core==1.6.1` in a research environment. No upstream checkout or network
access is needed: the geometry and the generator are retained here.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  tools/profile/study_color_check_scaling.py --trials 8 \
  --output /tmp/color-check-scaling.json
```

The default distances are 3, 5, 7, 9, and 11. `--distances 3 --trials 2` runs
a bounded smoke check. `constructed_color_checks.make_source` reproduces the
physical circuit text for any retained distance. The structural and bounded
core implementations remain those of the earlier Chan checkpoint.
