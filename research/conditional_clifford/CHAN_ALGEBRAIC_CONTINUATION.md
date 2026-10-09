# Algebraic continuation through the complete Chan circuit

This research checkpoint replaces physical-state tomography with a bounded
algebraic instrument update. It handles the supplied Chan d3 circuit and the
actual small-check, growth, large-check, and terminal-check sequence of the
four-round d5 circuit. Production compiler and executor behavior is unchanged.

The implementation is on `codex/chan-algebraic-continuation`, based on the
[earlier boundary study](CHAN_BOUNDARY.md) at `bcafbee9`. The input circuits and
their hashes remain in `chan-boundary-inputs.json.gz`. They contain 17 physical
qubits and 23 visible records for d3, and 55 physical qubits and 145 records for
four-round d5. No physical circuit, noise probability, or postselection rule
was changed to obtain the reduction.

## Algebraic certificate and update

The existing inverse-T parity audit discovers complete check regions directly
from circuit operations. It checks the inverse phase layers, inverse Clifford
networks, central measurement/reset, and closed flags. The binder preserves
the physical Pauli faults and reported-bit flips. For a fixed fault history
and physical central outcome m the full check is

```text
K_m = A_m (I + (-1)^m C) / 2,
C = D^dagger P D,     C = C^dagger,     C^2 = I.
```

The signed Pauli P includes the fault-dependent relative sign. The common
action A_m is Clifford and includes the central reset and its intervening
faults. Flags keep their own physical records and final states.

Let S be the known signed stabilizer subgroup of the incoming conditional
state. Compute the intersection

```text
R = S intersect C S C^dagger.
```

This is Gaussian elimination on Pauli generators, including their signs.
Because C is an involution, C preserves R. Both coherent terms, and hence the
entire projected state, lie in R's common eigenspace. If R has n-r independent
generators, only r quantum coordinates are needed to evaluate the instrument.
The implementation refuses r greater than three before allocating a dense
vector or matrix. It retains the interference between the two terms.

The Clifford action on that subspace comes from its conjugated Pauli
generators. A tableau alone omits the global phase of C, which is a relative
phase in the projector and cannot be dropped. The implementation fixes it
using one nonzero physical computational-basis amplitude. Affine support and
stabilizer-generator products evaluate that amplitude in polynomial storage;
there is no physical statevector expansion.

After applying the small projector, stabilizer extraction is restricted to
the at-most-three-coordinate vector. It produces an updated Clifford encoder
and requires at most one remaining coordinate. Export additionally checks the
remaining state is in the Clifford orbit of a stabilizer or T state. Ordinary
Clifford gates, measurements, resets, feedback, and later checks then consume
that actual conditional state. The initial T injection is also represented
algebraically. All quantum outcomes use their computed probabilities.

This certificate does not need a named code or a precomputed ideal eigenstate
at each boundary. It supplies a state-dependent eligibility check and an exact
update when accepted. The incompatible one-T input from the earlier study is
refused because its common subspace exceeds the bound.

## Evidence

The retained [validation artifact](chan-algebraic-validation.json) records
source and native-extension hashes, package versions, physical fault histories,
boundary dimensions, and probability errors. Its checks include:

- Independent Qiskit statevectors for 144 Pauli-measurement branches and 45
  admitted Clifford-projector branches on one through six qubits. Another
  99 Clifford-projector branches exceed the bounded representation and are
  explicitly refused. Stabilizer amplitude queries are checked against Stim.
- 246 complete original-circuit record queries across 20 d3 and 21 four-round
  d5 fault histories: ideal, freshly drawn, denser stress, and the retained
  witness. Each history checks records from both samplers and altered records
  that can be impossible.
  Native references preserve the original gates; fresh environment wires
  marginalize reset outcomes for the probability API.
- Independent Aer full physical states and conditional probabilities for all
  20 d3 histories, including the actual preparation, flagged check, and terminal
  measurements.
- Qubit-permuted complete inputs, independent central/flag readout flips,
  malformed inverse networks and phase layers, and the incompatible entry.
- Every record of the published four-fault witness under the supplied
  four-round d5 model: 128 records, total probability one, all rejected, with
  each probability checked against ordinary Clifft. The mapped fault history
  and source provenance are in [the witness manifest](chan-four-fault-history.json).

The carried core stays at width one through the tested complete sequences.
Noisy whole-check evaluation sometimes requires a temporary width-two core,
then returns to one. Exported boundary preparations are independently compiled
by Clifft and must also have width at most one. The terminal check is included;
the simulation does not stop at the large physical check.

The 128-record witness law is exhaustive for that fixed fault history. The
broader noise checks are selected histories and do not prove that every
possible d5 history is eligible. Every actual update rechecks its certificate;
unsupported states fail explicitly. This is not a logical-error-rate estimate.

## Scope and remaining limits

- This is offline host specialization. It performs per-shot tableau algebra
  outside ordinary execution and exports a conditional preparation plus the
  preserved records. It is not a public sampling mode, a static symbolic
  plan, or a new executor continuation architecture.
- The current front end accepts the audited inverse-T/CNOT parity structure
  and a restricted terminal Pauli-product sandwich. It refuses interleaved
  noise inside an outer T layer, unsupported middle operations, early flag
  readout, or a subspace/core exceeding its budget. Refusal never redraws a
  fault or silently substitutes a different trajectory. Automatic conversion
  to an ordinary fallback is not implemented in this driver. Reset, individual
  readout, and T instructions must each have one target, as in the retained
  exported inputs; grouped operands are explicitly refused.
- The small core update can carry a different Clifford encoding after each
  outcome. No cache of full fault histories, branch enumeration, or physical
  Pauli tomography is used for sampling. Enumeration appears only inside the
  bounded core and in the separate exhaustive validation driver.
- Reset branches of a product stabilizer factor are marginalized. A reset
  entangled with other coordinates is sampled and retained as a hidden
  trajectory choice. In that case the driver's returned probability is joint
  with that hidden branch, not a marginal visible-record probability. All
  retained complete-circuit query cases verify that no such unresolved hidden
  branch occurs. The sampled state law remains the reset channel in either
  case.
- This Python implementation is for capability and correctness. Its measured
  per-trajectory times are diagnostic, exclude some setup and fault-drawing
  costs, and establish no speed advantage over ordinary d3/d5 sampling.
  No d7/d9 protocol or new large-distance performance result is claimed.

The checkpoint supports investigating explicit larger-distance flagged checks.
The next meaningful question is whether the same certificate remains small
through their complete preparation/growth/check sequence. Static compilation
can be assessed against this concrete state-update contract. A general
coherent-sum representation is not needed by the tested Chan sequence.

## Reproduction and code map

Use a Python environment with the PR 552 Clifft build and the repository's
development dependencies. The implementation uses the unchanged native library;
no new C++ target is needed.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_chan_algebraic.py --trials 16 \
  --output /tmp/chan-algebraic-validation.json
```

The validator needs only retained inputs in this branch. Its complete witness
enumeration takes substantially longer than a single trajectory.

| Responsibility | File under `tools/profile/` |
| --- | --- |
| Signed intersection, stabilizer amplitudes, bounded instrument, state export | `conditional_clifford_core.py` |
| Complete-circuit discovery, fault binding, records and continuation | `chan_algebraic_continuation.py` |
| Reused structural recognizer from `codex/wan-corpus-study` at `01a8eec8` | `parity_envelope_audit.py` |
| Independent oracles and complete-circuit validation | `validate_chan_algebraic.py` |

The existing `study_chan_boundary.branch_frames` supplies the previously
validated two-path physical fault binding. Its exhaustive `classify` routine
is not used by the new algorithm. The circuit and witness provenance follows
the [pinned upstream corpus](https://github.com/kh428/exact_ler_for_msc/tree/b99452de7e300324e7bbf5cd9347241c3e289bae).
