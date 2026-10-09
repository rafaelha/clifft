# Flagged Chan checks: a conditional one-T boundary

2026-10-09. **A complete flagged check can return to the one-T state class,
even when noise takes it outside the original signed code space.** This
removes a suspected representation obstacle for these inputs. It does not
yet implement a fast whole-circuit reducer.

The useful condition is algebraic: the incoming state is a Pauli correction
of a one-T state that is an eigenstate of the check's Hermitian Clifford
operator. Under that condition the coherent two-term instrument becomes a
Clifford instrument on the incoming magic resource. A counterexample shows
that merely having one T at entry, or recognizing the check's gate pattern,
is insufficient.

## Inputs and question

This follows the [external cultivation audit in #495](https://github.com/unitaryfoundation/clifft/blob/01a8eec8/tools/profile/research/cultivation_applicability.md).
The retained inputs are the complete Chan d3 circuit and Chan four-round d5
circuit, exported at p=0.001 from upstream revision
`b99452de7e300324e7bbf5cd9347241c3e289bae`. The largest flagged check has seven
data qubits and two flags in d3, and 19 data qubits and 13 flags in d5.

The study uses the audit's supplied boundaries, then rechecks the inverse
Clifford networks, inverse T layers, flag endpoints and pulled-back parity.
It does not add automatic region discovery. The d3 state experiments execute
the complete physical preparation from all zero through the check, including
all realized prefix and check faults. They stop before terminal readout.

The question was whether this boundary needs a larger coherent-sum state
representation, or can return to the current Clifford-plus-one-T class.
The existing frontend itself still stops before these later phase regions.

## Sufficient condition

For a fixed Pauli fault history and central quantum outcome, the closed
flagged check has the form, up to a common global phase,

```text
K = A (I + s C) / 2,
C = D^dagger P D,        C = C^dagger, C^2 = I.
```

Here D is the first signed single-T layer, P is the central X pulled back
through the first CNOT network, and s includes the outcome and fault signs.
Prepared spectators contribute known signs. The two coherent paths have a
relative Pauli before conjugation by D. Their common action A is Clifford:
conjugating a Pauli by a single T on each affected wire gives a Clifford.
The central reset correction and faults between measurement and reset are
included in A. Closing flags must not reveal which coherent path occurred.

Suppose the entry is E|psi>, where E is Pauli and

```text
C|psi> = lambda|psi>,    lambda = +1 or -1.
```

Since C is Clifford, E' = C E C^dagger is another Hermitian Pauli. Therefore

```text
K E|psi> = A E (I + s lambda Q)|psi> / 2,
Q = E^dagger E'.
```

Retaining the phase of Q is essential:

- If Q is a signed Hermitian Pauli, this is a Pauli measurement projector.
- If Q is an imaginary multiple of a Hermitian Pauli, the normalized action
  is a Clifford quarter turn, with outcome probability 1/2.

Both are Clifford operations or Pauli measurements. Applied to a Clifford
image of one T state and stabilizer ancillas, they leave at most one T state.
For the measurement case, decode the input to one magic qubit and stabilizer
ancillas: a Pauli measurement either consumes the magic qubit or preserves it
in a new Clifford encoding. No additional magic resource is required.

This identity handles multiple simultaneous faults without a history cache.
It is a sufficient condition on the actual prepared state, not an identity
that replaces the check by Clifford gates on an arbitrary unknown input.

## Evidence

The [reproducible report](chan-boundary-study.json) records package and input
hashes. The experiment uses the PR 552 Python build, whose version and native
extension hash are recorded separately from the study script.

| Check | Result |
| --- | --- |
| Actual noisy d3 preparation and one complete flagged check | 32 conditional physical states reconstructed with one T; all emitted preparations compile at active width one |
| Constructed repetitions of the same physical d3 check | 32 trajectories with two checks and 32 with three; every final state still has a one-T reconstruction |
| Independent physical instruments | 192 check applications compared against Aer, including conditional probabilities, complete physical states and 384 flag records |
| Algebra at d3 and d5 | Ideal entry eigenstate relation checked; 32 Pauli entry corrections per size, both outcomes, verify the conditional identity |
| Physical fault binding at d3 and d5 | 32 local histories per size, both outcomes; the relative coherent path remains the certified Pauli |
| Input outside the sufficient condition | Same parity check on a product input containing one T produces states outside the one-T class for both outcomes |

The physical d3 experiments include the no-fault case, histories drawn from
the original p=0.001 channels, and deliberately denser stress histories.
No measurement result or flag value is postselected away. These are checks
of conditional states and instruments, not estimates of logical error rates.
Maximum full-state reconstruction error was below 7e-17 up to global phase;
the maximum conditional probability discrepancy was below 7e-16.

Tomography exhausts all 4^7 data Paulis, extracts the independent stabilizers,
and verifies the remaining logical Bloch vector is a Clifford image of T.
It then emits a one-T preparation, reconstructs the spectator state, and
compares the complete 17-qubit state against Aer. Clifft independently checks
the seven-qubit data preparation and the width of the complete reconstruction.
This establishes a physical-state boundary, not just a terminal distribution.
The exhaustive tomography is an oracle; it is not a scalable implementation.

The noisy entry probes for both supplied circuits found the original signed
code and the expected logical T axes on all 256 sampled trajectories. At exit,
52/256 d3 and 183/256 d5 trajectories left that original signed code space.
That observation alone would misleadingly suggest a larger state is needed:
the d3 full-state checks instead find a different Clifford encoding of one T.
The d5 algebra supports the same conclusion under the certified entry
condition, but full d5 conditional-state reconstruction was not performed.

The counterexample starts with a product of seven plus states and applies T
to one of them. After the same data parity projection, the two outcomes have
probabilities 9/16 and 7/16. Both output states have zero independent Pauli
stabilizers: stabilizer nullity seven, versus at most one for a one-T state.
Thus even allowing a different Clifford frame cannot fit these outputs into
the current carrier. An entry certificate is necessary for this reduction.

## What chaining has and has not established

The repeated-check controls retain physical flags, reset, Pauli noise and
both central outcomes. They repeat the same check, rather than representing
two independently sourced protocols. After this check the state is an
eigenstate of C again, up to sign: the common action A commutes or anticommutes
with C. This explains why another identical check can reuse the condition,
including Pauli errors between repetitions.

An arbitrary bridge to a different check need not preserve the eigenstate
condition. In particular, the d3-to-d5 growth and intervening measurements
of the supplied four-round protocol still need a compositional certificate.
Sampled d5 entry probes support the condition, but are not a proof for every
fault and measurement history. RP2's unresolved intermediate code and Sahay
Fold's different gadget remain outside this experiment.

## Assessment and next step

Keep the one-T bound for the next experiment. These results justify trying
an algebraic check reducer before investing in a general coherent-sum carrier.
The existing generic phase-region frontend and production executor remain
unchanged; this study does not supply a faster sampling mode.

The next bounded task is to replace dense tomography with a certificate and
direct Clifford/Pauli-instrument lowering, then compose it across the actual
small-check, growth and large-check sequence of the four-round d5 circuit.
Preserve every original visible record and physical fault location, including
reported-bit flips and rejected trajectories. Compare complete conditional
states or exact record probabilities against the physical reference. Only
after that closure works should larger-distance cost become the next question.

The algebraic condition can be phrased without naming a code family. Whether
it can be discovered and maintained cheaply from raw circuits is still an
implementation and applicability question. General coherent sums remain a
useful fallback direction for inputs that fail this condition.

## Reproduction

No external checkout or native profiling worker is needed. Use this branch's
installed Clifft package and its development dependencies:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  tools/profile/study_chan_boundary.py --trials 32 --probe-shots 256 \
  --output /tmp/chan-boundary-study.json
```

[Retained inputs](chan-boundary-inputs.json.gz) contain both complete circuit
texts, original circuit hashes, audit boundaries and upstream provenance.
No temporary VM path is required to rerun the experiment.
