# A second protocol family for confidence in conditional reduction

Date: 2026-10-09. Branch: `codex/quadcycle-factory-study`.

## Purpose and workload choice

The immediate goal is confidence in the simulation method. Reproducing an
author's exact numerical curve is a later goal; unspecified implementation
choices should be recorded rather than silently assumed to match.

Keep both application workloads:

- The [quadcycle factory diagnostic](QUADCYCLE_FACTORY_STUDY.md) tests transfer
  into a larger code and continuation through physical measurement/feedforward.
- The depth-two tricycle protocols in
  [the theory paper](https://arxiv.org/html/2610.03605v1) provide a second
  validation target. Supplement S6.4 gives analytic single-fault predictions.
  Its shared-ancilla protocol E is an attractive subsequent workload; protocol
  D uses separate ancillas and avoids cancellation between block syndromes.
  E's favorable reported tradeoff is specific to the studied biased noise.

This adds a different continuation and a different CCZ arrangement. It is not
evidence of generality beyond code-based phase circuits; the existing small
unstructured instrument checks remain useful for that purpose.

## Independent bounded reference

`tools/profile/validate_theory_ccz_faults.py` reconstructs three ideal encoded
plus states from all-zero physical input and applies the 54 physical CCZ gates
in two layers on 81 qubits. The code and gate matrices use the same pinned
Merlin CSS helpers as the factory diagnostic. The reduction receives only raw
circuit text and categorical fault histories.

For a fixed fault history, the independent reference propagates physical Pauli
errors through the remaining CCZ gates. The resulting error consists of an X
offset and diagonal Z/CZ operations. Each CZ is expanded in Pauli Z characters;
integer coefficients are combined modulo the preparation's Z stabilizers
before amplitudes are squared. Characters label both X-check outcomes and
logical Z actions. The X offset fixes the Z-check outcomes.

This produces the complete joint law of 24 independent X checks and 48
independent Z checks, using exact rational arithmetic. It does not call shared
phase analysis, Clifft optimization, or a sampler to calculate that reference.
Clifft's reduced circuit is separately optimized and queried for every outcome
in the reference support. The reference probabilities sum exactly to one, so
checking the whole support also constrains probability assigned elsewhere.
Clifft comparisons are floating-point checks, not a formal proof.

The reference is deliberately bounded: its Pauli expansion can grow
exponentially and is capped at 65,536 characters. It is not an implementation
of the paper's full polynomial-time syndrome/fidelity algorithm. Its purpose
is an independent correctness oracle for this workload.

An additional 32 small random circuits compare the physical-error propagation
identity against direct Qiskit Aer statevectors, including multiple faults.
This comparison allows an irrelevant global phase.

## Checks and interpretation

The integer reference reproduces the aggregate single-fault predictions in
Supplement S6.4: 3,645 faults in total, 513 with nonzero X-check acceptance,
54 capable of leaving a logical Z error after the X component is removed,
and none accepted by both ideal X and Z checks. Those counts are a check of
the reconstructed workload as well as the reference calculation.

The candidate comparison covers every one of those single-fault histories,
the fault-free case, and 40 multiple-fault histories. Of the latter, 36 are
seeded choices with two, three, or five faults. Four deliberate accepted cases
exercise cancellation, an X error on both sides of a CCZ, a weight-three
logical Z, and that logical error accompanied by a cancelling pair. This
avoids relying only on random histories that postselection rejects.

Fidelity queries additionally append an X correction, inverse ideal CCZ, and
inverse preparation. The X correction knows the injected fault's full X
component. These queries test the quantum state left after X postselection;
they are not a physical decoder or an operational fidelity estimate. The
record queried includes the actual ideal Z syndrome, which is not used to
reject a shot for this particular X-only-postselection comparison.

The fidelity checks cover distinct single-fault acceptance/error classes and
every multiple-fault history. Full syndrome-law checks cover all histories.
The retained [result artifact](theory-ccz-fault-validation.json) records the
query counts, maximum discrepancies, widths, source hashes, and a digest of
the reference laws.

The completed run compared 103,463 joint-syndrome probabilities and 51 joint
postselection/fidelity probabilities. Maximum absolute discrepancy was below
`7e-18`; all candidate and scoring circuits stayed at active width nine.
The independent Aer propagation checks had state infidelity below `5e-16`.

The preparation and final syndrome measurements in this checkpoint are ideal.
The noise probabilities in the generated source only label categorical
locations; the checks condition on explicit faults and do not estimate a
noise-weighted protocol rate. There is no Monte Carlo confidence interval
because the comparisons query probabilities directly.

## Next checkpoint across the two workloads

The first item below is now covered by the subsequent
[physical Steane continuation study](STEANE_CONTINUATION_VALIDATION.md).
The noisy preparation and decoder checkpoint remains outstanding.

1. Add the physical shared-ancilla extraction of protocol E, with D as a
   control. Check explicitly the cancellation of equal syndromes in different
   blocks and errors propagating back from the shared ancilla. Retain physical
   readouts while comparing their inferred syndrome to the independent
   reference. Start with ideal preparation and controlled faults.
2. Add specified noisy preparation/extraction, metacheck handling, and a
   decoder. Freeze those choices for both candidate and reference. Virtual
   measurements and hidden fault labels must not inform actual acceptance or
   recovery decisions.
3. Continue aligning the quadcycle diagnostic's schedule, idle noise, and
   permitted postselection information. It remains the test for a quantum
   state exiting the reduced region into another encoded block.

Agreement on explicitly identical assumptions is the success criterion for
the next confidence study. Exact author settings and published rate curves
can be addressed separately, including author clarification if needed.

Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#theory-protocol-fault-validation).
