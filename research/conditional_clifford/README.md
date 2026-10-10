# Conditional Clifford reduction: research review

The [complete-circuit checkpoint](COMPLETE_CIRCUIT_CHECKPOINT.md) freezes BT81
direct-X, Steane E, Chan four-round d5, RP2 d5, and reconstructed folded f7
against current Clifft `847757e2`. It reuses the existing programs and reports
complete attempts, state-interface failures, and total process-tree memory.
Its [consumer review and scope proposal](COMPLETE_CIRCUIT_CHECKPOINT.md#concrete-consumer-and-optional-mode-scope)
now validate decoder-based FEC before folded logical readout, first at f3 with
dense Aer and then at f7 with an independent physical oracle. This consumes the
existing live state without a generic output interface. The decision is to
propose a limited optional sampling mode with explicit per-route certificates,
refusals, output limitations, and full costs; production implementation remains
outside this checkpoint. The surveys below are prior evidence.

The [SOFT/RP2 transfer study](FAMILY_TRANSFER.md) validates the same bounded
update on complete supplied d3/d5 cultivation-only models. RP2's morphed code
remains compact on the tested histories. SOFT d5 requires preserving a final
physical T layer, so its exact output export still has width 8-10.

The [larger color-code scaling study](COLOR_CHECK_SCALING.md) tests the same
conditional update on constructed d7/d9/d11 checks, with explicit physical
noise and independent coherent-state validation. These are scaling controls;
the complete author-supplied Chan evidence remains at d3/d5.

The [algebraic Chan continuation checkpoint](CHAN_ALGEBRAIC_CONTINUATION.md)
now replaces the earlier exhaustive boundary reconstruction with a bounded
state update and tests the supplied four-round d5 circuit through growth and
both physical checks. This is research tooling; production execution is
unchanged. The broader survey below remains the preceding prototype's result.

This branch explores an optional way to sample circuits whose ordinary Clifft
active width is too large. It specializes on sampled Pauli faults and
measurement outcomes, using algebraic structure of the prepared state to
replace part of the circuit before invoking Clifft. The result may be Clifford
or may retain a much smaller non-Clifford computation.

This is a research prototype for review, not a merge-ready public mode. It is
based on [PR 552](https://github.com/unitaryfoundation/clifft/pull/552) at
`ed2a125ad7daf75deef7e21d66af2a4af4a366bf`. Relative to that base, the work is
in profiling tools, opt-in build targets, and research records; it does not
change production compiler/executor semantics. Review that incremental scope,
not the inherited PR 552 changes, when assessing this experiment.

## Reading guide

1. This summary: the current approach, useful cases, and limits.
2. [Latest capability survey](CAPABILITY_SURVEY.md): complete circuits, costs,
   validation, and the assessment checkpoint.
3. [Method and implementation](METHOD.md): algebra, interfaces, and code map.
4. [Lessons from approaches set aside](LESSONS.md): failed shortcuts, mixed
   performance results, and what those results actually rule out.
5. [Reproduction and evidence](REPRODUCING.md): run retained circuit inputs
   without this VM's temporary directories; inspect the selected evidence.
6. [Flagged Chan boundary study](CHAN_BOUNDARY.md): a later algebraic result
   that can retain one T across a complete measured check, under a certified
   entry condition; not yet integrated into the frontend.

## How it works for a Clifft user

Ordinary Clifft compiles a circuit once, retaining stochastic operations for
many shots. This host instead prepares reusable analysis once and constructs
a conditional program for each shot:

1. Discover a Clifford preparation and a supported phase region automatically.
   Derive the stabilizer state's affine support and a phase polynomial modulo
   eight. Dependency-safe measurement deferral can enlarge that region.
2. Separate the fixed non-Clifford phase from Clifford corrections controlled
   by faults and preparation outcomes. Draw the complete original noise once
   per shot and evaluate those corrections. Reuse does not require repeated
   fault histories.
3. Reconstruct an equivalent conditional physical state, restore earlier
   visible records, and retain the complete continuation. A Clifford exit can
   feed another reduction. One surviving Pauli rotation has a bounded carrier
   through supported Clifford measurements/resets.
4. Optimize and sample the remaining complete program with Clifft. Where
   eligible, reuse the optimized preparation, traced Clifford continuation,
   and a guarded squeeze permutation. Planning remains fresh per shot.

The all-zero assumption applies only at circuit entry. This is an equivalence
of prepared conditional states, not a unitary identity valid for arbitrary
unknown input. Measurements, hidden reset branches, feedback, detectors and
observables remain part of the sampling contract. No postselection or
fault-weight cutoff is introduced.

## Latest findings

The frozen survey covers 33 inputs: 24 complete protocol variants and nine
small controls. All 4,224 prototype shots fit the width-12 execution budget;
nine inputs exceed that budget under ordinary Clifft. These are observed
widths under the existing bounded optimizer, not lower bounds on every
possible simulator.

| Complete input | Ordinary active width | Conditional active width |
| --- | ---: | ---: |
| Noisy factory | 33 | 8-9 |
| Steane D / E | 33 | 8-9 / 6-9 |
| Noisy BT27, scored / direct X | 33 | 0 / 5-7 |
| Noisy BT81, scored / direct X | 87 | 0 / 3-7 |
| Noise-free BT81, scored / direct X | 56 | 0 / 6 |
| Cultivation d3 | 4 | 1 |
| Cultivation d5 | 10 | 10 |

BT81 direct X retains 23 optimized T gates: becoming fully Clifford is not
necessary for a useful reduction. The noise-free BT81 result also supports
pursuing noise-free optimizations independently.

The structural fit is a large CNOT/diagonal-phase computation whose action
simplifies on a stabilizer preparation's actual support, possibly after
legal measurement movement. The tested protocols are mostly related
Clifford+T error-correction/distillation constructions. Variants of their
noise and readout do not count as independent circuit families.

Ordinary Clifft is faster on all 24 cases it can sample within the survey
budget. Small distillation circuits simplify but were already cheap. Noisy
BT81 takes about 36-53 ms per prototype shot and 20 seconds of setup. For its
scored variant Merlin takes about 35 ms per shot and 0.19 seconds of setup;
the installed Merlin rejects the direct-X readout. Full comparisons, including
setup amortization and timing limitations, are in the survey.

## Applicability and workflow limits

- The input state is all zero. Supported noise consists of categorical Pauli
  events and readout flips, including adjacent correlated/ELSE chains.
  General quantum channels and arbitrary input states are outside scope.
- Discovery is conservative. The current bounds are 128 affine support
  variables, 200,000 phase terms, and eight regions. Unsupported structure
  retains an exact ordinary continuation where the input language permits it;
  that continuation may still be too wide.
- Only one surviving rotation has specialized transport. Multiple rotations
  remain for ordinary Clifft. Two/three-region chaining is independently
  checked on small controls; practical gains from chaining several interacting
  non-Clifford regions in a large protocol remain unestablished.
- This is a host workflow, not the public compile-once/batched sampler or a
  mid-dispatch executor continuation. Its control evaluation, tableau work,
  source reconstruction and planning finish before ordinary execution. It
  does not alter allocation or topology-planning rules inside hot dispatch.
- Setup and per-shot compilation can dominate. There is no automatic choice
  between ordinary Clifft, this prototype and Merlin. Seeded bitstrings need
  not coincide across different sampling implementations.

The checks include independent small quantum instruments, exact selected
large-circuit probabilities, and full recompilation comparisons for reused
fragments. The 128-shot timing batches are not rare-error-rate validation or
an exhaustive large-circuit equivalence proof.

## Assessment and next step

Compilation reuse has reached a useful checkpoint. The supplied Chan circuits
from #495 now provide the next structural test, without acquiring more inputs.
The [boundary study](CHAN_BOUNDARY.md) finds a sufficient algebraic condition
under which a complete noisy flagged check returns to a Clifford image of
one T. Independent d3 physical-state checks and repetitions support that
condition; a counterexample shows it cannot be assumed for arbitrary one-T
entries. The existing frontend and the frozen survey above are unchanged.

Keep the one-T bound. Next, replace the study's dense reconstruction oracle
with a certificate and direct lowering, and test composition through the actual
small-check, growth and large-check sequence of the four-round d5 protocol.
That will distinguish useful whole-circuit closure from another local saving.

## Historical provenance

This review snapshot consolidates the earlier incremental reports and removes
superseded output artifacts from the current tree. The complete earlier
research record, scripts and measurements remain at
[`d126c29e`](https://github.com/unitaryfoundation/clifft/tree/d126c29ec60d9dfcecfcbe0171632fd91f4809dc/research/conditional_clifford).
The retained survey and validation artifacts keep their original source and
binary hashes; they are measurements of those revisions, not newly relabeled
runs of the review packaging. Older helper modules remain where the current
prototype or validators import them. Standalone historical drivers may require
their archived inputs; use the reproduction guide for current entry points.
