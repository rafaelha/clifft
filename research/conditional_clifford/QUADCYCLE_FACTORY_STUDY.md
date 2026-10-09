# State-preserving continuation toward a quadcycle factory

Date: 2026-10-09. Branch: `codex/quadcycle-factory-study`, forked at `aa8199d5`.

## Objective and comparison with the theory method

The goal is to make circuits practical when ordinary Clifft requires too much
non-Clifford width. Ordinary Clifft can remain the preferred backend on other
circuits. Equal performance on already-cheap circuits is not a success criterion.
Circuit entry is all zero. Specialization may depend on sampled Pauli faults
and observed preparation outcomes. Production execution architecture is unchanged.

The application target is
[Single-shot magic state factories for quantum LDPC codes](https://arxiv.org/html/2610.10731v1).
The smallest instance couples three tricycle/fan-out blocks through CCZ, then
teleports into three quadcycle blocks. This adds a concrete continuation to the
previous isolated BT workload. The paper's reported numerical treatment uses
single-pair marginal channels; that does not retain the complete correlated
three-block channel.

[Efficient fidelity simulation of high-rate magic distillation circuits](https://arxiv.org/html/2610.03605v1)
retains CCZ effects algebraically for its supported syndrome and fidelity queries.
It produces syndrome samples; it does not generally emit every original logical
or physical readout. Supplement S5 handles specified teleportation/feedforward
gadgets, with assumptions on feedforward noise. We have not benchmarked that
method, so there is no measured speed comparison. Its narrower query contract
and polynomial guarantee make it a strong reference for acceptance/fidelity.

The additional capability sought here is the actual record together with the
conditional quantum state needed for further circuit execution. A record alone
does not specify that state. Potential uses include joint output distributions,
continuation in a different measurement basis, and corrections whose physical
implementation/noise depends on the realized logical result. Syndrome-based
decoding/postselection and individual Pauli expectations are already covered
by the theory approach; they are not unique motivations for a full sampler.

## What was implemented

`SharedPhase(..., preserve_state=True)` now exposes `render_state(controls)`.
It builds the reduced state, restores its embedding on the original physical
wires, and applies the fault/outcome-dependent affine offsets as physical X
operations. Original preparation records remain in their original slots. The
unmodified continuation can then execute quantum gates, measurements, resets,
and ordinary record-controlled Pauli gates in Clifft.

This export requires that no quantum measurement occur inside the phase region
after its initial Clifford prefix. The prefix itself may measure and reset.
State export disables terminal readout-basis absorption: replacing `H; M` by
`MX` preserves outcome probabilities but changes the collapsed physical state,
which matters if a continuation uses that wire. Constant computational-basis
wires also remain quantum at the interface, rather than becoming sampled bits.

This is an ensemble-state replacement for the specified circuit input and
sampled preparation trajectory. It is not a unitary identity valid for an
arbitrary unknown input. A second phase region can execute in the continuation,
but automatic reduction of that second region is not implemented.

The generic fault host additionally accepts adjacent `CORRELATED_ERROR` /
`ELSE_CORRELATED_ERROR` chains as a single categorical event. Each ELSE weight
is multiplied by the probability that all preceding alternatives failed. This
allows a physical CCZ location to draw one of 63 nonidentity three-qubit Paulis
without factorizing its correlations. CCZ is decomposed exactly by the existing
parser; there are no added fault locations inside that mathematical decomposition.

All conditional construction and analysis remain outside the production executor.
Neither Stim nor the Clifft source, public API, or dispatch contracts changed.

## Coupled factory diagnostic

The diagnostic reconstructs the smallest code pair from the factory paper's
matrices and polynomials, using pinned Merlin CSS linear-algebra helpers at
`097380fac1a3968ca47925146e211fe990f4c396`. It checks CSS commutation, ranks,
the logical escape map, and the CCZ tensor. The physical tensor uses the
orientation formula in the theory supplement. Its restriction to the code
depends on nine logical coordinates and gives the six-term determinant phase.
Fan-out distributes its 54 CCZ gates so each of the 162 source qubits participates
in exactly one gate.

The simulated diagnostic includes:

- Three 27-qubit tricycle preparations, with measured Z checks and linear
  feedback recovery, plus their 27-qubit fan-out copies.
- Three 54-qubit quadcycle zero-state preparations, with measured X checks
  and linear feedback recovery. Each uses 36 separate extraction ancillas.
- All 54 physical CCZ gates coupling the three source blocks, with correlated
  three-qubit depolarization at strength `2p`.
- Physical escape CNOTs, all 162 source X readouts, and Pauli-frame corrections
  selected by their logical parities.
- Ideal destination decoders and 18 logical X readouts for diagnostics.

There are 432 physical qubits and 369 records: 189 preparation, 162 escape,
and 18 destination logical readouts. The transformer receives raw circuit text;
the code matrices and logical labels belong only to the workload generator.

This is explicitly not the paper's final performance model. Extraction uses
a simple serial ordering, not the optimized circuit schedule. Recovery is a
linear right inverse on independent checks, not metacheck repair or a
most-likely decoder. There is no idle noise or postselection. Destination
diagnostic decoding/readout and virtual Pauli corrections are ideal. It does
not include the downstream extraction of two separate CCZ states or their
consumption in an algorithm. No logical-error-rate or yield estimate is made.

## Structural result

| Path | Ideal active width | Noisy active width | Noisy T count |
| --- | ---: | ---: | ---: |
| Ordinary default compilation | 9 | 33 | 378 |
| Shared state export, raw tracing | 21 | 21 | 216 |
| Shared state export plus existing optimizer | 9 | 9 | 33 |

Noise is `p=0.001` at the diagnostic preparation gates, resets, extraction
measurements, fan-out gates, escape gates, and source readout. Nine noisy-region
variants are exercised: identity, four fresh draws, three selected three-qubit
CCZ outcomes, and five simultaneous CCZ faults. Each retains a sampled actual
prefix branch and runs 16 shots of the live noisy continuation. An ideal case
adds another 16 shots. These are conditional batches, not 160 independent draws
of the complete noisy factory. No cache is used.

The shared affine support has 105 variables. Its synthesized state reconstructs
324 nonconstant physical wires; active width nine is a different quantity.
The raw state export is not yet compact enough for direct dense execution in
this diagnostic budget. Applying the existing optimizer after export and
attaching the continuation obtains width nine. Unlike the preceding terminal
BT study, this experiment therefore runs phase optimization per variant.

This is an execution-feasibility result, not a throughput improvement claim.
Wide ordinary/raw plans are inspected without dense-state allocation. The
artifact records construction and analysis times but does not establish a
comparison with the theory paper's simulator. The unchanged noisy continuation
does not require its faults to be known before this reduction.

## Correctness evidence

- **Independent quantum instruments:** 16 bounded cases cover 110 fixed fault
  histories and 129 enumerated prefix branches. Aer density matrices retain
  every output-qubit matrix element separately for every classical record.
  This checks phases and quantum/classical correlations, including reuse of a
  measured wire, reset branches, teleportation, and a noncommuting continuation.
  Maximum entry error is below `1.4e-16`. Clifft record probabilities are also
  checked on the exported branches.
- **Categorical noise:** nonuniform correlated chains, zero/one probabilities,
  and all 63 nonidentity three-qubit outcomes are tested. Bell readout makes
  every Pauli distinguishable; 65,536 Stim shots check the three-qubit channel.
  All 64 outcomes, including identity, additionally pass the instrument check
  through CCZ and a noncommuting continuation.
- **Ideal escape channel:** three encoded Bell halves are teleported into the
  destination. Exact Stim flow constraints certify 81 deterministic output
  stabilizers, including both Bell correlations for every logical pair, the
  destination code checks, and the unused logical zero states. This covers all
  source measurement branches and certifies the ideal channel, not just its
  action on an encoded plus state.
- **Full conditional factory records:** four combinations of region/tail
  faults and two prefix branches each are compared with direct conditional
  compilation of the original physical circuit. The reference evolves the
  actual fixed-fault Clifford preparation with Stim and keeps the original
  physical CCZ circuit. Across 128 sampled full records, replay probabilities
  agree to the reported precision (maximum relative discrepancy zero).
  Both directions supply records to check. This is a selected-probability
  check conditional on the existing optimizer, not equality of all output laws.
- **Regressions:** the prior shared-phase validator passes all small, renamed,
  distillation, and conditional BT checks. Existing generic channel and measured
  branch checks pass. Older result artifacts retain their original provenance.

The current exact-law checks retain the previously documented deterministic
padding normalization for the installed Stim flow oracle.

## Next checkpoint

The state interface composes on the bounded independent tests, and the coupled
noisy diagnostic stays within an executable width after optimization. This
removes the terminal-readout restriction for this tested continuation. It does
not establish efficient arbitrary-region composition or repeated optimization
after an unknown non-Clifford input.

Before quoting factory acceptance or fidelity, align extraction scheduling,
metacheck/recovery choices, idle noise, and permitted postselection information
with a specified protocol, and establish a matching reference calculation.
Use virtual diagnostic measurements only for evaluation, never as information
available to the physical decoder or acceptance rule. Compare the resulting
queries with the theory method under identical assumptions. Throughput tuning
and broader region discovery can then be judged against that concrete workload.

Retained artifacts:
[width study](quadcycle-factory-study.json),
[instrument validation](phase-continuation-validation.json), and
[conditional factory validation](quadcycle-factory-validation.json).
Reproduction commands are in the
[profiling guide](../../tools/profile/README.md#state-preserving-phase-continuation).
