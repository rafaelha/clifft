# Structural transfer to a held-out encoded IQP circuit

The phase algebra transfers unchanged to a complete reconstructed circuit
with 24 logical qubits. The stronger finding is that **conditioning the
faults alone already makes this circuit affordable**, and is faster than
the phase rewrite. This checkpoint supports a general conditional-compilation
host, but does not establish that phase reduction is necessary for this
input. It supplies no reason to build another family-specific simulator.

Folded/FEC work is paused. Existing reducers and production source are
unchanged. The reusable conditions, current implementation restrictions,
and family-specific facts are separated in [METHOD.md](METHOD.md#structural-conditions-versus-incidental-assumptions).

## Why this mechanism and input

Prior complete-circuit evidence gives the phase route the stronger claim to
opening otherwise large ordinary widths. The measured-check route has
transferred across cultivation families, but their supplied complete d3/d5
circuits remain ordinary-affordable; the larger d9/d11 constructions are
supporting controls. Phase reduction does not require one logical qubit or a
one-T output. This input tests that distinction directly.

The literature motivation is Bluvstein et al.'s
[logical neutral-atom processor](https://doi.org/10.1038/s41586-023-06927-3):
encoded IQP sampling using three logical qubits per eight-qubit block,
transversal diagonal gates, inter-block CNOTs with hypercube connectivity,
and X readout. Those ingredients motivate the reconstruction; this is not an
author export or a reproduction of the published experimental schedule.

The local specification is eight blocks, 64 physical/24 logical qubits,
four alternating diagonal layers, and CNOTs along three successive cube
dimensions. Each block uses the independently constructed affine encoder
`q_v = g XOR a*v0 XOR b*v1 XOR c*v2` and logical input signs minus, plus,
minus. Even layers use T-dagger on every vertex; odd layers use T on vertices
0-3 and T-dagger on 4-7. CNOT direction is low to high along each dimension.
The input includes initialization, all gates, all 64 physical readouts,
eight global-X parity detectors, and 24 logical-X parity observables.

Synthetic noise is p=0.001: reset X errors, one/two-qubit depolarization
after gates, and readout flips. All 656 categorical sites remain present.
The encoder schedule and diagonal pattern are explicit reconstruction
choices. Transport, idles, loss, leakage, coherent calibration error,
two-copy experiments, and a calibrated experimental noise model are omitted.
There is no decoded logical-error-rate or experimental-performance claim.

The source and existing reducer hashes were frozen before its reducer run
at research revision `d8bc2f3f`. Its source SHA-256 is
`9d2098ca5ebbe211a9192c6944f48ea0cf64248110d369759c11ac465a219c1a`.
"Held out" means a new complete input outside the retained reducer
development/evaluation corpus, not a claim that the paper was never discussed.
No circuit-name rule or code matrix was supplied to select a reduction.
The measured-check recognizer explicitly refuses it with
`no certified check envelope`; the phase certificate succeeds.

An earlier 32-physical/12-logical reconstruction was retained as a control:
ordinary width 16, phase width 10-11, ordinary 0.925 ms/attempt versus phase
2.224 ms/attempt. It did not meet the challenge criterion. The next hypercube
instance was therefore selected; both inputs and all results are retained.
There was no search over schedules for one that favored the reducer.

## Complete-attempt results

Both routes use pinned Clifft
`847757e2bc8c4f7439b9e92011550fd61719eb85`, the same build as the
[complete panel](COMPLETE_CIRCUIT_CHECKPOINT.md). Binary hashes are verified
in the manifest. Runs use one thread on the recorded EPYC host. Memory is
20 ms sampled peak process-tree RSS, including imports, setup, attempt
execution and serialization, excluding only the supervising parent and the
separate correctness audits. It is observed memory, not a hard upper bound.

| Complete route | Peak active width | Residual T rotations | Coefficient payload | Setup | Mean per attempt | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary noisy source | 32 | Not executed | 64 GiB required | 3.49 ms to width guard | Not measured | 37.8 MiB before allocation |
| Existing specialized phase worker | 20-24 planned | 806 | 16-256 MiB required | 400 ms | 9.33 ms to refusal | 111.3 MiB |
| Unchanged phase export + public Clifft compiler/executor | 20-24 | 806 | 16-256 MiB | 183 ms | 327 ms | 319.4 MiB |
| Original physical source with faults fixed, public compiler/executor | 21 | 112 | 32 MiB | 7.57 ms | 66.0 ms | 86.0 MiB |

The ordinary route was stopped before allocating width 32; its total
execution memory and attempt time are unknown. The allocation budget was
initially 20 for ordinary Clifft and 16 for the specialized worker.
**All 32 worker attempts refused at the width guard.** These remain labeled
`certificate_failure` in the original assessment artifact; specifically,
they are resource-guard failures, not failures of the phase identity.

Before adding assessment plumbing, its missing capability was identified:
execute the already available exact conditional source above the specialized
worker's hard width-16 limit. The adapter calls existing public trace,
optimization, lower and sample operations, with a declared width-24 budget.
The budget was selected after observing the worker's planned widths. Neither
the reducer nor the native worker was modified. No new handoff, sampler,
coordinate algorithm, or hot-execution behavior was introduced.

The same 32 draws and quantum seeds were replayed, with no history discarded
or redrawn. Every attempt completed; 24 passed detector postselection and
eight were rejected. All records, detectors, observables, faults, seeds and
exact conditional programs are retained. Widths were 20 on 29 histories,
21 on one, 23 on one, and 24 on one. Mean compilation/preparation cost was
18.2 ms and sampling cost 307 ms; complete attempt cost also includes fault
drawing, rendering and record handling. The slowest attempt took 2.87 s.
Setup and cold process wall time (11.55 s for 32 attempts) are separate.

Intermediate structure is not hidden behind a small reported core:
32 affine support variables, 1,654 phase monomials, 704 conditional Clifford
responses, 1,568 primitive Boolean fault controls, and 95,448 bytes of packed
dependency masks. There are 1,456 synthesized T gates before prefix
optimization and 806 afterward, plus the full physical encoder (96 CNOTs),
Clifford corrections and 64-qubit reconstruction. The conditional source
retains the quantum preparation before final readout. After all 64 X
measurements the quantum output is a product of their physical eigenstates;
reported readout flips must be undone to recover those physical signs.

The fault-only ablation uses the same existing categorical noise renderer
and public compiler, without phase analysis or a reduced preparation.
All 32 attempts complete at width 21 with 112 residual T gates, including
eight rejected attempts. Mean compile cost is 2.00 ms and sampling 62.3 ms;
complete attempt cost is 66.0 ms. Cold process wall time is 2.34 s. This route
is about five times faster than the phase export here. Different compiled
programs need not produce identical records from the same RNG seed; the
comparison below uses both programs' sampled records as probability queries.

## Correctness and limits of the evidence

Before the larger execution, a complete 16-physical/6-logical instance was
checked against dense Qiskit Aer 0.17.2. Eleven histories include the ideal
case, eight ordinary draws, all sites at their last nonidentity outcome,
and every readout flipped. The full live state before readout agrees up to
global phase to 3.50e-16 maximum amplitude error. All 65,536 output
probabilities agree to 7.05e-19 maximum absolute error. These selected stress
histories are correctness controls, not unconditional resource guarantees.

A planned independent large physical MPS oracle was not completed. The
small test first exposed inconsistent `save_amplitudes` results in the
installed Aer MPS path. Direct contraction of saved MPS tensors was checked
against the complete dense distribution and 260 statevector entries, but
the untruncated 64-qubit MPS run was interrupted before its first query
completed. No large-instance independent Aer agreement is claimed.

The larger validation instead compares the phase export with the original
fixed-fault physical circuit through Clifft's full-record probability
engine. This is independent of the research reducer but shares Clifft's
compiler and execution code. For each of the 32 histories, the queries are
the complete record sampled by each route and a one-bit mutation of the
phase route's record. Annotation removal and reset purification are used
only for probability queries; timed sampling retains all annotations and
records. All 96 comparisons pass: maximum nonzero relative error 4.99e-14,
with all 31 zero-probability queries also exactly zero.

The algebra covers supported Pauli histories, but 32 draws do not prove
that every history fits width 24, nor that other layer schedules or larger
algorithms remain affordable. No fault-weight cutoff is imposed. A future
mode must report a resource/certificate refusal on the actual history,
preserve that draw for a supported fallback, and never replace it with an
easier draw. This test does not exercise intermediate adaptive measurements
or general quantum input to the first phase region. Competitiveness against
bespoke IQP/tensor-network methods was not measured.

## Decision

**The phase mechanism works unchanged on the tested complete circuit; its
specialized worker needs a broadly applicable resource-budget extension,
not another circuit-family algorithm.** Existing conditional export plus
ordinary execution already supplies the needed capability within the
declared budget. However, the fault-only ablation is the better route for
this input, so this is not evidence that phase reduction itself causes the
capability gain.

The clearer optional-mode scope is a host that draws the complete Pauli
history once and first assesses the original conditional program. Certified
phase or measured-check reductions are optional transformations whose
**complete** costs and output contracts must justify using them. The phase
route needs affine/diagonal structure and an affordable residual; the
measured-check route needs a small signed common subspace and a supported
output. Ordinary static compilation remains preferable on the small control.
These are structural dispatch criteria, with explicit refusal and budget
semantics, not a list of supported circuit names. This is a scope proposal;
production implementation, broader scaling and compiler tuning stop here.

## Artifacts and reproduction

The [manifest](heldout-iqp-manifest.json) records baseline, build, source and
artifact hashes. Inputs/results are `heldout-iqp-{input,results}.json.gz`;
the result index references lossless per-route sidecars for the public
executors to keep individual files below the repository size limit. The
earlier affordable control adds `control-`; validation is in
`heldout-iqp-{small,large}-audit.json.gz`. The results retain the original
worker failures alongside both successful public-executor routes.

Build/package the pinned baseline and native tools using the
[complete-panel instructions](COMPLETE_CIRCUIT_CHECKPOINT.md#reproduction).
Expose binaries as `phase_exporter` and `phase_worker` in a directory passed
with `--bin`, and set `PYTHONPATH` to that pinned Python package followed by
`tools/profile`. All commands below accept that `--bin` option. For example:

```bash
python tools/profile/audit_heldout_encoded_iqp.py small --output /tmp/iqp-small.json.gz
python tools/profile/study_heldout_encoded_iqp.py run --output /tmp/iqp-results.json.gz
python tools/profile/study_heldout_encoded_iqp.py export --max-width 24 --output /tmp/iqp-results.json.gz
python tools/profile/study_heldout_encoded_iqp.py ablate --max-width 24 --output /tmp/iqp-results.json.gz
python tools/profile/audit_heldout_encoded_iqp.py large \
  --input research/conditional_clifford/heldout-iqp-input.json.gz \
  --results /tmp/iqp-results.json.gz --output /tmp/iqp-large.json.gz
```

The final command runs the native physical-source comparisons, not the
abandoned large MPS reference. Keep `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` for comparable timing. Use `freeze --dimensions 3
--input /tmp/new-iqp-input.json.gz` only to regenerate a separate input;
replaying the retained one checks that the existing reducers are unchanged.
