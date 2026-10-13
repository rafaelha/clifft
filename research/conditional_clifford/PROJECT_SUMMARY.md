# Clifft conditional simulation research: review handoff

Prepared 2026-10-13. Research results are frozen at
[`dbbc8865`](https://github.com/unitaryfoundation/clifft/tree/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford),
the last completed Aggregator checkpoint. This document consolidates the
research and the project owner's goals; it reports no new simulation results.
It is intended to be reviewable from public web pages without running code.

## 1. Motivation and the actual objective

[Clifft](https://github.com/unitaryfoundation/clifft/blob/847757e2bc8c4f7439b9e92011550fd61719eb85/README.md)
is an exact near-Clifford circuit simulator. Its dense coefficient state has
`2^k` complex amplitudes, where `k` is the active width, rather than the number
of physical qubits. Large physical circuits can be cheap when `k` stays small;
an apparently modest circuit can become impractical when `k` grows. Throughout
this document, `k` means Clifft active width, not a code's logical-qubit count.

The starting point was [PR 552](https://github.com/unitaryfoundation/clifft/pull/552),
which recovers stabilizer constraints through measurements and feedback for
static optimization. Its usable fixed constraints must hold across the
trajectories covered by the compiled program. The research asks whether
conditioning on realized faults and measurement outcomes permits stronger
state-specific simplifications than that static representation exposes.

The project owner's priorities are:

- **Unlock exact sampling of complete circuits beyond current Clifft's
  practical active width**, motivated by recent literature.
- **Support several meaningful circuit families through reusable structure.**
  A collection of one-family shortcuts is not the desired endpoint.
- Changing what is compiled once versus specialized per shot is acceptable.
  Conditional states and explicit compilation/continuation boundaries are
  legitimate possibilities.
- Improving existing small-width circuits or existing magic-state-cultivation
  (MSC) examples is not required. Ordinary Clifft may remain best for them.
- Competitiveness with bespoke literature methods is desirable, but follows
  establishing capability and breadth. A width reduction alone is insufficient:
  setup, per-attempt cost, total memory, and the output contract matter.
- Avoid open-ended tuning, increasingly elaborate reconstructions of one
  protocol, or building interfaces before identifying what they enable.

The work originated in the chats "LLM driven study", "LLM driven - flag
circuits", and "LLM Driven - Lukin paper". "Aggregator" subsequently
consolidated and extended it. The public reports linked here replace those
private conversations as the evidence a reviewer needs.

## 2. Current assessment

There are **four substantive research approaches**, plus ordinary Clifft as
the baseline. Different scripts, caching policies, state exporters, and the
folded recovery experiment are not additional approaches.

| Approach | Assessment against the owner's objective |
| --- | --- |
| Conditional phase reduction | Strongest direction: substantial sampling capability gains on BT/factory inputs and unchanged algebraic transfer to an encoded IQP computation. Scope remains structurally restricted. |
| Sample faults, then compile the original circuit | Promising simple alternative: sufficient and faster on the held-out IQP example, insufficient on the earlier hard panel. Breadth of useful cases is under-tested. |
| Measured-check/common-stabilizer reduction | Reusable algebra across Chan/SOFT/RP2, with positive larger constructed controls. No additional sampling unlock on the complete Chan/RP2 panel cases; substantial current overhead. |
| Folded specialized contractions | Strong capability within the reconstructed folded family, including f7 and a recovery continuation. Cross-family generality has not been established. |

The evidence supports investigating a bounded conditional-compilation
extension. It does **not** establish one efficient state representation for
all these methods, or an unrestricted multi-stage simulator. A common runner
interface would be engineering infrastructure, not evidence of unification.
These assessments are judgments for review, not established optimality claims.

## 3. What the methods actually do

### Conditional phase reduction

The frontend accepts complete physical circuit text and discovers a supported
region. It does not require code matrices, logical-gate labels, or manually
specified boundaries. For a Clifford preparation, its computational-basis
support is represented as

```text
q = A x XOR b_h
```

Here `h` includes sampled Pauli faults and preparation outcomes; `x` labels
the remaining quantum superposition and is not sampled as classical input.
CNOT gates transform the affine expressions. Diagonal gates contribute a
weighted Boolean phase polynomial modulo eight. Evaluating these phases on
the actual preparation support exposes cancellations and smaller residual
non-Clifford work.

Within the supported structure, the phase can be separated into

```text
P_h(x) = P_fixed(x) + Q_h(x) mod 8
```

where `Q_h` is a diagonal Clifford correction. For example, changing a T-phase
bit from `l` to `l XOR 1` changes its exponent by `1 - 2*l`: a global phase
and an S-dagger phase on that parity. Supported faults therefore change
Clifford corrections around reusable non-Clifford work. This does not imply
that every fault-dependent change in a general circuit is Clifford.

Setup derives the support, phase algebra, fault responses, and reusable
preparation/continuation fragments. Each attempt draws the original
categorical noise, samples preparation outcomes, evaluates corrections,
reconstructs the physical state and earlier records, and compiles/samples
the remaining program with Clifft. Planning still occurs per attempt.

The replacement is an equality of conditional prepared states, up to global
phase, for the specified input. It is not a unitary identity on arbitrary
unknown inputs. A Clifford exit can permit another reduction. One surviving
Pauli rotation has specialized transport through supported operations;
general reduction of a subsequent region with a larger non-Clifford input
remains unresolved. See the
[method and implementation map](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/METHOD.md).

### Fault-only conditional compilation

Draw the physical Pauli/readout history once, replace stochastic operations
with those outcomes, and compile the original remaining circuit using the
existing optimizer. Quantum measurements still have to be sampled with their
correct probabilities. This route performs no phase-specific state replacement.
Its value is both a useful policy and an essential ablation: it reveals which
gains require the more elaborate reducer.

### Measured-check/common-stabilizer reduction

A recognized noisy check has a coherent conditional operation of the form

```text
K_m = A_m (I + (-1)^m C) / 2,
C = D^dagger P D, with C Hermitian, Clifford, and C^2 = I.
```

The binder retains faults, flag records, and the relative phase between the
two branches. Given the incoming signed Pauli stabilizers `S`, it computes
`S intersect C S C^dagger`. A large intersection leaves only a small quantum
subspace in which to evaluate the projector. It then extracts the outgoing
stabilizers and preserves the physical Clifford encoding.

The efficiency condition is small codimension of that **signed** intersection
and a compact output on the actual history. A small logical-qubit count or a
transversal gate is insufficient by itself. The prototype allows at most three
temporary coordinates, compresses to at most one, and restricts export to the
Clifford orbit of a stabilizer or T state. Those are implementation bounds,
not mathematical impossibility results. The relative phase cannot be discarded
by treating the two Clifford branches as a classical mixture.
[Algebra and validation](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/CHAN_ALGEBRAIC_CONTINUATION.md).

### Folded contractions

This existing specialized research runner uses coherent-path contractions,
signed-code data, a small logical state, and a dedicated growth transfer for
reconstructed folded cultivation. Aggregator reused it rather than deriving
a generic folded reduction. The f7 complete runner includes a width-eight
ordinary prefix and four coherent paths; neither figure is its total memory.

There is also a separate
[opt-in integration experiment](https://github.com/unitaryfoundation/clifft/blob/b87c0da396b4bbde54fbee5535aeac1116170605/docs/guide/folded-specialization.md)
with strict terminal-region recognition. Its scope differs from the complete
research runner; in particular, that integration does not include the research
runner's specialized f5-to-f7 growth handoff. It should not be mistaken for a
general interface connecting all four approaches.

## 4. Complete-circuit results that drive the decision

The latest five-input panel uses Clifft baseline
[`847757e2`](https://github.com/unitaryfoundation/clifft/tree/847757e2bc8c4f7439b9e92011550fd61719eb85),
one CPU/thread, and 16 retained fault histories per input. The subsequent IQP
experiment uses the same native build and 32 histories. The panel's original
ordinary allocation budget was width 20; the fault-only ablation allows width
24. Refusals occur before allocating the exponential state.

| Input | Ordinary noisy width | Fault-only width | Existing reduction | Complete-attempt evidence |
| --- | ---: | ---: | --- | --- |
| BT81 direct-X | 87 | 56 | Phase width 4-7; 23 residual T gates | Phase 67.7 ms; fault-only remains over budget. |
| Steane E diagnostic | 33 | 30 | Phase width 9; 33 residual T gates | Phase 5.28 ms; fault-only remains over budget. |
| Chan four-round d5 | 10 | 10 | Measured-check carried core 1, temporary core 2 | Ordinary 0.0367 ms; fault-only 7.91 ms; measured-check 462.6 ms. |
| RP2 d5 | 18 | 18 | Measured-check carried core 1, temporary core 2 | Ordinary 5.12 ms; fault-only 19.43 ms; measured-check 4,382 ms. |
| Reconstructed folded f7 | 44 | 44 | Specialized prefix width 8 plus contractions | Specialized 174.7 ms; fault-only remains over budget. |
| Reconstructed encoded IQP: 64 physical, 24 logical qubits | 32 | 21 | Phase export width 20-24; 806 residual T gates | Fault-only 66.0 ms with 112 T gates; phase export 327 ms. |

The reported reduced cores and ordinary active widths describe different
representations. RP2's final export has width zero **after readout**; its
pre-readout quantum state still has one active coordinate. Times include
accepted and rejected attempts, with setup separate. A time to a resource
guard is not a completed sampling time.

Resource and scope qualifications:

- BT81 phase setup is about 18.3 seconds and peak process RSS about 3 GiB.
  Steane E setup is 0.44 seconds and peak RSS about 113 MiB. Small native cores
  do not eliminate host tables, frames, or Python overhead.
- Folded f7 setup is about 33.9 seconds and peak RSS about 352 MiB. Its ordinary
  width would require 256 TiB of coefficients alone.
- IQP ordinary width 32 requires 64 GiB of coefficients. Fault-only used about
  86 MiB peak RSS and 7.57 ms setup; phase export used about 319 MiB and 183 ms
  setup. Both completed all 32 draws, including eight rejected attempts.
- The specialized phase worker refused every IQP history at its width-16
  guard. The **unchanged phase export** succeeded through the public compiler
  with a width-24 budget selected after observing the planned widths. This is
  algebraic transfer plus an execution adapter, not an unchanged-worker success.
- The fault-only panel retained all 80 histories: 32 completed on Chan/RP2 and
  48 were resource refusals on BT81/E/f7. E's 16 GiB coefficient requirement is
  a declared-budget refusal, not a claim that no larger machine can run it.

These results show both that fault conditioning alone can unlock a workload
and that the whole phase route adds capability beyond fault conditioning on
other workloads. They do not isolate which individual phase/preparation pass
causes each gain. BT81 also conditions on 162 prefix records; E has no prefix
records, so its gain cannot be attributed to that measurement conditioning.

Primary reports:
[complete panel](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/COMPLETE_CIRCUIT_CHECKPOINT.md),
[fault-only ablation](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/FAULT_ONLY_PANEL.md),
and [held-out IQP](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/HELDOUT_IQP.md).

## 5. Other findings worth retaining

The earlier
[33-input survey](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/CAPABILITY_SURVEY.md)
covered 24 protocol variants and nine controls, with 4,224 fresh prototype
shots. Nine inputs exceeded its ordinary width-12 budget and became affordable.
These are **not 33 independent families**. Every complete protocol used one
discovered phase region; two/three-region chaining was demonstrated on small
constructed controls. Ordinary Clifft was faster on every input it could
sample within that survey budget.

Noise-free BT81 direct-X also improved from width 56 to six. This supports a
separate static optimization opportunity: some gains may come from richer
state analysis or phase-region capacity, not inherently from per-shot work.
All width comparisons are against the tested bounded optimizer, not lower
bounds on the circuit or on other optimizers.

The measured-check method transferred to complete supplied SOFT/RP2 d3/d5
models across 52 retained histories, with general binding/extensions rather
than code-name rules. SOFT d5 retains an unmatched terminal physical T layer:
its exact exported state still has width 8-10. A small internal check core
does not establish a small arbitrary continuation.
[Transfer study](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/FAMILY_TRANSFER.md).

Constructed color-code checks at d9/d11 have ordinary widths 31/46 and were
sampled with a carried one-coordinate core. This is positive scaling evidence,
but these are **constructed checks, not supplied complete larger cultivation
protocols with growth**. Independent coherent-state checks support them; they
do not fill the missing complete-protocol evidence.
[Scaling study](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/COLOR_CHECK_SCALING.md).

A folded continuation adds decoder-controlled Pauli recovery before logical
`H_XY` readout. One controlled history changes its final distribution from
50/50 to deterministic, showing a genuine need to consume the quantum state.
Small f3 checks used dense Aer; f7 used an independent physical coherent oracle.
No improved acceptance was demonstrated. This is still a specialized folded
capability, and further folded/FEC work was paused.
[Recovery experiment](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/COMPLETE_CIRCUIT_CHECKPOINT.md#concrete-consumer-and-optional-mode-scope).

## 6. Literature, circuit provenance, and comparison contracts

| Source | Role in this project and limits |
| --- | --- |
| Cercelescu, Koch, Pesah, [Polynomial-time simulation of non-Clifford quantum error correction](https://arxiv.org/abs/2610.06811), and [Merlin code at the pinned revision](https://github.com/mark-koch/merlin/tree/097380fac1a3968ca47925146e211fe990f4c396) | Circuit generators/CSS helpers and a comparison backend for distillation, cultivation, and BT examples. Whole-circuit synthetic noise and readout choices are specified in the reports. |
| Menon et al., [Single-shot magic state factories for quantum LDPC codes](https://arxiv.org/abs/2610.10731) | Motivates the reconstructed 432-qubit tricycle/fan-out/quadcycle factory with CCZ, transfer, and feedback. Scheduling, recovery, idle noise, and diagnostic readout differ from an experimental/full-performance model. |
| Xiao et al., [Efficient fidelity simulation of high-rate magic distillation circuits](https://arxiv.org/abs/2610.03605) | Supplies the D/E protocol motivation and analytic checks. Its supported syndrome/fidelity/observable queries do not equal a general sampler of all logical measurements and their joint outputs. Our D/E inputs are diagnostics with specified noise weights and ideal probes, not reproduced logical-error-rate estimates. |
| Chan et al., [Diagnosing and Restoring the Degraded Fault Distance of Magic State Cultivation](https://arxiv.org/abs/2609.17706) | Motivates flagged measured checks. Actual supplied d3/four-round d5 inputs come through the pinned corpus below; larger locally constructed checks have separate provenance. |
| Wan/Zapirain [cultivation corpus](https://github.com/kh428/exact_ler_for_msc/tree/b99452de7e300324e7bbf5cd9347241c3e289bae), [SOFT](https://github.com/haoliri0/SOFT), and [RP2 source](https://github.com/Zihan-Chen-PhMA/Cultiv_T_RP2) | Released physical cultivation inputs. SOFT is retained verbatim; RP2 uses documented actual-T conversions of S-proxy models, not an author-supplied actual-T export. Escape/decoding are omitted. SOFT d5's actual noise is 0.005 despite its misleading filename. |
| Takada, Bartlett, Williamson, [Exact efficient simulation of noisy logical magic states using Clifford stabilizers](https://arxiv.org/abs/2609.16929) | Motivation for folded cultivation/checks and recovery-before-readout. The Clifft folded schedules are reconstructions, not author exports; reproducing the paper's rates is not claimed. |
| Bluvstein et al., [Logical quantum processor based on reconfigurable atom arrays](https://arxiv.org/abs/2312.03982) | Motivates encoded IQP using eight-qubit/three-logical-qubit blocks, diagonal gates, inter-block CNOTs, and X readout. The tested 64-qubit circuit is a deterministic local reconstruction with synthetic Pauli noise, not the published experimental schedule. |

The original factory source, assumptions, and bounded independent references
are in the
[factory handoff](https://github.com/unitaryfoundation/clifft/blob/1bcbf96044824ed5d1602318d435707f641f0446/research/conditional_clifford/BENCHMARK_HANDOFF.md),
[quadcycle study](https://github.com/unitaryfoundation/clifft/blob/1bcbf96044824ed5d1602318d435707f641f0446/research/conditional_clifford/QUADCYCLE_FACTORY_STUDY.md),
and [D/E validation](https://github.com/unitaryfoundation/clifft/blob/1bcbf96044824ed5d1602318d435707f641f0446/research/conditional_clifford/STEANE_CONTINUATION_VALIDATION.md).
The reference calculation is a bounded oracle, not an implementation of the
theory paper's full polynomial-time algorithm.

Limited Merlin comparisons exist. On the older scored BT81 survey, Clifft's
prototype took about 53 ms/shot versus Merlin's 35 ms, with about 21 seconds
versus 0.19 seconds setup. The installed Merlin rejected the tested direct-X
variant; that is an implementation/input result, not an impossibility theorem
for Merlin's theory. No broad head-to-head superiority over bespoke methods,
or IQP tensor-network methods, has been established.

## 7. Correctness evidence and remaining uncertainty

Validation includes independent small physical states and complete measurement
instruments using Aer/Stim, bounded exact rational references for selected
D/E workloads, physical coherent-state references for folded/scaling studies,
and large full-record probability comparisons using original physical circuits.
Fresh full-compilation comparisons additionally validate compilation reuse.
Those checks have different independence and coverage; they should not be
combined into a claim of exhaustive large-instance correctness.

For IQP, a 16-physical-qubit instance passed independent dense Aer state and
full-distribution checks on 11 selected histories. The 64-qubit instance passed
96 original-source versus phase-export probability comparisons, with maximum
nonzero relative error about `5e-14`; those share Clifft's compiler/executor.
An independent large Aer MPS calculation was not completed. The 32 timing
histories do not prove every history fits width 24.

Accepted and rejected attempts are retained. A resource refusal is neither
a physical rejection nor permission to redraw an easier history. A fallback
after quantum outcomes have been sampled must preserve those outcomes and
their conditional state. Finite feasibility batches do not estimate rare
logical-error rates or establish all-history resource bounds.

See the [validation map](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/REPRODUCING.md)
and each later report's validation section. Machine-readable inputs, hashes,
histories, results, and manifests are committed alongside the reports; many
are gzip archives. The prose reports are the primary access path for a reviewer
that cannot download/decompress artifacts.

## 8. Lessons and alternatives still open

- **Full fault-history caching is not a general solution.** In a retained
  10,000-draw experiment, full-noise BT81 had no repeated complete histories.
  Reuse algebra and responses, not an assumption that whole histories recur.
- **A noiseless plan plus output flips is insufficient.** Faults can change
  the rank of the full record law and the required plan, not just signs.
- **Small internal state does not guarantee cheap complete execution.**
  Tableau work, corrections, reconstruction, planning, and continuation count.
- **Matching terminal probabilities is weaker than preserving a state.**
  Reset mixtures, physical frames, records, and relative phases matter at a
  continuation boundary.
- **Several region reductions are not yet a demonstrated large-protocol win.**
  All large phase-survey protocols used one reduced region. Cultivation d5's
  first region becomes small while its complete width stays ten.

These are tested limitations of particular shortcuts, not impossibility
theorems. Details and counterexamples are in
[Lessons](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/research/conditional_clifford/LESSONS.md).

**Compile once with symbolic corrections remains an alternative worth
assessing.** The desired program would preserve small width while accepting
fault/outcome controls at execution. Existing sign expressions alone do not
establish this: an optional S on `|+>` changes the stabilizer axis from X to Y.
A fixed plan covering such possibilities can require more active storage than
any individually specialized branch. Conversely, the current per-shot cost
does not prove that fresh planning is intrinsically necessary. No complete
static replacement preserving the demonstrated width savings has been
validated by these checkpoints.

The literature thread also identified modified hidden-shift circuits from
[Koch, Yeung, and Wang](https://arxiv.org/abs/2307.01803), and successive encoded
H/T teleportation operations in
[Bluvstein et al.'s universal architecture](https://www.nature.com/articles/s41586-025-09848-5),
as possible multi-stage challenges. They are **candidate directions, not
successful benchmarks in this research panel**. Neither should be counted
as evidence of broader chaining. Another circuit search is not automatically
the next step.

## 9. Current decision boundary and questions for an independent reviewer

Aggregator stopped after proposing ordinary Clifft as the default, with an
optional host selecting fault-only compilation or a certified phase
transformation. Folded stays a separate provider; measured-check sampling
integration and a universal state interface are not justified by this panel.
**That proposed combined mode has not been implemented.** The overseer's
recommendation is narrower: prioritize the reusable phase capability and keep
fault-only compilation as the simple alternative, while assessing whether the
supported structural class is broad enough for the original goal.

Any implementation must respect Clifft's
[architecture constraints](https://github.com/unitaryfoundation/clifft/blob/dbbc8865fe2babe0855ea28024790c509ff44463/AGENTS.md):
immutable Stim, allocation/exception-free ordinary hot execution, deterministic
RNG, and no runtime topology discovery inside ordinary dispatch. Current
research planning happens in setup or outside that dispatch. A richer dynamic
executor would require an explicit architectural decision.

Useful review questions, in priority order:

1. What is the strongest defensible common structural claim, and is it broad
   enough to justify a Clifft extension beyond the successful examples?
2. Which gains require trajectory conditioning, and which could instead come
   from better static support/phase analysis, wider representations, or symbolic
   conditional Clifford handling?
3. What distinguishes this contribution from Merlin/DCP and the recent
   Clifford-stabilizer and syndrome/fidelity methods: input class, output
   contract, continuation support, or practical integration? Separate evidence
   from conjectured novelty and speed.
4. Does the measured-check signed-intersection method offer a credible common
   representation beyond its one-coordinate implementation, and is there a
   concrete complete workload requiring it?
5. Is there one decisive missing analysis or experiment that would change the
   go/no-go decision? Specify the hypothesis and stopping criterion rather than
   proposing another open-ended benchmark or framework effort.

An independent reviewer should feel free to recommend narrowing, redirecting,
or stopping the project. The requested outcome is a reasoned research decision,
not endorsement of the current prototype or a plan to integrate every runner.
