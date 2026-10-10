# Fault-only compilation on the frozen complete-circuit panel

**Decision: keep an optional conditional-compilation host with a certified
phase route. Fault conditioning alone is insufficient for this panel.**
It unlocks none of the three inputs that exceed ordinary Clifft's allocation
budget. The existing phase route adds sampling capability for BT81 direct-X
and Steane E. Chan and RP2 are better served by ordinary static Clifft for
sampling. Reconstructed folded f7 still requires its existing specialized
contractions; this experiment does not generalize that algorithm.

This qualifies the [held-out IQP result](HELDOUT_IQP.md), where fault-only
compilation was sufficient and faster than the phase rewrite. Both findings
belong in the implementation decision. No circuits were added, no reducers
were modified, and folded/FEC work remains paused.

## Frozen inputs and comparison policy

The inputs, omissions, native build and original measurements are exactly
those in the [five-circuit checkpoint](COMPLETE_CIRCUIT_CHECKPOINT.md).
The source archive and result archive are checked against its manifest.
Clifft is pinned to `847757e2bc8c4f7439b9e92011550fd61719eb85`; its imported
extension hash is verified before every worker. One CPU and one
OpenMP/OpenBLAS thread are used, matching the retained panel policy.

Each input reuses all 16 retained **fault** histories, for 80 assessments.
Quantum measurement outcomes remain sampled by the original circuit; the
baseline neither fixes the earlier measurement records nor substitutes an
ideal state at a boundary. The previous full records remain in the unchanged
panel archive. For the four generic inputs the original fault RNG stream is
replayed and every draw checked for exact equality, so fault-drawing cost is
included without introducing different histories. Their saved quantum seeds
are reused, although different compiled programs need not generate identical
records from the same seed.

The existing `FaultModel` renders each physical circuit with only the
categorical noise outcomes fixed. All quantum gates, measurements, resets,
feedback, detector parities and observable declarations remain. In
particular, the Steane diagnostic's correlated/ELSE chain remains a joint
categorical event, not independently drawn Pauli components.

For folded f7, the existing pinned `bind_faults` handles the retained
`DEPOLARIZE3` outcomes and four saved fault partitions. All 2,189 physical
noise operations are covered exactly once, with their complete categorical
labels checked. The regenerated physical source must equal the frozen source
byte for byte before binding. This reuses the existing fault binder, not the
contractions, growth transfer or FEC code. Its native interleaved RNG stream
has no saved per-attempt quantum seed; this is not the obstruction here,
because all 16 bound inputs exceed the width guard before execution.

The new assessment adapter supplies only retained-history binding, default
Clifft compilation, guarded full execution and cost/provenance accounting.
Its allocation budget is width 24 (256 MiB of complex coefficients), matching
the IQP follow-up and exceeding the earlier panel's ordinary width-20 budget.
No declined history is dropped, redrawn or treated as a rejected measurement
outcome. No static compiler tuning or additional reduction is performed.

## Capability and residual work

| Frozen input | Ordinary noisy width | Fault-only width, all 16 histories | Fault-only residual T gates | Existing reduction | Conclusion |
| --- | ---: | ---: | ---: | --- | --- |
| BT81 direct-X | 87 | 56 | 648 | Phase: width 4-7, 23 T gates | Phase route adds sampling capability; fault-only refuses all 16 |
| Steane E diagnostic | 33 | 30 | 216 | Phase: width 9, 33 T gates | Phase route adds sampling capability; fault-only refuses all 16 |
| Chan four-round d5 | 10 | 10 | 53 | Measured check: carried core 1, temporary core 2, final width 1 | All methods can sample; ordinary Clifft is fastest |
| RP2 d5 | 18 | 18 | 102 | Measured check: carried core 1, temporary core 2, final width 0 | All methods can sample; ordinary Clifft is fastest |
| Reconstructed folded f7 | 44 | 44 | 585 | Existing folded contractions: prefix width 8, four coherent paths, one logical coordinate plus syndrome data | Specialized route remains necessary within this budget |

Unreduced coefficient storage alone would be **1 EiB** for BT81, **16 GiB**
for E, and **256 TiB** for f7 after fault binding. No allocation or full
sampling-time estimate is attempted for these rows. E's requirement is a
budget refusal here, not an assertion that no larger machine could run it.
The previously measured reduced BT81 host still uses about 3 GiB in total,
despite its small native execution width.

The comparison establishes the value of the **whole existing phase route**,
including preparation-state analysis and, for BT81, its 162 sampled prefix
records. It does not isolate the marginal contribution of polynomial
synthesis from other steps in that route. E has no prefix records, so its
improvement also demonstrates the value of the stabilizer-support reduction
without that source of measurement conditioning.

Measured-check reduction genuinely supplies certified compact live states
on Chan and RP2, as already validated before final readout in the earlier
checkpoint. This is an additional state-output capability, not an additional
sampling unlock on these two complete circuits. No larger constructed
checks are included to strengthen this panel's result.

## Complete costs and total memory

The ordinary and reducer columns below reuse the retained measurements on
the same pinned build; only the fault-only column is newly timed. Times
include every accepted and rejected attempt. Each route has separate setup.
The research-host costs include rendering, planning, output handling and
record-parity checks; they are not native kernel timings.

| Input | Ordinary complete attempt | Fault-only complete attempt | Existing reducer complete attempt |
| --- | ---: | ---: | ---: |
| BT81 direct-X | Width refusal | Width refusal; 74.56 ms to guard | Phase: 67.71 ms |
| Steane E | Width refusal | Width refusal; 17.08 ms to guard | Phase: 5.28 ms |
| Chan four-round d5 | 0.03669 ms | 7.91 ms | Measured check: 462.55 ms |
| RP2 d5 | 5.117 ms | 19.43 ms | Measured check: 4,382.01 ms |
| Folded f7 | Width refusal | Width refusal; 14.11 ms to guard | Folded: 174.71 ms |

Times to a refusal are **not complete attempt costs**. In particular, f7
only replays its saved native fault labels and does not include native fault
draw time; no complete baseline timing is claimed for that row. The complete
costs for Chan/RP2 include the verified original draws. Their compile/prepare
means are 2.31/12.53 ms, while native sample calls average 0.0585/3.25 ms.
This separates the cost of per-attempt compilation from the execution width.

| Input | Setup: ordinary / fault-only / reducer (s) | Fault-only peak RSS / PSS (MiB) | Existing reducer peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| BT81 direct-X | 0.8126 / 0.4810 / 18.326 | 95.1 / 82.7 | 3,057.6 |
| Steane E | 0.007773 / 0.01867 / 0.437 | 56.4 / 43.9 | 112.8 |
| Chan four-round d5 | 0.02353 / 0.04276 / 0.334 | 58.0 / 45.4 | 106.9 |
| RP2 d5 | 0.1462 / 0.04670 / 0.261 | 60.3 / 47.6 | 103.0 |
| Folded f7 | 0.03189 / 0.05682 / 33.888 | 58.6 / 45.7 | 352.3 |

New cold worker wall times are 1.879, 0.464, 0.335, 0.551 and 0.484 seconds
in panel order. RSS/PSS are process-tree observations sampled every 20 ms,
including imports, archive loading, setup, records and result serialization;
the supervisor and separate audits are excluded. Per-process high-water RSS
is also retained. Refused rows measure compilation memory, not the much
larger unallocated quantum state. Coefficient budgets are not total-memory
budgets; the phase host's tables and Python storage matter separately.

Both Chan and RP2 complete all 16 attempts. Chan retains 2 accepted and
14 rejected attempts; RP2 retains 0 accepted and 16 rejected attempts after
applying its 17 nonzero ideal detector signs. No cost per accepted output is
inferred from these small counts. BT81, E and f7 each retain 16 resource
refusals, alongside their previous outcomes, rather than reclassifying a
known rejected history as a completed baseline attempt.

## Binding and correctness checks

Every executed or declined bound source has a retained hash and a complete
fault mapping. The original full source plus this mapping reconstructs the
conditional physical circuit; large source strings are not duplicated in
each result. For Chan/RP2, the whole-circuit binding also agrees byte for
byte with the existing segmented binder on all 32 histories.

All 32 previous measured-check full-record probabilities agree with the
bound original physical circuits to at most 6.67e-16 relative error. All
32 newly sampled records have positive full-record probability. These are
64 queries through the existing Clifft physical probability engine, not a
new independent simulator. Record, detector and observable parity checks
also pass. The prior independent state and folded-trajectory evidence is
unchanged and is not counted as newly run validation.

For BT81, E and f7, full physical probability replay is explicitly declined
at the declared width budget. All bindings and retained identifiers are
verified, including the folded partition coverage and labels, but a resource
refusal is not a new correctness proof of a successful simulation. Sixteen
histories per input do not prove an all-history resource bound or acceptance
rate. No conditional post-shot state object is exported by the fault-only
baseline; the complete source, faults, seeds and records remain available.

## Smallest shared implementation proposal

1. **Keep ordinary static compilation as the default.** For an affordable
   complete circuit, reuse its plan across shots. Chan/RP2 provide no sampling
   justification for replacing it with either research host.
2. **Add an opt-in cold host for conditional programs.** Reuse the existing
   categorical Pauli/readout model, immutable physical site identifiers and
   exact renderer. The shared attempt record retains faults, quantum decisions
   when sampled, complete outputs or an explicit refusal, resource use and
   timing. Use the unchanged Clifft compiler and executor for the resulting
   program. IQP establishes a useful fault-only policy; this panel establishes
   that it is insufficient as the only policy.
3. **Provide the existing certified phase transformation as the second
   generic policy.** Reuse its analysis, fixed preparation and continuation
   machinery. Admission depends on verified affine/diagonal structure and the
   complete resource/output contract. Explicit policy selection plus
   certificates is enough initially; no circuit-name dispatcher or new
   adaptive selection framework is needed. The present useful phase cases
   fit the existing worker limit, so this evidence does not require raising
   that limit or tuning static compilation.
4. **Do not require a common quantum-state representation.** Measured-check
   export remains useful research/state-analysis capability, but its sampling
   integration is not justified by these panel costs. The existing folded
   runner remains a separate certified provider with its current fault and
   state machinery. A common outer budget/result wrapper can report its
   attempts; arbitrary-source discovery and new handoffs are outside the
   minimal mode. The generic fault/phase host must explicitly refuse f7 if
   that provider and its source-linked certificate are unavailable.

Route selection should happen before quantum decisions whenever possible.
Always attempting an unreduced compile on every shot would add 75 ms to
BT81 and 17 ms to E merely to rediscover the refusal. Assess fault-only
compilation when choosing the input's policy, then check the selected route's
actual per-history limits, including setup and total memory as well as active
width. If a route fails after sampling prefix outcomes, any fallback must
preserve those outcomes and its exact conditional state. Restarting the
original circuit with new quantum outcomes would discard a difficult history;
an explicit retained refusal is the minimum correct behavior when continuation
is unavailable. Acceptance never authorizes dropping a resource failure.

This proposal keeps planning and certificate work outside ordinary hot
execution. It changes no architectural invariant and requires no new kernels,
generic output interface, folded/FEC refinement or additional circuit search.
**The checkpoint stops at this decision; no production mode is implemented.**

## Reproduction and retained artifacts

The [manifest](panel-fault-only-manifest.json) pins the unchanged input/result
archives, native build, previous research source and new adapter. Results and
audits are [panel-fault-only-results.json.gz](panel-fault-only-results.json.gz)
and [panel-fault-only-audits.json.gz](panel-fault-only-audits.json.gz).

Use the pinned Python package and folded reference from the complete-panel
reproduction instructions. With that package followed by `tools/profile` on
`PYTHONPATH`, and both thread environment variables set to one:

```bash
python tools/profile/compare_panel_fault_conditioning.py run \
  --reference "$PANEL_REFERENCE" --max-width 24 --output /tmp/panel-fault-only.json.gz
python tools/profile/compare_panel_fault_conditioning.py audit \
  --reference "$PANEL_REFERENCE" --case chan-four-round-d5 --max-width 24 \
  --results /tmp/panel-fault-only.json.gz --output /tmp/chan-fault-audit.json.gz
```

Repeat the audit with each frozen case name to reproduce the five retained
audit groups. No new input generation or folded build is required.
