# Complete-circuit checkpoint across families

2026-10-10. **Decision: sufficient cross-family capability to propose a limited,
optional conditional sampling mode.** This is a scope proposal, not production
implementation or a claim of universal conditional-state support.

The existing programs sample the five frozen complete inputs on all 80 fresh
attempts tested. The subsequent
[consumer review and optional-mode scope](#concrete-consumer-and-optional-mode-scope)
add 32 validated complete FEC attempts at f3/f7. Decoder recovery consumes the
existing folded state after the physical syndrome measurements and before
logical readout. It supplies a concrete missing calculation without a generic
output interface. These results support a bounded mode proposal with explicit
structural admission, per-history resource guards, and output limitations.
This checkpoint stops at that proposal; production architecture is unchanged.

## What is actually common

All three existing methods exploit an actual conditional state to simplify a
complete input. They preserve physical noise and records while using different
representations for the remaining quantum computation:

- The phase reducer factors a fixed non-Clifford preparation from conditional
  Clifford corrections, then executes the complete residual program.
- The measured-check update finds the signed stabilizers shared by the incoming
  state and its Clifford-transformed image. It evaluates the coherent projector
  in a small temporary space and retains the resulting encoding and state.
- The folded contractions retain coherent cat paths and a logical amplitude
  pair, with signed syndrome/ancilla information. The existing growth transfer
  propagates that representation through a certified physical bridge.

These are reused implementations, not a newly invented common simulator. A
useful shared interface must specify the *conditioning and allowed continuation*
as well as the state representation and resource cost. A small logical vector
by itself does not supply this interface.

The phase route can already express its state and continuation as an executable
conditional program; it need not fit a one-coordinate state object. The folded
route already composes its specialized chain internally. The new FEC consumer
uses that live representation directly. The proposed sampling scope therefore
does not require all routes to share a generic quantum-state object.

## Frozen panel and provenance

The [input archive](complete-panel-inputs.json.gz) contains complete source text.
The [manifest](complete-panel-manifest.json) pins every input hash, original
archive, and the complete retained folded research source tree. Display names
select files only; they are not reducer-selection keys.

| Input | Provenance and included scope | Explicit qualifications and omissions |
| --- | --- | --- |
| BT81 direct-X | Existing Merlin-helper construction at `097380fac1a3968ca47925146e211fe990f4c396`; complete retained preparation, phase, decoder, and direct-X records; 405 wires, 405 records, 24,492 noise sites | Synthetic Pauli noise across the retained protocol; no substituted ideal boundary or new literature circuit. This is the retained direct-X variant, not the scored variant. |
| Steane E diagnostic | Exact supplied `steane-E.stim`, from the earlier factory package/exporter checkout `1bcbf960`; 108 wires, 99 records, 351 noise sites | Original nonuniform diagnostic location weights, not a uniform physical error-rate model. Includes ideal diagnostic data probes. No factory throughput, decoder, or published logical-rate claim. |
| Chan four-round d5 | Complete released model retained in `chan-boundary-inputs.json.gz`, from Wan/Zapirain corpus `b99452de7e300324e7bbf5cd9347241c3e289bae`; preparation, growth, both flagged checks and supplied readout; 55 wires, 145 records, 5,537 sites | Complete supplied cultivation input; no added escape/storage/decoder. This is not a constructed d9/d11 protocol. |
| RP2 d5 | Same pinned corpus; actual-T conversion of the authors' S-proxy model, using the existing `wan_corpus.py` exporter; original model hash, export metadata and MIT notice remain in `family-transfer-inputs.json.gz`; p=0.001 | Complete supplied cultivation-only input, not an author-supplied actual-T export. 61 used wires have index extent 257; 141 records and 2,405 sites. No escape or decoder. Seventeen ideal detector bits are nonzero and are accounted for. |
| Reconstructed folded f7 | Existing reconstruction and contraction/growth code at `b87c0da396b4bbde54fbee5535aeac1116170605`; p=0.001; injection, d3/d5/d7 growth and checks, FED and logical evaluation; 99 wires, 351 visible records and 436 hidden reset outcomes | Not an author-supplied circuit or a verified reproduction of a paper's logical-error rates. Sequential extraction order is a reconstruction choice. No idle noise, escape, storage, or decoder-based FEC. |

Folded f7 has exactly the retained hash
`fff72e8f48a0509a00d52133d8cb334a1743fc7c74ce1f40eeeccc9785909545`.
Its 2,189 physical noise sites remain partitioned as 535 prefix, 134 f5-check,
694 growth, and 826 f7-check/continuation sites. Its source-derived annotations
must match the entire physical source before the existing compiler verifies
branch, CSS, record, and growth identities.

The earlier [constructed d9/d11 checks](COLOR_CHECK_SCALING.md) remain supporting
controls only. They are not added to this panel or counted as complete-circuit
capability. SOFT, smaller circuits, alternate BT readouts, and factory D are
also outside this frozen panel; their earlier evidence remains historical.

## Minimum adapters and certificates

The two adapters' missing capabilities were stated before implementation:

1. `complete_circuit_panel.py` supplies a common complete-source assessment,
   structural selection/refusal record, retained attempts, and process-tree
   memory accounting across the existing programs. It reuses all state updates,
   noise binders, reductions, and state preparation exporters.
2. `retain_folded_attempts.cpp` supplies retained faults and full records for
   **every timed folded attempt**. The previous benchmark kept audit traces but
   only aggregate counts for timed shots. It includes the pinned implementation
   and calls its existing `Protocol`; it changes no contraction or handoff.

The common selector first tries the existing measured-envelope certificates,
then phase eligibility, then a source-linked annotated folded candidate. Both
generic frontends explicitly refuse folded `DEPOLARIZE3`; its existing binder
handles all those sites. Folded discovery still needs the annotated model.
This is a common *assessment workflow*, not automatic folded discovery from
arbitrary raw circuit text. Altered physical source with the old annotation
is refused. Inside the folded compiler, annotations propose candidates;
verified physical identities authorize them.

No new quantum handoff was added for the frozen five-input panel. Measured-check
export still has the existing
one-coordinate/stabilizer-or-T-orbit restriction and temporary bound of three.
All per-shot Python tableau work remains in research host preparation, outside
ordinary Clifft hot execution. Production sources and Stim are unchanged.

## Current baseline, complete cost, and memory

Current main was resolved and pinned to
`847757e2bc8c4f7439b9e92011550fd61719eb85` (2026-10-09, release-candidate docs).
Its clean source was built in an isolated Release directory. All native helpers
were rebuilt against that same core; the Python extension hash is retained in
every result. This replaces PR 552 as the measurement baseline without changing
the historical provenance of the research algorithms.

One CPU, one OpenMP/OpenBLAS thread, seed 7102026, 16 fresh original-noise attempts
per input, no early rejection. Ordinary Clifft uses its current default passes
and complete source. It compiles once and samples serial one-shot API calls.
The conditional timer includes fault draws, host specialization, per-attempt
planning where required, complete quantum execution, records, and retention
of state reconstruction data. Correctness audits and final artifact compression
are outside attempt timings. There are no warmups or selected-history replacements.

These are short local measurements, not steady-state throughput estimates.
The native folded driver reports complete execution plus trace construction;
its outer process wall time is also retained. All rows have separate setup
costs. Python folded model/certificate/contraction-plan construction is included
in setup, not hidden behind the native loader's much smaller setup time.

| Input | Ordinary width | Conditional resources observed | Setup, ordinary / conditional (s) | Complete attempt, ordinary / conditional (ms) |
| --- | ---: | --- | ---: | ---: |
| BT81 direct-X | 87 | Residual native execution width 4-7; 23 optimized T gates | 0.8126 / 18.326 | budget / 67.71 |
| Steane E | 33 | Residual native execution width 9; 33 optimized T gates | 0.007773 / 0.437 | budget / 5.28 |
| Chan four-round d5 | 10 | Carried core 1; temporary check core 2; final preparation width 1 | 0.02353 / 0.334 | 0.03669 / 462.55 |
| RP2 d5 | 18 | Carried core 1; temporary check core 2; final preparation width 0 after readout | 0.1462 / 0.261 | 5.117 / 4,382.01 |
| Folded f7 | 44 | Native prefix width 8; four coherent paths; one logical coordinate with syndrome/ancilla data | 0.03189 / 33.888 | budget / 174.71 |

The ordinary allocation budget is width 20; the existing phase worker has a
separate width-16 guard. All widths above budget are reported before dense
allocation. BT81 also exceeds the native planner's width limit: width 87 is
measured on optimized HIR before lowering. Estimated **coefficient storage
alone** is 2.48e27 bytes for BT81, 128 GiB for E, and 256 TiB for folded f7.
No ordinary sampling time or speedup ratio is invented for these guarded cases.

At just 16 attempts, conditional setup plus attempts totals approximately
19.41, 0.52, 7.73, 70.37, and 36.68 s respectively. The one-coordinate
RP2 state does not make its complete Python-host execution competitive with
ordinary Clifft at this size.

Total cold worker wall times for these 16-attempt jobs are 20.10, 0.71, 8.04, 70.82,
and 37.72 s in panel order. These additionally include process startup,
imports, input/build integrity checks, result serialization and compression.
Those costs are excluded from the timed compiler/frontend setup and attempt
columns, and are included in the process-tree memory observations.

| Input | Ordinary observed process-tree RSS / PSS (MiB) | Conditional observed process-tree RSS / PSS (MiB) |
| --- | ---: | ---: |
| BT81 direct-X | 89.2 / 78.8 | 3,057.6 / 3,047.6 |
| Steane E | 41.9 / 32.1 | 112.8 / 99.0 |
| Chan four-round d5 | 53.1 / 43.4 | 106.9 / 96.8 |
| RP2 d5 | 50.8 / 41.9 | 103.0 / 95.9 |
| Folded f7 | 43.6 / 36.5 | 352.3 / 341.4 |

These are aggregate resident memory observations across the assessment child
and all its live native children, including imports, setup, tables, tableaux,
records, and retained state data. The supervising launcher is excluded. RSS
can double-count shared pages; PSS apportions them. Sampling is nominally every
20 ms and can miss short peaks; per-process high-water RSS is retained too.
Guarded ordinary rows are compiler-memory measurements, not executed-state
memory. Audit-process memory is separate and excluded from this comparison.

Intermediate costs that a small-core claim would miss:

- BT81's phase analysis has 87 support variables, 90,525 primitive fault
  controls, 13,869 conditional Clifford responses, and 111,935,856 bytes of
  dependency-mask payload. Its full host memory peaks near 3 GiB.
- E has 33 support variables and 81 phase-region quantum variables before
  reduction. Its residual computation retains 33 T gates even at width 9.
- Chan/RP2 retain physical Clifford frames, including RP2's extent of 257.
  Temporary width two entails a four-entry vector and 4-by-4 complex operator;
  neither accounts for the host's total storage or tableau work.
- Folded's prefix coefficient payload is 4 KiB, but its two kernels have
  9,023,524 bytes of lookup payload and 2,570,720 bytes of contraction workspace.
  The existing growth map adds 249,344 packed bytes, excluding its metadata.
  These payloads are not substitutes for the process-tree totals above.

## Records, state, failures, and validation limits

The [results](complete-panel-results.json.gz) retain all 80 fresh attempts.
No difficult history is dropped, redrawn, or replaced. BT81 has 0 accepted
and 16 rejected attempts under its supplied detector predicate; Chan has 2/14,
RP2 0/16, and folded 3/13. E has diagnostic records and no acceptance predicate.
These tiny counts are bookkeeping, not acceptance or logical-error estimates.
RP2 detector events use the retained nonzero ideal reference signs.

Each attempt retains its exact fault history and full visible records.
Folded additionally retains all 436 hidden reset outcomes. Measured-check
attempts have no unresolved entangling hidden-reset branch; irrelevant pure
local reset outcomes are marginalized by the existing implementation.

Output-state capability is reported separately from successful sampling:

| Route | Retained conditional output | Residual complexity and limitation |
| --- | --- | --- |
| Phase | Exact executable conditional circuit, full records, prefix/execution seeds and faults | 23/33 residual T gates and observed complete execution width 4-7/9. Native worker does not export a post-shot state object. The conditioned circuit retains the physical output state for replay/composition within that program. No smaller final-state rank is claimed. |
| Measured check | Explicit physical preparation, records and live boundary preparations | Chan final width 1; RP2 final width 0, but verified pre-readout width 1. No unmatched output layer in these two panel inputs. |
| Folded | Original physical circuit, complete fault labels, visible/hidden records, signed-sector/logical representation inside the existing chain | Exact conditional state remains defined by this reconstruction data. A compact usable physical output state is **not exported or certified**. Reconstructing through the unreduced full program retains width 44; a smaller final-state bound is not established by its scalar readout calculation. |

Thus there are zero sampling-certificate failures on the 80 observed fresh
attempts, but explicit **state-interface failures on the phase and folded
worker outputs**. These must not be reported as 80 complete state-interface
successes. The measured-check route supplies the strongest generic live-state
evidence in this checkpoint.

The [separate audit archive](complete-panel-audits.json.gz) records:

- All 32 fresh Chan/RP2 full-record probabilities against the original physical
  gate sequences on current Clifft. Maximum relative error is 6.67e-16.
- Ideal, every-site-nonidentity, and p=0.03-selected stress histories for each
  measured-check input. All six pass, including 5,537 and 2,405 simultaneous
  selected faults. Maximum relative error is 2.45e-15. These are validation
  histories, not draws used in the performance/noise-law claim.
- Four live state exports: first and last fresh Chan/RP2 histories, at 127/145
  and 104/141 records respectively. Each compares all dormant stabilizers and
  all three logical Pauli axes against both the original physical prefix and
  exported preparation. Maximum relative discrepancy is 1.12e-15. The quantum
  coordinate is still present; RP2's measured final state is not the evidence.
- Six phase-worker equivalence checks: first/last fresh history plus every-site
  stress for BT81 and E. Complete records reproduce, optimized conditional
  programs agree, and native plans agree with the baseline planner. These are
  not an independent full-physical BT81 oracle. The existing 24 preparation
  and 14 original/renamed small-instrument controls also pass on the pinned
  build, including their enumerated noise/measurement branches.
- All 16 fresh complete folded histories against the existing independent
  coherent-Clifford physical oracle, with native replay of the prefix.
  Maximum conditional log-probability discrepancy is 2.67e-15. Records,
  acceptance, logical output and all four fault partitions are checked.
- The existing 36 live growth handoff comparisons against Stim, covering
  logical X/Y/Z eigenstates, random syndrome/ancilla signs, and up to 31 faults.
  They test the certified bridge, not arbitrary new continuations.

No exhaustive support theorem for all measurement/fault histories is inferred
from these successes. Structural certificates are checked where the existing
programs provide them; measured core/orbit bounds remain per-history guards.
Certificates that are absent are recorded as absent. Earlier independent Aer
and larger selected-reference results remain in the linked studies; they were
not all rerun and are not counted as new checks here.

## Conditioning obligation identified by the initial checkpoint

The existing `sample_body` samples a syndrome sector and retains two coherent
logical amplitudes. Its f5-to-f7 growth certificate proves that later physical
d7 measurements already resolve those input sectors. Sampling them early is
therefore valid for this complete continuation. The 36 Stim checks include
nontrivial logical information, so the existing handoff is substantive work
to reuse, not a terminal-state export to reinvent.

However, a state conditioned on future syndrome outcomes is not automatically
the state conditioned only on records observed at an earlier physical cut.
A different suffix may invalidate early sector resolution. A small retained
counterexample illustrates the obligation: preparing `|+>`, applying H, then
measuring Z gives zero deterministically; measuring Z early before H instead
makes the final Z outcome fair. This is not a counterexample to the certified
folded bridge. It explains why that certificate cannot be silently generalized.

The current terminal `finish` processes post-check records, adjusts the physical
Pauli frame, and samples the final logical measurement. It does not return the
post-measurement quantum state. Its internal pre-readout boundary is the precise
place to test a justified continuation: after the resolving physical checks,
while the logical amplitudes remain live. The export must retain any pending
conditioning obligations if an earlier cut is used.

The initial proposal treated a general exit as the next decision point. The
consumer review below instead tests a specific missing calculation at this
cut, using the existing representation directly.

## Reproduction

Use Python with NumPy, Stim, and the current built Clifft extension. The folded
oracle additionally uses Cirq. The retained run used NumPy 2.4.2, Stim 1.16.0,
Cirq 1.6.1, GCC 13.3, Release `-O3 -march=native`, and one thread. The installed
package path and native binary hashes are in the artifacts.

1. Check out baseline `847757e2` into a separate source directory and build its
   Release Python extension and `clifft_core` library without modifying source.
   Set `PYTHONPATH` so that this extension, not a previous PR 552 installation,
   is imported.
2. Extract `tools/profile` from folded reference `b87c0da3` into `$PANEL_REFERENCE`.
   The manifest pins every file; `freeze` checks them against that Git revision.
3. Compile the existing `export_optimized_prefix.cpp` as `phase_exporter`, and
   `profile_prefix_trace_reuse.cpp` plus `coordinate_reuse.cpp` as `phase_worker`.
   Compile `retain_folded_attempts.cpp` as `folded_retained` and the reference's
   `replay_cultivation.cpp` as `replay`. Link all four against the pinned current
   core, using current `src`/generated includes, reference `tools/profile`
   includes for folded sources, C++20, `-O3 -march=native -fopenmp`.
4. Run the commands below with absolute paths. `freeze` is optional when replaying
   the retained inputs. Use a fresh work directory. Native binary compilation
   is development setup, not the per-input setup measured in the table.

The worker verifies its imported extension and helper binaries against the
manifest's `build` hashes before measuring. A rebuild can legitimately change
binary hashes; first verify the clean baseline source revision and record the
new build provenance/hashes in a copy of the manifest. Do not label an old
PR 552 installation as the pinned baseline. Regenerating inputs with `freeze`
also requires recording that build section again.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/profile/complete_circuit_panel.py run \
  --reference "$PANEL_REFERENCE" --bin "$PANEL_BIN" \
  --work /tmp/complete-panel-run --shots 16 \
  --output research/conditional_clifford/complete-panel-results.json.gz
python tools/profile/audit_complete_circuit_panel.py \
  --reference "$PANEL_REFERENCE" --bin "$PANEL_BIN" \
  --output /tmp/complete-panel-audits.json.gz
python "$PANEL_REFERENCE/tools/profile/audit_folded_growth_handoff.py" \
  --output /tmp/complete-panel-growth.json
python tools/profile/validate_compiled_prefix_reuse.py \
  --exporter "$PANEL_BIN/phase_exporter" --output /tmp/complete-panel-phase.json
```

The compact audit archive also includes the latter two existing validators and
the documented two-circuit early-sector counterexample. None of these audit
costs is used as a sampling benchmark.

## Concrete consumer and optional-mode scope

The concrete consumer is **decoder-based fictitious error correction (FEC),
followed by logical H_XY evaluation**. The complete input includes noisy
injection, cultivation, physical growth, both final folded checks, noisy CSS
extraction, noiseless final syndrome extraction, decoder-controlled physical
Pauli recovery, and the final noiseless logical measurement. This extends the
retained reconstructed cultivation calculation from FED to FEC. It does not
add escape or a fault-tolerant storage protocol.

Takada, Bartlett, and Williamson describe this continuation in
[Appendix A.2](https://arxiv.org/html/2609.16929v1#A1.SS2): decode the final
syndrome with PyMatching and propagate the recovery through the subsequent
non-Clifford logical measurement. The retained generator accepts only `fed`
and `none`; the existing terminal sampler has no decoder/recovery step.
Changing which records reject an attempt would therefore be insufficient.

The missing capability and its complete input were stated before adding the
adapter. The new research adapter inserts exactly one bounded recovery call
in the existing `finish`, after all resolving physical checks and before
`bind(final_, boundary)`. It updates the existing signed-code state and resumes
the existing final contraction. It neither exports a generic state object nor
changes the contraction, measured-check update, phase reducer, or growth map.
The pinned reference remains untouched; the build creates an explicitly
recorded source overlay. Production Clifft and Stim remain unchanged.

The decoder policy is PyMatching 2.3.1, separate X/Z check matrices, unit weight
per physical data qubit, and final noiseless syndrome records only. Recovery
applies X then Z, with no additional recovery noise. This is a fully specified
evaluation instrument, not a claimed reproduction of the authors' decoder
weights, tie choices, or logical-error rates. The complete input is the
physical source **plus this conditional recovery policy**; source text alone
omits the adaptive operation. The source, matrices, record slots, every chosen
recovery, and their placement immediately before `final_logical` are retained.

The first retained f3 stress history provides a physical witness: without
recovery its final logical distribution is `(0.5, 0.5)`; with recovery it is
`(1, 0)`, to numerical precision. The same prior fault and measurement history
is used in both evaluations. Flipping an already sampled terminal bit cannot
produce that change. This is a rejected stress history used to test the
quantum operation, not evidence of improved acceptance or an error-rate result.

### Validation and resource limits

The [FEC evidence archive](folded-fec-evidence.json.gz) retains the complete
inputs, policy, all original attempts, audits, the witness, build provenance,
and an invalid-recovery refusal. Small-instance validation preceded f7 testing.
The final audit observes the existing physical oracle during its full circuit
traversal; it does not reconstruct its state from the native logical amplitudes.

All 32 new complete attempts pass. At f3, eight fresh attempts at p=0.001 and
eight at p=0.03 were checked with dense Aer, including each hidden reset
projection and all code-stabilizer/logical X/Y/Z expectations immediately
before and after recovery. Maximum Aer discrepancy is 1.47e-14. Eight stress
histories have nonzero final syndromes; seven change their logical Bloch vector.

At f7, all 16 p=0.001 attempts agree with the independent physical coherent
Clifford oracle and native prefix replay, conditioned on each retained Pauli
fault history. Maximum complete record log-probability
error is 2.67e-15 and live-state expectation error is 2.78e-15. All 84 data
stabilizers and all three logical axes are checked on both sides of recovery.
Eleven attempts have nonzero final syndromes and two change the logical Bloch
vector. Four attempts pass the original noisy-protocol predicate and twelve
fail; none of the four accepted attempts has nonzero final syndrome. The new
capability is validated, but improved acceptance is not demonstrated here.

There are zero sampling or recovery-certificate failures on these draws.
As a separate negative check, a deliberately invalid zero recovery on the first
nonzero-syndrome f3 stress history is refused before logical readout, with
`FEC recovery does not clear the measured syndrome`. Its request and failure
are retained; it is not counted as a physical protocol rejection.

No attempt is selected for replacement, discarded from the archive, or redrawn. The
higher-noise f3 run is a separate declared stress input. These finite histories
do not establish universal support or acceptance/error-rate estimates.

At f7 the native prefix still peaks at width 8, the folded computation has four
coherent paths and one live logical coordinate plus the signed code and
ancillas, and recovery keeps that representation. Residual quantum work is the
existing terminal H_XY instrument. Lookup payload is 9,023,524 bytes, contraction
workspace 2,570,720 bytes, and growth transfer payload 249,344 bytes. None of
these small internal figures replaces total process memory or setup cost.

The pinned baseline remains current Clifft `847757e2`. The unchanged noisy
physical schedules have ordinary widths 8 (f3) and 44 (f7); the latter requires
256 TiB of coefficient storage alone. Ordinary Clifft does not accept this
external MWPM-controlled instrument as one static input. There is consequently
**no measured ordinary complete-FEC time or speedup ratio**. The five-family
table above remains the direct comparison on common complete static inputs.

For 16 original-noise f7 FEC attempts, host/compiler setup plus native loading
takes 32.10 s; complete native attempts average 173.70 ms, including the blocking
decoder request, recovery, final measurement, all faults and records, and trace
construction. Decoder computation averages 0.494 ms within that total. Cold
worker wall time is 36.52 s and observed aggregate RSS/PSS is 310.24/299.52 MiB.
The same 20 ms process-tree observer is used as above. Audits are excluded.
These short measurements are not evidence that FEC is faster than FED.

### Draft scope of an optional mode

The proposed scope is **opt-in conditional trajectory sampling of complete
inputs whose entire reduced execution has certificates and bounded resources**.
The complete input can include an explicitly specified decoder instrument,
as in this experiment. Ordinary compilation remains the default. The mode's
purpose is access to otherwise infeasible inputs; the current Python measured
check route is much slower than ordinary Clifft on Chan d5 and RP2 d5.

| Existing route | Structural admission and useful complete inputs | Boundary of the proposal |
| --- | --- | --- |
| Phase reduction | Certified stabilizer preparation, supported phase polynomial and conditional Clifford responses; BT81 direct-X and Steane E | Preserve the complete residual executable program, including its 23/33 T gates. No small-core or final-state export promise. |
| Measured-check update | Verified measured Clifford envelope, shared signed stabilizers, supported outgoing state and width/orbit bounds; supplied Chan four-round d5 and RP2 d5 | Per-history checks remain necessary. Unsupported branches fail explicitly. Constructed d9/d11 remain controls. |
| Folded contraction and growth | Exact source coverage, monomial paths, signed CSS entry, resolving measurements, verified growth map; reconstructed f7 | FED terminal evaluation and the specific FEC recovery/readout consumer. No generic continuation or arbitrary folded-circuit recognition claim. |

Admission must use verified operations and state structure. Names select panel
fixtures only. The folded research frontend currently needs source-linked
annotations; the mode must disclose that requirement instead of presenting
it as automatic discovery from arbitrary source text.

All original noise channels and records, including hidden outcomes needed for
the retained trajectory, must survive reduction. Rejection remains an output
of the complete protocol. A failed certificate must identify the attempt and
the failed obligation and stop that request with an incomplete result. It must
never be treated as protocol rejection, silently omitted, or replaced by a
new shot. Any eventual fallback must preserve the same conditioning and fault
history; absent such a fallback, the request fails explicitly.

The output contract is complete records and sufficient retained reconstruction
data for each conditional quantum state. Compact state preparation/export is
reported only where separately certified. Sampling success is not a claim
that all three representations implement a common public quantum-state API.
Peak conditional/intermediate resources, residual output complexity, total
memory, setup, and complete attempt cost remain mandatory diagnostics.

Outside scope: arbitrary non-Pauli noise, universal adaptive quantum circuits,
unverified folded suffixes, escape/storage protocols, a universal support
theorem, GPU/packed execution guarantees, fixed-fault public API integration,
and competitive performance claims against bespoke literature simulators.
f7 remains an independent reconstruction with sequential extraction and no
idle noise. The FEC decoder and noiseless evaluation are explicit modeling
choices, not an implementation of a full operational decoder/escape pipeline.

There is already a pinned
[folded integration experiment](https://github.com/unitaryfoundation/clifft/blob/b87c0da396b4bbde54fbee5535aeac1116170605/docs/guide/folded-specialization.md)
with `specialize_folded=True`, operation recognition, and a preallocated native
kernel action. It excludes the native f5-to-f7 growth transfer and decoder FEC,
and it is absent from the pinned current baseline. Any implementation proposal
should reuse that work alongside the existing research routes. This checkpoint
does not implement a new production dispatcher, general output interface, or
executor architecture. Python host planning and blocking decoder IPC here are
research machinery, not a production hot-path design.

The resulting decision is to propose this bounded optional sampling mode for
review. The concrete FEC consumer removes the need for an architecture-driven
generic folded exit experiment. It does not remove the recorded output-state
limitations of the other routes. This checkpoint stops at the proposal; it
does not start additional scaling controls, another simulator, or static
compilation tuning.

### Reproducing the bounded consumer

Use the same extracted reference and baseline source/build as above; put the
reference `tools/profile`, this checkout's `tools/profile`, and the pinned
Clifft Python extension on `PYTHONPATH`. In addition to the earlier audit
dependencies, install PyMatching 2.3.1. The build verifies the reference tree
and baseline source revision; retain the resulting core/binary hashes.

```bash
python tools/profile/folded_fec_consumer.py build \
  --reference "$PANEL_REFERENCE" --work /tmp/folded-fec-build \
  --baseline-source "$PANEL_BASELINE_SOURCE" --baseline-build "$PANEL_BASELINE_BUILD"
python tools/profile/folded_fec_consumer.py run \
  --distance 3 --shots 8 --probability 0.001 --work /tmp/fec-f3 \
  --executable /tmp/folded-fec-build/folded_fec --output /tmp/fec-f3.json.gz
python tools/profile/audit_folded_fec.py \
  --input /tmp/fec-f3.json.gz --reference "$PANEL_BIN/replay" \
  --output /tmp/fec-f3-audit.json.gz
```

Repeat the small run/audit at p=0.03 in a separate directory. Only after the
small checks pass, run/audit distance 7, 16 shots, p=0.001. All runs use seed
10102026 and one OpenMP/OpenBLAS thread. The Python audit forces each original
measurement/reset branch; it does not draw a more convenient reference history.
