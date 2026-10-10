# Conditional-state transfer to supplied SOFT and RP2 circuits

The Chan update transfers to the complete supplied SOFT d3/d5 and RP2 d3/d5
cultivation-only models on all 52 retained histories. Each measured check
returns to a one-coordinate core, sometimes using two coordinates temporarily.
In particular, RP2's intermediate morphed code does not require a supplied
code matrix or a family-specific state assumption. The same signed-stabilizer
intersection derives its conditional encoding from the actual incoming state.

There is an important output qualification. SOFT d5 ends with an unmatched
physical T layer. Its measurement can be evaluated in a small core, but the
physical output includes that layer. The exact exported preparation therefore
still has ordinary active width 8-10. This result does not establish a
one-coordinate physical output for every family, or a large-distance protocol.
Production Clifft and the native PR 552 library are unchanged.

## Inputs and provenance

The [retained inputs](family-transfer-inputs.json.gz) include the original
circuit texts, hashes, upstream paths, RP2 export metadata, and the upstream
RP2 MIT notice. They come from Wan and Zapirain's
[frozen corpus](https://github.com/kh428/exact_ler_for_msc/tree/b99452de7e300324e7bbf5cd9347241c3e289bae),
revision `b99452de7e300324e7bbf5cd9347241c3e289bae`. All 402 upstream manifest
hashes were checked before extraction.

- SOFT d3 is `data/inputs/clifft_d3_p001.stim`, attributed upstream to
  [SOFT](https://github.com/haoliri0/SOFT). SOFT d5 is
  `data/inputs/soft_cultivation_d5_p0005.stim`. Both are retained verbatim.
  Despite the latter filename, its actual noise parameters are **0.005**;
  d3 uses 0.001. No noise parameter was inferred from a filename or silently
  normalized for this study.
- RP2 uses the corpus's actual-T `rp2-d3/model.json` and `rp2-d5/model.json`,
  exported at p=0.001 with the unchanged `wan_corpus.py` from `01a8eec8`.
  These are documented conversions of the authors' S proxies, not
  author-supplied actual-T exports. The exporter hash and original model hashes
  are retained. The original source is
  [Cultiv_T_RP2](https://github.com/Zihan-Chen-PhMA/Cultiv_T_RP2); the conversion
  and scope are described in the corpus's
  [model documentation](https://github.com/kh428/exact_ler_for_msc/blob/b99452de7e300324e7bbf5cd9347241c3e289bae/docs/OTHER_CULTIVATION.md).

These are the complete supplied cultivation-only inputs, including preparation,
growth where present, checks, feedback, terminal projection and logical readout.
They exclude escape and decoding. Sampling retains rejected attempts and all
physical noise locations. Stress histories are validation inputs, not estimates
under the supplied physical noise law.

## What had to change

The initial refusals were representation and binding restrictions:

1. SOFT has grouped resets, T gates and measurements. The study expands parsed
   instructions without moving any operation. It checks equality of every
   categorical noise site and every rendered instruction against the original.
   The underlying conditional driver still requires expanded instructions.
2. RP2 uses X-basis flags. The binder now supports certified X- or Z-basis flag
   preparations and readouts, preserving physical Pauli faults, relative phases,
   and reported-bit flips. An X flag and the central X readout are distinguished
   by their qubit identifiers. Early flag readout is still refused.
3. RP2's final logical Pauli rotation acts inside the already certified core.
   A quarter-Pauli rotation can now update that core when its decoded Pauli
   preserves every dormant coordinate. An additional rotation that would
   activate a dormant coordinate is explicitly refused.
4. SOFT d5 ends with `D; MPP`, without the inverse phase layer. The host
   evaluates the projector of `D^dagger P D`, retaining the result in D's input
   frame, then restores D in the exported physical preparation. This rule is
   restricted to a terminal measurement. Further quantum operations after such
   an open layer are not silently treated as a supported continuation.

The temporary bound remains three coordinates, and compression still requires
at most one. No code lookup, larger bound, enumerated fault cache, redraw,
postselection, physical-state tomography, or automatic expensive fallback was
introduced. This remains an offline Python host with per-shot tableau work,
outside ordinary Clifft execution.

## Results

Each input has one ideal, eight fresh original-noise and four stress histories.
All 52 histories pass; no selected history is refused or redrawn. Across their
104 measured-check reductions, the maximum temporary core is two and the
maximum carried core is one. No retained query has an unresolved hidden reset
branch.

| Input | Used physical wires | Visible records | Noise sites | Ordinary width | Exported output width | Median seconds per attempt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SOFT d3 | 15 | 21 | 518 | 4 | 1 | 0.0253 |
| SOFT d5 | 42 | 112 | 3,564 | 10 | 8-10 | 0.241 |
| RP2 d3 | 25 | 47 | 678 | 8 | 0 | 0.279 |
| RP2 d5 | 61 | 141 | 2,405 | 18 | 0 | 3.79 |

RP2's sparse identifiers have extents 107 and 257, respectively; those are not
counts of used physical wires. The prototype retains those identifiers, which
also increases its tableau work. RP2's width-zero final export is after its
logical measurement; its quantum state between checks still carries one
coordinate. SOFT d5's open output layer contains 19 physical T/T-dagger gates.

The [validation artifact](family-transfer-validation.json) retains complete
fault histories, records, boundary widths, source/native hashes and timings.
The native comparisons evaluate the original physical gate sequences; they do
not use the new check reduction. Validation includes:

- 156 complete-record probability queries: conditional samples, independently
  sampled native records, and altered records. Maximum nonzero relative error
  is `1.00e-14`; impossible records are checked separately.
- Eight complete conditional-state characterizations, at the ideal and final
  stress history of each input. These measure every dormant stabilizer jointly
  and all three core Bloch coordinates where applicable, totaling 850 Pauli
  checks including unused zero wires. For SOFT d5, undoing its final physical D
  before the reference probes verifies the state whose export restores D.
  Maximum relative probability error is `1.00e-14`.
- Independent Aer physical states and probabilities for all 13 SOFT d3
  histories, with maximum state error `6.44e-17` up to global phase.
- Twenty small physical Aer trajectories covering every single-fault
  alternative of an X-flag control, including faults around central reset,
  readout flips and faults after flag readout. Four further Aer checks verify
  the open output layer. Maximum state error is `2.55e-16`.
- Native logical-rotation checks and explicit refusals for unsupported angles,
  expanding rotations and quantum continuation after an open terminal layer.

The original [Chan regression](family-transfer-chan-regression.json) also
passes: 25 complete histories, 150 original-circuit probability queries, the
independent primitive suite and the full 128-record four-fault witness. The
witness probabilities sum to one and every branch is rejected.

The raw RP2 detector bits have nonzero reference parities supplied by its
model. In the artifact, `detector_events` counts raw detector ones, not
disagreements with those reference parities. Ideal raw counts of 4/17 for
d3/d5 must not be interpreted as failed checks. The full record contract and
the original detector expressions are preserved.

Times are diagnostic medians for eight original-noise histories, pinned to one
CPU. They include conditional evolution, state export, native compilation and
one native shot; setup, fault-history generation and correctness checks are
excluded. No throughput advantage is claimed. These d3/d5 cases are affordable
under ordinary Clifft; the result establishes transfer of the representation,
not a newly inaccessible large-distance benchmark. Selected histories do not
prove eligibility for every possible fault/outcome history or validate rare
logical-error rates.

## Consequence and next target

The same state-derived certificate now works on supplied models from three
families: Chan, SOFT and RP2. RP2's 15/35-qubit intermediate phase supports do
not obstruct the tested updates, despite differing from its final 9/25-data
codes. This resolves the specific transfer question left by the earlier local
parity audit, on the retained complete trajectories.

The [folded f7 input audit](family-transfer-next-target.json) identifies a
different next step. That existing reconstructed circuit has 351 visible
records and ordinary width 44. The present frontend finds no inverse-T check
regions there. Its ideal trajectory refuses an expanding ungrouped T gate;
its noisy source also requires three-qubit categorical channels. These are
frontend failures, not an intrinsic lower bound on its conditional core.
The separately validated folded contraction remains the relevant existing
method; the reconstructed input is not an author-supplied export.

The stronger design direction is a common conditional-state interface fed by
different certified reductions. It should preserve records, the conditional
Clifford encoding, the bounded core, and any exact residual quantum tail.
SOFT's terminal D is a concrete reason the residual tail belongs in that
contract. The next scientific test is whether the folded check can rejoin
this interface after its existing native prefix and coherent contraction,
without assuming that its internal state fits the Chan projector model.
No production continuation architecture is implemented or approved here.

## Reproduction

Use the same PR 552 Clifft build, NumPy, Stim, Qiskit and Aer as the earlier
Chan validation. The retained inputs make these commands independent of the
external upstream checkouts:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  tools/profile/study_conditional_family_transfer.py --trials 8 \
  --output /tmp/family-transfer-validation.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  tools/profile/validate_chan_algebraic.py --trials 8 \
  --output /tmp/family-transfer-chan-regression.json
```

`--cases soft-d3 --trials 1` runs a smaller transfer check. The input archive's
export metadata pins the existing RP2 exporter and upstream source files.
