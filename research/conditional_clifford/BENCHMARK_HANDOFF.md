# Factory benchmark handoff

Use these workloads to test an optimizer on complete physical circuits from
the all-zero input state. The benchmark work is on branch
`codex/quadcycle-factory-study`; the validated implementation checkpoint before
this export/handoff addition is `a213cdfe`. No production executor changes are
part of this branch.

In this VM the worktree is
`/home/exedev/.codex/worktrees/e82c/clifft-codex`. Another worktree can read these
files and run the exporter here, then feed its output to its own optimizer;
merging this research branch is not required. The branch is local; do not assume
these commits or paths are available on a remote machine.

Current direction: freeze circuit complexity for an initial optimizer
comparison. Another effort owns optimization. This effort supplies inputs,
references, and targeted extensions when benchmark results identify a gap.
Noisy preparation, decoder reconstruction, and matching published rate curves
are deferred; earlier reports' "next checkpoint" sections are historical plans.

## Obtain complete input circuits

From the repository root, using a Python environment with the built Clifft
extension and development dependencies (including Stim, NumPy, and Qiskit Aer):

```bash
env OMP_NUM_THREADS=1 .venv/bin/python tools/profile/export_factory_benchmarks.py \
  --merlin-checkout /path/to/pinned/merlin \
  --output-dir /tmp/clifft-factory-benchmarks \
  --draws 4 --seed 2610109
```

The output directory must not already exist. Merlin must be at revision
`097380fac1a3968ca47925146e211fe990f4c396`, with unchanged
`benchmarks/protocols/css.py`. The repository is
<https://github.com/mark-koch/merlin>. In this VM an existing checkout is at
`/tmp/merlin-research-2026-10-06`.

The default export contains 19 circuit files and `manifest.json`:

| Source | Physical qubits | Visible records | Purpose |
| --- | ---: | ---: | --- |
| `quadcycle-ideal.stim` | 432 | 369 | Ideal state-transfer control |
| `quadcycle-noisy.stim` | 432 | 369 | Noisy preparation, CCZ, escape, and feedback |
| `steane-D.stim` | 162 | 153 | Separate extraction ancillas and ideal data probes |
| `steane-E.stim` | 108 | 99 | Reused extraction ancilla and ideal data probes |

Each of the three noisy sources also has five `-fixed-N.stim` variants:
identity at index zero, then four reproducible draws from its noise channels.
Each file contains the complete unreduced physical circuit and original record
ordering. Fixed-fault rendering uses the existing parser's canonical gate form,
including exact CCZ decomposition; it adds no noise sites inside that decomposition.
No phase-region boundary, code matrix, or reduced-state replacement is supplied
to the optimizer. The `.stim` extension uses Clifft's
extended circuit syntax, including CCZ; these are not directly executable by
Stim alone.

The manifest includes source hashes, qubit/record counts, seed, categorical
fault histories, and selected exact full-record probability queries for all
D/E fixed variants. Probabilities are rational strings, including zero.
The record order for D/E is raw ancilla Z readout (81 or 27 bits), 24 ideal
data X checks, then 48 ideal data Z checks. Those final checks are diagnostic
probes, not additional information available to a physical decoder.

The exporter uses the existing workload constructor, which initializes the
old shared-phase analysis. That analysis does not alter the exported circuit
text. Export/reference construction is benchmark setup, not optimizer timing.

An export is already available in this VM at
`/tmp/clifft-factory-benchmarks-20261009`. It is disposable; the command above
regenerates it under another directory name. The default export was generated
twice with byte-identical results. All 19 files passed hash, record-count, and
fault-rendering checks. All 50 exported D/E probability queries matched the
existing reduced baseline, whose maximum active width was nine for these cases.

## Integrate another optimizer

Start with the fixed variants. Feed the whole source to the candidate, optimize
under the all-zero entry assumption, and preserve the visible record contract.
For each D/E variant, evaluate the manifest queries on the candidate result.
The helper `validate_automatic_specialization.visible_probability(program,
record)` supports Clifft lowered programs, including a bounded number of hidden
reset records. Import research helpers with `PYTHONPATH=tools/profile`.

For example, after producing a candidate `program` for one manifest entry:

```python
import math
from fractions import Fraction
from validate_automatic_specialization import visible_probability

for query in entry["record_probability_queries"]:
    expected = float(Fraction(query["probability"]))
    actual = visible_probability(program, query["record"])
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-300)
```

These are unconditional probabilities of complete records for the specified
fixed Pauli history. A candidate that specializes on measurement outcomes must
include branch probabilities when reconstructing the full record law. A result
conditioned on one sampled preparation branch cannot be compared directly.
Tiny raw-record probabilities require relative tolerance; a loose absolute
tolerance could incorrectly accept zero everywhere. The exported queries are
selected checks, not a certificate of full distribution equality.

For stronger validation, adapt the candidate construction in
`validate_steane_continuation.py`, preserving its independent reference side:

- Replace `state = workload.shared.render_state(...)` and subsequent candidate
  compilation with the new optimizer applied to the original fixed-fault
  source, available from `workload.shared.model.render(region)` followed by
  `extraction.model.render(tail)` and the same diagnostic probes.
- Do the analogous replacement for the separate coarse-syndrome and scoring
  circuits. Preserve their distinct output contracts. The existing loop checks
  complete inferred-syndrome laws, records proposed by both candidate and
  reference, and conditional-state probes in two logical bases.
- Keep fault-aware scoring separate from operational inputs: its correction
  intentionally knows hidden fault information for validation only.

The validators currently invoke the old reducer. Running them unchanged checks
the baseline; it does not validate a different optimizer. The export utility
does not provide a universal optimizer plugin or timing harness.

After fixed-case debugging, use the live-noise sources. Fresh complete shots
must redraw faults and measurement outcomes. Repeated shots from a fixed fault
file estimate a conditional law, not the original noisy circuit's law. Keep
identity controls separate from noise-weighted draws, retain duplicate draws,
and count fallback/failure cases rather than discarding them. Sampling faults
outside the executor is allowed; account for that cost. Existing `FaultModel`
can supply independent categorical Pauli/readout histories for these sources.

## Measurements and baseline limits

Record circuit hash, optimizer revision, environment, shots, and seeds. Measure
one-time setup, per-history or per-outcome specialization/planning, executor
preparation, and sampling separately, then report complete per-shot cost.
Include active width, actual peak memory, fallback counts, and correctness
results. Memory estimates for coefficient arrays alone are not total memory.

Inspect plans before allocating a dense state. The factory baseline study found
ordinary noisy width 33 versus width nine after shared state export and existing
optimization. Both D/E continuations stayed at width nine for the tested fixed
histories. These are bounded feasibility observations, not a throughput promise
or a guarantee for every history. Do not allocate an ordinary width-33 baseline
merely to complete a timing table; report the resource limit explicitly.

Use the candidate's own rebuilt extension when testing native optimizer changes.
This branch previously reused the built environment at
`/home/exedev/.codex/worktrees/93cf/clifft-codex/.venv` because its production
sources matched. That environment is suitable for reproducing the old baseline,
not automatically for measuring another worktree's changes.

## Reference map and reproduction

| Resource | What it provides |
| --- | --- |
| [Steane study](STEANE_CONTINUATION_VALIDATION.md) | D/E assumptions, exact versus selected checks, 2,908 validated histories |
| [Steane results](steane-continuation-validation.json) | Counts, errors, widths, and mechanism controls |
| [Factory study](QUADCYCLE_FACTORY_STUDY.md) | Larger state-transfer circuit, noise model, width comparison, limitations |
| [Factory results](quadcycle-factory-study.json) | Conditional variant metadata and structural results |
| [Factory validation](quadcycle-factory-validation.json) | Selected full-record checks against direct conditional compilation |
| [Theory validation](THEORY_PROTOCOL_VALIDATION.md) | Independent physical error propagation and rational syndrome oracle |
| [Small instrument results](phase-continuation-validation.json) | Aer comparisons of complete classical/quantum output states |
| [Shared-phase study](SHARED_PHASE_SPECIALIZATION.md) | Prototype algorithm, BT/distillation controls, and limitations |
| [Profiling guide](../../tools/profile/README.md#state-preserving-phase-continuation) | Full reproduction commands |

Quick baseline smoke checks, still including the mixed-fault and mechanism
controls in the D/E validator:

```bash
env OMP_NUM_THREADS=1 .venv/bin/python tools/profile/validate_phase_continuation.py \
  --output /tmp/phase-continuation-smoke.json
env OMP_NUM_THREADS=1 .venv/bin/python tools/profile/validate_steane_continuation.py \
  --merlin-checkout /path/to/pinned/merlin --tail-limit 0 \
  --output /tmp/steane-continuation-smoke.json
```

D/E use ideal encoded preparation with residual, gate, and readout faults;
their diagnostic channel weights are 0.01 (0.02 at CCZ). The factory uses
`p=0.001`, with CCZ noise at `2p`, simplified recovery/scheduling, and no idle
noise or postselection. Neither model is a validated reproduction of published
operational acceptance or logical-error rates. Use both families as benchmarks,
and retain small/random/distillation controls to detect circuit-family overfit.
