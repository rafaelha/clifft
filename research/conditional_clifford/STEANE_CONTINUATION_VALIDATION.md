# Physical Steane extraction after conditional phase reduction

Date: 2026-10-09. Branch: `codex/quadcycle-factory-study`.

## Purpose and circuit boundary

This checkpoint adds the physical extraction circuits of protocols D and E
from [the theory paper, Supplement S6](https://arxiv.org/html/2610.03605v1)
to the preceding [CCZ fault study](THEORY_PROTOCOL_VALIDATION.md).
D couples each of three data blocks to its own encoded plus-state ancilla.
E reuses one such ancilla in three successive coupling layers. Both have
81 extraction CNOTs. The ancillas are physically measured in Z; their full
readout strings are retained.

The data region is the same 81-qubit, two-layer CCZ circuit. Its ideal encoded
preparation starts from zero, and fixed Pauli histories may include faults
before or inside the CCZ layer. The generic shared-phase transformer exports
the physical data state. The continuation then prepares fresh ancillas,
executes the extraction CNOTs with specified faults, and measures the ancillas.
The reducer has no knowledge of the code matrices or protocol labels.

Ancilla preparation is ideal in this checkpoint. Controlled residual faults
are inserted after preparation, after extraction gates, and in readout. They
do not represent a fully simulated noisy preparation procedure. There is no
metacheck repair, decoder, idle schedule, or noise-weighted rate estimate.

## Independent reference and quantum state checks

The reference keeps the preceding exact integer expansion of the propagated
CCZ error in physical Z/CZ terms. These diagonal data terms commute with the
data-to-ancilla extraction CNOTs. Stim propagates the remaining Pauli frame,
including extraction faults, to the end of the Clifford continuation.

For ideal extraction, the encoded plus ancilla is invariant under translations
by any data codeword. A fixed fault trajectory therefore leaves an independent
ancilla state whose Z readout is uniform on a shifted binary code space. The
data state includes the phase errors propagated back from the ancilla. This
factorization is not assumed for general noisy or entangled ancilla inputs.

Two exact stabilizer checks certify the ideal extraction channel by preparing
nine encoded Bell halves, their reference partners, and the ancillas. The
complete joint stabilizer state is unchanged after the extraction CNOTs.
This tests arbitrary logical input and ancilla decoupling, independently of
the specific CCZ state. Propagated Pauli frames account for the controlled
faults around that ideal map.

Validation uses three complementary queries:

- **Complete inferred-syndrome laws.** A diagnostic copy of the physical
  coupling circuit measures the independent ancilla Z checks instead of all
  raw Z bits. Reported readout flips are mapped to parity inversions. Together
  with 24 ideal data X checks and 48 ideal data Z checks, every outcome in the
  rational reference support is checked. Normalization constrains outcomes
  outside that support. This is the coarse-grained marginal of the raw records.
- **Individual physical records.** Both the candidate and the reference
  generate records, and their probabilities are checked in Clifft. A deliberately
  invalid ancilla record also tests zero probability. These selected queries
  do not enumerate the full raw distribution: there are 11 random ancilla
  bits in E and 33 in D even for a fixed fault history.
- **Conditional data-state probes.** After the raw ancilla measurements and
  ideal data probes, a diagnostic correction removes the known X fault, then
  the ideal CCZ and encoding are inverted. Readout in two logical bases checks
  both squared amplitudes and interference between them. An integer Walsh
  transform supplies the independent reference for the second basis. These
  are selected state probes, not full tomography at physical scale.

The state probes use hidden fault information only for validation. It is
never passed to a physical decoder or the acceptance calculation. Ideal data
Z information is likewise diagnostic and does not enter acceptance. The
bounded acceptance rule uses the extracted ancilla syndrome and ideal data
X syndrome; it is not yet an end-to-end noisy implementation of the protocol.

Six small shared/separate extraction instruments also compare the entire
conditional output density matrix, including the physical record, against
Qiskit Aer. Each covers two region histories and a noncommuting continuation.
These independently exercise measurement collapse and the quantum-state exit
interface without relying on a large-circuit fidelity score alone.

## Fault coverage and mechanistic controls

For each protocol, the panel covers every nonidentity Pauli outcome at all
81 extraction CNOTs, every single-qubit Pauli on every ancilla at extraction
entry, and every individual ancilla readout flip. Additional controls include
the ideal case, matching data errors, ancilla faults at different times, and
18 mixed region/extraction histories. Mixed histories have one, two, or three
faults in each part; the same region histories are used for D and E.

| Protocol | Physical qubits | Raw ancilla records | Single extraction faults | Total histories |
| --- | ---: | ---: | ---: | ---: |
| D, separate ancillas | 162 | 81 | 1,539 | 1,562 |
| E, shared ancilla | 108 | 27 | 1,323 | 1,346 |

The controls distinguish two physical effects:

1. Identical X errors on corresponding qubits in two data blocks cancel in
   E's extracted syndrome. D detects them separately. In this ideal-check
   control, E accepts with probability one and D rejects with probability one.
2. A Z fault on the ancilla after its first coupling propagates into the later
   data blocks in E. In D, that ancilla never couples to those blocks. This
   control is rejected by E's data X checks and accepted by D.

These contrasting controls are not a ranking of protocols. Real performance
depends on fault weights, preparation, recovery, and the noise model.

The retained [result artifact](steane-continuation-validation.json) records
the full run's probability counts, discrepancies, active widths, channel
certificates, witnesses, and source hashes. Probability comparisons use
relative tolerance so rare raw records, especially D's 33 random bits, are
not accepted merely because their probabilities are smaller than a loose
absolute tolerance.

All 2,908 histories passed. The run checked 6,220 coarse-syndrome probabilities,
14,540 individual physical-record probabilities, and 368 conditional state
probe probabilities. Maximum absolute discrepancy was below `7e-18`; maximum
relative discrepancy was below `4e-14`. All candidate continuations and state
probes stayed at active width nine. The six small instruments covered twelve
fixed histories, with maximum density-matrix entry error below `1e-17`.

## Next checkpoint

The next change should introduce a specified noisy preparation/extraction
model and an explicit syndrome-processing rule, using the same assumptions
for candidate and reference. First validate the classical information used
for preparation repair, acceptance, and final recovery; then estimate rates.
The present fault-conditioned comparisons establish useful correctness
evidence but do not validate an operational decoder.

Keep the quadcycle factory as the larger state-transfer workload. Neither
this study nor the previous one requires exact reproduction of a published
performance curve before advancing. Production dispatch and automatic
discovery of multiple reducible regions remain outside this checkpoint.

Reproduction is documented in the
[profiling guide](../../tools/profile/README.md#physical-steane-continuation-validation).
