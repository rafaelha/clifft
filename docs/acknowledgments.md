# Acknowledgments

Clifft benefited from Bluvstein et al.'s
[logical neutral-atom processor](https://doi.org/10.1038/s41586-023-06927-3)
through the encoded IQP gate vocabulary and block connectivity used to
construct the [held-out conditional-compilation input](../research/conditional_clifford/HELDOUT_IQP.md).
This is a literature-inspired reconstruction, not reused author code or an
exact experimental schedule. Qiskit Aer supplies its independent small
physical-state and full-distribution validation.

Clifft benefited from the existing
[folded research implementation](https://github.com/unitaryfoundation/clifft/tree/b87c0da396b4bbde54fbee5535aeac1116170605/tools/profile)
through direct reuse of its reconstructed physical f7 circuit, coherent-path
contractions, signed-code growth transfer, and independent physical-history
oracle in the [complete-circuit checkpoint](../research/conditional_clifford/COMPLETE_CIRCUIT_CHECKPOINT.md).
Its categorical outcome ordering and physical fault binder are also reused
unchanged in the [fault-only comparison](../research/conditional_clifford/FAULT_ONLY_PANEL.md).
That reconstruction credits Takada, Bartlett, and Williamson's
[fold-transversal cultivation construction](https://arxiv.org/html/2609.16929v1#A1)
for its check/cat schedules, and Higgott et al.'s
[surface-code growth construction](https://quantum-journal.org/papers/q-2021-08-05-517/)
for its growth circuit. These are reused reconstruction artifacts and design
influences, not author-supplied f7 circuits or new algorithms in this checkpoint.

Clifft benefited from Takada, Bartlett, and Williamson's
[FEC evaluation procedure](https://arxiv.org/html/2609.16929v1#A1.SS2)
through the concrete decoder-recovery-then-logical-measurement consumer tested
in the same checkpoint. Clifft benefited from
[PyMatching](https://github.com/oscarhiggott/PyMatching) through direct use of
its minimum-weight matching implementation for that research decoder. Its
unit-weight CSS policy is specified locally; this is not a reproduction of
the paper's decoder configuration or published error rates.

Clifft benefited from Wan and Zapirain's
[exact cultivation corpus](https://github.com/kh428/exact_ler_for_msc/tree/b99452de7e300324e7bbf5cd9347241c3e289bae)
through the released Chan circuit models and four-fault witness used to
validate the conditional Clifford research. The retained models preserve
their physical noise sites and measurement conventions. The algebraic
continuation is a Clifft research implementation, not reused upstream
simulation code.

Clifft benefited from the same corpus through its actual-T RP2 model conversions
and retained SOFT circuits in the
[cross-family conditional-state study](../research/conditional_clifford/FAMILY_TRANSFER.md).
Clifft benefited from [Chen et al.'s RP2 code](https://github.com/Zihan-Chen-PhMA/Cultiv_T_RP2)
through those source-derived morphed-code and X-flag schedules, and from
[SOFT](https://github.com/haoliri0/SOFT) through its physical cultivation circuits
used directly for full-record and conditional-output validation. The RP2
upstream MIT notice is retained inside the input archive. Actual-T conversion
and the model's omission of escape and decoding remain explicit.

Clifft benefited from [Stim](https://github.com/quantumlib/Stim) through direct
use of its Clifford tableaux, stabilizer generators, and independent
stabilizer-state references in this research. Clifft benefited from
[Qiskit Aer](https://github.com/Qiskit/qiskit-aer) through independent physical
state and measurement-probability validation. These uses are recorded in the
[algebraic continuation study](../research/conditional_clifford/CHAN_ALGEBRAIC_CONTINUATION.md).

Clifft benefited from Gidney, Shutty, and Jones's
[magic-state-cultivation code](https://github.com/Strilanc/magic-state-cultivation/blob/871e68ff6df2f75190b1bfd6351459d1b5a037e3/src/cultiv/_construction/_color_code.py)
through direct use of its color-code coordinates and CSS check supports in
the [larger scaling controls](../research/conditional_clifford/COLOR_CHECK_SCALING.md).
The new encoder, flag network, and noisy schedule are constructed research
controls; they are not larger protocols supplied by those authors or by Chan.
Clifft benefited from [Cirq](https://github.com/quantumlib/Cirq) through its
phase-preserving CH-form stabilizer representation in an independent
coherent-state probability and final-state reference.
