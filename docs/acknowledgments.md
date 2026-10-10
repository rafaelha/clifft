# Acknowledgments

Clifft benefited from Wan and Zapirain's
[exact cultivation corpus](https://github.com/kh428/exact_ler_for_msc/tree/b99452de7e300324e7bbf5cd9347241c3e289bae)
through the released Chan circuit models and four-fault witness used to
validate the conditional Clifford research. The retained models preserve
their physical noise sites and measurement conventions. The algebraic
continuation is a Clifft research implementation, not reused upstream
simulation code.

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
