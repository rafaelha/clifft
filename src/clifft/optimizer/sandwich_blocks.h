#pragma once

// Recognition of T-sandwich blocks in the Heisenberg IR.
//
// A layer of T gates about mutually commuting Pauli axes, a Pauli measurement,
// and the inverse layer together measure a Clifford observable: conjugating a
// Pauli by pi/4 rotations yields a Clifford. A later rewrite can replace such a
// block by a single projection. This analysis only recognizes the blocks and
// checks that the projection stays inside the dormant stabilizer subspace.

#include "clifft/frontend/hir.h"
#include "clifft/util/pauli_arena.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft {

struct SandwichBlock {
    // T_GATE op indices of the opening layer, ascending.
    std::vector<size_t> opening_ops;
    // T_GATE op indices of the closing layer, in pairing order:
    // closing_ops[i] inverts opening_ops[i] (same unsigned axis, same mask
    // sign, opposite dagger). This is ascending only when the closing layer
    // lists its gates in the order of the opening layer. Empty for a terminal
    // block, which has no closing layer.
    std::vector<size_t> closing_ops;
    // The one visible MEASURE op after the opening layer.
    size_t check_op = 0;
    // Hidden MEASURE ops in the interior whose unsigned Pauli equals the
    // check's, such as the reset of the check qubit.
    std::vector<size_t> echo_ops;
    // NOISE and CONDITIONAL_PAULI ops in the interior: strictly between the
    // last opening op and the first closing op, or after the last opening op
    // when there is no closing layer.
    std::vector<size_t> interior_noise_ops;
    std::vector<size_t> interior_conditional_ops;
    // One flag per opening op: 1 when its axis anticommutes with the check Pauli.
    std::vector<uint8_t> axis_anticommutes_with_check;
    // True when the closure check below passed.
    bool supported = false;
    // Dormant-subspace active width just before the first opening op.
    uint32_t active_width_at_opening = 0;
};

// Finds every recognized block, in op order. Blocks never overlap.
//
// Recognition rules:
//
// 1. A layer is a maximal group of T_GATE ops that are consecutive after
//    ignoring NOISE, READOUT_NOISE, DETECTOR and OBSERVABLE ops between them.
//    Any other op kind ends a layer.
// 2. Layers are scanned left to right. A layer L_open starts a block when the
//    next layer L_close has the same size, and each closing op can be paired
//    with an opening op that has the same unsigned axis, the same mask sign
//    and the opposite is_dagger(), and the opening axes pairwise commute.
//    Pairing is by unsigned axis, so a layer in which two ops share an axis
//    is skipped as a block start.
// 3. Between the last opening op and the first closing op there must be
//    exactly one visible MEASURE (the check) and no T_GATE, PHASE_ROTATION,
//    INSTRUMENT or EXP_VAL. Every hidden MEASURE there must have the check's
//    unsigned Pauli (an echo, such as the reset of the check qubit). Every
//    CONDITIONAL_PAULI there must commute with every opening axis. NOISE,
//    READOUT_NOISE, DETECTOR and OBSERVABLE ops are allowed.
// 4. The check Pauli must anticommute with at least one opening axis.
//    Otherwise the layer pair is an ordinary cancellation, not a block.
// 5. After a block, scanning resumes after L_close. A layer whose successor
//    is not a valid closing layer is skipped; it may be a lone T layer such
//    as a state injection.
// 6. Terminal form: the last layer, when it did not start a block under rule 2,
//    starts a terminal block if the ops after its last op satisfy rules 3 and
//    4 with the interior extended to the end of the HIR: exactly one visible
//    MEASURE, hidden MEASUREs with the check's unsigned Pauli, and otherwise
//    only NOISE, READOUT_NOISE, DETECTOR, OBSERVABLE and CONDITIONAL_PAULI ops
//    that commute with every opening axis. closing_ops is empty. No quantum op
//    follows the check, so the inverse layer is not needed to reproduce the
//    records and the block measures the same Clifford observable. An earlier
//    layer never qualifies because the layers after it put a T_GATE behind the
//    check.
//
// Closure check (sets `supported`): the HIR is walked from the start with a
// DormantSubspace, paused before each block's first opening op. Let P be the
// check Pauli and A the opening axes anticommuting with P. The block measures
// the Clifford observable C = P * prod_{j in A} exp(-i pi/4 s_j P_j), which
// conjugates a dormant generator g to g times the product of the axes in A
// that anticommute with g, up to sign. The block is supported when that
// unsigned image lies in the dormant subspace for every generator g, so the
// projection onto C = +1 or -1 leaves the subspace intact.
[[nodiscard]] std::vector<SandwichBlock> find_sandwich_blocks(const HirModule& hir);

// One flag per opening op of `block`: 1 when `mask` anticommutes with that
// op's axis. The flags are indexed like opening_ops; callers key noise
// realizations on them.
[[nodiscard]] std::vector<uint8_t> axis_anticommutation(const HirModule& hir,
                                                        const SandwichBlock& block,
                                                        PauliMaskView mask);

}  // namespace clifft
