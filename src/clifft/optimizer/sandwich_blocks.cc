#include "clifft/optimizer/sandwich_blocks.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/tableau/pauli_string.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>

namespace clifft {

namespace {

// Noisy circuits interleave channels and bookkeeping with the gates of one
// layer, so these ops must not split it.
bool is_layer_transparent(OpType type) {
    return type == OpType::NOISE || type == OpType::READOUT_NOISE || type == OpType::DETECTOR ||
           type == OpType::OBSERVABLE;
}

std::vector<std::vector<size_t>> collect_layers(const HirModule& hir) {
    std::vector<std::vector<size_t>> layers;
    std::vector<size_t> current;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const OpType type = hir.ops[i].op_type();
        if (type == OpType::T_GATE) {
            current.push_back(i);
        } else if (!is_layer_transparent(type) && !current.empty()) {
            layers.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) {
        layers.push_back(std::move(current));
    }
    return layers;
}

int compare_words(MaskView a, MaskView b) {
    assert(a.num_words() == b.num_words());
    for (uint32_t i = 0; i < a.num_words(); ++i) {
        if (a.words[i] != b.words[i]) {
            return a.words[i] < b.words[i] ? -1 : 1;
        }
    }
    return 0;
}

// Orders ops by unsigned axis. Sorting one layer by this key lets the
// opposite layer find each partner by binary search.
int compare_axes(const HirModule& hir, size_t a, size_t b) {
    const HeisenbergOp& op_a = hir.ops[a];
    const HeisenbergOp& op_b = hir.ops[b];
    const int x_order = compare_words(hir.destab_mask(op_a), hir.destab_mask(op_b));
    return x_order != 0 ? x_order : compare_words(hir.stab_mask(op_a), hir.stab_mask(op_b));
}

bool ops_anticommute(const HirModule& hir, size_t a, size_t b) {
    const HeisenbergOp& op_a = hir.ops[a];
    const HeisenbergOp& op_b = hir.ops[b];
    return anti_commute(hir.destab_mask(op_a), hir.stab_mask(op_a), hir.destab_mask(op_b),
                        hir.stab_mask(op_b));
}

// Returns the closing partner of each opening op, or nullopt when the layers
// are not inverses of each other.
std::optional<std::vector<size_t>> pair_layers(const HirModule& hir,
                                               const std::vector<size_t>& opening,
                                               const std::vector<size_t>& closing) {
    if (opening.size() != closing.size()) {
        return std::nullopt;
    }
    const size_t n = opening.size();

    // by_axis holds positions in `opening`, ordered by axis.
    std::vector<size_t> by_axis(n);
    std::iota(by_axis.begin(), by_axis.end(), size_t{0});
    std::ranges::sort(
        by_axis, [&](size_t a, size_t b) { return compare_axes(hir, opening[a], opening[b]) < 0; });

    constexpr size_t kUnpaired = std::numeric_limits<size_t>::max();
    std::vector<size_t> partner(n, kUnpaired);
    for (const size_t closing_op : closing) {
        const auto it = std::lower_bound(
            by_axis.begin(), by_axis.end(), closing_op, [&](size_t position, size_t op_index) {
                return compare_axes(hir, opening[position], op_index) < 0;
            });
        if (it == by_axis.end() || compare_axes(hir, opening[*it], closing_op) != 0) {
            return std::nullopt;
        }
        const size_t position = *it;
        // A second closing op on the same axis lands here too, which also
        // rejects layers whose opening ops share an axis: the pairing would
        // be ambiguous.
        if (partner[position] != kUnpaired) {
            return std::nullopt;
        }
        const HeisenbergOp& open_op = hir.ops[opening[position]];
        const HeisenbergOp& close_op = hir.ops[closing_op];
        if (hir.sign(open_op) != hir.sign(close_op) ||
            open_op.is_dagger() == close_op.is_dagger()) {
            return std::nullopt;
        }
        partner[position] = closing_op;
    }
    // Equal sizes and distinct partners leave no opening op unpaired.
    return partner;
}

bool axes_pairwise_commute(const HirModule& hir, const std::vector<size_t>& opening) {
    for (size_t i = 0; i < opening.size(); ++i) {
        for (size_t j = i + 1; j < opening.size(); ++j) {
            if (ops_anticommute(hir, opening[i], opening[j])) {
                return false;
            }
        }
    }
    return true;
}

struct Interior {
    size_t check_op = 0;
    std::vector<size_t> echo_ops;
    std::vector<size_t> noise_ops;
    std::vector<size_t> conditional_ops;
};

// Classifies ops in [first, last). Returns nullopt when they cannot sit
// between the layers of a block.
std::optional<Interior> scan_interior(const HirModule& hir, size_t first, size_t last,
                                      const std::vector<size_t>& opening) {
    Interior interior;
    std::optional<size_t> check;
    std::vector<size_t> hidden;
    for (size_t i = first; i < last; ++i) {
        const HeisenbergOp& op = hir.ops[i];
        switch (op.op_type()) {
            case OpType::T_GATE:
            case OpType::PHASE_ROTATION:
            case OpType::INSTRUMENT:
            case OpType::EXP_VAL:
                return std::nullopt;
            case OpType::MEASURE:
                if (op.is_hidden()) {
                    hidden.push_back(i);
                } else if (check.has_value()) {
                    return std::nullopt;
                } else {
                    check = i;
                }
                break;
            case OpType::CONDITIONAL_PAULI:
                for (const size_t axis : opening) {
                    if (ops_anticommute(hir, i, axis)) {
                        return std::nullopt;
                    }
                }
                interior.conditional_ops.push_back(i);
                break;
            case OpType::NOISE:
                interior.noise_ops.push_back(i);
                break;
            case OpType::READOUT_NOISE:
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
            case OpType::NUM_OP_TYPES:
                break;
        }
    }
    if (!check.has_value()) {
        return std::nullopt;
    }
    interior.check_op = *check;
    for (const size_t h : hidden) {
        if (compare_axes(hir, h, *check) != 0) {
            return std::nullopt;
        }
    }
    interior.echo_ops = std::move(hidden);
    return interior;
}

std::optional<SandwichBlock> try_form_block(const HirModule& hir,
                                            const std::vector<size_t>& opening,
                                            const std::vector<size_t>& closing) {
    std::optional<std::vector<size_t>> partner = pair_layers(hir, opening, closing);
    if (!partner.has_value() || !axes_pairwise_commute(hir, opening)) {
        return std::nullopt;
    }
    std::optional<Interior> interior =
        scan_interior(hir, opening.back() + 1, closing.front(), opening);
    if (!interior.has_value()) {
        return std::nullopt;
    }

    SandwichBlock block;
    block.opening_ops = opening;
    block.closing_ops = std::move(*partner);
    block.check_op = interior->check_op;
    block.echo_ops = std::move(interior->echo_ops);
    block.interior_noise_ops = std::move(interior->noise_ops);
    block.interior_conditional_ops = std::move(interior->conditional_ops);
    block.axis_anticommutes_with_check =
        axis_anticommutation(hir, block, hir.mask_view(hir.ops[block.check_op]));
    const bool any_anticommuting =
        std::ranges::any_of(block.axis_anticommutes_with_check, [](uint8_t f) { return f != 0; });
    if (!any_anticommuting) {
        return std::nullopt;
    }
    return block;
}

// True when every dormant generator maps back into the subspace under the
// block's Clifford observable. The image of g is g times the product of the
// check-anticommuting axes that anticommute with g.
bool closure_holds(const HirModule& hir, const SandwichBlock& block,
                   const DormantSubspace& subspace) {
    std::vector<size_t> axes;
    for (size_t j = 0; j < block.opening_ops.size(); ++j) {
        if (block.axis_anticommutes_with_check[j] != 0) {
            axes.push_back(block.opening_ops[j]);
        }
    }

    const uint32_t words = hir.pauli_masks.num_words();
    std::vector<uint64_t> image_x(words);
    std::vector<uint64_t> image_z(words);
    for (const PauliString& generator : subspace.generators()) {
        std::ranges::copy(generator.x().words, image_x.begin());
        std::ranges::copy(generator.z().words, image_z.begin());
        for (const size_t axis_index : axes) {
            const HeisenbergOp& axis = hir.ops[axis_index];
            // The axes commute pairwise, so the order of accumulation is free.
            if (anti_commute(generator.x(), generator.z(), hir.destab_mask(axis),
                             hir.stab_mask(axis))) {
                MutableMaskView{image_x}.xor_with(hir.destab_mask(axis));
                MutableMaskView{image_z}.xor_with(hir.stab_mask(axis));
            }
        }
        if (!subspace.contains(MaskView{image_x}, MaskView{image_z})) {
            return false;
        }
    }
    return true;
}

// Blocks are ordered and disjoint, so one walk visits every opening layer.
void evaluate_closure(const HirModule& hir, std::vector<SandwichBlock>& blocks) {
    if (blocks.empty()) {
        return;
    }
    DormantSubspace subspace(hir.num_qubits);
    size_t next = 0;
    for (size_t i = 0; i < hir.ops.size() && next < blocks.size(); ++i) {
        if (i == blocks[next].opening_ops.front()) {
            SandwichBlock& block = blocks[next++];
            block.active_width_at_opening = subspace.active_width();
            block.supported = closure_holds(hir, block, subspace);
        }
        static_cast<void>(classify_and_apply(hir, hir.ops[i], subspace));
    }
}

}  // namespace

std::vector<uint8_t> axis_anticommutation(const HirModule& hir, const SandwichBlock& block,
                                          PauliMaskView mask) {
    std::vector<uint8_t> flags(block.opening_ops.size(), 0);
    for (size_t j = 0; j < flags.size(); ++j) {
        const HeisenbergOp& axis = hir.ops[block.opening_ops[j]];
        flags[j] =
            anti_commute(mask.x(), mask.z(), hir.destab_mask(axis), hir.stab_mask(axis)) ? 1 : 0;
    }
    return flags;
}

std::vector<SandwichBlock> find_sandwich_blocks(const HirModule& hir) {
    const std::vector<std::vector<size_t>> layers = collect_layers(hir);
    std::vector<SandwichBlock> blocks;
    size_t i = 0;
    while (i + 1 < layers.size()) {
        if (std::optional<SandwichBlock> block = try_form_block(hir, layers[i], layers[i + 1])) {
            blocks.push_back(std::move(*block));
            i += 2;
        } else {
            ++i;
        }
    }
    evaluate_closure(hir, blocks);
    return blocks;
}

}  // namespace clifft
