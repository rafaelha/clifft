#pragma once

// Maps one shot's presampled noise to the SandwichKey of every recognized
// T-sandwich block, so a dispatcher can pick the plan specialized for that
// realization class.
//
// A specialized plan is valid for the realizations whose accumulated
// pre-block error reverses the same opening axes (axis_flip) and whose
// interior errors anticommute with the same axes (internal_support). Reset
// feedback carries an error onto axes it never touches directly, so the
// reversals cannot be read off the fired channels. They are the presampled
// parities of the opening-axis signs in the planner's own frame. Sampled
// outcomes in those signs are handled inside the specialized plan, so keys
// depend on presampled symbols only.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/sandwich_specialization_pass.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace clifft::sampling {

// One opening axis of one block, indexed like find_sandwich_blocks and the
// block's opening_ops.
struct SandwichAxisRef {
    uint32_t block = 0;
    uint32_t axis = 0;

    friend bool operator==(const SandwichAxisRef&, const SandwichAxisRef&) = default;
};

// Presampled ordinals index the plan's Presampled symbols in ascending
// SymbolId order, as Executor::run_shot(presampled_values) consumes them.
// Every row of a block has one entry per opening axis. A block whose
// projection is not supported cannot be specialized, so it has no rows and
// its key is empty.
struct SandwichKeyRows {
    // Sorted ordinals whose parity reverses the axis at the opening in the
    // unspecialized HIR.
    std::vector<std::vector<uint32_t>> axis_flip_terms;
    // Sorted ordinals of interior channels that anticommute with the axis.
    std::vector<std::vector<uint32_t>> internal_support_terms;
    // Later axes whose reversal toggles when this axis is internally
    // supported but not reversed. Specializing the block absorbs its
    // correction K = prod exp(+i pi/4 s_j P_j) over the supported axes into
    // every later op but not into the frame accumulated before K. Relative
    // to the conjugated ops, that frame acts as if multiplied by each
    // supported P_j whose frame parity is 1 where K is absorbed. That parity
    // is the axis's reversal plus its interior support, which is 1, so it is
    // 1 exactly when the axis was not reversed. The targets are the later
    // axes that P_j, inserted where K is absorbed, anticommutes with after
    // the feedback it triggers.
    std::vector<std::vector<SandwichAxisRef>> support_flip_targets;

    [[nodiscard]] size_t width() const { return axis_flip_terms.size(); }
};

class SandwichKeyMap {
  public:
    SandwichKeyMap() = default;
    // Throws std::invalid_argument when a row is ragged, an ordinal is out of
    // range, or a support target does not name an axis of a later block.
    SandwichKeyMap(std::vector<SandwichKeyRows> blocks, uint32_t num_presampled);

    // Rows per block, in find_sandwich_blocks order.
    [[nodiscard]] const std::vector<SandwichKeyRows>& blocks() const { return blocks_; }
    [[nodiscard]] uint32_t num_presampled() const { return num_presampled_; }

    // Keys of the realization with one value per presampled ordinal.
    [[nodiscard]] std::vector<SandwichKey> keys_for(
        std::span<const uint8_t> presampled_values) const;
    // Keys of the realization in which exactly these distinct ordinals fire.
    [[nodiscard]] std::vector<SandwichKey> keys_for_fired(
        std::span<const uint32_t> fired_ordinals) const;
    // Keys shaped for this map with every flag clear.
    [[nodiscard]] std::vector<SandwichKey> zero_keys() const;
    // Overwrites `out`, which must be shaped like zero_keys(), with the keys
    // of the realization in which exactly these distinct ordinals fire. It
    // never allocates, so a sampler can reuse one key vector across shots.
    void assign_keys(std::span<const uint32_t> fired_ordinals,
                     std::vector<SandwichKey>& out) const noexcept;

  private:
    // One flag toggled by a firing ordinal.
    struct Toggle {
        uint32_t block = 0;
        uint32_t axis = 0;
        bool internal = false;
    };

    std::vector<SandwichKeyRows> blocks_;
    uint32_t num_presampled_ = 0;
    // Inverted index: the toggles of ordinal o are
    // toggles_[toggle_offsets_[o], toggle_offsets_[o + 1]).
    std::vector<uint32_t> toggle_offsets_;
    std::vector<Toggle> toggles_;
};

// Computes the key map of an unspecialized HIR from the planner's frame at
// each block opening. Throws std::invalid_argument when the HIR already holds
// sandwich ops, has instrument sites, or has noise moved from its logical
// position, or when planning it fails.
[[nodiscard]] SandwichKeyMap compute_sandwich_key_map(const HirModule& hir);

}  // namespace clifft::sampling
