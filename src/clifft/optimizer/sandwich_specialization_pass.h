#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft {

// The noise-realization class one T-sandwich block is specialized for. Both
// vectors are indexed like the block's opening ops. A key with both vectors
// empty leaves its block unchanged.
struct SandwichKey {
    // 1 when the error accumulated before the block, including corrections
    // it triggers through feedback, anticommutes with that opening axis,
    // which reverses the axis's rotation. The axis is the one in the module
    // with the earlier keyed blocks specialized, since their corrections
    // conjugate later masks.
    std::vector<uint8_t> axis_flip;
    // 1 when the product of the errors inside the block anticommutes with
    // that opening axis.
    std::vector<uint8_t> internal_support;

    [[nodiscard]] bool empty() const { return axis_flip.empty() && internal_support.empty(); }

    friend bool operator==(const SandwichKey&, const SandwichKey&) = default;
};

// Hashes the per-block keys of one realization class so callers can look up
// the plan specialized for it.
struct SandwichKeysHash {
    [[nodiscard]] size_t operator()(const std::vector<SandwichKey>& keys) const noexcept;
};

// Replaces recognized T-sandwich blocks with projections onto the Clifford
// observables they measure, for one noise-realization class per block.
//
// A block U^-1 (interior) U, with U = prod_j exp(-i pi/8 s_j P_j), measures
// C = U^dag P U for its check P. Within a class the block is equivalent to:
// a projection onto an eigenspace of C at the opening, the interior noise as
// it stands, the check and echo records as the projection's branch combined
// with the frame sign of P at their positions, and the Clifford correction
// K = prod exp(+i pi/4 s_j P_j) over the axes that the product of the
// interior errors anticommutes with, applied after the interior (for a
// terminal block, only to the final frame). The rewrite drops both T
// layers, turns the check and echo measurements into SANDWICH_RECORD ops,
// inserts a SANDWICH_PROJECTION whose site keeps the opening axes and the
// key's axis_flip, and absorbs K into every later op.
//
// The result samples correctly only shots whose realization lies in the
// keyed class: the planner fixes each axis direction from axis_flip, and K
// is fixed by internal_support. Choosing the class per shot is the caller's
// responsibility, so the pass takes its keys as arguments and is not part of
// the registered pipeline.
//
// Keys are matched to find_sandwich_blocks(hir) in block order; trailing
// blocks without a key are left unchanged. run() throws std::invalid_argument
// when noise positions are not current, when there are more keys than
// blocks, when a nonempty key's vectors do not match its block's opening
// layer, when a keyed block is not supported, or when a keyed block has
// noise between the gates of one of its layers.
class SandwichSpecializationPass : public HirPass {
  public:
    explicit SandwichSpecializationPass(std::vector<SandwichKey> keys);
    void run(HirModule& hir) override;

    [[nodiscard]] size_t blocks_found() const { return blocks_found_; }
    [[nodiscard]] size_t blocks_specialized() const { return blocks_specialized_; }
    [[nodiscard]] const std::vector<SandwichKey>& keys() const { return keys_; }

  private:
    std::vector<SandwichKey> keys_;
    size_t blocks_found_ = 0;
    size_t blocks_specialized_ = 0;
};

}  // namespace clifft
