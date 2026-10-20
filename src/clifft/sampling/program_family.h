#pragma once

// A program family pairs the generic plan of a circuit with plans
// specialized for chosen sandwich realization classes. Each shot draws its
// presampled noise once, computes the keys of that realization, and runs the
// member specialized for them, or the generic plan when no member matches.
// Every plan consumes the same presampled values by ordinal: the
// specialization pass keeps NOISE ops in circuit order, so one drawn vector
// names the same realization in every plan of the family.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/sandwich_specialization_pass.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sandwich_keys.h"
#include "clifft/util/xoshiro.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace clifft::sampling {

struct FamilyMember {
    // One key per block of the family's key map.
    std::vector<SandwichKey> keys;
    ExecutablePlan plan;
};

class ProgramFamily {
  public:
    // Throws std::invalid_argument when the generic plan has instruments or
    // presampled symbols without a distribution, when the key map does not
    // cover the generic plan's presampled symbols, when a member's outputs or
    // presampled noise differ from the generic plan's, or when member keys
    // are misshapen or repeated.
    ProgramFamily(ExecutablePlan generic, SandwichKeyMap key_map,
                  std::vector<FamilyMember> members);

    [[nodiscard]] const ExecutablePlan& generic() const { return generic_; }
    [[nodiscard]] size_t num_members() const { return members_.size(); }
    [[nodiscard]] const FamilyMember& member(size_t i) const { return members_.at(i); }
    [[nodiscard]] const SandwichKeyMap& key_map() const { return key_map_; }
    // The member specialized for exactly these keys, if any.
    [[nodiscard]] std::optional<size_t> find(const std::vector<SandwichKey>& keys) const;

    // Draws one realization from the generic plan's site distributions
    // exactly as scalar execution draws them and replaces `fired` with its
    // presampled ordinals in ascending order. `fired` must have capacity for
    // one ordinal per noise site, so a reused vector never allocates.
    void draw_presampled(Xoshiro256PlusPlus& rng, std::vector<uint32_t>& fired) const noexcept;

  private:
    ExecutablePlan generic_;
    SandwichKeyMap key_map_;
    std::vector<FamilyMember> members_;
    std::unordered_map<std::vector<SandwichKey>, size_t, SandwichKeysHash> index_;
    // Presampled ordinal of each generic-plan symbol that has one.
    std::vector<uint32_t> ordinal_of_symbol_;
};

// Plans `hir` as is for the generic plan and, for each entry of
// `member_keys`, runs SandwichSpecializationPass on a copy of `hir` and plans
// the result. Every plan uses `options`. Throws what the pass, planning, or
// the family constructor throws.
[[nodiscard]] ProgramFamily build_program_family(
    const HirModule& hir, std::span<const std::vector<SandwichKey>> member_keys,
    const SamplingPlanOptions& options = {});

}  // namespace clifft::sampling
