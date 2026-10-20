#include "clifft/sampling/program_family.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft::sampling {

namespace {

void require_matching_outputs(const ExecutablePlan& generic, const ExecutablePlan& plan,
                              size_t member) {
    const std::string name = "program family member " + std::to_string(member);
    if (plan.num_visible_records() != generic.num_visible_records() ||
        plan.num_hidden_records() != generic.num_hidden_records() ||
        plan.num_detectors() != generic.num_detectors() ||
        plan.num_observables() != generic.num_observables() ||
        plan.num_exp_vals() != generic.num_exp_vals() ||
        plan.has_postselection() != generic.has_postselection()) {
        throw std::invalid_argument(name + " has outputs that differ from the generic plan");
    }
    if (plan.has_instruments()) {
        throw std::invalid_argument(name + " has instrument sites");
    }
    if (plan.num_presampled_symbols() != generic.num_presampled_symbols() ||
        !plan.has_same_presampled_noise(generic)) {
        throw std::invalid_argument(name + " draws presampled noise unlike the generic plan");
    }
}

void require_key_shape(const SandwichKeyMap& key_map, const std::vector<SandwichKey>& keys,
                       size_t member) {
    const std::string name = "program family member " + std::to_string(member);
    if (keys.size() != key_map.blocks().size()) {
        throw std::invalid_argument(name + " has " + std::to_string(keys.size()) + " keys for " +
                                    std::to_string(key_map.blocks().size()) + " sandwich blocks");
    }
    const auto is_flag = [](uint8_t flag) { return flag <= 1; };
    for (size_t b = 0; b < keys.size(); ++b) {
        const size_t width = key_map.blocks()[b].width();
        if (keys[b].axis_flip.size() != width || keys[b].internal_support.size() != width ||
            !std::ranges::all_of(keys[b].axis_flip, is_flag) ||
            !std::ranges::all_of(keys[b].internal_support, is_flag)) {
            throw std::invalid_argument(name + " key " + std::to_string(b) +
                                        " does not have one 0 or 1 flag per axis of its block");
        }
    }
}

}  // namespace

ProgramFamily::ProgramFamily(ExecutablePlan generic, SandwichKeyMap key_map,
                             std::vector<FamilyMember> members)
    : generic_(std::move(generic)), key_map_(std::move(key_map)), members_(std::move(members)) {
    if (generic_.has_instruments()) {
        throw std::invalid_argument("program families do not support instrument sites");
    }
    if (generic_.num_unbound_presampled_symbols() != 0) {
        throw std::invalid_argument(
            "program families require a distribution for every presampled symbol");
    }
    if (key_map_.num_presampled() != generic_.num_presampled_symbols()) {
        throw std::invalid_argument(
            "program family key map does not cover the generic plan's presampled symbols");
    }
    index_.reserve(members_.size());
    for (size_t m = 0; m < members_.size(); ++m) {
        require_matching_outputs(generic_, members_[m].plan, m);
        require_key_shape(key_map_, members_[m].keys, m);
        if (!index_.emplace(members_[m].keys, m).second) {
            throw std::invalid_argument("program family member " + std::to_string(m) +
                                        " repeats the keys of an earlier member");
        }
    }
    ordinal_of_symbol_.assign(generic_.num_symbols(), 0);
    const std::span<const uint32_t> presampled = generic_.presampled_symbols();
    for (uint32_t ordinal = 0; ordinal < presampled.size(); ++ordinal) {
        ordinal_of_symbol_[presampled[ordinal]] = ordinal;
    }
}

std::optional<size_t> ProgramFamily::find(const std::vector<SandwichKey>& keys) const {
    const auto found = index_.find(keys);
    if (found == index_.end()) {
        return std::nullopt;
    }
    return found->second;
}

void ProgramFamily::draw_presampled(Xoshiro256PlusPlus& rng,
                                    std::vector<uint32_t>& fired) const noexcept {
    assert(fired.capacity() >= generic_.num_noise_sites() &&
           "fired storage must have room for every noise site");
    fired.clear();
    generic_.draw_noise(
        0, generic_.num_noise_sites(), [&rng]() noexcept { return rng.next_double(); },
        [&](uint32_t symbol) noexcept { fired.push_back(ordinal_of_symbol_[symbol]); });
}

ProgramFamily build_program_family(const HirModule& hir,
                                   std::span<const std::vector<SandwichKey>> member_keys,
                                   const SamplingPlanOptions& options) {
    SandwichKeyMap key_map = compute_sandwich_key_map(hir);
    ExecutablePlan generic(plan_sampling(hir, options));
    std::vector<FamilyMember> members;
    members.reserve(member_keys.size());
    for (const std::vector<SandwichKey>& keys : member_keys) {
        HirModule specialized = hir;
        SandwichSpecializationPass pass(keys);
        pass.run(specialized);
        members.push_back(FamilyMember{keys, ExecutablePlan(plan_sampling(specialized, options))});
    }
    return ProgramFamily(std::move(generic), std::move(key_map), std::move(members));
}

}  // namespace clifft::sampling
