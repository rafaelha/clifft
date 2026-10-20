#include "clifft/sampling/sandwich_keys.h"

#include "clifft/optimizer/sandwich_blocks.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace clifft::sampling {

namespace {

void require_ordinals(const std::vector<std::vector<uint32_t>>& rows, uint32_t num_presampled,
                      const char* name) {
    for (const std::vector<uint32_t>& row : rows) {
        if (!std::ranges::is_sorted(row) || std::ranges::adjacent_find(row) != row.end() ||
            std::ranges::any_of(row, [&](uint32_t ordinal) { return ordinal >= num_presampled; })) {
            throw std::invalid_argument(std::string("SandwichKeyMap ") + name +
                                        " rows must hold sorted distinct presampled ordinals");
        }
    }
}

// The Pauli error a single inserted Pauli has become at a later point, after
// the records it flips trigger their conditional corrections. The planner's
// frame carries an inserted Pauli the same way.
class InsertedPauli {
  public:
    InsertedPauli(const HirModule& hir, PauliMaskView mask)
        : x_(mask.x().words.begin(), mask.x().words.end()),
          z_(mask.z().words.begin(), mask.z().words.end()),
          record_flips_(static_cast<size_t>(hir.num_measurements) + hir.num_hidden_measurements,
                        0) {}

    void advance(const HirModule& hir, const HeisenbergOp& op) {
        switch (op.op_type()) {
            case OpType::MEASURE:
                record_flips_.at(static_cast<uint32_t>(op.meas_record_idx())) =
                    anticommutes(hir.mask_view(op)) ? 1 : 0;
                break;
            case OpType::CONDITIONAL_PAULI:
                if (record_flips_.at(static_cast<uint32_t>(op.controlling_meas())) != 0) {
                    const PauliMaskView mask = hir.mask_view(op);
                    MutableMaskView{x_}.xor_with(mask.x());
                    MutableMaskView{z_}.xor_with(mask.z());
                }
                break;
            default:
                break;
        }
    }

    [[nodiscard]] bool anticommutes(PauliMaskView mask) const {
        return anti_commute(MaskView{x_}, MaskView{z_}, mask.x(), mask.z());
    }

  private:
    std::vector<uint64_t> x_;
    std::vector<uint64_t> z_;
    std::vector<uint8_t> record_flips_;
};

std::vector<uint32_t> presampled_ordinals(const SamplingPlan& plan) {
    std::vector<uint32_t> ordinal(plan.symbols.size(), 0);
    uint32_t count = 0;
    for (size_t symbol = 0; symbol < plan.symbols.size(); ++symbol) {
        ordinal[symbol] = count;
        count += plan.symbols[symbol] == SymbolKind::Presampled ? 1 : 0;
    }
    return ordinal;
}

uint32_t count_presampled(const SamplingPlan& plan) {
    return static_cast<uint32_t>(std::ranges::count(plan.symbols, SymbolKind::Presampled));
}

// Interior channels anticommuting with each opening axis. Their symbols are
// the planner's, so a channel without probability has none and cannot fire.
std::vector<std::vector<uint32_t>> internal_support_rows(const HirModule& hir,
                                                         const SandwichBlock& block,
                                                         const SamplingPlan& plan,
                                                         const std::vector<uint32_t>& ordinal) {
    std::vector<std::vector<uint32_t>> rows(block.opening_ops.size());
    for (const size_t op : block.interior_noise_ops) {
        const auto site = static_cast<uint32_t>(hir.ops[op].noise_site_idx());
        const PresampledNoiseSite& planned = plan.presampled_noise_sites.at(site);
        uint32_t local = 0;
        for (const NoiseChannel& channel : hir.noise_sites.at(site).channels) {
            if (channel.prob == 0.0) {
                continue;
            }
            const uint32_t channel_ordinal = ordinal.at(index(planned.outcomes.at(local).symbol));
            ++local;
            const std::vector<uint8_t> flags =
                axis_anticommutation(hir, block, hir.noise_channel_masks.at(channel.mask));
            for (size_t j = 0; j < flags.size(); ++j) {
                if (flags[j] != 0) {
                    rows[j].push_back(channel_ordinal);
                }
            }
        }
    }
    for (std::vector<uint32_t>& row : rows) {
        std::ranges::sort(row);
    }
    return rows;
}

// Later keyed axes that each axis of `block`, inserted where specialization
// absorbs the block's correction, anticommutes with at their openings.
std::vector<std::vector<SandwichAxisRef>> support_flip_rows(
    const HirModule& hir, const std::vector<SandwichBlock>& blocks,
    const std::vector<uint8_t>& keyed, size_t b) {
    const SandwichBlock& block = blocks[b];
    std::vector<std::vector<SandwichAxisRef>> rows(block.opening_ops.size());
    if (block.closing_ops.empty()) {
        // A terminal block's correction reaches only the final frame.
        return rows;
    }
    const size_t absorbed_at = *std::ranges::min_element(block.closing_ops);
    for (size_t j = 0; j < block.opening_ops.size(); ++j) {
        InsertedPauli error(hir, hir.mask_view(hir.ops[block.opening_ops[j]]));
        size_t next_op = absorbed_at;
        for (size_t later = b + 1; later < blocks.size(); ++later) {
            if (keyed[later] == 0) {
                continue;
            }
            const SandwichBlock& target = blocks[later];
            for (; next_op < target.opening_ops.front(); ++next_op) {
                error.advance(hir, hir.ops[next_op]);
            }
            for (size_t k = 0; k < target.opening_ops.size(); ++k) {
                if (error.anticommutes(hir.mask_view(hir.ops[target.opening_ops[k]]))) {
                    rows[j].push_back(
                        SandwichAxisRef{static_cast<uint32_t>(later), static_cast<uint32_t>(k)});
                }
            }
        }
    }
    return rows;
}

}  // namespace

SandwichKeyMap::SandwichKeyMap(std::vector<SandwichKeyRows> blocks, uint32_t num_presampled)
    : blocks_(std::move(blocks)), num_presampled_(num_presampled) {
    for (size_t b = 0; b < blocks_.size(); ++b) {
        const SandwichKeyRows& rows = blocks_[b];
        const size_t width = rows.width();
        if (rows.internal_support_terms.size() != width ||
            rows.support_flip_targets.size() != width) {
            throw std::invalid_argument("SandwichKeyMap block " + std::to_string(b) +
                                        " has rows of different widths");
        }
        require_ordinals(rows.axis_flip_terms, num_presampled_, "axis flip");
        require_ordinals(rows.internal_support_terms, num_presampled_, "internal support");
        for (const std::vector<SandwichAxisRef>& targets : rows.support_flip_targets) {
            for (const SandwichAxisRef target : targets) {
                if (target.block <= b || target.block >= blocks_.size() ||
                    target.axis >= blocks_[target.block].width()) {
                    throw std::invalid_argument(
                        "SandwichKeyMap support targets must name axes of later blocks");
                }
            }
        }
    }

    // Counting sort by ordinal builds the inverted index in two passes.
    toggle_offsets_.assign(static_cast<size_t>(num_presampled_) + 1, 0);
    const auto for_each_toggle = [&](auto&& visit) {
        for (uint32_t b = 0; b < blocks_.size(); ++b) {
            const SandwichKeyRows& rows = blocks_[b];
            for (uint32_t j = 0; j < rows.width(); ++j) {
                for (const uint32_t ordinal : rows.axis_flip_terms[j]) {
                    visit(ordinal, Toggle{b, j, false});
                }
                for (const uint32_t ordinal : rows.internal_support_terms[j]) {
                    visit(ordinal, Toggle{b, j, true});
                }
            }
        }
    };
    for_each_toggle([&](uint32_t ordinal, Toggle) { ++toggle_offsets_[ordinal + 1]; });
    for (size_t o = 0; o < num_presampled_; ++o) {
        toggle_offsets_[o + 1] += toggle_offsets_[o];
    }
    toggles_.resize(toggle_offsets_.back());
    std::vector<uint32_t> cursor(toggle_offsets_.begin(), toggle_offsets_.end() - 1);
    for_each_toggle([&](uint32_t ordinal, Toggle toggle) { toggles_[cursor[ordinal]++] = toggle; });
}

std::vector<SandwichKey> SandwichKeyMap::zero_keys() const {
    std::vector<SandwichKey> keys(blocks_.size());
    for (size_t b = 0; b < blocks_.size(); ++b) {
        keys[b].axis_flip.assign(blocks_[b].width(), 0);
        keys[b].internal_support.assign(blocks_[b].width(), 0);
    }
    return keys;
}

void SandwichKeyMap::assign_keys(std::span<const uint32_t> fired_ordinals,
                                 std::vector<SandwichKey>& out) const noexcept {
    assert(out.size() == blocks_.size() && "key storage must be shaped by zero_keys");
    for (size_t b = 0; b < blocks_.size(); ++b) {
        assert(out[b].axis_flip.size() == blocks_[b].width() &&
               out[b].internal_support.size() == blocks_[b].width() &&
               "key storage must be shaped by zero_keys");
        std::ranges::fill(out[b].axis_flip, uint8_t{0});
        std::ranges::fill(out[b].internal_support, uint8_t{0});
    }
    for (const uint32_t ordinal : fired_ordinals) {
        assert(ordinal < num_presampled_ && "fired ordinal must name a presampled symbol");
        for (uint32_t t = toggle_offsets_[ordinal]; t < toggle_offsets_[ordinal + 1]; ++t) {
            const Toggle& toggle = toggles_[t];
            SandwichKey& key = out[toggle.block];
            (toggle.internal ? key.internal_support : key.axis_flip)[toggle.axis] ^= 1;
        }
    }
    // Targets lie in later blocks, so each block's flags are final when the
    // loop reaches it.
    for (size_t b = 0; b < blocks_.size(); ++b) {
        const SandwichKeyRows& rows = blocks_[b];
        for (size_t j = 0; j < rows.width(); ++j) {
            if (out[b].internal_support[j] == 0 || out[b].axis_flip[j] != 0) {
                continue;
            }
            for (const SandwichAxisRef target : rows.support_flip_targets[j]) {
                out[target.block].axis_flip[target.axis] ^= 1;
            }
        }
    }
}

std::vector<SandwichKey> SandwichKeyMap::keys_for_fired(
    std::span<const uint32_t> fired_ordinals) const {
    std::vector<uint32_t> sorted(fired_ordinals.begin(), fired_ordinals.end());
    std::ranges::sort(sorted);
    if (std::ranges::adjacent_find(sorted) != sorted.end() ||
        (!sorted.empty() && sorted.back() >= num_presampled_)) {
        throw std::invalid_argument("fired ordinals must be distinct presampled ordinals");
    }
    std::vector<SandwichKey> keys = zero_keys();
    assign_keys(sorted, keys);
    return keys;
}

std::vector<SandwichKey> SandwichKeyMap::keys_for(
    std::span<const uint8_t> presampled_values) const {
    if (presampled_values.size() != num_presampled_) {
        throw std::invalid_argument("presampled values must hold one value per presampled symbol");
    }
    std::vector<uint32_t> fired;
    for (uint32_t ordinal = 0; ordinal < presampled_values.size(); ++ordinal) {
        if (presampled_values[ordinal] > 1) {
            throw std::invalid_argument("presampled values must be Boolean");
        }
        if (presampled_values[ordinal] != 0) {
            fired.push_back(ordinal);
        }
    }
    std::vector<SandwichKey> keys = zero_keys();
    assign_keys(fired, keys);
    return keys;
}

SandwichKeyMap compute_sandwich_key_map(const HirModule& hir) {
    if (!hir.sandwich_sites.empty()) {
        throw std::invalid_argument("sandwich keys require an HIR without sandwich operations");
    }
    if (!hir.instrument_sites.empty()) {
        throw std::invalid_argument("sandwich keys do not support instrument sites");
    }
    // The probes read the frame where each block opens. Noise moved across
    // the block would be counted on the wrong side of it.
    if (!hir.logical_noise_prefix_matches_schedule()) {
        throw std::invalid_argument(
            "sandwich keys require noise operations at their logical positions");
    }

    const std::vector<SandwichBlock> blocks = find_sandwich_blocks(hir);
    std::vector<uint8_t> keyed(blocks.size(), 0);
    std::vector<FrameSignProbe> probes;
    for (size_t b = 0; b < blocks.size(); ++b) {
        if (!blocks[b].supported) {
            continue;
        }
        keyed[b] = 1;
        probes.push_back(FrameSignProbe{blocks[b].opening_ops.front(), blocks[b].opening_ops});
    }
    const ProbedSamplingPlan probed = plan_sampling_with_probes(hir, probes);
    const std::vector<uint32_t> ordinal = presampled_ordinals(probed.plan);

    std::vector<SandwichKeyRows> rows(blocks.size());
    size_t probe = 0;
    for (size_t b = 0; b < blocks.size(); ++b) {
        if (keyed[b] == 0) {
            continue;
        }
        SandwichKeyRows& block_rows = rows[b];
        for (const std::vector<SymbolId>& terms : probed.presampled_terms.at(probe)) {
            std::vector<uint32_t> ordinals;
            ordinals.reserve(terms.size());
            for (const SymbolId term : terms) {
                ordinals.push_back(ordinal.at(index(term)));
            }
            block_rows.axis_flip_terms.push_back(std::move(ordinals));
        }
        ++probe;
        block_rows.internal_support_terms =
            internal_support_rows(hir, blocks[b], probed.plan, ordinal);
        block_rows.support_flip_targets = support_flip_rows(hir, blocks, keyed, b);
    }
    return SandwichKeyMap(std::move(rows), count_presampled(probed.plan));
}

}  // namespace clifft::sampling
