#pragma once

// Helpers shared by the sandwich specialization and dispatch tests: toy and
// fixture circuits, a feedback-aware reference for the keys of a noise
// realization, and replay comparisons between generic and specialized plans.

#include "clifft/circuit/circuit.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/sandwich_blocks.h"
#include "clifft/optimizer/sandwich_specialization_pass.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace clifft::sandwich_test {

using sampling::ExecutablePlan;
using sampling::Executor;
using sampling::ReplayResult;
using sampling::SamplingPlan;

// A three-axis T sandwich around an ancilla check over data qubits 0-2 in
// |0>. The ancilla's X measurement picks up the data X parity through the
// ladder, so the check anticommutes with every Z axis.
inline constexpr const char* kToySandwich =
    "RX 3\n"
    "T 0 1 2\n"
    "CX 3 0 3 1 3 2\n"
    "MX 3\n"
    "RX 3\n"
    "CX 3 0 3 1 3 2\n"
    "T_DAG 0 1 2\n"
    "MX 0 1 2\n";

inline HirModule hir_of(const std::string& text) {
    return trace(parse(text));
}

inline HirModule specialize(const HirModule& hir, std::vector<SandwichKey> keys) {
    HirModule result = hir;
    SandwichSpecializationPass pass(std::move(keys));
    pass.run(result);
    return result;
}

// One fired Pauli channel. `channel` indexes the site's channel list.
struct FiredChannel {
    uint32_t site = 0;
    uint32_t channel = 0;
};

inline PauliMaskView channel_mask(const HirModule& hir, const FiredChannel& fired) {
    return hir.noise_channel_masks.at(hir.noise_sites[fired.site].channels[fired.channel].mask);
}

// The error a realization has accumulated at some point of an HIR: the fired
// channels plus the feedback they trigger by flipping the records that
// conditional Paulis read. The planner's frame carries the same noise terms.
class AccumulatedError {
  public:
    explicit AccumulatedError(const HirModule& hir)
        : x_(hir.pauli_masks.num_words(), 0),
          z_(hir.pauli_masks.num_words(), 0),
          record_flips_(static_cast<size_t>(hir.num_measurements) + hir.num_hidden_measurements,
                        0) {}

    void advance(const HirModule& hir, const HeisenbergOp& op,
                 const std::vector<FiredChannel>& fired) {
        switch (op.op_type()) {
            case OpType::NOISE:
                for (const FiredChannel& channel : fired) {
                    if (channel.site == static_cast<uint32_t>(op.noise_site_idx())) {
                        apply(channel_mask(hir, channel));
                    }
                }
                break;
            case OpType::MEASURE:
                record_flips_.at(static_cast<uint32_t>(op.meas_record_idx())) =
                    anticommutes(hir.mask_view(op)) ? 1 : 0;
                break;
            case OpType::SANDWICH_RECORD:
                record_flips_.at(static_cast<uint32_t>(op.sandwich_record_idx())) =
                    anticommutes(hir.mask_view(op)) ? 1 : 0;
                break;
            case OpType::CONDITIONAL_PAULI:
                if (record_flips_.at(static_cast<uint32_t>(op.controlling_meas())) != 0) {
                    apply(hir.mask_view(op));
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
    void apply(PauliMaskView mask) {
        MutableMaskView{x_}.xor_with(mask.x());
        MutableMaskView{z_}.xor_with(mask.z());
    }

    std::vector<uint64_t> x_;
    std::vector<uint64_t> z_;
    std::vector<uint8_t> record_flips_;
};

// The class of a realization: per block, the axes the accumulated error
// reverses at its opening, and the axes the product of its fired interior
// channels anticommutes with (interior feedback commutes with every axis).
// Feedback can carry an error onto an axis it does not touch directly, so
// the flips come from the accumulated error rather than from the fired
// channels alone. Each block is keyed in the module with the earlier blocks
// specialized, because their corrections conjugate the later masks the
// planner reads. Noise sites keep their indices through the pass.
inline std::vector<SandwichKey> keys_for(const HirModule& hir,
                                         const std::vector<FiredChannel>& fired) {
    const std::vector<SandwichBlock> original = find_sandwich_blocks(hir);
    std::vector<SandwichKey> keys;
    for (size_t b = 0; b < original.size(); ++b) {
        const HirModule current = keys.empty() ? hir : specialize(hir, keys);
        const SandwichBlock block = find_sandwich_blocks(current).at(0);
        REQUIRE(block.opening_ops.size() == original[b].opening_ops.size());
        const size_t width = block.opening_ops.size();
        SandwichKey key{std::vector<uint8_t>(width, 0), std::vector<uint8_t>(width, 0)};

        AccumulatedError error(current);
        for (size_t i = 0; i < block.opening_ops.front(); ++i) {
            error.advance(current, current.ops[i], fired);
        }
        for (size_t j = 0; j < width; ++j) {
            key.axis_flip[j] =
                error.anticommutes(current.mask_view(current.ops[block.opening_ops[j]])) ? 1 : 0;
        }
        for (const size_t op : block.interior_noise_ops) {
            const auto site = static_cast<uint32_t>(current.ops[op].noise_site_idx());
            for (const FiredChannel& channel : fired) {
                if (channel.site != site) {
                    continue;
                }
                const std::vector<uint8_t> flags =
                    axis_anticommutation(current, block, channel_mask(current, channel));
                for (size_t j = 0; j < width; ++j) {
                    key.internal_support[j] ^= flags[j];
                }
            }
        }
        keys.push_back(std::move(key));
    }
    return keys;
}

// Presampled values in ascending SymbolId order with the fired channels set.
inline std::vector<uint8_t> presampled_values(const HirModule& hir, const SamplingPlan& plan,
                                              const std::vector<FiredChannel>& fired) {
    std::vector<uint32_t> ordinal(plan.symbols.size(), 0);
    uint32_t count = 0;
    for (size_t symbol = 0; symbol < plan.symbols.size(); ++symbol) {
        ordinal[symbol] = count;
        count += plan.symbols[symbol] == sampling::SymbolKind::Presampled ? 1 : 0;
    }
    std::vector<uint8_t> values(count, 0);
    for (const FiredChannel& channel : fired) {
        uint32_t local = 0;
        const auto& channels = hir.noise_sites[channel.site].channels;
        for (uint32_t c = 0; c < channel.channel; ++c) {
            local += channels[c].prob != 0.0 ? 1 : 0;
        }
        const auto symbol = plan.presampled_noise_sites.at(channel.site).outcomes.at(local).symbol;
        values[ordinal[sampling::index(symbol)]] = 1;
    }
    return values;
}

// Generic and specialized plans of one HIR for one noise realization. The
// specialized plan uses `keys`, by default the class of the realization.
struct PlanPair {
    SamplingPlan generic_plan;
    SamplingPlan specialized_plan;
    ExecutablePlan generic;
    ExecutablePlan specialized;
    std::vector<uint8_t> generic_presampled;
    std::vector<uint8_t> specialized_presampled;

    PlanPair(const HirModule& hir, const std::vector<FiredChannel>& fired)
        : PlanPair(hir, fired, keys_for(hir, fired)) {}

    PlanPair(const HirModule& hir, const std::vector<FiredChannel>& fired,
             std::vector<SandwichKey> keys)
        : generic_plan(sampling::plan_sampling(hir)),
          specialized_plan(sampling::plan_sampling(specialize(hir, std::move(keys)))),
          generic(generic_plan),
          specialized(specialized_plan),
          generic_presampled(presampled_values(hir, generic_plan, fired)),
          specialized_presampled(presampled_values(hir, specialized_plan, fired)) {
        specialized_plan.validate();
    }
};

// Every realization with at most two fired channels on distinct sites.
inline std::vector<std::vector<FiredChannel>> small_realizations(const HirModule& hir) {
    std::vector<FiredChannel> singles;
    for (uint32_t site = 0; site < hir.noise_sites.size(); ++site) {
        for (uint32_t c = 0; c < hir.noise_sites[site].channels.size(); ++c) {
            singles.push_back(FiredChannel{site, c});
        }
    }
    std::vector<std::vector<FiredChannel>> realizations{{}};
    for (size_t a = 0; a < singles.size(); ++a) {
        realizations.push_back({singles[a]});
        for (size_t b = a + 1; b < singles.size(); ++b) {
            if (singles[b].site != singles[a].site) {
                realizations.push_back({singles[a], singles[b]});
            }
        }
    }
    return realizations;
}

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

inline std::string fixture_path(const std::string& name) {
    return std::string(CLIFFT_FIXTURES_DIR) + "/" + name;
}

enum class Pauli1 : uint8_t { X = 1, Y = 2, Z = 3 };

// The channel applying `pauli` to `qubit` from a single-qubit noise
// instruction traced into ops [begin, end). With `nearest_end` the search
// runs backwards, so a window that closes at a block picks the error
// closest to it.
inline FiredChannel find_channel(const Circuit& circuit, const HirModule& hir, size_t begin,
                                 size_t end, uint32_t qubit, Pauli1 pauli, bool nearest_end) {
    std::vector<size_t> candidates;
    for (size_t i = begin; i < end; ++i) {
        candidates.push_back(i);
    }
    if (nearest_end) {
        std::ranges::reverse(candidates);
    }
    for (const size_t i : candidates) {
        if (hir.ops[i].op_type() != OpType::NOISE) {
            continue;
        }
        // An instruction traces into one NOISE op per target, consecutively.
        size_t first = i;
        while (first > 0 && hir.ops[first - 1].op_type() == OpType::NOISE &&
               hir.source_map[first - 1] == hir.source_map[i]) {
            --first;
        }
        REQUIRE(hir.source_map[i].size() == 1);
        const uint32_t line = hir.source_map[i][0];
        // The parser may split one line into several nodes of one gate.
        std::vector<uint32_t> targets;
        GateType gate = GateType::TICK;
        for (const AstNode& node : circuit.nodes) {
            if (node.source_line == line) {
                gate = node.gate;
                for (const Target& target : node.targets) {
                    targets.push_back(target.value());
                }
            }
        }
        const bool depolarizing = gate == GateType::DEPOLARIZE1;
        const bool matching_flip = (gate == GateType::X_ERROR && pauli == Pauli1::X) ||
                                   (gate == GateType::Z_ERROR && pauli == Pauli1::Z);
        if ((!depolarizing && !matching_flip) || targets.at(i - first) != qubit) {
            continue;
        }
        const auto site = static_cast<uint32_t>(hir.ops[i].noise_site_idx());
        return FiredChannel{site, depolarizing ? static_cast<uint32_t>(pauli) - 1 : 0};
    }
    FAIL("no matching noise channel in the window");
    return {};
}

// Replay forces final record values and cannot attribute them to readout
// flips, so the replayed fixtures keep only their Pauli noise.
inline HirModule without_readout_noise(HirModule hir) {
    std::vector<HeisenbergOp> ops;
    std::vector<std::vector<uint32_t>> sources;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        if (hir.ops[i].op_type() == OpType::READOUT_NOISE) {
            continue;
        }
        ops.push_back(hir.ops[i]);
        sources.push_back(hir.source_map[i]);
    }
    hir.ops = std::move(ops);
    hir.source_map = std::move(sources);
    hir.readout_noise.clear();
    return hir;
}

inline size_t interior_end(const SandwichBlock& block) {
    return *std::ranges::min_element(block.closing_ops);
}

inline size_t block_end(const SandwichBlock& block) {
    return block.closing_ops.empty() ? block.check_op + 1
                                     : *std::ranges::max_element(block.closing_ops) + 1;
}

// Samples visible-then-hidden record rows from the generic plan under the
// fixed realization and requires both plans to replay each row with the
// same probability.
inline void check_sampled_rows_agree(PlanPair& plans, uint64_t num_shots) {
    const uint32_t visible = plans.generic.num_visible_records();
    const uint32_t hidden = plans.generic.num_hidden_records();
    REQUIRE(plans.specialized.num_visible_records() == visible);
    REQUIRE(plans.specialized.num_hidden_records() == hidden);
    Executor generic(plans.generic);
    Executor specialized(plans.specialized);
    std::vector<uint8_t> records(static_cast<size_t>(visible) + hidden);
    for (uint64_t seed = 0; seed < num_shots; ++seed) {
        Executor sampler(plans.generic, seed);
        sampler.run_shot(plans.generic_presampled);
        std::ranges::copy(sampler.visible_records(), records.begin());
        std::ranges::copy(sampler.hidden_records(), records.begin() + visible);
        INFO("seed " << seed);
        const ReplayResult expected = generic.replay_shot(records, plans.generic_presampled);
        const ReplayResult actual = specialized.replay_shot(records, plans.specialized_presampled);
        // The rows come from the generic plan, so an unreachable reference
        // would make the comparison vacuous.
        REQUIRE(expected.reachable);
        REQUIRE(actual.reachable);
        CHECK(std::abs(actual.log_probability - expected.log_probability) < 1e-9);
    }
}

}  // namespace clifft::sandwich_test
