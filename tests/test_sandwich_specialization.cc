// Checks the sandwich projection and record ops: their HIR representation,
// how other passes treat them, the pass that replaces recognized T-sandwich
// blocks with them, and the sampling plans built from the result. Plans are
// compared against the unspecialized plan of the same module by replaying
// record rows under one fixed noise realization.

#include "clifft/circuit/circuit.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/drop_non_unitary_pass.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/sandwich_blocks.h"
#include "clifft/optimizer/sandwich_specialization_pass.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/hir_introspection.h"
#include "clifft/util/symplectic.h"

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;
using sampling::ExecutablePlan;
using sampling::Executor;
using sampling::ReplayResult;
using sampling::SamplingPlan;

namespace {

// A three-axis T sandwich around an ancilla check over data qubits 0-2 in
// |0>. The ancilla's X measurement picks up the data X parity through the
// ladder, so the check anticommutes with every Z axis.
constexpr const char* kToySandwich =
    "RX 3\n"
    "T 0 1 2\n"
    "CX 3 0 3 1 3 2\n"
    "MX 3\n"
    "RX 3\n"
    "CX 3 0 3 1 3 2\n"
    "T_DAG 0 1 2\n"
    "MX 0 1 2\n";

HirModule hir_of(const std::string& text) {
    return trace(parse(text));
}

SandwichKey uniform_key(size_t width, bool flip, bool support) {
    return SandwichKey{std::vector<uint8_t>(width, flip ? 1 : 0),
                       std::vector<uint8_t>(width, support ? 1 : 0)};
}

HirModule specialize(const HirModule& hir, std::vector<SandwichKey> keys) {
    HirModule result = hir;
    SandwichSpecializationPass pass(std::move(keys));
    pass.run(result);
    return result;
}

size_t count_ops(const HirModule& hir, OpType type) {
    return static_cast<size_t>(std::ranges::count_if(
        hir.ops, [type](const HeisenbergOp& op) { return op.op_type() == type; }));
}

PauliMaskHandle claim_axis(HirModule& hir, uint64_t destab, uint64_t stab, bool sign) {
    return hir.claim_side_mask([&](MutablePauliMaskView slot) {
        slot.x().words[0] = destab;
        slot.z().words[0] = stab;
        slot.set_sign(sign);
    });
}

void fill_x0(MutablePauliMaskView slot) {
    slot.x().words[0] = test::X(0);
}

// One site whose axes are Z0 and X1, a projection on X0 and a visible
// record of it. Extra mask capacity lets callers append more ops.
HirModule projection_module(size_t extra_masks = 0) {
    HirModule hir(3, 4 + extra_masks);
    SandwichSite site;
    site.axes = {claim_axis(hir, 0, test::Z(0), false), claim_axis(hir, test::X(1), 0, true)};
    site.axis_dagger = {0, 1};
    site.axis_flip = {0, 0};
    site.check_record = 0;
    hir.sandwich_sites.push_back(site);
    hir.num_measurements = 1;
    hir.append_sandwich_projection(SandwichSiteIdx{0}, fill_x0);
    hir.append_sandwich_record(MeasRecordIdx{0}, SandwichSiteIdx{0}, false, fill_x0);
    return hir;
}

std::string format_op(const HirModule& hir, size_t index) {
    return format_hir_op(hir.ops[index], hir.mask_view(hir.ops[index]));
}

// One fired Pauli channel. `channel` indexes the site's channel list.
struct FiredChannel {
    uint32_t site = 0;
    uint32_t channel = 0;
};

PauliMaskView channel_mask(const HirModule& hir, const FiredChannel& fired) {
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
std::vector<SandwichKey> keys_for(const HirModule& hir, const std::vector<FiredChannel>& fired) {
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
std::vector<uint8_t> presampled_values(const HirModule& hir, const SamplingPlan& plan,
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

// Generic and specialized plans of one HIR for one noise realization.
struct PlanPair {
    SamplingPlan generic_plan;
    SamplingPlan specialized_plan;
    ExecutablePlan generic;
    ExecutablePlan specialized;
    std::vector<uint8_t> generic_presampled;
    std::vector<uint8_t> specialized_presampled;

    PlanPair(const HirModule& hir, const std::vector<FiredChannel>& fired)
        : generic_plan(sampling::plan_sampling(hir)),
          specialized_plan(sampling::plan_sampling(specialize(hir, keys_for(hir, fired)))),
          generic(generic_plan),
          specialized(specialized_plan),
          generic_presampled(presampled_values(hir, generic_plan, fired)),
          specialized_presampled(presampled_values(hir, specialized_plan, fired)) {
        specialized_plan.validate();
    }
};

// Replays every record pattern through both plans. Returns the total
// probability of the reachable patterns under the specialized plan.
double check_all_patterns_agree(PlanPair& plans) {
    const uint32_t num_records =
        plans.generic.num_visible_records() + plans.generic.num_hidden_records();
    REQUIRE(num_records <= 16);
    Executor generic(plans.generic);
    Executor specialized(plans.specialized);
    std::vector<uint8_t> records(num_records);
    double total = 0.0;
    for (uint32_t pattern = 0; pattern < (uint32_t{1} << num_records); ++pattern) {
        for (uint32_t r = 0; r < num_records; ++r) {
            records[r] = static_cast<uint8_t>((pattern >> r) & 1U);
        }
        const ReplayResult expected = generic.replay_shot(records, plans.generic_presampled);
        const ReplayResult actual = specialized.replay_shot(records, plans.specialized_presampled);
        INFO("record pattern " << pattern);
        REQUIRE(actual.reachable == expected.reachable);
        if (expected.reachable) {
            CHECK(std::abs(actual.log_probability - expected.log_probability) < 1e-9);
            total += std::exp(actual.log_probability);
        }
    }
    return total;
}

// Every realization with at most two fired channels on distinct sites.
std::vector<std::vector<FiredChannel>> small_realizations(const HirModule& hir) {
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

// A planned module whose only quantum content is one hand-built projection
// on one qubit in |0>, with `check` the projected Pauli and `axis` the single
// opening axis.
HirModule single_projection(uint64_t check_x, uint64_t check_z, uint64_t axis_x, uint64_t axis_z) {
    HirModule hir(2, 3);
    SandwichSite site;
    site.axes = {claim_axis(hir, axis_x, axis_z, false)};
    site.axis_dagger = {0};
    site.axis_flip = {0};
    hir.sandwich_sites.push_back(site);
    hir.num_measurements = 1;
    const auto fill = [&](MutablePauliMaskView slot) {
        slot.x().words[0] = check_x;
        slot.z().words[0] = check_z;
    };
    hir.append_sandwich_projection(SandwichSiteIdx{0}, fill);
    hir.append_sandwich_record(MeasRecordIdx{0}, SandwichSiteIdx{0}, false, fill);
    return hir;
}

}  // namespace

TEST_CASE("Sandwich ops carry their site and record payloads", "[sandwich]") {
    HirModule hir = projection_module(1);
    hir.append_sandwich_record(MeasRecordIdx{1}, SandwichSiteIdx{0}, true, fill_x0);

    const HeisenbergOp& projection = hir.ops[0];
    CHECK(projection.op_type() == OpType::SANDWICH_PROJECTION);
    CHECK(projection.sandwich_site_idx() == SandwichSiteIdx{0});
    CHECK(op_type_to_str(OpType::SANDWICH_PROJECTION) == "SANDWICH_PROJECTION");
    CHECK(format_op(hir, 0) == "SANDWICH_PROJECTION +X0 site=0");

    const HeisenbergOp& record = hir.ops[1];
    CHECK(record.op_type() == OpType::SANDWICH_RECORD);
    CHECK(record.sandwich_record_idx() == MeasRecordIdx{0});
    CHECK(record.sandwich_site_idx() == SandwichSiteIdx{0});
    CHECK_FALSE(record.is_hidden());
    CHECK(op_type_to_str(OpType::SANDWICH_RECORD) == "SANDWICH_RECORD");
    CHECK(format_op(hir, 1) == "SANDWICH_RECORD +X0 -> rec[0] site=0");
    CHECK(format_op(hir, 2) == "SANDWICH_RECORD +X0 -> rec[1] site=0 (hidden)");

    CHECK_FALSE(hir.is_deterministic());
}

TEST_CASE("Growing the Pauli arena keeps every existing slot", "[sandwich]") {
    const HirModule hir = projection_module();
    const HirModule grown = hir.with_additional_pauli_masks(2);

    REQUIRE(grown.pauli_masks.size() == hir.pauli_masks.size() + 2);
    for (size_t i = 0; i < hir.pauli_masks.size(); ++i) {
        const auto handle = static_cast<PauliMaskHandle>(i);
        CHECK(grown.pauli_masks.at(handle) == hir.pauli_masks.at(handle));
    }
    CHECK(grown.sandwich_sites == hir.sandwich_sites);
    CHECK(grown.ops.size() == hir.ops.size());

    // The copy claims after the original's last slot.
    HirModule extended = grown;
    const PauliMaskHandle fresh = claim_axis(extended, 0, test::Z(2), true);
    CHECK(static_cast<size_t>(fresh) == hir.pauli_masks.size());
}

TEST_CASE("Sandwich ops never swap with a commuting neighbour", "[sandwich]") {
    HirModule hir = projection_module(1);
    // Z2 commutes with every mask in the module.
    test::append_tgate(hir, 0, test::Z(2), false);

    CHECK_FALSE(can_swap(hir.ops[0], hir.ops[2], hir));
    CHECK_FALSE(can_swap(hir.ops[2], hir.ops[0], hir));
    CHECK_FALSE(can_swap(hir.ops[1], hir.ops[2], hir));
    CHECK_FALSE(can_swap(hir.ops[2], hir.ops[1], hir));
}

TEST_CASE("Clifford absorption conjugates sandwich masks and site axes", "[sandwich]") {
    HirModule hir = projection_module();
    optimizer_detail::CliffordAbsorption cliffords;
    PauliString axis(hir.num_qubits);
    axis.set_pauli(0, false, true);
    axis.set_pauli(1, false, true);
    // exp(-i pi/4 Z0 Z1) sends X0 to Y0 Z1 and X1 to Z0 Y1 up to sign, and
    // leaves Z0 alone.
    cliffords.absorb(axis, 2);
    for (const HeisenbergOp& op : hir.ops) {
        cliffords.transform(hir, op);
    }

    const auto expect_body = [&](PauliMaskView mask, uint64_t destab, uint64_t stab) {
        CHECK(mask.x().words[0] == destab);
        CHECK(mask.z().words[0] == stab);
    };
    expect_body(hir.mask_view(hir.ops[0]), test::X(0), test::Z(0) | test::Z(1));
    expect_body(hir.mask_view(hir.ops[1]), test::X(0), test::Z(0) | test::Z(1));
    const SandwichSite& site = hir.sandwich_sites[0];
    expect_body(hir.pauli_masks.at(site.axes[0]), 0, test::Z(0));
    expect_body(hir.pauli_masks.at(site.axes[1]), test::X(1), test::Z(0) | test::Z(1));
}

TEST_CASE("Peephole fusion conjugates sandwich masks downstream", "[sandwich]") {
    HirModule hir(3, 6);
    // T T on Z0 fuses into S on Z0, which is absorbed into later ops.
    test::append_tgate(hir, 0, test::Z(0), false);
    test::append_tgate(hir, 0, test::Z(0), false);
    SandwichSite site;
    site.axes = {claim_axis(hir, test::X(0), 0, false)};
    site.axis_dagger = {0};
    site.axis_flip = {0};
    hir.sandwich_sites.push_back(site);
    hir.num_measurements = 1;
    hir.append_sandwich_projection(SandwichSiteIdx{0}, fill_x0);
    hir.append_sandwich_record(MeasRecordIdx{0}, SandwichSiteIdx{0}, false, fill_x0);

    PeepholeFusionPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 2);
    REQUIRE(hir.ops[0].op_type() == OpType::SANDWICH_PROJECTION);
    REQUIRE(hir.ops[1].op_type() == OpType::SANDWICH_RECORD);
    for (const PauliMaskView mask : {hir.mask_view(hir.ops[0]), hir.mask_view(hir.ops[1]),
                                     hir.pauli_masks.at(hir.sandwich_sites[0].axes[0])}) {
        CHECK(mask.x().words[0] == test::X(0));
        CHECK(mask.z().words[0] == test::Z(0));
    }
}

TEST_CASE("DropNonUnitaryPass removes sandwich ops and their sites", "[sandwich]") {
    HirModule hir = projection_module(1);
    test::append_tgate(hir, 0, test::Z(2), false);

    DropNonUnitaryPass pass;
    pass.run(hir);

    REQUIRE(hir.ops.size() == 1);
    CHECK(hir.ops[0].op_type() == OpType::T_GATE);
    CHECK(hir.sandwich_sites.empty());
    CHECK(hir.num_measurements == 0);
}

TEST_CASE("SandwichSpecializationPass replaces a block with a projection and records",
          "[sandwich]") {
    const HirModule hir = hir_of(kToySandwich);
    const std::vector<SandwichBlock> blocks = find_sandwich_blocks(hir);
    REQUIRE(blocks.size() == 1);
    const SandwichBlock& block = blocks[0];
    REQUIRE(block.supported);

    SandwichSpecializationPass pass({uniform_key(3, false, false)});
    HirModule result = hir;
    pass.run(result);
    CHECK(pass.blocks_found() == 1);
    CHECK(pass.blocks_specialized() == 1);

    // Both layers go; the projection takes the first opening slot and the
    // check and echo keep their positions as records.
    REQUIRE(result.ops.size() == hir.ops.size() - 5);
    CHECK(result.num_t_gates() == 0);
    CHECK(count_ops(result, OpType::MEASURE) == count_ops(hir, OpType::MEASURE) - 2);
    const size_t opening = block.opening_ops.front();
    REQUIRE(result.ops[opening].op_type() == OpType::SANDWICH_PROJECTION);
    const HeisenbergOp& check = result.ops[opening + 1];
    const HeisenbergOp& echo = result.ops[opening + 2];
    REQUIRE(check.op_type() == OpType::SANDWICH_RECORD);
    REQUIRE(echo.op_type() == OpType::SANDWICH_RECORD);
    CHECK(check.sandwich_record_idx() == hir.ops[block.check_op].meas_record_idx());
    CHECK_FALSE(check.is_hidden());
    CHECK(echo.sandwich_record_idx() == hir.ops[block.echo_ops[0]].meas_record_idx());
    CHECK(echo.is_hidden());
    CHECK(result.mask_view(result.ops[opening]) == hir.mask_view(hir.ops[block.check_op]));
    CHECK(result.mask_view(check) == hir.mask_view(hir.ops[block.check_op]));

    REQUIRE(result.sandwich_sites.size() == 1);
    const SandwichSite& site = result.sandwich_sites[0];
    CHECK(result.ops[opening].sandwich_site_idx() == SandwichSiteIdx{0});
    CHECK(check.sandwich_site_idx() == SandwichSiteIdx{0});
    CHECK(site.check_record == static_cast<uint32_t>(hir.ops[block.check_op].meas_record_idx()));
    CHECK(site.axis_dagger == std::vector<uint8_t>{0, 0, 0});
    CHECK(site.axis_flip == std::vector<uint8_t>{0, 0, 0});
    REQUIRE(site.axes.size() == 3);
    for (size_t j = 0; j < 3; ++j) {
        CHECK(result.pauli_masks.at(site.axes[j]) == hir.mask_view(hir.ops[block.opening_ops[j]]));
    }

    // The projection names every gate of both layers as its source.
    std::vector<uint32_t> layer_lines;
    for (const auto* layer : {&block.opening_ops, &block.closing_ops}) {
        for (const size_t op : *layer) {
            layer_lines.insert(layer_lines.end(), hir.source_map[op].begin(),
                               hir.source_map[op].end());
        }
    }
    std::ranges::sort(layer_lines);
    layer_lines.erase(std::unique(layer_lines.begin(), layer_lines.end()), layer_lines.end());
    REQUIRE(result.source_map.size() == result.ops.size());
    CHECK(result.source_map[opening] == layer_lines);
    CHECK(result.source_map[opening + 1] == hir.source_map[block.check_op]);
    CHECK(result.logical_noise_prefix.empty());
}

TEST_CASE("SandwichSpecializationPass copies the key flips and opening directions", "[sandwich]") {
    // T_DAG opens this block, so every axis records a dagger.
    std::string text = kToySandwich;
    text.replace(text.find("T_DAG 0 1 2"), 11, "T 0 1 2");
    text.replace(text.find("T 0 1 2"), 7, "T_DAG 0 1 2");
    const HirModule hir = hir_of(text);
    REQUIRE(find_sandwich_blocks(hir).size() == 1);

    const HirModule result = specialize(hir, {SandwichKey{{1, 0, 1}, {0, 0, 0}}});
    REQUIRE(result.sandwich_sites.size() == 1);
    CHECK(result.sandwich_sites[0].axis_dagger == std::vector<uint8_t>{1, 1, 1});
    CHECK(result.sandwich_sites[0].axis_flip == std::vector<uint8_t>{1, 0, 1});
}

TEST_CASE("SandwichSpecializationPass folds the internal correction into later ops", "[sandwich]") {
    const HirModule hir = hir_of(kToySandwich);
    const SandwichBlock block = find_sandwich_blocks(hir).at(0);
    const HirModule result = specialize(hir, {SandwichKey{{0, 0, 0}, {1, 0, 0}}});
    const size_t after_closing = *std::ranges::max_element(block.closing_ops) + 1;

    // Ops after the closing layer shift left by the five dropped gates.
    const PauliMaskView axis = hir.mask_view(hir.ops[block.opening_ops[0]]);
    size_t anticommuting = 0;
    for (size_t i = after_closing; i < hir.ops.size(); ++i) {
        const PauliMaskView before = hir.mask_view(hir.ops[i]);
        const PauliMaskView after = result.mask_view(result.ops[i - 5]);
        if (anti_commute(before.x(), before.z(), axis.x(), axis.z())) {
            // Conjugating by exp(i pi/4 P) multiplies an anticommuting Pauli by P.
            ++anticommuting;
            for (uint32_t w = 0; w < before.x().num_words(); ++w) {
                CHECK(after.x().words[w] == (before.x().words[w] ^ axis.x().words[w]));
                CHECK(after.z().words[w] == (before.z().words[w] ^ axis.z().words[w]));
            }
        } else {
            CHECK(after == before);
        }
    }
    CHECK(anticommuting > 0);

    // Without internal support the later ops are untouched.
    const HirModule plain = specialize(hir, {uniform_key(3, false, false)});
    for (size_t i = after_closing; i < hir.ops.size(); ++i) {
        CHECK(plain.mask_view(plain.ops[i - 5]) == hir.mask_view(hir.ops[i]));
    }
}

TEST_CASE("SandwichSpecializationPass leaves blocks with empty keys unchanged", "[sandwich]") {
    // Two copies of the toy block, with the data reset between them.
    const HirModule hir = hir_of(std::string(kToySandwich) + "R 0 1 2\n" + kToySandwich);
    const std::vector<SandwichBlock> blocks = find_sandwich_blocks(hir);
    REQUIRE(blocks.size() == 2);
    REQUIRE(blocks[1].supported);

    SECTION("no keys") {
        SandwichSpecializationPass pass({});
        HirModule result = hir;
        pass.run(result);
        CHECK(pass.blocks_found() == 2);
        CHECK(pass.blocks_specialized() == 0);
        CHECK(result.ops.size() == hir.ops.size());
        CHECK(result.num_t_gates() == hir.num_t_gates());
    }

    SECTION("only the second block") {
        SandwichSpecializationPass pass({SandwichKey{}, uniform_key(3, false, false)});
        HirModule result = hir;
        pass.run(result);
        CHECK(pass.blocks_specialized() == 1);
        CHECK(result.num_t_gates() == 6);
        CHECK(count_ops(result, OpType::SANDWICH_PROJECTION) == 1);
        // The first block's layers keep their ops.
        for (const size_t op : blocks[0].opening_ops) {
            CHECK(result.ops[op].op_type() == OpType::T_GATE);
        }
        CHECK(result.ops[blocks[1].opening_ops.front()].op_type() == OpType::SANDWICH_PROJECTION);
    }
}

TEST_CASE("SandwichSpecializationPass rejects invalid keys and blocks", "[sandwich]") {
    const HirModule hir = hir_of(kToySandwich);

    SECTION("key length mismatch") {
        HirModule copy = hir;
        SandwichSpecializationPass pass({SandwichKey{{0, 0}, {0, 0, 0}}});
        CHECK_THROWS_WITH(pass.run(copy), ContainsSubstring("2 axis flips"));
        SandwichSpecializationPass support({SandwichKey{{0, 0, 0}, {}}});
        CHECK_THROWS_AS(support.run(copy), std::invalid_argument);
    }

    SECTION("more keys than blocks") {
        HirModule copy = hir;
        SandwichSpecializationPass pass(
            {uniform_key(3, false, false), uniform_key(3, false, false)});
        CHECK_THROWS_WITH(pass.run(copy), ContainsSubstring("2 keys for 1 recognized blocks"));
    }

    SECTION("unsupported block") {
        // Data qubits in |+> are stabilized by X, which the Z axes rotate away.
        std::string text = kToySandwich;
        text.replace(0, 4, "RX 0 1 2 3");
        HirModule copy = hir_of(text);
        const std::vector<SandwichBlock> blocks = find_sandwich_blocks(copy);
        REQUIRE(blocks.size() == 1);
        REQUIRE_FALSE(blocks[0].supported);
        SandwichSpecializationPass pass({uniform_key(3, false, false)});
        CHECK_THROWS_WITH(pass.run(copy), ContainsSubstring("not supported"));
    }

    SECTION("noise between the gates of a layer") {
        std::string text = kToySandwich;
        text.replace(text.find("T 0 1 2"), 7, "T 0\nX_ERROR(0.1) 0\nT 1 2");
        HirModule copy = hir_of(text);
        REQUIRE(find_sandwich_blocks(copy).size() == 1);
        SandwichSpecializationPass pass({uniform_key(3, false, false)});
        CHECK_THROWS_WITH(pass.run(copy), ContainsSubstring("interleaves noise"));
    }

    SECTION("a measurement moved across noise") {
        HirModule copy = hir_of(std::string(kToySandwich) + "X_ERROR(0.1) 0\nM 1\n");
        copy.materialize_logical_noise_prefix();
        const size_t last = copy.ops.size() - 1;
        REQUIRE(copy.ops[last].op_type() == OpType::MEASURE);
        REQUIRE(copy.ops[last - 1].op_type() == OpType::NOISE);
        std::swap(copy.ops[last], copy.ops[last - 1]);
        std::swap(copy.source_map[last], copy.source_map[last - 1]);
        std::swap(copy.logical_noise_prefix[last], copy.logical_noise_prefix[last - 1]);
        REQUIRE_FALSE(copy.logical_noise_prefix_matches_schedule());
        SandwichSpecializationPass pass({uniform_key(3, false, false)});
        CHECK_THROWS_WITH(pass.run(copy), ContainsSubstring("logical positions"));
    }
}

TEST_CASE("Specialized sandwich plans replay every record of a noiseless block", "[sandwich]") {
    const HirModule hir = hir_of(kToySandwich);
    PlanPair plans(hir, {});
    CHECK(plans.specialized.peak_active_width() <= plans.generic.peak_active_width());
    CHECK(plans.specialized.requires_scalar_execution());
    CHECK_THAT(plans.specialized_plan.inspect(), ContainsSubstring("PROJECT_ACTIVE_UNITARY"));
    CHECK(std::abs(check_all_patterns_agree(plans) - 1.0) < 1e-9);
}

TEST_CASE("Specialized sandwich plans replay every record of a noisy block", "[sandwich]") {
    // Depolarizing sites before the opening, inside the block before the
    // check and after the reset, and after the closing layer. The terminal
    // variant has no closing layer, so its interior runs to the end.
    const auto [name, text] = GENERATE(table<std::string, std::string>({
        {"closed T block",
         "RX 3\nDEPOLARIZE1(0.1) 0\nT 0 1 2\nDEPOLARIZE1(0.1) 2 3\nCX 3 0 3 1 3 2\nMX 3\n"
         "RX 3\nDEPOLARIZE1(0.1) 1\nCX 3 0 3 1 3 2\nT_DAG 0 1 2\nDEPOLARIZE1(0.1) 0\n"
         "MX 0 1 2\n"},
        {"closed mixed block",
         "RX 3\nDEPOLARIZE1(0.1) 1\nT_DAG 0 1\nT 2\nDEPOLARIZE1(0.1) 0 3\nCX 3 0 3 1 3 2\n"
         "MX 3\nRX 3\nDEPOLARIZE1(0.1) 2\nCX 3 0 3 1 3 2\nT 0 1\nT_DAG 2\nM 0 1 2\n"},
        {"terminal block",
         "RX 3\nDEPOLARIZE1(0.1) 0\nT 0 1 2\nCX 3 0 3 1 3 2\nDEPOLARIZE1(0.1) 1 3\nMX 3\n"
         "RX 3\nDEPOLARIZE1(0.1) 2\n"},
    }));
    CAPTURE(name);
    const HirModule hir = hir_of(text);
    const std::vector<SandwichBlock> blocks = find_sandwich_blocks(hir);
    REQUIRE(blocks.size() == 1);
    REQUIRE(blocks[0].supported);
    REQUIRE_FALSE(blocks[0].interior_noise_ops.empty());

    size_t flipped = 0;
    size_t supported = 0;
    for (const std::vector<FiredChannel>& fired : small_realizations(hir)) {
        const std::vector<SandwichKey> keys = keys_for(hir, fired);
        flipped += std::ranges::count(keys[0].axis_flip, 1) > 0 ? 1 : 0;
        supported += std::ranges::count(keys[0].internal_support, 1) > 0 ? 1 : 0;
        PlanPair plans(hir, fired);
        CHECK(std::abs(check_all_patterns_agree(plans) - 1.0) < 1e-9);
    }
    // Both parts of the key vary across the realizations.
    CHECK(flipped > 0);
    CHECK(supported > 0);
}

TEST_CASE("The sandwich planner rejects malformed projections", "[sandwich]") {
    SECTION("a record before its projection") {
        HirModule hir = single_projection(test::X(0), 0, 0, test::Z(0));
        std::swap(hir.ops[0], hir.ops[1]);
        CHECK_THROWS_WITH(sampling::plan_sampling(hir), ContainsSubstring("precedes"));
    }

    SECTION("a projection without its check record") {
        HirModule hir = single_projection(test::X(0), 0, 0, test::Z(0));
        hir.ops.pop_back();
        CHECK_THROWS_WITH(sampling::plan_sampling(hir), ContainsSubstring("check record"));
    }

    SECTION("a site whose flags do not match its axes") {
        HirModule hir = single_projection(test::X(0), 0, 0, test::Z(0));
        hir.sandwich_sites[0].axis_flip.push_back(0);
        CHECK_THROWS_WITH(sampling::plan_sampling(hir), ContainsSubstring("malformed"));
    }

    SECTION("axes that all commute with the check") {
        const HirModule hir = single_projection(test::X(0), 0, 0, test::Z(1));
        CHECK_THROWS_WITH(sampling::plan_sampling(hir), ContainsSubstring("no axis"));
    }

    SECTION("an observable that leaves the dormant state") {
        // C = Z0 exp(-i pi/4 X0) maps the dormant stabilizer Z0 to a Y.
        const HirModule hir = single_projection(0, test::Z(0), test::X(0), 0);
        CHECK_THROWS_WITH(sampling::plan_sampling(hir), ContainsSubstring("not closed"));
    }

    SECTION("a projection wider than the dense matrix limit") {
        // Nine T gates on |+> qubits are active before the toy block opens.
        const HirModule hir = specialize(
            hir_of("H 4 5 6 7 8 9 10 11 12\nT 4 5 6 7 8 9 10 11 12\n" + std::string(kToySandwich)),
            {uniform_key(3, false, false)});
        REQUIRE(count_ops(hir, OpType::SANDWICH_PROJECTION) == 1);
        CHECK_THROWS_WITH(sampling::plan_sampling(hir), ContainsSubstring("exceeds the maximum"));
    }
}

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

std::string fixture_path(const std::string& name) {
    return std::string(CLIFFT_FIXTURES_DIR) + "/" + name;
}

enum class Pauli1 : uint8_t { X = 1, Y = 2, Z = 3 };

// The channel applying `pauli` to `qubit` from a single-qubit noise
// instruction traced into ops [begin, end). With `nearest_end` the search
// runs backwards, so a window that closes at a block picks the error
// closest to it.
FiredChannel find_channel(const Circuit& circuit, const HirModule& hir, size_t begin, size_t end,
                          uint32_t qubit, Pauli1 pauli, bool nearest_end) {
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
HirModule without_readout_noise(HirModule hir) {
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

size_t interior_end(const SandwichBlock& block) {
    return *std::ranges::min_element(block.closing_ops);
}

size_t block_end(const SandwichBlock& block) {
    return block.closing_ops.empty() ? block.check_op + 1
                                     : *std::ranges::max_element(block.closing_ops) + 1;
}

// Samples visible-then-hidden record rows from the generic plan under the
// fixed realization and requires both plans to replay each row with the
// same probability.
void check_sampled_rows_agree(PlanPair& plans, uint64_t num_shots) {
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

}  // namespace

TEST_CASE("The cultivation bench fixtures contain supported sandwich blocks", "[sandwich]") {
    // The d3 workload holds the cat check and the final T MPP T_DAG check.
    // The d5 workload adds the d5 cat check between them.
    const auto [name, expected_blocks] = GENERATE(table<std::string, size_t>({
        {"cultivation_d3_bench.stim", 2},
        {"cultivation_d5_bench.stim", 3},
    }));
    CAPTURE(name);
    const HirModule hir = trace(parse_file(fixture_path(name)));
    const std::vector<SandwichBlock> blocks = find_sandwich_blocks(hir);
    REQUIRE(blocks.size() == expected_blocks);
    size_t block_gates = 0;
    for (const SandwichBlock& block : blocks) {
        CHECK(block.supported);
        block_gates += block.opening_ops.size() + block.closing_ops.size();
    }

    std::vector<SandwichKey> keys;
    for (const SandwichBlock& block : blocks) {
        keys.push_back(uniform_key(block.opening_ops.size(), false, false));
    }
    const HirModule result = specialize(hir, keys);
    CHECK(result.num_t_gates() == hir.num_t_gates() - block_gates);
    CHECK(count_ops(result, OpType::SANDWICH_PROJECTION) == expected_blocks);
    const SamplingPlan plan = sampling::plan_sampling(without_readout_noise(result));
    CHECK_THAT(plan.inspect(), ContainsSubstring("PROJECT_ACTIVE_UNITARY"));
}

TEST_CASE("Specialized d3 cultivation plans replay sampled records", "[sandwich]") {
    const Circuit circuit = parse_file(fixture_path("cultivation_d3_bench.stim"));
    const HirModule hir = without_readout_noise(trace(circuit));
    const std::vector<SandwichBlock> blocks = find_sandwich_blocks(hir);
    REQUIRE(blocks.size() == 2);
    const SandwichBlock& cat = blocks[0];
    const SandwichBlock& final_check = blocks[1];
    REQUIRE(cat.echo_ops.size() == 1);

    // Qubit 0 carries a T axis, and qubit 11 is a cat ancilla. The final
    // check has no noise between its layers.
    const auto before_cat = [&](Pauli1 pauli) {
        return find_channel(circuit, hir, 0, cat.opening_ops.front(), 0, pauli, true);
    };
    const FiredChannel inside_before_check =
        find_channel(circuit, hir, cat.opening_ops.back(), cat.check_op, 0, Pauli1::X, false);
    const FiredChannel inside_after_reset =
        find_channel(circuit, hir, cat.echo_ops.back(), interior_end(cat), 0, Pauli1::X, false);
    const FiredChannel cat_ancilla =
        find_channel(circuit, hir, cat.opening_ops.back(), cat.check_op, 11, Pauli1::X, false);
    const FiredChannel between = find_channel(circuit, hir, block_end(cat),
                                              final_check.opening_ops.front(), 0, Pauli1::X, true);

    // An error that reverses an axis before a check needs one promoted
    // coordinate; the others leave the plan at the noiseless width.
    const auto [name, fired, expected_peak] =
        GENERATE_COPY(table<std::string, std::vector<FiredChannel>, uint32_t>({
            {"no error", {}, 1},
            {"X before the cat check", {before_cat(Pauli1::X)}, 2},
            {"Y before the cat check", {before_cat(Pauli1::Y)}, 2},
            {"Z before the cat check", {before_cat(Pauli1::Z)}, 1},
            {"X inside before the check", {inside_before_check}, 1},
            {"X inside after the reset", {inside_after_reset}, 1},
            {"X on a cat ancilla inside", {cat_ancilla}, 1},
            {"X before and X inside", {before_cat(Pauli1::X), inside_after_reset}, 2},
            {"X between the blocks", {between}, 2},
        }));
    CAPTURE(name);
    PlanPair plans(hir, fired);
    CHECK(plans.specialized.peak_active_width() == expected_peak);
    CHECK(plans.specialized.peak_active_width() < plans.generic.peak_active_width());
    check_sampled_rows_agree(plans, 50);
}

TEST_CASE("Specialized d5 cultivation plans replay sampled records", "[sandwich]") {
    const Circuit circuit = parse_file(fixture_path("cultivation_d5_bench.stim"));
    const HirModule hir = without_readout_noise(trace(circuit));
    const std::vector<SandwichBlock> blocks = find_sandwich_blocks(hir);
    REQUIRE(blocks.size() == 3);
    const SandwichBlock& d3_check = blocks[0];
    const SandwichBlock& d5_check = blocks[1];
    const SandwichBlock& final_check = blocks[2];

    // The d5 check opens with T_DAG on qubit 3 and T on qubit 0. Earlier
    // stabilizer outcomes reverse some of its axes in the frame.
    const FiredChannel before_d3 =
        find_channel(circuit, hir, 0, d3_check.opening_ops.front(), 0, Pauli1::X, true);
    const FiredChannel before_d5 = find_channel(circuit, hir, block_end(d3_check),
                                                d5_check.opening_ops.front(), 3, Pauli1::X, true);
    const FiredChannel inside_d5_positive = find_channel(circuit, hir, d5_check.opening_ops.back(),
                                                         d5_check.check_op, 0, Pauli1::X, false);
    const FiredChannel inside_d5_dagger = find_channel(circuit, hir, d5_check.opening_ops.back(),
                                                       d5_check.check_op, 3, Pauli1::X, false);
    const FiredChannel before_final = find_channel(
        circuit, hir, block_end(d5_check), final_check.opening_ops.front(), 0, Pauli1::X, true);

    const auto [name, fired, expected_peak] =
        GENERATE_COPY(table<std::string, std::vector<FiredChannel>, uint32_t>({
            {"no error", {}, 1},
            {"X before the d3 check", {before_d3}, 2},
            {"X before the d5 check", {before_d5}, 2},
            {"X inside the d5 check on a T axis", {inside_d5_positive}, 1},
            {"X inside the d5 check on a T_DAG axis", {inside_d5_dagger}, 1},
            {"X before the final check", {before_final}, 2},
        }));
    CAPTURE(name);
    PlanPair plans(hir, fired);
    CHECK(plans.specialized.peak_active_width() == expected_peak);
    CHECK(plans.specialized.peak_active_width() < plans.generic.peak_active_width());
    check_sampled_rows_agree(plans, 50);
}
