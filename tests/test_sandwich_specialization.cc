// Checks the sandwich projection and record ops: their HIR representation,
// how other passes treat them, the pass that replaces recognized T-sandwich
// blocks with them, and the sampling plans built from the result.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/drop_non_unitary_pass.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/sandwich_blocks.h"
#include "clifft/optimizer/sandwich_specialization_pass.h"
#include "clifft/util/hir_introspection.h"
#include "clifft/util/symplectic.h"

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace clifft;
using Catch::Matchers::ContainsSubstring;

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

HirModule specialized(const HirModule& hir, std::vector<SandwichKey> keys) {
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
    hir.append_sandwich_projection(
        SandwichSiteIdx{0}, [](MutablePauliMaskView slot) { slot.x().words[0] = test::X(0); });
    hir.append_sandwich_record(MeasRecordIdx{0}, SandwichSiteIdx{0}, false,
                               [](MutablePauliMaskView slot) { slot.x().words[0] = test::X(0); });
    return hir;
}

std::string format_op(const HirModule& hir, size_t index) {
    return format_hir_op(hir.ops[index], hir.mask_view(hir.ops[index]));
}

}  // namespace

TEST_CASE("Sandwich ops carry their site and record payloads", "[sandwich]") {
    HirModule hir = projection_module(1);
    hir.append_sandwich_record(MeasRecordIdx{1}, SandwichSiteIdx{0}, true,
                               [](MutablePauliMaskView slot) { slot.x().words[0] = test::X(0); });

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
    hir.append_sandwich_projection(
        SandwichSiteIdx{0}, [](MutablePauliMaskView slot) { slot.x().words[0] = test::X(0); });
    hir.append_sandwich_record(MeasRecordIdx{0}, SandwichSiteIdx{0}, false,
                               [](MutablePauliMaskView slot) { slot.x().words[0] = test::X(0); });

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

    const HirModule result = specialized(hir, {SandwichKey{{1, 0, 1}, {0, 0, 0}}});
    REQUIRE(result.sandwich_sites.size() == 1);
    CHECK(result.sandwich_sites[0].axis_dagger == std::vector<uint8_t>{1, 1, 1});
    CHECK(result.sandwich_sites[0].axis_flip == std::vector<uint8_t>{1, 0, 1});
}

TEST_CASE("SandwichSpecializationPass folds the internal correction into later ops", "[sandwich]") {
    const HirModule hir = hir_of(kToySandwich);
    const SandwichBlock block = find_sandwich_blocks(hir).at(0);
    const HirModule result = specialized(hir, {SandwichKey{{0, 0, 0}, {1, 0, 0}}});
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
    const HirModule plain = specialized(hir, {uniform_key(3, false, false)});
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
