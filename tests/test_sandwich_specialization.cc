// Checks the sandwich projection and record ops: their HIR representation,
// how other passes treat them, the pass that replaces recognized T-sandwich
// blocks with them, and the sampling plans built from the result.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/commutation.h"
#include "clifft/optimizer/drop_non_unitary_pass.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/util/hir_introspection.h"

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace clifft;

namespace {

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
