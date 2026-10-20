#include "clifft/sampling/cuda/executable_plan.h"
#include "clifft/sampling/hip/executable_plan.h"
#include "clifft/sampling/plan.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using clifft::sampling::AffineBool;
using clifft::sampling::PlannedAction;
using clifft::sampling::ProjectActiveUnitary;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordSlot;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolKind;

namespace {

using Complex = std::complex<double>;
// Row-major dense matrix with element (row, column) at row * dimension + column.
using Matrix = std::vector<Complex>;

constexpr double kInvSqrt2 = std::numbers::sqrt2 / 2.0;

Matrix identity(size_t dimension) {
    Matrix result(dimension * dimension);
    for (size_t i = 0; i < dimension; ++i) {
        result[i * dimension + i] = 1.0;
    }
    return result;
}

Matrix pauli_x() {
    return {0.0, 1.0, 1.0, 0.0};
}

Matrix pauli_y() {
    return {0.0, Complex{0.0, -1.0}, Complex{0.0, 1.0}, 0.0};
}

Matrix pauli_z() {
    return {1.0, 0.0, 0.0, -1.0};
}

Matrix scaled_sum(const Matrix& left, Complex left_scale, const Matrix& right,
                  Complex right_scale) {
    Matrix result(left.size());
    for (size_t i = 0; i < left.size(); ++i) {
        result[i] = left_scale * left[i] + right_scale * right[i];
    }
    return result;
}

// Builds a plan that promotes `half_turns.size()` coordinates and then applies
// one projection over all of them, recording its outcome in visible record 0.
SamplingPlan projection_plan(const std::vector<double>& half_turns, Matrix matrix) {
    const uint32_t width = static_cast<uint32_t>(half_turns.size());
    SamplingPlan plan;
    plan.num_qubits = width;
    plan.peak_active_width = width;
    plan.num_visible_records = 1;
    plan.symbols = {SymbolKind::Branch};
    for (uint32_t coordinate = 0; coordinate < width; ++coordinate) {
        plan.actions.push_back(PlannedAction{coordinate, coordinate + 1,
                                             PromoteDormantRotation{half_turns[coordinate], {}}});
    }
    const SymbolId branch{0};
    plan.actions.push_back(
        PlannedAction{width, width,
                      ProjectActiveUnitary{width, std::move(matrix), branch,
                                           AffineBool::symbol(branch), RecordSlot{0}}});
    return plan;
}

ProjectActiveUnitary& projection_of(SamplingPlan& plan) {
    return std::get<ProjectActiveUnitary>(plan.actions.back().action);
}

void require_rejected(const SamplingPlan& plan, std::string_view fragment) {
    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    REQUIRE_THROWS_WITH(plan.validate(), ContainsSubstring(std::string(fragment)));
}

}  // namespace

TEST_CASE("Projection action validation accepts Hermitian unitary matrices", "[projection]") {
    SECTION("one coordinate") {
        REQUIRE_NOTHROW(projection_plan({0.25}, pauli_y()).validate());
        REQUIRE_NOTHROW(
            projection_plan({0.25}, scaled_sum(pauli_x(), kInvSqrt2, pauli_y(), kInvSqrt2))
                .validate());
    }

    SECTION("the largest supported width") {
        const SamplingPlan plan = projection_plan(std::vector<double>(8, 0.1), identity(256));
        REQUIRE_NOTHROW(plan.validate());
    }

    SECTION("small deviations within the tolerance") {
        Matrix matrix = pauli_z();
        matrix[0] += 1e-11;
        REQUIRE_NOTHROW(projection_plan({0.25}, matrix).validate());
    }
}

TEST_CASE("Projection action validation rejects malformed matrices", "[projection]") {
    SECTION("non-Hermitian unitary matrix") {
        // i X is unitary but its adjoint is -i X.
        require_rejected(projection_plan({0.25}, {0.0, Complex{0.0, 1.0}, Complex{0.0, 1.0}, 0.0}),
                         "Hermitian");
    }

    SECTION("Hermitian matrix that is not unitary") {
        require_rejected(projection_plan({0.25}, {1.0, 0.0, 0.0, 0.0}), "unitary");
        require_rejected(projection_plan({0.25}, scaled_sum(pauli_z(), 2.0, pauli_z(), 0.0)),
                         "unitary");
    }

    SECTION("entries that are not finite") {
        Matrix matrix = pauli_z();
        matrix[0] = std::nan("");
        require_rejected(projection_plan({0.25}, matrix), "Hermitian");
    }

    SECTION("matrix of the wrong size") {
        require_rejected(projection_plan({0.25}, identity(1)), "entries");
        require_rejected(projection_plan({0.25, 0.25}, pauli_z()), "entries");
        require_rejected(projection_plan({0.25}, {}), "entries");
    }

    SECTION("width that disagrees with the active width") {
        SamplingPlan plan = projection_plan({0.25, 0.25}, identity(4));
        projection_of(plan).width = 1;
        projection_of(plan).matrix = pauli_z();
        require_rejected(plan, "active width");
    }

    SECTION("width above the supported maximum") {
        require_rejected(projection_plan(std::vector<double>(9, 0.1), identity(512)), "maximum");
    }
}

TEST_CASE("Projection action validation checks its branch symbol and record", "[projection]") {
    SECTION("branch symbol of another kind") {
        SamplingPlan plan = projection_plan({0.25}, pauli_z());
        plan.symbols[0] = SymbolKind::Derived;
        REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    }

    SECTION("outcome that omits the branch") {
        SamplingPlan plan = projection_plan({0.25}, pauli_z());
        projection_of(plan).outcome = AffineBool{};
        require_rejected(plan, "omits branch");
    }

    SECTION("record slot written more than once") {
        SamplingPlan plan = projection_plan({0.25}, pauli_z());
        plan.symbols.push_back(SymbolKind::Branch);
        plan.actions.push_back(
            PlannedAction{1, 1,
                          ProjectActiveUnitary{1, pauli_z(), SymbolId{1},
                                               AffineBool::symbol(SymbolId{1}), RecordSlot{0}}});
        require_rejected(plan, "more than once");
    }

    SECTION("branch symbol defined twice") {
        SamplingPlan plan = projection_plan({0.25}, pauli_z());
        plan.num_visible_records = 2;
        plan.actions.push_back(
            PlannedAction{1, 1,
                          ProjectActiveUnitary{1, pauli_z(), SymbolId{0},
                                               AffineBool::symbol(SymbolId{0}), RecordSlot{1}}});
        require_rejected(plan, "defined more than once");
    }
}

TEST_CASE("Projection action inspection omits matrix entries", "[projection]") {
    const SamplingPlan plan =
        projection_plan({0.25}, scaled_sum(pauli_x(), kInvSqrt2, pauli_y(), kInvSqrt2));
    const std::string text = plan.inspect();

    REQUIRE_THAT(
        text, ContainsSubstring("PROJECT_ACTIVE_UNITARY width=1 branch=s0 outcome=s0 record=r0"));
    REQUIRE_THAT(plan.inspect_action(1), ContainsSubstring("w1 dense_passes=2 "));
    REQUIRE_THAT(plan.inspect_action_compact(1),
                 ContainsSubstring("w1 PROJECT_ACTIVE_UNITARY width=1 branch=s0"));
    REQUIRE_THAT(plan.inspect_action_compact(1), ContainsSubstring("passes=2"));
    REQUIRE(text.find("0.707") == std::string::npos);
    REQUIRE(clifft::sampling::predicted_dense_passes(plan.actions.back().action) == 2);
}

TEST_CASE("GPU executable plans reject projection actions", "[projection]") {
    const SamplingPlan plan = projection_plan({0.25}, pauli_y());

    REQUIRE_THROWS_WITH(clifft::sampling::cuda::ExecutablePlan(plan),
                        ContainsSubstring("projection"));
    REQUIRE_THROWS_WITH(clifft::sampling::hip::ExecutablePlan(plan),
                        ContainsSubstring("projection"));
    REQUIRE_THROWS_AS(clifft::sampling::cuda::ExecutablePlan(plan), std::invalid_argument);
    REQUIRE_THROWS_AS(clifft::sampling::hip::ExecutablePlan(plan), std::invalid_argument);
}
