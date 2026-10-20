#include "clifft/sampling/batch/executor.h"
#include "clifft/sampling/batch/policy.h"
#include "clifft/sampling/cuda/executable_plan.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/hip/executable_plan.h"
#include "clifft/sampling/plan.h"
#include "clifft/sampling/sampler.h"

#include "test_helpers.h"

#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using clifft::sampling::ActivePauli;
using clifft::sampling::AffineBool;
using clifft::sampling::BatchExecutionPolicy;
using clifft::sampling::BatchExecutor;
using clifft::sampling::BatchOutputMode;
using clifft::sampling::BatchSamplingMode;
using clifft::sampling::ExecutablePlan;
using clifft::sampling::Executor;
using clifft::sampling::MeasureActivePauli;
using clifft::sampling::PlannedAction;
using clifft::sampling::PresampledNoiseOutcome;
using clifft::sampling::PresampledNoiseSite;
using clifft::sampling::ProjectActiveUnitary;
using clifft::sampling::PromoteDormantRotation;
using clifft::sampling::RecordSlot;
using clifft::sampling::ReplayResult;
using clifft::sampling::RotateActivePauli;
using clifft::sampling::SamplingPlan;
using clifft::sampling::SymbolId;
using clifft::sampling::SymbolKind;

namespace {

using Complex = std::complex<double>;
// Row-major dense matrix with element (row, column) at row * dimension + column.
using Matrix = std::vector<Complex>;
using Vector = std::vector<Complex>;

constexpr double kInvSqrt2 = std::numbers::sqrt2 / 2.0;
constexpr double kTolerance = 1e-12;

Matrix identity(size_t dimension) {
    Matrix result(dimension * dimension);
    for (size_t i = 0; i < dimension; ++i) {
        result[i * dimension + i] = 1.0;
    }
    return result;
}

Matrix diagonal(const std::vector<double>& entries) {
    Matrix result(entries.size() * entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        result[i * entries.size() + i] = entries[i];
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

Matrix hadamard() {
    return {kInvSqrt2, kInvSqrt2, kInvSqrt2, -kInvSqrt2};
}

Matrix scaled_sum(const Matrix& left, Complex left_scale, const Matrix& right,
                  Complex right_scale) {
    Matrix result(left.size());
    for (size_t i = 0; i < left.size(); ++i) {
        result[i] = left_scale * left[i] + right_scale * right[i];
    }
    return result;
}

Matrix negated(const Matrix& matrix) {
    return scaled_sum(matrix, -1.0, matrix, 0.0);
}

// x X + y Y + z Z, a Hermitian unitary when (x, y, z) is a unit vector.
Matrix bloch_observable(double x, double y, double z) {
    Matrix result(4);
    for (size_t i = 0; i < 4; ++i) {
        result[i] = x * pauli_x()[i] + y * pauli_y()[i] + z * pauli_z()[i];
    }
    return result;
}

// Tensor product with `high` acting on the more significant coordinate bits.
Matrix kron(const Matrix& high, size_t high_dimension, const Matrix& low, size_t low_dimension) {
    const size_t dimension = high_dimension * low_dimension;
    Matrix result(dimension * dimension);
    for (size_t high_row = 0; high_row < high_dimension; ++high_row) {
        for (size_t high_column = 0; high_column < high_dimension; ++high_column) {
            for (size_t low_row = 0; low_row < low_dimension; ++low_row) {
                for (size_t low_column = 0; low_column < low_dimension; ++low_column) {
                    const size_t row = high_row * low_dimension + low_row;
                    const size_t column = high_column * low_dimension + low_column;
                    result[row * dimension + column] =
                        high[high_row * high_dimension + high_column] *
                        low[low_row * low_dimension + low_column];
                }
            }
        }
    }
    return result;
}

// Permutation matrix sending basis state b to image[b].
Matrix permutation(const std::vector<size_t>& image) {
    Matrix result(image.size() * image.size());
    for (size_t basis = 0; basis < image.size(); ++basis) {
        result[image[basis] * image.size() + basis] = 1.0;
    }
    return result;
}

// i^popcount(x & z) X^x Z^z, the operator an ActivePauli denotes.
Matrix active_pauli_matrix(ActivePauli pauli, uint32_t width) {
    const size_t dimension = size_t{1} << width;
    constexpr std::array<Complex, 4> kPhases = {Complex{1.0, 0.0}, Complex{0.0, 1.0},
                                                Complex{-1.0, 0.0}, Complex{0.0, -1.0}};
    const Complex phase = kPhases[std::popcount(pauli.x & pauli.z) % 4];
    Matrix result(dimension * dimension);
    for (uint64_t basis = 0; basis < dimension; ++basis) {
        const double sign = std::popcount(basis & pauli.z) % 2 == 0 ? 1.0 : -1.0;
        result[(basis ^ pauli.x) * dimension + basis] = sign * phase;
    }
    return result;
}

Vector apply(const Matrix& matrix, const Vector& vector) {
    Vector result(vector.size());
    for (size_t row = 0; row < vector.size(); ++row) {
        for (size_t column = 0; column < vector.size(); ++column) {
            result[row] += matrix[row * vector.size() + column] * vector[column];
        }
    }
    return result;
}

Complex inner_product(const Vector& left, const Vector& right) {
    Complex result;
    for (size_t i = 0; i < left.size(); ++i) {
        result += std::conj(left[i]) * right[i];
    }
    return result;
}

// Probability of the +1 eigenspace of a Hermitian unitary: (1 + <C>) / 2.
double plus_probability(const Matrix& matrix, const Vector& state) {
    return 0.5 * (1.0 + inner_product(state, apply(matrix, state)).real());
}

Vector state_of(const Executor& executor) {
    const auto real = executor.state().real();
    const auto imag = executor.state().imag();
    Vector result(real.size());
    for (size_t i = 0; i < real.size(); ++i) {
        result[i] = Complex{real[i], imag[i]};
    }
    return result;
}

// Dormant coordinates promoted one at a time, each with its own rotation angle,
// followed by rotations over the whole active set that entangle them.
struct StatePreparation {
    std::vector<double> promotion_half_turns;
    std::vector<RotateActivePauli> rotations;
};

uint32_t width_of(const StatePreparation& preparation) {
    return static_cast<uint32_t>(preparation.promotion_half_turns.size());
}

std::vector<PlannedAction> preparation_actions(const StatePreparation& preparation) {
    const uint32_t width = width_of(preparation);
    std::vector<PlannedAction> actions;
    for (uint32_t coordinate = 0; coordinate < width; ++coordinate) {
        actions.push_back(PlannedAction{
            coordinate, coordinate + 1,
            PromoteDormantRotation{preparation.promotion_half_turns[coordinate], {}}});
    }
    for (const RotateActivePauli& rotation : preparation.rotations) {
        actions.push_back(PlannedAction{width, width, rotation});
    }
    return actions;
}

SamplingPlan base_plan(const StatePreparation& preparation, uint32_t visible_records) {
    SamplingPlan plan;
    plan.num_qubits = width_of(preparation);
    plan.peak_active_width = width_of(preparation);
    plan.num_visible_records = visible_records;
    plan.actions = preparation_actions(preparation);
    return plan;
}

// The state a projection acts on, with no measurement afterwards.
SamplingPlan preparation_plan(const StatePreparation& preparation) {
    return base_plan(preparation, 0);
}

// Appends one projection over every prepared coordinate. Its record is the
// branch XOR `correction`, written to visible record 0.
SamplingPlan projection_plan(const StatePreparation& preparation, Matrix matrix,
                             AffineBool correction = {}) {
    const uint32_t width = width_of(preparation);
    SamplingPlan plan = base_plan(preparation, 1);
    const SymbolId branch{0};
    plan.symbols = {SymbolKind::Branch};
    plan.actions.push_back(PlannedAction{
        width, width,
        ProjectActiveUnitary{width, std::move(matrix), branch,
                             std::move(correction) ^ AffineBool::symbol(branch), RecordSlot{0}}});
    return plan;
}

SamplingPlan projection_plan(const std::vector<double>& half_turns, Matrix matrix) {
    return projection_plan(StatePreparation{half_turns, {}}, std::move(matrix));
}

// Measures a Pauli of the same state with the existing Pauli measurement, which
// retires one coordinate. Used as an independent source of branch probabilities.
SamplingPlan pauli_measurement_plan(const StatePreparation& preparation, ActivePauli pauli) {
    const uint32_t width = width_of(preparation);
    SamplingPlan plan = base_plan(preparation, 1);
    const SymbolId branch{0};
    plan.symbols = {SymbolKind::Branch};
    const uint32_t pivot =
        static_cast<uint32_t>(std::countr_zero(pauli.x != 0 ? pauli.x : pauli.z));
    plan.actions.push_back(PlannedAction{
        width, width - 1,
        MeasureActivePauli{pauli, pivot, branch, AffineBool::symbol(branch), RecordSlot{0}}});
    return plan;
}

ProjectActiveUnitary& projection_of(SamplingPlan& plan) {
    return std::get<ProjectActiveUnitary>(plan.actions.back().action);
}

void require_rejected(const SamplingPlan& plan, std::string_view fragment) {
    REQUIRE_THROWS_AS(plan.validate(), std::invalid_argument);
    REQUIRE_THROWS_WITH(plan.validate(), ContainsSubstring(std::string(fragment)));
}

// Probability the plan assigns to a single visible record value, found by
// replaying that value so it does not depend on random sampling.
double replayed_probability(const SamplingPlan& plan, uint8_t record) {
    const ExecutablePlan executable(plan);
    Executor executor(executable);
    const std::array<uint8_t, 1> forced = {record};
    const ReplayResult result = executor.replay_shot(forced);
    return result.reachable ? std::exp(result.log_probability) : 0.0;
}

// The state before any projection, as the executor reaches it.
Vector prepared_state(const StatePreparation& preparation) {
    const ExecutablePlan executable(preparation_plan(preparation));
    Executor executor(executable);
    executor.run_shot();
    return state_of(executor);
}

// Fidelity with the reference state, which ignores the global phase.
double fidelity(const Vector& reference, const Vector& state) {
    return std::norm(inner_product(reference, state));
}

// Requires the state to be a unit vector with C v = eigenvalue v, which is what
// projecting onto an eigenspace of C must leave behind.
void require_eigenvector(const Matrix& matrix, const Vector& state, double eigenvalue) {
    REQUIRE_THAT(inner_product(state, state).real(), WithinAbs(1.0, kTolerance));
    const Vector image = apply(matrix, state);
    for (size_t i = 0; i < state.size(); ++i) {
        REQUIRE(std::abs(image[i] - eigenvalue * state[i]) < kTolerance);
    }
}

// Runs a projection plan and requires every shot to record `expected` and leave
// the prepared state unchanged up to a global phase.
void require_deterministic_projection(const StatePreparation& preparation, const Matrix& matrix,
                                      uint8_t expected) {
    const Vector reference = prepared_state(preparation);
    const SamplingPlan plan = projection_plan(preparation, matrix);
    const ExecutablePlan executable(plan);
    Executor executor(executable, 11);
    for (uint32_t shot = 0; shot < 200; ++shot) {
        executor.run_shot();
        REQUIRE(executor.visible_records()[0] == expected);
        REQUIRE(executor.symbols()[0] == expected);
        REQUIRE_THAT(fidelity(reference, state_of(executor)), WithinAbs(1.0, kTolerance));
    }

    const std::array<uint8_t, 1> matching = {expected};
    const std::array<uint8_t, 1> opposite = {static_cast<uint8_t>(expected ^ 1)};
    const ReplayResult reachable = executor.replay_shot(matching);
    REQUIRE(reachable.reachable);
    REQUIRE(reachable.log_probability == 0.0);
    REQUIRE_FALSE(executor.replay_shot(opposite).reachable);
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
        matrix[0] = clifft::test::opaque_nan();
        require_rejected(projection_plan({0.25}, matrix), "finite");
        matrix[0] = Complex{0.0, clifft::test::opaque_infinity()};
        require_rejected(projection_plan({0.25}, matrix), "finite");
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

TEST_CASE("Projection onto an eigenspace containing the state is deterministic", "[projection]") {
    // Promoting by +1/4 half turns gives exp(-i pi/8 X)|0>, whose Bloch vector
    // is (0, -1, 1) / sqrt(2); a negative angle flips the Y component.
    const StatePreparation positive{{0.25}, {}};
    const StatePreparation negative{{-0.25}, {}};
    const Matrix z_minus_y = bloch_observable(0.0, -kInvSqrt2, kInvSqrt2);
    const Matrix z_plus_y = bloch_observable(0.0, kInvSqrt2, kInvSqrt2);

    SECTION("positive angle in the plus eigenspace") {
        require_deterministic_projection(positive, z_minus_y, 0);
    }

    SECTION("positive angle in the minus eigenspace") {
        require_deterministic_projection(positive, negated(z_minus_y), 1);
    }

    SECTION("negative angle swaps the roles of the Y sign") {
        require_deterministic_projection(negative, z_plus_y, 0);
        require_deterministic_projection(negative, negated(z_plus_y), 1);
    }

    SECTION("a scalar state projects onto the sign of a one by one matrix") {
        const StatePreparation scalar{{}, {}};
        require_deterministic_projection(scalar, {1.0}, 0);
        require_deterministic_projection(scalar, {-1.0}, 1);
    }

    SECTION("an observable with zero expectation splits evenly") {
        const SamplingPlan plan = projection_plan(positive, z_plus_y);
        REQUIRE_THAT(replayed_probability(plan, 0), WithinAbs(0.5, kTolerance));
        REQUIRE_THAT(replayed_probability(plan, 1), WithinAbs(0.5, kTolerance));

        const ExecutablePlan executable(plan);
        Executor executor(executable, 5);
        uint32_t ones = 0;
        constexpr uint32_t kShots = 400;
        for (uint32_t shot = 0; shot < kShots; ++shot) {
            executor.run_shot();
            ones += executor.visible_records()[0];
        }
        REQUIRE(ones > kShots / 4);
        REQUIRE(ones < 3 * kShots / 4);
    }
}

TEST_CASE("Projection by a Pauli matches a Pauli measurement", "[projection]") {
    struct Case {
        StatePreparation preparation;
        std::vector<ActivePauli> paulis;
    };
    // Rotations about the active Paulis below make the states generic enough
    // that every tested Pauli has a nontrivial outcome distribution.
    const std::vector<Case> cases = {
        {StatePreparation{{0.3}, {RotateActivePauli{ActivePauli{0, 1}, 0.37, {}}}},
         {ActivePauli{1, 0}, ActivePauli{0, 1}, ActivePauli{1, 1}}},
        {StatePreparation{{0.3, 0.17},
                          {RotateActivePauli{ActivePauli{1, 2}, 0.41, {}},
                           RotateActivePauli{ActivePauli{0, 1}, 0.23, {}},
                           RotateActivePauli{ActivePauli{2, 0}, 0.31, {}}}},
         {ActivePauli{1, 2}, ActivePauli{3, 0}, ActivePauli{1, 3}, ActivePauli{2, 2},
          ActivePauli{3, 3}, ActivePauli{0, 3}, ActivePauli{2, 1}}},
    };

    for (const Case& test_case : cases) {
        const uint32_t width = width_of(test_case.preparation);
        const Vector state = prepared_state(test_case.preparation);
        for (const ActivePauli pauli : test_case.paulis) {
            CAPTURE(width, pauli.x, pauli.z);
            const Matrix matrix = active_pauli_matrix(pauli, width);
            const double expected = plus_probability(matrix, state);
            REQUIRE(expected > 0.01);
            REQUIRE(expected < 0.99);

            const SamplingPlan projection = projection_plan(test_case.preparation, matrix);
            const SamplingPlan measurement = pauli_measurement_plan(test_case.preparation, pauli);
            const double measured_zero = replayed_probability(measurement, 0);
            const double measured_one = replayed_probability(measurement, 1);
            const double projected_zero = replayed_probability(projection, 0);
            const double projected_one = replayed_probability(projection, 1);
            REQUIRE_THAT(measured_zero, WithinAbs(expected, kTolerance));
            REQUIRE_THAT(projected_zero, WithinAbs(expected, kTolerance));
            REQUIRE_THAT(projected_one, WithinAbs(1.0 - expected, kTolerance));
            REQUIRE_THAT(projected_zero, WithinAbs(measured_zero, kTolerance));
            REQUIRE_THAT(projected_one, WithinAbs(measured_one, kTolerance));
        }
    }
}

TEST_CASE("Projection by a non-Pauli unitary leaves an eigenvector of the matrix", "[projection]") {
    struct Case {
        std::string name;
        StatePreparation preparation;
        Matrix matrix;
    };
    const Matrix x_plus_y = bloch_observable(kInvSqrt2, kInvSqrt2, 0.0);
    const StatePreparation two_coordinates{{0.3, 0.17},
                                           {RotateActivePauli{ActivePauli{1, 2}, 0.41, {}},
                                            RotateActivePauli{ActivePauli{0, 1}, 0.23, {}},
                                            RotateActivePauli{ActivePauli{2, 0}, 0.31, {}}}};
    const StatePreparation three_coordinates{{0.8, 0.7, 0.45},
                                             {RotateActivePauli{ActivePauli{1, 6}, 0.41, {}},
                                              RotateActivePauli{ActivePauli{2, 5}, 0.23, {}},
                                              RotateActivePauli{ActivePauli{4, 3}, 0.31, {}}}};
    const std::vector<Case> cases = {
        {"rotated X plus Y on the high coordinate with Z on the low one", two_coordinates,
         kron(x_plus_y, 2, pauli_z(), 2)},
        {"two Hadamards", two_coordinates, kron(hadamard(), 2, hadamard(), 2)},
        {"swap", two_coordinates, permutation({0, 2, 1, 3})},
        {"controlled Z", two_coordinates, diagonal({1.0, 1.0, 1.0, -1.0})},
        {"Toffoli", three_coordinates, permutation({0, 1, 2, 3, 4, 5, 7, 6})},
    };

    for (const Case& test_case : cases) {
        SECTION(test_case.name) {
            const uint32_t width = width_of(test_case.preparation);
            const Vector state = prepared_state(test_case.preparation);
            const double plus = plus_probability(test_case.matrix, state);
            REQUIRE(plus > 0.05);
            REQUIRE(plus < 0.95);

            const SamplingPlan plan = projection_plan(test_case.preparation, test_case.matrix);
            REQUIRE_THAT(replayed_probability(plan, 0), WithinAbs(plus, kTolerance));
            REQUIRE_THAT(replayed_probability(plan, 1), WithinAbs(1.0 - plus, kTolerance));

            const ExecutablePlan executable(plan);
            Executor executor(executable, 29);
            constexpr uint32_t kShots = 4000;
            uint32_t ones = 0;
            for (uint32_t shot = 0; shot < kShots; ++shot) {
                executor.run_shot();
                const uint8_t record = executor.visible_records()[0];
                ones += record;
                if (shot < 200) {
                    // The state is projected, not merely measured: it must be a
                    // normalized eigenvector whose eigenvalue matches the record.
                    REQUIRE(executor.state().active_width() == width);
                    require_eigenvector(test_case.matrix, state_of(executor),
                                        record == 0 ? 1.0 : -1.0);
                }
            }
            const double minus = 1.0 - plus;
            const double sigma = std::sqrt(plus * minus / kShots);
            REQUIRE_THAT(static_cast<double>(ones) / kShots, WithinAbs(minus, 5.0 * sigma));
        }
    }
}

TEST_CASE("Projection evaluates its correction expression", "[projection]") {
    SECTION("constant correction inverts the record") {
        const StatePreparation state{{0.25}, {}};
        const SamplingPlan plan =
            projection_plan(state, bloch_observable(0.0, -kInvSqrt2, kInvSqrt2), AffineBool{true});
        const ExecutablePlan executable(plan);
        Executor executor(executable, 3);
        for (uint32_t shot = 0; shot < 50; ++shot) {
            executor.run_shot();
            REQUIRE(executor.symbols()[0] == 0);
            REQUIRE(executor.visible_records()[0] == 1);
        }
        REQUIRE(replayed_probability(plan, 1) == 1.0);
        REQUIRE(replayed_probability(plan, 0) == 0.0);
    }

    SECTION("correction on an earlier branch") {
        // A second Z projection repeats the first one, so its branch equals the
        // earlier branch and the corrected record is always zero.
        const SymbolId first{0};
        const SymbolId second{1};
        SamplingPlan plan;
        plan.num_qubits = 1;
        plan.peak_active_width = 1;
        plan.num_visible_records = 2;
        plan.symbols = {SymbolKind::Branch, SymbolKind::Branch};
        plan.actions = {
            PlannedAction{0, 1, PromoteDormantRotation{0.3, {}}},
            PlannedAction{1, 1,
                          ProjectActiveUnitary{1, pauli_z(), first, AffineBool::symbol(first),
                                               RecordSlot{0}}},
            PlannedAction{1, 1,
                          ProjectActiveUnitary{1, pauli_z(), second,
                                               AffineBool(false, {first, second}), RecordSlot{1}}},
        };
        const double sin_squared = std::pow(std::sin(std::numbers::pi * 0.3 / 2.0), 2);

        const ExecutablePlan executable(plan);
        Executor executor(executable, 8);
        constexpr uint32_t kShots = 2000;
        uint32_t ones = 0;
        for (uint32_t shot = 0; shot < kShots; ++shot) {
            executor.run_shot();
            ones += executor.visible_records()[0];
            REQUIRE(executor.visible_records()[1] == 0);
        }
        const double sigma = std::sqrt(sin_squared * (1.0 - sin_squared) / kShots);
        REQUIRE_THAT(static_cast<double>(ones) / kShots, WithinAbs(sin_squared, 5.0 * sigma));

        const ReplayResult zero = executor.replay_shot(std::array<uint8_t, 2>{0, 0});
        REQUIRE(zero.reachable);
        REQUIRE_THAT(std::exp(zero.log_probability), WithinAbs(1.0 - sin_squared, kTolerance));
        const ReplayResult one = executor.replay_shot(std::array<uint8_t, 2>{1, 0});
        REQUIRE(one.reachable);
        REQUIRE_THAT(std::exp(one.log_probability), WithinAbs(sin_squared, kTolerance));
        REQUIRE_FALSE(executor.replay_shot(std::array<uint8_t, 2>{0, 1}).reachable);
        REQUIRE_FALSE(executor.replay_shot(std::array<uint8_t, 2>{1, 1}).reachable);
    }
}

TEST_CASE("Projection executes at the maximum width", "[projection]") {
    // The matrix image needs 2^8 entries when the projection spans the peak
    // active width, which is more than the state's own scratch can hold.
    StatePreparation preparation;
    for (uint32_t coordinate = 0; coordinate < 8; ++coordinate) {
        preparation.promotion_half_turns.push_back(0.1 + 0.05 * coordinate);
    }
    std::vector<size_t> flip_all(256);
    for (size_t basis = 0; basis < flip_all.size(); ++basis) {
        flip_all[basis] = basis ^ 255;
    }
    const Matrix matrix = permutation(flip_all);
    const SamplingPlan plan = projection_plan(preparation, matrix);
    const ExecutablePlan executable(plan);
    Executor executor(executable, 17);
    for (uint32_t shot = 0; shot < 20; ++shot) {
        executor.run_shot();
        const Vector projected = state_of(executor);
        REQUIRE(projected.size() == 256);
        require_eigenvector(matrix, projected, executor.visible_records()[0] == 0 ? 1.0 : -1.0);
    }
    // Every promoted coordinate lies in the Y-Z plane, so the product of their
    // X expectations vanishes and both outcomes are equally likely.
    const std::array<uint8_t, 1> zero = {0};
    const ReplayResult replay = executor.replay_shot(zero);
    REQUIRE(replay.reachable);
    REQUIRE_THAT(std::exp(replay.log_probability), WithinAbs(0.5, kTolerance));
}

TEST_CASE("Executable plans report projection actions", "[projection]") {
    const StatePreparation state{{0.25}, {}};
    const ExecutablePlan without(preparation_plan(state));
    const ExecutablePlan with(projection_plan(state, pauli_z()));

    REQUIRE_FALSE(without.requires_scalar_execution());
    REQUIRE(with.requires_scalar_execution());
    REQUIRE_THAT(with.inspect(),
                 ContainsSubstring("PROJECT_ACTIVE_UNITARY width=1 branch=s0 record=r0"));
}

TEST_CASE("Packed execution rejects projection plans", "[projection]") {
    const ExecutablePlan executable(projection_plan({0.25}, pauli_z()));

    REQUIRE_THROWS_AS(BatchExecutor(executable, 64), std::invalid_argument);
    REQUIRE_THROWS_WITH(BatchExecutor(executable, 64), ContainsSubstring("projection"));
}

TEST_CASE("Batch policy selects scalar execution for projection plans", "[projection]") {
    const StatePreparation state{{0.25}, {}};
    const ExecutablePlan executable(projection_plan(state, pauli_z()));
    // The same state without a projection batches, so the scalar choice below
    // comes from the projection and not from the plan's size.
    const ExecutablePlan control(preparation_plan(state));

    for (const BatchOutputMode output_mode :
         {BatchOutputMode::Rows, BatchOutputMode::AggregateSurvivors}) {
        for (const BatchSamplingMode sampling_mode :
             {BatchSamplingMode::Ordinary, BatchSamplingMode::FixedFaults}) {
            REQUIRE(clifft::sampling::resolve_batch_execution_policy(
                        control, 8192, 4, 1, output_mode, std::nullopt, sampling_mode)
                        .lane_capacity > 1);
            REQUIRE(clifft::sampling::resolve_batch_execution_policy(
                        executable, 8192, 4, 1, output_mode, std::nullopt, sampling_mode)
                        .lane_capacity == 1);
            REQUIRE(clifft::sampling::resolve_batch_execution_policy(
                        executable, 8192, 4, 1, output_mode, 256, sampling_mode)
                        .lane_capacity == 1);
        }
    }
}

TEST_CASE("Sampling a projection plan falls back to scalar execution", "[projection]") {
    const StatePreparation state{{0.25}, {}};
    const SamplingPlan plan = projection_plan(state, bloch_observable(0.0, -kInvSqrt2, kInvSqrt2));
    const ExecutablePlan executable(plan);
    constexpr uint32_t kShots = 5000;

    const auto require_records = [](const std::vector<uint8_t>& records, uint32_t shots,
                                    uint8_t expected) {
        REQUIRE(records.size() == shots);
        REQUIRE(std::ranges::all_of(records, [&](uint8_t value) { return value == expected; }));
    };

    SECTION("automatic batch size") {
        require_records(clifft::sampling::sample(executable, kShots, 41).measurements, kShots, 0);
        require_records(clifft::sampling::sample_records(executable, kShots, 41), kShots, 0);
    }

    SECTION("explicit batch size and several workers") {
        require_records(
            clifft::sampling::sample(executable, kShots, 41, 2, std::nullopt, 128).measurements,
            kShots, 0);
    }

    SECTION("survivor sampling") {
        const auto result =
            clifft::sampling::sample_survivors(executable, kShots, 41, true, 1, std::nullopt, 128);
        REQUIRE(result.passed_shots == kShots);
        require_records(result.measurements, kShots, 0);
    }

    SECTION("batch tuning keeps the scalar candidate") {
        const auto result =
            clifft::sampling::sample(executable, 2000, 41, 1, std::nullopt, std::nullopt, 0.05);
        require_records(result.measurements, 2000, 0);
        REQUIRE(result.batch_tuning.has_value());
        REQUIRE(result.batch_tuning->batch_size == 1);
    }

    SECTION("forced faults") {
        // One presampled flip on the record: zero faults leave the projection
        // outcome alone and one fault always inverts it.
        SamplingPlan noisy = plan;
        noisy.symbols.push_back(SymbolKind::Presampled);
        noisy.presampled_noise_sites = {
            PresampledNoiseSite{0.5, {PresampledNoiseOutcome{SymbolId{1}, 0.5}}}};
        projection_of(noisy).outcome = AffineBool(false, {SymbolId{0}, SymbolId{1}});
        const ExecutablePlan noisy_executable(noisy);

        require_records(
            clifft::sampling::sample_k(noisy_executable, 1000, 0, 41, 1, std::nullopt, 128)
                .measurements,
            1000, 0);
        require_records(
            clifft::sampling::sample_k(noisy_executable, 1000, 1, 41, 1, std::nullopt, 128)
                .measurements,
            1000, 1);
        const auto survivors = clifft::sampling::sample_k_survivors(noisy_executable, 1000, 1, 41,
                                                                    true, 1, std::nullopt, 128);
        REQUIRE(survivors.passed_shots == 1000);
        require_records(survivors.measurements, 1000, 1);
    }
}
