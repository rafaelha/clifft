#include "clifft/sampling/planner.h"

#include "clifft/sampling/planner_frame.h"
#include "clifft/util/hir_introspection.h"
#include "clifft/util/numeric.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace clifft::sampling {

namespace {

using Pauli = internal::PlannerPauli;
using Tableau = internal::PlannerTableau;
using internal::CoordinateFrame;
using internal::SymbolicPauliFrame;

Pauli pauli_from_hir(const HirModule& hir, const HeisenbergOp& op) {
    Pauli result(hir.num_qubits);
    result.mut_x().xor_with(hir.destab_mask(op));
    result.mut_z().xor_with(hir.stab_mask(op));
    result.set_sign(false);
    return result;
}

Pauli noise_pauli_from_hir(const HirModule& hir, PauliMaskHandle handle) {
    Pauli result(hir.num_qubits);
    const PauliMaskView mask = hir.noise_channel_masks.at(handle);
    result.mut_x().xor_with(mask.x());
    result.mut_z().xor_with(mask.z());
    result.set_sign(false);
    return result;
}

Pauli pauli_from_mask(const HirModule& hir, PauliMaskHandle handle) {
    Pauli result(hir.num_qubits);
    const PauliMaskView mask = hir.pauli_masks.at(handle);
    result.mut_x().xor_with(mask.x());
    result.mut_z().xor_with(mask.z());
    result.set_sign(false);
    return result;
}

struct ResolvedPauli {
    Pauli body;
    AffineBool sign;
};

ResolvedPauli resolve_pauli(const Pauli& initial_body, const AffineBool& initial_sign,
                            CoordinateFrame& coordinates, SymbolicPauliFrame& symbolic_frame) {
    AffineBool sign = initial_sign;
    sign ^= symbolic_frame.sign_for(initial_body);
    Pauli body = coordinates.to_current(initial_body);
    sign ^= body.sign();
    body.set_sign(false);
    return ResolvedPauli{std::move(body), std::move(sign)};
}

std::optional<uint32_t> first_x_at_or_above(const Pauli& pauli, uint32_t begin) {
    for (uint32_t q = begin; q < pauli.num_qubits(); ++q) {
        if (pauli.x().bit_get(q)) {
            return q;
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> last_x_below(const Pauli& pauli, uint32_t end) {
    for (uint32_t q = end; q > 0; --q) {
        if (pauli.x().bit_get(q - 1)) {
            return q - 1;
        }
    }
    return std::nullopt;
}

std::optional<uint32_t> first_z_below(const Pauli& pauli, uint32_t end) {
    for (uint32_t q = 0; q < end; ++q) {
        if (pauli.z().bit_get(q)) {
            return q;
        }
    }
    return std::nullopt;
}

ActivePauli active_projection(const Pauli& pauli, uint32_t active_width) {
    ActivePauli result;
    for (uint32_t q = 0; q < active_width; ++q) {
        result.x |= static_cast<uint64_t>(pauli.x().bit_get(q)) << q;
        result.z |= static_cast<uint64_t>(pauli.z().bit_get(q)) << q;
    }
    return result;
}

SymbolId reserve_symbol(SamplingPlan& plan) {
    const SymbolId symbol{static_cast<uint32_t>(plan.symbols.size())};
    plan.symbols.emplace_back();
    return symbol;
}

std::span<const uint32_t> source_lines_for(const HirModule& hir, size_t operation_index) {
    assert(hir.source_map.size() == hir.ops.size() &&
           "retained HIR provenance must remain parallel to operations");
    return hir.source_map[operation_index];
}

void append_action(SamplingPlan& plan, PlannedAction action,
                   std::span<const uint32_t> source_lines) {
    plan.actions.push_back(std::move(action));
    if (plan.source_map.has_value()) {
        plan.source_map->append(source_lines);
    }
}

void define_symbol(SamplingPlan& plan, SymbolId symbol, SymbolKind kind) {
    SymbolKind& stored_kind = plan.symbols.at(index(symbol));
    if (stored_kind != SymbolKind::Unused) {
        throw std::logic_error("sampling planner attempted to redefine a reserved symbol");
    }
    stored_kind = kind;
}

bool option_bit(std::span<const uint8_t> values, uint32_t index) {
    return index < values.size() && values[index] != 0;
}

RecordParity make_record_parity(std::span<const uint32_t> records, bool constant) {
    std::vector<RecordSlot> slots;
    slots.reserve(records.size());
    for (uint32_t record : records) {
        slots.push_back(RecordSlot{record});
    }
    return RecordParity{constant, std::move(slots)};
}

struct ObservableRecordReference {
    uint32_t record = 0;
    uint32_t generation = 0;
};

void canonicalize_record_snapshots(std::vector<ObservableRecordReference>& references) {
    std::ranges::sort(references, [](const auto& left, const auto& right) {
        if (left.record != right.record) {
            return left.record < right.record;
        }
        return left.generation < right.generation;
    });
    size_t output = 0;
    for (size_t begin = 0; begin < references.size();) {
        size_t end = begin + 1;
        while (end < references.size() && references[end].record == references[begin].record &&
               references[end].generation == references[begin].generation) {
            ++end;
        }
        if ((end - begin) % 2 != 0) {
            references[output++] = references[begin];
        }
        begin = end;
    }
    references.resize(output);
}

struct PlanningRecord {
    std::optional<AffineBool> value;
    uint32_t generation = 0;
};

struct PlannedSandwich {
    size_t action = 0;
    SymbolId branch{};
    AffineBool outcome_correction;
    bool check_recorded = false;
};

struct PendingObservable {
    AffineBool historical_value;
    std::vector<ObservableRecordReference> record_snapshots;
    std::vector<uint32_t> source_lines;
};

// Only rotations and measurements may change their position relative to noise.
void validate_logical_noise_prefix(const HirModule& hir) {
    if (!hir.has_logical_noise_prefix()) {
        return;
    }
    const auto num_noise_sites = static_cast<uint32_t>(hir.noise_sites.size());
    uint32_t schedule_count = 0;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const HeisenbergOp& op = hir.ops[i];
        const uint32_t entry = hir.logical_noise_prefix[i];
        if (entry > num_noise_sites) {
            throw std::invalid_argument("sampling planner logical noise prefix at operation " +
                                        std::to_string(i) + " exceeds the noise site count");
        }
        const OpType type = op.op_type();
        const bool may_cross_noise =
            type == OpType::T_GATE || type == OpType::PHASE_ROTATION || type == OpType::MEASURE;
        if (!may_cross_noise && entry != schedule_count) {
            throw std::invalid_argument(
                "sampling planner logical noise prefix at operation " + std::to_string(i) +
                " must equal its schedule position for an operation that cannot move across "
                "noise");
        }
        if (type == OpType::NOISE) {
            ++schedule_count;
        }
    }
}

// Resolve noise-dependent signs without exposing optional HIR metadata to
// the planning loop. Symbol bases also allow references to later noise sites.
class NoiseSigns {
  public:
    explicit NoiseSigns(const HirModule& hir) : hir_(hir) {
        validate_logical_noise_prefix(hir);
        if (hir.has_logical_noise_prefix()) {
            symbol_bases_.resize(hir.noise_sites.size());
        }
    }

    void record_symbols(uint32_t site, uint32_t first_symbol) {
        if (!symbol_bases_.empty()) {
            symbol_bases_[site] = first_symbol;
        }
    }

    void check_symbols([[maybe_unused]] uint32_t site, [[maybe_unused]] size_t first_symbol) const {
        assert((symbol_bases_.empty() || symbol_bases_[site] == first_symbol) &&
               "noise symbol allocation disagrees with the planning prepass");
    }

    // Moving a Z rotation across an X error reverses its angle if the error
    // fires. XOR the crossed anticommuting channels to correct either direction.
    [[nodiscard]] AffineBool sign_for(size_t operation_index, uint32_t noise_count) const {
        const HeisenbergOp& op = hir_.ops[operation_index];
        AffineBool sign(hir_.sign(op));
        if (symbol_bases_.empty()) {
            return sign;
        }
        const uint32_t original_count = hir_.logical_noise_prefix[operation_index];
        if (noise_count == original_count) {
            return sign;
        }
        const MaskView x = hir_.destab_mask(op);
        const MaskView z = hir_.stab_mask(op);
        const uint32_t begin = std::min(noise_count, original_count);
        const uint32_t end = std::max(noise_count, original_count);
        for (uint32_t site = begin; site < end; ++site) {
            uint32_t local_index = 0;
            for (const NoiseChannel& channel : hir_.noise_sites[site].channels) {
                if (channel.prob == 0.0) {
                    continue;
                }
                const PauliMaskView channel_view = hir_.noise_channel_masks.at(channel.mask);
                if (anti_commute(x, z, channel_view.x(), channel_view.z())) {
                    sign ^= AffineBool::symbol(SymbolId{symbol_bases_[site] + local_index});
                }
                ++local_index;
            }
        }
        return sign;
    }

  private:
    const HirModule& hir_;
    // First symbol for each noise site's nonzero channels. Unneeded without
    // recorded positions, so ordinary planning does not allocate this table.
    std::vector<uint32_t> symbol_bases_;
};

struct PlanningRequirements {
    uint32_t symbol_count = 0;
    bool supports_final_state_queries = true;
};

bool operation_supports_final_state_queries(OpType type) {
    switch (type) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
        case OpType::EXP_VAL:
            return true;
        case OpType::MEASURE:
        case OpType::CONDITIONAL_PAULI:
        case OpType::NOISE:
        case OpType::READOUT_NOISE:
        case OpType::INSTRUMENT:
        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::SANDWICH_PROJECTION:
        case OpType::SANDWICH_RECORD:
        case OpType::NUM_OP_TYPES:
            return false;
    }
    return false;
}

PlanningRequirements inspect_planning_requirements(const HirModule& hir, NoiseSigns& noise_signs) {
    PlanningRequirements result;
    auto add = [&](size_t amount) {
        if (amount > std::numeric_limits<uint32_t>::max() - result.symbol_count) {
            throw std::length_error("sampling planner symbol count exceeds uint32 range");
        }
        result.symbol_count += static_cast<uint32_t>(amount);
    };

    for (size_t i = 0; i < hir.ops.size(); ++i) {
        const HeisenbergOp& op = hir.ops[i];
        result.supports_final_state_queries &= operation_supports_final_state_queries(op.op_type());
        switch (op.op_type()) {
            case OpType::MEASURE:
            case OpType::READOUT_NOISE:
            case OpType::INSTRUMENT:
                add(1);
                break;
            case OpType::NOISE: {
                const uint32_t site_index = static_cast<uint32_t>(op.noise_site_idx());
                if (site_index >= hir.noise_sites.size()) {
                    throw std::invalid_argument("sampling planner noise site is out of range");
                }
                noise_signs.record_symbols(site_index, result.symbol_count);
                const NoiseSite& site = hir.noise_sites[site_index];
                add(std::ranges::count_if(site.channels, [](const NoiseChannel& channel) {
                    return channel.prob != 0.0;
                }));
                break;
            }
            case OpType::T_GATE:
            case OpType::PHASE_ROTATION:
            case OpType::EXP_VAL:
                break;
            case OpType::CONDITIONAL_PAULI:
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
                break;
            case OpType::SANDWICH_PROJECTION:
                add(1);
                break;
            case OpType::SANDWICH_RECORD:
                break;
            case OpType::NUM_OP_TYPES:
                throw std::invalid_argument("sampling planner does not support HIR operation " +
                                            op_type_to_str(op.op_type()) + " at index " +
                                            std::to_string(i));
        }
    }
    return result;
}

void initialize_site_metadata(const HirModule& hir, SamplingPlan& plan) {
    plan.presampled_noise_sites.resize(hir.noise_sites.size());
    for (uint32_t site = 0; site < hir.noise_sites.size(); ++site) {
        plan.presampled_noise_sites[site].total_probability =
            hir.noise_sites[site].total_probability;
    }
    plan.instrument_distributions.reserve(hir.instrument_sites.size());
    for (uint32_t site = 0; site < hir.instrument_sites.size(); ++site) {
        const InstrumentProbabilities& probabilities = hir.instrument_sites[site].probabilities;
        InstrumentDistribution distribution;
        for (uint8_t source = 0; source < 2; ++source) {
            distribution.p_fire[source] = probabilities.p_fire[source];
            for (uint8_t destination = 0; destination < 2; ++destination) {
                distribution.p_computational_dest[source][destination] =
                    probabilities.p_computational_dest[source][destination];
            }
        }
        plan.instrument_distributions.push_back(distribution);
    }
}

bool process_rotation(const Pauli& body, double half_turns, const AffineBool& sign,
                      SamplingPlan& plan, uint32_t& active_width, CoordinateFrame& coordinates,
                      SymbolicPauliFrame& symbolic_frame, std::span<const uint32_t> source_lines) {
    ResolvedPauli resolved = resolve_pauli(body, sign, coordinates, symbolic_frame);
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(resolved.body, active_width);
    if (!dormant_pivot.has_value()) {
        const ActivePauli active = active_projection(resolved.body, active_width);
        if (active.is_identity()) {
            return false;
        }
        append_action(
            plan,
            PlannedAction{active_width, active_width,
                          RotateActivePauli{active, half_turns, std::move(resolved.sign)}},
            source_lines);
        return false;
    }

    if (active_width + 1 >= kDenseActiveWidthLimit) {
        throw std::overflow_error(
            "sampling planner active width would reach " + std::to_string(active_width + 1) +
            ", but the dense-state limit is " + std::to_string(kDenseActiveWidthLimit));
    }

    coordinates.promote_dormant(resolved.body, active_width, *dormant_pivot);
    append_action(plan,
                  PlannedAction{active_width, active_width + 1,
                                PromoteDormantRotation{half_turns, std::move(resolved.sign)}},
                  source_lines);
    ++active_width;
    plan.peak_active_width = std::max(plan.peak_active_width, active_width);
    return true;
}

AffineBool process_measurement(const Pauli& body, const AffineBool& sign, RecordSlot record,
                               SymbolId branch, SamplingPlan& plan, uint32_t& active_width,
                               CoordinateFrame& coordinates, SymbolicPauliFrame& symbolic_frame,
                               std::span<const uint32_t> source_lines) {
    ResolvedPauli resolved = resolve_pauli(body, sign, coordinates, symbolic_frame);
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(resolved.body, active_width);
    if (dormant_pivot.has_value()) {
        coordinates.measure_dormant(resolved.body, *dormant_pivot);

        define_symbol(plan, branch, SymbolKind::Branch);
        Pauli correction(coordinates.current_to_initial().x_output(*dormant_pivot));
        correction.set_sign(false);
        symbolic_frame.apply(correction, AffineBool::symbol(branch));
        const AffineBool outcome = resolved.sign ^ AffineBool::symbol(branch);
        append_action(plan,
                      PlannedAction{active_width, active_width,
                                    MeasureDormantRandom{*dormant_pivot, branch, outcome, record}},
                      source_lines);
        return outcome;
    }

    const ActivePauli active = active_projection(resolved.body, active_width);
    if (active.is_identity()) {
        append_action(
            plan, PlannedAction{active_width, active_width, RecordClassical{resolved.sign, record}},
            source_lines);
        return resolved.sign;
    }

    std::optional<uint32_t> pivot = last_x_below(resolved.body, active_width);
    if (!pivot.has_value()) {
        pivot = first_z_below(resolved.body, active_width);
    }
    if (!pivot.has_value()) {
        throw std::logic_error("sampling planner could not select an active measurement pivot");
    }

    Pauli active_body(resolved.body.num_qubits());
    for (uint32_t q = 0; q < active_width; ++q) {
        active_body.set_pauli(q, resolved.body.x().bit_get(q), resolved.body.z().bit_get(q));
    }
    active_body.set_sign(false);
    coordinates.measure_active(active_body, active_width, *pivot);

    define_symbol(plan, branch, SymbolKind::Branch);
    Pauli correction(coordinates.current_to_initial().x_output(active_width - 1));
    correction.set_sign(false);
    symbolic_frame.apply(correction, AffineBool::symbol(branch));
    const AffineBool outcome = resolved.sign ^ AffineBool::symbol(branch);
    append_action(plan,
                  PlannedAction{active_width, active_width - 1,
                                MeasureActivePauli{active, *pivot, branch, outcome, record}},
                  source_lines);
    --active_width;
    return outcome;
}

void process_instrument(const HirModule& hir, const HeisenbergOp& op, uint32_t next_noise_site,
                        SamplingPlan& plan, uint32_t& active_width, CoordinateFrame& coordinates,
                        SymbolicPauliFrame& symbolic_frame,
                        std::span<const uint32_t> source_lines) {
    const InstrumentSiteId site{static_cast<uint32_t>(op.instrument_site_idx())};
    const InstrumentSite& hir_site = hir.instrument_sites.at(index(site));
    const SymbolId destination_flip_symbol = reserve_symbol(plan);
    const uint32_t symbol_prefix_size = static_cast<uint32_t>(plan.symbols.size());
    const Pauli body = pauli_from_hir(hir, op);
    const Pauli destination_flip = pauli_from_mask(hir, hir_site.destination_flip_mask);
    const InstrumentDistribution& distribution = plan.instrument_distributions.at(index(site));
    ResolvedPauli resolved =
        resolve_pauli(body, AffineBool(hir.sign(op)), coordinates, symbolic_frame);
    const std::optional<uint32_t> dormant_pivot = first_x_at_or_above(resolved.body, active_width);

    InstrumentMode mode = InstrumentMode::Classical;
    ActivePauli source;
    AffineBool sign = std::move(resolved.sign);
    uint32_t active_after = active_width;

    if (dormant_pivot.has_value()) {
        // Equal no-fire factors normalize away. Otherwise exact damping needs
        // the dormant-coherent source represented in the dense state.
        if (hir.neglect_instrument_damping || distribution.p_fire[0] == distribution.p_fire[1]) {
            mode = InstrumentMode::DormantTrap;
            sign = AffineBool{};
        } else {
            if (active_width + 1 >= kDenseActiveWidthLimit) {
                throw std::overflow_error("sampling planner active width would reach " +
                                          std::to_string(active_width + 1) +
                                          ", but the dense-state limit is " +
                                          std::to_string(kDenseActiveWidthLimit));
            }
            coordinates.promote_dormant(resolved.body, active_width, *dormant_pivot);
            ++active_after;
            mode = InstrumentMode::Activate;
            // The promotion frame installs this observable as the next X
            // generator, so rediscovering it through a local inverse is wasteful.
            source.x = uint64_t{1} << active_width;
        }
    } else {
        source = active_projection(resolved.body, active_width);
        // An identity projection means the source is already determined by
        // the symbolic sign; otherwise its active Pauli needs coefficient work.
        mode = source.is_identity() ? InstrumentMode::Classical : InstrumentMode::Active;
    }

    std::optional<SymbolId> destination_symbol;
    if (mode != InstrumentMode::DormantTrap) {
        // Only an in-line computational destination needs the reserved flip;
        // a trapped continuation resolves its destination instead.
        destination_symbol = destination_flip_symbol;
        define_symbol(plan, destination_flip_symbol, SymbolKind::Instrument);
    }
    append_action(plan,
                  PlannedAction{active_width, active_after,
                                ApplyInstrument{site, mode, source, sign, destination_symbol}},
                  source_lines);

    if (destination_symbol.has_value()) {
        symbolic_frame.apply(destination_flip, AffineBool::symbol(*destination_symbol));
    }
    active_width = active_after;
    plan.peak_active_width = std::max(plan.peak_active_width, active_width);
    // A trapped shot resumes here so it cannot execute the instrument twice.
    append_action(plan,
                  PlannedAction{active_width, active_width,
                                InstrumentBoundary{site, next_noise_site, symbol_prefix_size}},
                  source_lines);
}

// A sandwich block measures C = P * prod_{j in A} exp(-i pi/4 s_j P_j), where
// A holds the opening axes that anticommute with the check P and s_j is the
// rotation direction in effect for the specialized noise class. Bodies are
// Hermitian with positive sign in the current coordinates.
struct SandwichAxis {
    Pauli body;
    bool positive = true;
    // Earlier sampled outcomes that also reverse this axis in the frame.
    std::vector<SymbolId> sampled_terms;
};

struct SandwichObservable {
    Pauli check;
    std::vector<SandwichAxis> axes;
};

// The specialized class fixes which pre-block errors fired, so the parity of
// the noise symbols in an axis sign is the key's flip. Terms from outcomes
// sampled during the shot stay with the axis; the caller verifies that they
// cannot change the projection.
SandwichObservable resolve_sandwich_observable(const HirModule& hir, const HeisenbergOp& op,
                                               const SandwichSite& site, const SamplingPlan& plan,
                                               CoordinateFrame& coordinates,
                                               SymbolicPauliFrame& symbolic_frame) {
    SandwichObservable result{
        resolve_pauli(pauli_from_hir(hir, op), AffineBool{}, coordinates, symbolic_frame).body, {}};
    for (size_t j = 0; j < site.axes.size(); ++j) {
        const PauliMaskHandle handle = site.axes[j];
        ResolvedPauli axis = resolve_pauli(pauli_from_mask(hir, handle),
                                           AffineBool(hir.pauli_masks.at(handle).sign()),
                                           coordinates, symbolic_frame);
        // An axis that commutes with the check cancels from U^dag P U.
        if (axis.body.view().commutes(result.check.view())) {
            continue;
        }
        std::vector<SymbolId> sampled_terms;
        for (const SymbolId term : axis.sign.terms()) {
            if (plan.symbols.at(index(term)) != SymbolKind::Presampled) {
                sampled_terms.push_back(term);
            }
        }
        const bool negative =
            axis.sign.constant() ^ (site.axis_flip[j] != 0) ^ (site.axis_dagger[j] != 0);
        result.axes.push_back(
            SandwichAxis{std::move(axis.body), !negative, std::move(sampled_terms)});
    }
    return result;
}

// Sampled outcomes are values the plan cannot fix. Enumerating their
// assignments bounds the plan-time work of proving the projection
// independent of them.
constexpr size_t kMaxSandwichSampledSymbols = 10;

std::vector<SymbolId> sampled_symbols(const SandwichObservable& observable) {
    std::vector<SymbolId> symbols;
    for (const SandwichAxis& axis : observable.axes) {
        symbols.insert(symbols.end(), axis.sampled_terms.begin(), axis.sampled_terms.end());
    }
    std::ranges::sort(symbols);
    symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());
    if (symbols.size() > kMaxSandwichSampledSymbols) {
        throw std::invalid_argument("sampling planner sandwich axes depend on " +
                                    std::to_string(symbols.size()) + " sampled outcomes; at most " +
                                    std::to_string(kMaxSandwichSampledSymbols) + " are supported");
    }
    return symbols;
}

// The observable the frame describes when the sampled symbols take the
// values in `assignment`, bit k standing for symbols[k].
SandwichObservable with_sampled_values(const SandwichObservable& observable,
                                       const std::vector<SymbolId>& symbols, uint32_t assignment) {
    SandwichObservable result = observable;
    for (SandwichAxis& axis : result.axes) {
        bool reversed = false;
        for (const SymbolId term : axis.sampled_terms) {
            const auto k =
                static_cast<size_t>(std::ranges::lower_bound(symbols, term) - symbols.begin());
            reversed ^= ((assignment >> k) & 1U) != 0;
        }
        axis.positive ^= reversed;
    }
    return result;
}

// Returns the dormant coordinates q whose stabilizer Z_q the observable maps
// to -Z^v. Each anticommuting factor exp(-i pi/4 s P) sends Z_q to
// Z_q (i s P), and conjugation by P negates an anticommuting image. A closed
// block maps every dormant Z_q to a signed product of dormant Z's.
std::vector<uint32_t> projection_sign_mismatches(const SandwichObservable& observable,
                                                 uint32_t active_width) {
    const uint32_t n = observable.check.num_qubits();
    std::vector<uint32_t> mismatched;
    for (uint32_t q = active_width; q < n; ++q) {
        Pauli image(n);
        image.set_pauli(q, false, true);
        for (const SandwichAxis& axis : observable.axes) {
            if (axis.body.x().bit_get(q)) {
                image.right_multiply(axis.body.view());
                image.add_phase(axis.positive ? 1 : 3);
            }
        }
        if (!image.view().commutes(observable.check.view())) {
            image.add_phase(2);
        }
        bool closed = image.x().is_zero();
        for (uint32_t low = 0; closed && low < active_width; ++low) {
            closed = !image.z().bit_get(low);
        }
        if (!closed) {
            throw std::invalid_argument("sandwich block is not closed in the current frame");
        }
        // Without X support the canonical Hermitian phase is zero.
        if (image.phase() == 2) {
            mismatched.push_back(q);
        } else if (image.phase() != 0) {
            throw std::logic_error("sampling planner sandwich image is not Hermitian");
        }
    }
    return mismatched;
}

using BitRow = std::vector<uint64_t>;

void xor_into(BitRow& target, const BitRow& source) {
    for (size_t w = 0; w < target.size(); ++w) {
        target[w] ^= source[w];
    }
}

bool row_is_zero(const BitRow& row) {
    return std::ranges::all_of(row, [](uint64_t word) { return word == 0; });
}

BitRow dormant_x(const Pauli& pauli, uint32_t active_width) {
    BitRow row(pauli.x().words.begin(), pauli.x().words.end());
    for (uint32_t q = 0; q < active_width; ++q) {
        row[q / 64] &= ~(uint64_t{1} << (q % 64));
    }
    return row;
}

// Solutions of XOR_{j in T} x_dormant(P_j) = x_dormant(P) over subsets T of
// the axes: a particular subset and a basis of the homogeneous solutions.
// Terms of the product expansion outside this affine space move the dormant
// |0...0> to an orthogonal state, so only these reach the active matrix.
struct AxisSubsets {
    BitRow particular;
    std::vector<BitRow> null_basis;
};

AxisSubsets solve_dormant_cancellation(const SandwichObservable& observable,
                                       uint32_t active_width) {
    const size_t num_axes = observable.axes.size();
    const size_t subset_words = (num_axes + 63) / 64;
    struct Pivot {
        BitRow vector;
        BitRow subset;
        size_t bit;
    };
    std::vector<Pivot> pivots;
    AxisSubsets result;

    const auto reduce = [&](BitRow& vector, BitRow& subset) {
        for (const Pivot& pivot : pivots) {
            if ((vector[pivot.bit / 64] >> (pivot.bit % 64)) & 1U) {
                xor_into(vector, pivot.vector);
                xor_into(subset, pivot.subset);
            }
        }
    };
    const auto lowest_bit = [](const BitRow& row) {
        for (size_t w = 0; w < row.size(); ++w) {
            if (row[w] != 0) {
                return w * 64 + static_cast<size_t>(std::countr_zero(row[w]));
            }
        }
        return row.size() * 64;
    };

    for (size_t j = 0; j < num_axes; ++j) {
        BitRow vector = dormant_x(observable.axes[j].body, active_width);
        BitRow subset(subset_words, 0);
        subset[j / 64] |= uint64_t{1} << (j % 64);
        reduce(vector, subset);
        if (row_is_zero(vector)) {
            result.null_basis.push_back(std::move(subset));
            continue;
        }
        const size_t bit = lowest_bit(vector);
        // Keep earlier pivots reduced so a single pass reduces later vectors.
        for (Pivot& pivot : pivots) {
            if ((pivot.vector[bit / 64] >> (bit % 64)) & 1U) {
                xor_into(pivot.vector, vector);
                xor_into(pivot.subset, subset);
            }
        }
        pivots.push_back(Pivot{std::move(vector), std::move(subset), bit});
    }

    BitRow target = dormant_x(observable.check, active_width);
    result.particular.assign(subset_words, 0);
    reduce(target, result.particular);
    if (!row_is_zero(target)) {
        throw std::logic_error(
            "sampling planner sandwich observable has no support on the dormant state");
    }
    return result;
}

std::complex<double> i_power(uint8_t exponent) {
    static constexpr std::array<std::complex<double>, 4> kPowers{
        std::complex<double>{1.0, 0.0}, std::complex<double>{0.0, 1.0},
        std::complex<double>{-1.0, 0.0}, std::complex<double>{0.0, -1.0}};
    return kPowers[exponent & 3U];
}

// Expands C = 2^(-|A|/2) sum_T P prod_{j in T} (-i s_j P_j) and keeps the
// terms that act on the dormant |0...0> as +1. Each remaining term is
// i^phase X^x Z^z on the active coordinates.
std::vector<std::complex<double>> projection_matrix(const SandwichObservable& observable,
                                                    uint32_t active_width) {
    const AxisSubsets subsets = solve_dormant_cancellation(observable, active_width);
    const size_t dimension = size_t{1} << active_width;
    // Coefficients of X^x Z^z, indexed by (x << width) | z.
    std::vector<std::complex<double>> coefficients(dimension * dimension);
    const double scale = std::pow(2.0, -0.5 * static_cast<double>(observable.axes.size()));

    // Gray-code order changes one null-space generator per step. The factors
    // commute, so leaving T multiplies by the inverse factor i s_j P_j.
    BitRow subset(subsets.particular.size(), 0);
    Pauli term = observable.check;
    term.set_sign(false);
    const auto toggle = [&](const BitRow& change) {
        for (size_t j = 0; j < observable.axes.size(); ++j) {
            if (((change[j / 64] >> (j % 64)) & 1U) == 0) {
                continue;
            }
            const SandwichAxis& axis = observable.axes[j];
            const bool entering = ((subset[j / 64] >> (j % 64)) & 1U) == 0;
            term.right_multiply(axis.body.view());
            term.add_phase(entering == axis.positive ? 3 : 1);
            subset[j / 64] ^= uint64_t{1} << (j % 64);
        }
    };
    const auto accumulate = [&]() {
        uint64_t x = 0;
        uint64_t z = 0;
        for (uint32_t q = 0; q < active_width; ++q) {
            x |= static_cast<uint64_t>(term.x().bit_get(q)) << q;
            z |= static_cast<uint64_t>(term.z().bit_get(q)) << q;
        }
        coefficients[(x << active_width) | z] += scale * i_power(term.phase());
    };

    toggle(subsets.particular);
    const size_t null_dimension = subsets.null_basis.size();
    if (null_dimension >= 63) {
        throw std::overflow_error("sampling planner sandwich expansion has too many terms");
    }
    const uint64_t num_terms = uint64_t{1} << null_dimension;
    accumulate();
    for (uint64_t step = 1; step < num_terms; ++step) {
        toggle(subsets.null_basis[static_cast<size_t>(std::countr_zero(step))]);
        accumulate();
    }

    std::vector<std::complex<double>> matrix(dimension * dimension);
    for (uint64_t x = 0; x < dimension; ++x) {
        for (uint64_t z = 0; z < dimension; ++z) {
            const std::complex<double> coefficient = coefficients[(x << active_width) | z];
            if (coefficient == std::complex<double>{}) {
                continue;
            }
            for (uint64_t column = 0; column < dimension; ++column) {
                const bool negative = (std::popcount(z & column) & 1) != 0;
                matrix[(column ^ x) * dimension + column] += negative ? -coefficient : coefficient;
            }
        }
    }
    return matrix;
}

constexpr double kSandwichMatrixTolerance = 1e-9;

// Within numerical tolerance, C restricted to the dormant |0...0> subspace
// must be the Hermitian unitary the plan action requires.
void require_hermitian_unitary(const std::vector<std::complex<double>>& matrix, size_t dimension) {
    constexpr double kTolerance = kSandwichMatrixTolerance;
    const auto entry = [&](size_t row, size_t column) { return matrix[row * dimension + column]; };
    for (size_t row = 0; row < dimension; ++row) {
        for (size_t column = 0; column < dimension; ++column) {
            if (!(std::abs(entry(row, column) - std::conj(entry(column, row))) <= kTolerance)) {
                throw std::logic_error("sampling planner sandwich matrix is not Hermitian");
            }
            std::complex<double> product{};
            for (size_t middle = 0; middle < dimension; ++middle) {
                product += entry(row, middle) * entry(middle, column);
            }
            const std::complex<double> expected{row == column ? 1.0 : 0.0, 0.0};
            if (!(std::abs(product - expected) <= kTolerance)) {
                throw std::logic_error("sampling planner sandwich matrix is not unitary");
            }
        }
    }
}

// Returns true when `variant` equals -`base`, false when it equals `base`,
// and nullopt otherwise.
std::optional<bool> relative_sign(const std::vector<std::complex<double>>& variant,
                                  const std::vector<std::complex<double>>& base) {
    bool same = true;
    bool negated = true;
    for (size_t entry = 0; entry < base.size(); ++entry) {
        same &= std::abs(variant[entry] - base[entry]) <= kSandwichMatrixTolerance;
        negated &= std::abs(variant[entry] + base[entry]) <= kSandwichMatrixTolerance;
    }
    if (same) {
        return false;
    }
    if (negated) {
        return true;
    }
    return std::nullopt;
}

struct PlannedProjection {
    size_t action = 0;
    // Sampled symbols whose values negate the projected observable on the
    // dense subspace; every record of the block includes them.
    AffineBool outcome_correction;
};

// Plans the projection at the opening of a specialized block. The returned
// action's outcome is completed when the check record is reached.
PlannedProjection process_sandwich_projection(const HirModule& hir, const HeisenbergOp& op,
                                              const SandwichSite& site, SymbolId branch,
                                              SamplingPlan& plan, uint32_t& active_width,
                                              CoordinateFrame& coordinates,
                                              SymbolicPauliFrame& symbolic_frame,
                                              std::span<const uint32_t> source_lines) {
    SandwichObservable observable =
        resolve_sandwich_observable(hir, op, site, plan, coordinates, symbolic_frame);
    if (observable.axes.empty()) {
        // The block would be a plain Pauli measurement, which the recognizer
        // never reports as a sandwich.
        throw std::invalid_argument(
            "sampling planner sandwich projection has no axis anticommuting with its check");
    }

    // Outcomes sampled during the shot may reverse axes in the frame. The
    // plan is fixed, so every assignment must project onto the same pair of
    // eigenspaces of the dense subspace: the reversal can at most exchange
    // them, as when the outcome's correction acts on the dormant state as a
    // signed stabilizer. An exchange is linear in the outcomes and moves
    // into the record expressions.
    const std::vector<SymbolId> sampled = sampled_symbols(observable);
    const uint32_t num_assignments = uint32_t{1} << sampled.size();
    std::vector<uint32_t> mismatched = projection_sign_mismatches(observable, active_width);
    for (uint32_t assignment = 1; assignment < num_assignments; ++assignment) {
        if (projection_sign_mismatches(with_sampled_values(observable, sampled, assignment),
                                       active_width) != mismatched) {
            throw std::invalid_argument(
                "sampling planner sandwich projection depends on a sampled outcome");
        }
    }
    if (!mismatched.empty()) {
        // The observable maps the dormant state to an orthogonal one. Adding
        // the coordinate whose X flips exactly the mismatched stabilizers
        // leaves a dormant group that the observable preserves with sign +1.
        if (active_width + 1 >= kDenseActiveWidthLimit) {
            throw std::overflow_error(
                "sampling planner active width would reach " + std::to_string(active_width + 1) +
                ", but the dense-state limit is " + std::to_string(kDenseActiveWidthLimit));
        }
        Pauli promoted(hir.num_qubits);
        for (const uint32_t q : mismatched) {
            promoted.set_pauli(q, true, false);
        }
        coordinates.promote_dormant(promoted, active_width, mismatched.front());
        append_action(plan,
                      PlannedAction{active_width, active_width + 1,
                                    PromoteDormantRotation{0.0, AffineBool{}}},
                      source_lines);
        ++active_width;
        plan.peak_active_width = std::max(plan.peak_active_width, active_width);
        observable = resolve_sandwich_observable(hir, op, site, plan, coordinates, symbolic_frame);
        if (!projection_sign_mismatches(observable, active_width).empty()) {
            throw std::logic_error(
                "sampling planner sandwich promotion left a dormant sign mismatch");
        }
    }

    if (active_width > kMaxProjectionWidth) {
        throw std::overflow_error("sampling planner sandwich projection width " +
                                  std::to_string(active_width) + " exceeds the maximum of " +
                                  std::to_string(kMaxProjectionWidth));
    }
    std::vector<std::complex<double>> matrix = projection_matrix(observable, active_width);
    require_hermitian_unitary(matrix, size_t{1} << active_width);
    uint32_t negating = 0;
    for (uint32_t assignment = 1; assignment < num_assignments; ++assignment) {
        const std::optional<bool> negated = relative_sign(
            projection_matrix(with_sampled_values(observable, sampled, assignment), active_width),
            matrix);
        // Single-symbol assignments come first in each power-of-two range,
        // so every combined assignment is checked against their parity.
        if (std::has_single_bit(assignment) && negated.value_or(false)) {
            negating |= assignment;
        }
        const bool expected = (std::popcount(assignment & negating) & 1) != 0;
        if (!negated.has_value() || *negated != expected) {
            throw std::invalid_argument(
                "sampling planner sandwich projection depends on a sampled outcome");
        }
    }
    AffineBool outcome_correction;
    for (size_t k = 0; k < sampled.size(); ++k) {
        if (((negating >> k) & 1U) != 0) {
            outcome_correction ^= AffineBool::symbol(sampled[k]);
        }
    }

    define_symbol(plan, branch, SymbolKind::Branch);
    const size_t action = plan.actions.size();
    // The outcome is completed when the check record is reached.
    append_action(plan,
                  PlannedAction{active_width, active_width,
                                ProjectActiveUnitary{active_width, std::move(matrix), branch,
                                                     AffineBool::symbol(branch),
                                                     RecordSlot{site.check_record}}},
                  source_lines);
    return PlannedProjection{action, std::move(outcome_correction)};
}

void validate_probes(const HirModule& hir, std::span<const FrameSignProbe> probes) {
    for (size_t p = 0; p < probes.size(); ++p) {
        const FrameSignProbe& probe = probes[p];
        if (probe.before_op >= hir.ops.size() ||
            (p > 0 && probe.before_op < probes[p - 1].before_op)) {
            throw std::invalid_argument(
                "sampling planner frame probes must name operations in ascending order");
        }
        for (const size_t op : probe.pauli_ops) {
            if (op >= hir.ops.size() || !hir.ops[op].has_mask()) {
                throw std::invalid_argument(
                    "sampling planner frame probe names an operation without a Pauli mask");
            }
        }
    }
}

// The coordinate change in resolve_pauli contributes only a constant sign,
// and its lookups can materialize a cached inverse. Reading the symbolic
// frame alone keeps probed planning identical to ordinary planning.
std::vector<std::vector<SymbolId>> read_probe(const HirModule& hir, const FrameSignProbe& probe,
                                              const SamplingPlan& plan,
                                              SymbolicPauliFrame& symbolic_frame) {
    std::vector<std::vector<SymbolId>> result;
    result.reserve(probe.pauli_ops.size());
    for (const size_t op : probe.pauli_ops) {
        const AffineBool sign = symbolic_frame.sign_for(pauli_from_hir(hir, hir.ops[op]));
        std::vector<SymbolId> presampled;
        for (const SymbolId term : sign.terms()) {
            if (plan.symbols.at(index(term)) == SymbolKind::Presampled) {
                presampled.push_back(term);
            }
        }
        result.push_back(std::move(presampled));
    }
    return result;
}

SamplingPlan plan_sampling_impl(const HirModule& hir, SamplingPlanOptions options,
                                std::span<const FrameSignProbe> probes,
                                std::vector<std::vector<std::vector<SymbolId>>>* probe_terms) {
    NoiseSigns noise_signs(hir);

    SamplingPlan plan;
    plan.num_qubits = hir.num_qubits;
    plan.num_visible_records = hir.num_measurements;
    plan.num_hidden_records = hir.num_hidden_measurements;
    plan.num_detectors = hir.num_detectors;
    plan.num_observables = hir.num_observables;
    plan.num_exp_vals = hir.num_exp_vals;
    if (options.retain_source_map) {
        if (hir.source_map.size() != hir.ops.size()) {
            throw std::invalid_argument(
                "sampling source provenance requires a complete HIR source map");
        }
        plan.source_map.emplace();
        plan.source_map->reserve(hir.ops.size() + hir.instrument_sites.size() +
                                 hir.num_observables);
    }

    // The packed symbolic frame needs its row count up front. Count only;
    // operations and their Paulis are consumed directly from HIR below.
    const PlanningRequirements requirements = inspect_planning_requirements(hir, noise_signs);
    plan.symbols.reserve(requirements.symbol_count);
    initialize_site_metadata(hir, plan);
    if (requirements.supports_final_state_queries) {
        plan.final_tableau = hir.final_tableau;
    }
    CoordinateFrame coordinates(hir.num_qubits);
    SymbolicPauliFrame symbolic_frame(hir.num_qubits, requirements.symbol_count);
    std::vector<PlanningRecord> records(static_cast<size_t>(hir.num_measurements) +
                                        hir.num_hidden_measurements);
    std::vector<PendingObservable> observables(hir.num_observables);
    std::vector<std::optional<PlannedSandwich>> sandwiches(hir.sandwich_sites.size());

    auto require_record = [&](RecordSlot record, size_t operation_index) -> const AffineBool& {
        const uint32_t record_index = index(record);
        if (record_index >= records.size() || !records[record_index].value.has_value()) {
            throw std::invalid_argument("sampling planner operation " +
                                        std::to_string(operation_index) + " reads record " +
                                        std::to_string(record_index) + " before assignment");
        }
        return *records[record_index].value;
    };

    auto assign_record = [&](RecordSlot record, AffineBool value, size_t operation_index) {
        const uint32_t record_index = index(record);
        if (record_index >= records.size()) {
            throw std::invalid_argument("sampling planner operation " +
                                        std::to_string(operation_index) + " writes record " +
                                        std::to_string(record_index) + " out of range");
        }
        records[record_index].value = std::move(value);
    };

    auto record_parity = [&](const std::vector<uint32_t>& records,
                             size_t operation_index) -> AffineBool {
        AffineBool result;
        for (uint32_t record : records) {
            result ^= require_record(RecordSlot{record}, operation_index);
        }
        return result;
    };

    uint32_t active_width = plan.initial_active_width;
    uint32_t detector_index = 0;
    uint32_t next_noise_site = 0;
    uint32_t next_instrument_site = 0;
    bool supports_final_state_queries = true;
    bool final_coordinates_changed = false;
    size_t next_probe = 0;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        for (; next_probe < probes.size() && probes[next_probe].before_op == i; ++next_probe) {
            probe_terms->push_back(read_probe(hir, probes[next_probe], plan, symbolic_frame));
        }
        const HeisenbergOp& op = hir.ops[i];
        const std::span<const uint32_t> source_lines =
            plan.source_map.has_value() ? source_lines_for(hir, i) : std::span<const uint32_t>{};
        supports_final_state_queries &= operation_supports_final_state_queries(op.op_type());
        switch (op.op_type()) {
            case OpType::T_GATE: {
                const double half_turns = op.is_dagger() ? -0.25 : 0.25;
                const Pauli body = pauli_from_hir(hir, op);
                final_coordinates_changed |=
                    process_rotation(body, half_turns, noise_signs.sign_for(i, next_noise_site),
                                     plan, active_width, coordinates, symbolic_frame, source_lines);
                break;
            }
            case OpType::PHASE_ROTATION: {
                const Pauli body = pauli_from_hir(hir, op);
                final_coordinates_changed |=
                    process_rotation(body, op.alpha(), noise_signs.sign_for(i, next_noise_site),
                                     plan, active_width, coordinates, symbolic_frame, source_lines);
                break;
            }
            case OpType::MEASURE: {
                const RecordSlot record{static_cast<uint32_t>(op.meas_record_idx())};
                const SymbolId branch = reserve_symbol(plan);
                const Pauli body = pauli_from_hir(hir, op);
                const AffineBool outcome = process_measurement(
                    body, noise_signs.sign_for(i, next_noise_site), record, branch, plan,
                    active_width, coordinates, symbolic_frame, source_lines);
                assign_record(record, outcome, i);
                break;
            }
            case OpType::CONDITIONAL_PAULI: {
                const RecordSlot controller{static_cast<uint32_t>(op.controlling_meas())};
                const Pauli body = pauli_from_hir(hir, op);
                symbolic_frame.apply(body, require_record(controller, i));
                break;
            }
            case OpType::NOISE: {
                const uint32_t site_index = static_cast<uint32_t>(op.noise_site_idx());
                if (site_index >= hir.noise_sites.size()) {
                    throw std::invalid_argument("sampling planner noise site is out of range");
                }
                if (site_index != next_noise_site) {
                    throw std::invalid_argument(
                        "sampling planner noise sites are not in circuit order");
                }
                ++next_noise_site;
                const NoiseSite& hir_site = hir.noise_sites[site_index];
                noise_signs.check_symbols(site_index, plan.symbols.size());
                PresampledNoiseSite& plan_site = plan.presampled_noise_sites[site_index];
                plan_site.outcomes.reserve(hir_site.channels.size());
                for (const NoiseChannel& channel : hir_site.channels) {
                    if (channel.prob == 0.0) {
                        continue;
                    }
                    const SymbolId symbol{static_cast<uint32_t>(plan.symbols.size())};
                    plan.symbols.push_back(SymbolKind::Presampled);
                    plan_site.outcomes.push_back(PresampledNoiseOutcome{symbol, channel.prob});
                    const Pauli body = noise_pauli_from_hir(hir, channel.mask);
                    symbolic_frame.apply(body, AffineBool::symbol(symbol));
                }
                break;
            }
            case OpType::READOUT_NOISE: {
                const uint32_t entry_index = static_cast<uint32_t>(op.readout_noise_idx());
                if (entry_index >= hir.readout_noise.size()) {
                    throw std::invalid_argument("sampling planner readout entry is out of range");
                }
                const ReadoutNoiseEntry& entry = hir.readout_noise[entry_index];
                const RecordSlot record{entry.meas_idx};
                const SymbolId flip = reserve_symbol(plan);
                const AffineBool source = require_record(record, i);
                if (entry.prob_zero_to_one == 0.0 && entry.prob_one_to_zero == 0.0) {
                    break;
                }
                define_symbol(plan, flip, SymbolKind::Readout);
                append_action(
                    plan,
                    PlannedAction{active_width, active_width,
                                  ApplyReadoutNoise{flip, source, record, entry.prob_zero_to_one,
                                                    entry.prob_one_to_zero}},
                    source_lines);
                PlanningRecord& updated = records[index(record)];
                updated.value = source ^ AffineBool::symbol(flip);
                ++updated.generation;
                break;
            }
            case OpType::INSTRUMENT: {
                const uint32_t site_index = static_cast<uint32_t>(op.instrument_site_idx());
                if (site_index >= hir.instrument_sites.size() ||
                    site_index != next_instrument_site) {
                    throw std::invalid_argument(
                        "sampling planner instrument site is out of range or circuit order");
                }
                const InstrumentSite& site = hir.instrument_sites[site_index];
                if (site.destination_flip_mask == kNoMask) {
                    throw std::invalid_argument(
                        "sampling planner instrument omits its destination flip");
                }
                process_instrument(hir, op, next_noise_site, plan, active_width, coordinates,
                                   symbolic_frame, source_lines);
                ++next_instrument_site;
                break;
            }
            case OpType::DETECTOR: {
                const uint32_t targets_index = static_cast<uint32_t>(op.detector_idx());
                if (targets_index >= hir.detector_targets.size() ||
                    detector_index >= hir.num_detectors) {
                    throw std::invalid_argument("sampling planner detector is out of range");
                }
                const std::vector<uint32_t>& targets = hir.detector_targets[targets_index];
                for (uint32_t record : targets) {
                    (void)require_record(RecordSlot{record}, i);
                }
                const bool expected = option_bit(options.expected_detectors, detector_index);
                const bool postselected = option_bit(options.postselection_mask, detector_index);
                append_action(
                    plan,
                    PlannedAction{active_width, active_width,
                                  WriteDetector{make_record_parity(targets, expected),
                                                DetectorSlot{detector_index}, postselected}},
                    source_lines);
                ++detector_index;
                break;
            }
            case OpType::OBSERVABLE: {
                const uint32_t targets_index = op.observable_target_list_idx();
                const uint32_t observable_index = static_cast<uint32_t>(op.observable_idx());
                if (targets_index >= hir.observable_targets.size() ||
                    observable_index >= hir.num_observables) {
                    throw std::invalid_argument("sampling planner observable is out of range");
                }
                PendingObservable& pending = observables[observable_index];
                pending.historical_value ^= record_parity(hir.observable_targets[targets_index], i);
                for (uint32_t record : hir.observable_targets[targets_index]) {
                    pending.record_snapshots.push_back({record, records.at(record).generation});
                }
                if (plan.source_map.has_value()) {
                    pending.source_lines.insert(pending.source_lines.end(), source_lines.begin(),
                                                source_lines.end());
                }
                break;
            }
            case OpType::EXP_VAL: {
                const uint32_t exp_val_index = static_cast<uint32_t>(op.exp_val_idx());
                if (exp_val_index >= hir.num_exp_vals) {
                    throw std::invalid_argument(
                        "sampling planner expectation slot is out of range");
                }
                const Pauli body = pauli_from_hir(hir, op);
                ResolvedPauli resolved =
                    resolve_pauli(body, AffineBool(hir.sign(op)), coordinates, symbolic_frame);
                const bool is_zero = first_x_at_or_above(resolved.body, active_width).has_value();
                append_action(
                    plan,
                    PlannedAction{active_width, active_width,
                                  WriteExpectationValue{
                                      is_zero ? std::nullopt
                                              : std::optional<ActiveExpectation>{ActiveExpectation{
                                                    active_projection(resolved.body, active_width),
                                                    std::move(resolved.sign)}},
                                      ExpValSlot{exp_val_index}}},
                    source_lines);
                break;
            }
            case OpType::SANDWICH_PROJECTION: {
                const uint32_t site_index = static_cast<uint32_t>(op.sandwich_site_idx());
                if (site_index >= hir.sandwich_sites.size() || sandwiches[site_index].has_value()) {
                    throw std::invalid_argument(
                        "sampling planner sandwich site is out of range or projected twice");
                }
                const SandwichSite& site = hir.sandwich_sites[site_index];
                if (site.axes.size() != site.axis_dagger.size() ||
                    site.axes.size() != site.axis_flip.size() ||
                    site.check_record >= records.size()) {
                    throw std::invalid_argument("sampling planner sandwich site is malformed");
                }
                const SymbolId branch = reserve_symbol(plan);
                PlannedProjection projection =
                    process_sandwich_projection(hir, op, site, branch, plan, active_width,
                                                coordinates, symbolic_frame, source_lines);
                sandwiches[site_index] = PlannedSandwich{projection.action, branch,
                                                         std::move(projection.outcome_correction)};
                break;
            }
            case OpType::SANDWICH_RECORD: {
                const uint32_t site_index = static_cast<uint32_t>(op.sandwich_site_idx());
                if (site_index >= hir.sandwich_sites.size() ||
                    !sandwiches[site_index].has_value()) {
                    throw std::invalid_argument(
                        "sampling planner sandwich record precedes its projection");
                }
                PlannedSandwich& planned = *sandwiches[site_index];
                const RecordSlot record{static_cast<uint32_t>(op.sandwich_record_idx())};
                // The projection fixed the eigenvalue of C, and P's frame sign
                // here already includes the interior errors it has crossed.
                ResolvedPauli resolved =
                    resolve_pauli(pauli_from_hir(hir, op), noise_signs.sign_for(i, next_noise_site),
                                  coordinates, symbolic_frame);
                AffineBool outcome = std::move(resolved.sign) ^ AffineBool::symbol(planned.branch);
                outcome ^= planned.outcome_correction;
                if (index(record) == hir.sandwich_sites[site_index].check_record) {
                    if (planned.check_recorded) {
                        throw std::invalid_argument(
                            "sampling planner sandwich check is recorded twice");
                    }
                    std::get<ProjectActiveUnitary>(plan.actions[planned.action].action).outcome =
                        outcome;
                    planned.check_recorded = true;
                } else {
                    append_action(
                        plan,
                        PlannedAction{active_width, active_width, RecordClassical{outcome, record}},
                        source_lines);
                }
                assign_record(record, std::move(outcome), i);
                break;
            }
            case OpType::NUM_OP_TYPES:
                throw std::invalid_argument("sampling planner does not support HIR operation " +
                                            op_type_to_str(op.op_type()) + " at index " +
                                            std::to_string(i));
        }
    }

    if (std::ranges::any_of(sandwiches, [](const std::optional<PlannedSandwich>& planned) {
            return !planned.has_value() || !planned->check_recorded;
        })) {
        throw std::invalid_argument(
            "sampling planner sandwich site lacks its projection or check record");
    }
    if (plan.symbols.size() != requirements.symbol_count) {
        throw std::logic_error("sampling planner symbol prepass disagrees with HIR lowering");
    }
    if (supports_final_state_queries != requirements.supports_final_state_queries) {
        throw std::logic_error("sampling planner final-state prepass disagrees with HIR lowering");
    }
    if (detector_index != hir.num_detectors) {
        throw std::invalid_argument("sampling planner detector count is inconsistent with HIR");
    }
    if (next_noise_site != hir.noise_sites.size()) {
        throw std::invalid_argument("sampling planner noise-site table is inconsistent with HIR");
    }
    if (next_instrument_site != hir.instrument_sites.size()) {
        throw std::invalid_argument(
            "sampling planner instrument-site table is inconsistent with HIR");
    }
    for (uint32_t observable = 0; observable < observables.size(); ++observable) {
        PendingObservable& pending = observables[observable];
        pending.historical_value ^= option_bit(options.expected_observables, observable);
        canonicalize_record_snapshots(pending.record_snapshots);
        const bool records_are_current =
            std::ranges::all_of(pending.record_snapshots, [&](const auto& reference) {
                return reference.generation == records[reference.record].generation;
            });
        ObservableValue outcome;
        if (records_are_current) {
            std::vector<uint32_t> snapshot_records;
            snapshot_records.reserve(pending.record_snapshots.size());
            for (const ObservableRecordReference& reference : pending.record_snapshots) {
                snapshot_records.push_back(reference.record);
            }
            outcome = make_record_parity(snapshot_records,
                                         option_bit(options.expected_observables, observable));
        } else {
            outcome = std::move(pending.historical_value);
        }
        const std::span<const uint32_t> source_lines =
            plan.source_map.has_value() ? std::span<const uint32_t>(pending.source_lines)
                                        : std::span<const uint32_t>{};
        append_action(
            plan,
            PlannedAction{active_width, active_width,
                          WriteObservable{std::move(outcome), ObservableSlot{observable}}},
            source_lines);
    }

    if (plan.final_tableau.has_value() && final_coordinates_changed) {
        const Tableau& coordinates_to_physical = coordinates.current_to_initial();
        plan.final_tableau = coordinates_to_physical.then(*plan.final_tableau);
    }

    // CPU and HIP executable construction validate both compiler-produced and
    // externally assembled plans before execution. Repeating that full scan
    // here makes the normal compile path validate the same plan twice.
    return plan;
}

}  // namespace

SamplingPlan plan_sampling(const HirModule& hir, SamplingPlanOptions options) {
    return plan_sampling_impl(hir, options, {}, nullptr);
}

ProbedSamplingPlan plan_sampling_with_probes(const HirModule& hir,
                                             std::span<const FrameSignProbe> probes,
                                             SamplingPlanOptions options) {
    validate_probes(hir, probes);
    ProbedSamplingPlan result;
    result.presampled_terms.reserve(probes.size());
    result.plan = plan_sampling_impl(hir, options, probes, &result.presampled_terms);
    return result;
}

}  // namespace clifft::sampling
