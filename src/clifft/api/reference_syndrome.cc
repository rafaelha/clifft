#include "clifft/api/reference_syndrome.h"

#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/remove_noise_pass.h"
#include "clifft/sampling/planner.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace clifft {

ReferenceSyndrome compute_reference_syndrome(const HirModule& hir) {
    if (!hir.instrument_sites.empty()) {
        throw std::invalid_argument(
            "Noiseless reference computation does not support instrument continuations");
    }
    if (hir.num_detectors == 0 && hir.num_observables == 0) {
        return {};
    }

    HirModule clean_hir = hir;
    RemoveNoisePass strip;
    strip.run(clean_hir);

    // Removing noise can expose cancellations needed to prove determinism in
    // the symbolic frame, which does not inspect active-state amplitudes.
    if (std::ranges::any_of(clean_hir.ops, [](const HeisenbergOp& op) {
            return op.op_type() == OpType::T_GATE || op.op_type() == OpType::PHASE_ROTATION;
        })) {
        auto passes = default_hir_pass_manager();
        passes.run(clean_hir);
    }

    const auto plan = sampling::plan_sampling(clean_hir, {.retain_syndrome_constants = true});
    const auto require_constants = [](const auto& values, const char* kind) {
        std::vector<uint8_t> result;
        result.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            if (!values[i].has_value()) {
                throw std::invalid_argument(
                    "Cannot prove noiseless " + std::string(kind) + " " + std::to_string(i) +
                    " deterministic. Supply explicit expected_detectors and expected_observables "
                    "with normalize_syndromes=False to choose reference parities.");
            }
            result.push_back(static_cast<uint8_t>(*values[i]));
        }
        return result;
    };
    return {require_constants(plan.syndrome_constants->detectors, "detector"),
            require_constants(plan.syndrome_constants->observables, "observable")};
}

}  // namespace clifft
