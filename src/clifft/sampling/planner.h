#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/sampling/plan.h"

#include <cstddef>
#include <span>
#include <vector>

namespace clifft::sampling {

// Builds the executor-independent sampling plan for the supported optimized
// HIR subset. The planner performs all stabilizer-coordinate changes and
// symbolic dependency discovery before execution.
struct SamplingPlanOptions {
    std::span<const uint8_t> postselection_mask{};
    std::span<const uint8_t> expected_detectors{};
    std::span<const uint8_t> expected_observables{};
    // Tooling can request an action-to-source sidecar when the HIR carries a
    // complete parallel source map. Sampling and ordinary lowering do not
    // require or retain it.
    bool retain_source_map = false;
};

// EXP_VAL probes become plan actions. Exact final-state queries instead retain
// a coordinate map only for eligible unitary HIR. Instruments become explicit
// state actions followed by continuation boundaries.
[[nodiscard]] SamplingPlan plan_sampling(const HirModule& hir, SamplingPlanOptions options = {});

// Reads presampled-symbol parities off the planner's own frame. Callers that
// key noise realizations on such parities, as sandwich dispatch does, need
// the frame that planned the circuit, including corrections that feedback
// carries onto Paulis no fired channel touches directly.
struct FrameSignProbe {
    // The signs are read just before this operation is planned.
    size_t before_op = 0;
    // Operations whose signed Pauli masks are resolved against the frame.
    std::vector<size_t> pauli_ops;
};

struct ProbedSamplingPlan {
    SamplingPlan plan;
    // Per probe and per entry of its pauli_ops: the Presampled symbols in the
    // resolved sign, ascending. Sampled outcomes in the sign are omitted.
    std::vector<std::vector<std::vector<SymbolId>>> presampled_terms;
};

// Plans `hir` exactly as plan_sampling does and evaluates `probes` on the
// way. Probes must be sorted by before_op and name operations of `hir` that
// carry a Pauli mask; otherwise this throws std::invalid_argument.
[[nodiscard]] ProbedSamplingPlan plan_sampling_with_probes(const HirModule& hir,
                                                           std::span<const FrameSignProbe> probes,
                                                           SamplingPlanOptions options = {});

}  // namespace clifft::sampling
