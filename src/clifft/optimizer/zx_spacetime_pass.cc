#include "clifft/optimizer/zx_spacetime_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/planner.h"
#include "clifft/zx/emit.h"
#include "clifft/zx/from_hir.h"
#include "clifft/zx/pbc.h"
#include "clifft/zx/schedule.h"
#include "clifft/zx/simplify.h"

#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clifft {
namespace {

// The peak the default pipeline and the planner reach on a copy of `hir`.
// The planner's coordinate changes can land below the structural trace, so
// comparing structural traces could accept a rewrite that gains nothing.
uint32_t default_peak(HirModule hir) {
    default_hir_pass_manager().run(hir);
    try {
        return sampling::plan_sampling(hir).peak_active_width;
    } catch (const std::exception&) {
        // Beyond the dense limit or otherwise unplannable: the structural
        // trace still orders the candidates.
        return analyze_active_width(hir).peak_width;
    }
}

}  // namespace

ZxSpacetimePass::ZxSpacetimePass(ZxSpacetimeOptions options) : options_(options) {
    if (options_.max_qubits == 0) {
        throw std::invalid_argument("ZxSpacetimePass: max_qubits must be positive");
    }
    if (options_.max_reduce_work == 0) {
        throw std::invalid_argument("ZxSpacetimePass: max_reduce_work must be positive");
    }
}

void ZxSpacetimePass::run(HirModule& hir) {
    applied_ = false;
    input_peak_ = 0;
    result_peak_ = 0;
    reason_.clear();

    if (hir.forced_traceout_slot) {
        reason_ = "a forced trace-out names a hidden measurement";
        return;
    }
    if (hir.num_measurements == 0) {
        reason_ = "the circuit has no measurement records";
        return;
    }
    const auto diagram = zx::build_single_copy_diagram(hir, &reason_, options_.max_qubits);
    if (!diagram) {
        return;
    }
    zx::Graph reduced = diagram->graph;
    if (!zx::full_reduce(reduced, {.max_work = options_.max_reduce_work})) {
        reason_ = "the diagram is too dense to reduce within the work limit";
        return;
    }
    const auto program = zx::read_pbc(*diagram, reduced, &reason_);
    if (!program) {
        return;
    }
    std::vector<std::vector<uint32_t>> orders;
    orders.reserve(program->groups.size());
    for (const zx::PbcGroup& group : program->groups) {
        orders.push_back(zx::greedy_schedule(group).order);
    }
    auto rewritten = zx::emit_hir(hir, *diagram, *program, orders, &reason_);
    if (!rewritten) {
        return;
    }
    if (rewritten->num_qubits > options_.max_qubits) {
        reason_ = "the rewrite would exceed the qubit limit";
        return;
    }
    input_peak_ = default_peak(hir);
    result_peak_ = default_peak(*rewritten);
    if (result_peak_ >= input_peak_) {
        reason_ = "the rewrite is not narrower";
        return;
    }
    hir = std::move(*rewritten);
    applied_ = true;
}

}  // namespace clifft
