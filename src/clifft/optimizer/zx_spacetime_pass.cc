#include "clifft/optimizer/zx_spacetime_pass.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/zx/emit.h"
#include "clifft/zx/from_hir.h"
#include "clifft/zx/pbc.h"
#include "clifft/zx/schedule.h"
#include "clifft/zx/simplify.h"

#include <utility>
#include <vector>

namespace clifft {
namespace {

// The peak the default pipeline reaches on a copy of `hir`.
uint32_t default_peak(HirModule hir) {
    default_hir_pass_manager().run(hir);
    return analyze_active_width(hir).peak_width;
}

}  // namespace

void ZxSpacetimePass::run(HirModule& hir) {
    applied_ = false;
    input_peak_ = 0;
    result_peak_ = 0;
    reason_.clear();

    if (hir.forced_traceout_slot) {
        reason_ = "a forced trace-out names a hidden measurement";
        return;
    }
    const auto diagram = zx::build_single_copy_diagram(hir, &reason_);
    if (!diagram) {
        return;
    }
    zx::Graph reduced = diagram->graph;
    zx::full_reduce(reduced);
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
