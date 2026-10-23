#pragma once

// Rewrites the HIR through single-copy ZX calculus when that narrows the
// sampled active state.
//
// The circuit becomes a ZX diagram over measurement outcomes and noise bits,
// is reduced, and is read back as Pauli rotations and measurements on fresh
// lines (src/clifft/zx/). Unlike the standard pipeline, which can only
// reorder the traced operations, this changes the spacetime structure: a
// qubit's state is never carried past the point where the diagram no longer
// needs it.
//
// The rewrite preserves the distribution of measurement records, detectors
// and observables, but not the quantum state: the result has no final
// tableau, so final-state queries are unavailable. It keeps every noise site
// and readout noise entry one to one. Run it first, on the traced HIR.
//
// The pass applies the rewrite only if, after the default passes, the
// planner's peak active width for the rewritten module is strictly below the
// input's;
// otherwise, or when the circuit is outside what the lowering covers, it
// leaves the module unchanged and records why. A circuit without measurement
// records is left unchanged too: its only output is the state, which the
// rewrite does not keep.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/hir_pass.h"

#include <cstdint>
#include <string>

namespace clifft {

struct ZxSpacetimeOptions {
    // Upper bound on the qubits of the diagram, where every reset continues
    // on a fresh qubit, and on the rewritten module, sign lines included.
    // Must be positive. Larger circuits are left unchanged.
    uint32_t max_qubits = 4096;
    // Edge toggles allowed while reducing the diagram. Dense circuits make
    // local complementation cubic in the degree; cultivation d5 needs about
    // 3e4 and distillation about 2.4e6, while quantum volume on 20 qubits
    // passes 1e8 without finishing. Must be positive.
    uint64_t max_reduce_work = uint64_t{1} << 23;
};

class ZxSpacetimePass : public HirPass {
  public:
    explicit ZxSpacetimePass(ZxSpacetimeOptions options = {});

    void run(HirModule& hir) override;

    // Statistics from the last run() call. The peaks are the planner's, after
    // the default passes; result_peak is zero when no rewrite was built.
    [[nodiscard]] bool applied() const { return applied_; }
    [[nodiscard]] uint32_t input_peak() const { return input_peak_; }
    [[nodiscard]] uint32_t result_peak() const { return result_peak_; }
    // Empty when the rewrite was applied.
    [[nodiscard]] const std::string& reason() const { return reason_; }

  private:
    ZxSpacetimeOptions options_;
    bool applied_ = false;
    uint32_t input_peak_ = 0;
    uint32_t result_peak_ = 0;
    std::string reason_;
};

}  // namespace clifft
