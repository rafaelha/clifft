#pragma once

#include "clifft/frontend/hir.h"

#include <cstdint>
#include <vector>

namespace clifft {

/// Noiseless reference syndrome for error syndrome normalization.
struct ReferenceSyndrome {
    std::vector<uint8_t> detectors;
    std::vector<uint8_t> observables;
};

/// Compute the noiseless reference syndrome for an HirModule.
/// Strips stochastic and readout noise from a copy, optimizes remaining
/// non-Clifford operations, and extracts symbolically proven constant parities.
/// Throws std::invalid_argument if any detector or observable cannot be proved
/// deterministic. Coherent rotations are retained as circuit operations.
[[nodiscard]] ReferenceSyndrome compute_reference_syndrome(const HirModule& hir);

}  // namespace clifft
