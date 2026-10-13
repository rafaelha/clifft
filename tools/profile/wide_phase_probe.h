#pragma once

#include "clifft/optimizer/tohpe.h"
#include "clifft/util/bitmask.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

// Bounded compiler experiment. The ordinary optimizer retains its 64-bit path.
namespace clifft::wide_phase_probe {
using Mask = BitMask<128>;
struct Less {
    bool operator()(const Mask& a, const Mask& b) const {
        return a.w[1] != b.w[1] ? a.w[1] < b.w[1] : a.w[0] < b.w[0];
    }
};
using Polynomial = std::map<Mask, uint8_t, Less>;

struct Budget {
    size_t max_updates = 50000000;
    size_t max_terms = 200000;
    size_t updates = 0;
    size_t peak_terms = 0;
    size_t synthesis_input_t = 0;

    void add(Polynomial& p, Mask monomial, int coefficient) {
        if (++updates > max_updates) {
            throw std::length_error("phase update budget exhausted");
        }
        const auto found = p.find(monomial);
        const auto value = uint8_t((coefficient + (found == p.end() ? 0 : found->second)) & 7);
        if (value) {
            p[monomial] = value;
        } else if (found != p.end()) {
            p.erase(found);
        }
        peak_terms = std::max(peak_terms, p.size());
        if (p.size() > max_terms) {
            throw std::length_error("phase term budget exhausted");
        }
    }
};

inline Mask bit(uint32_t index) {
    Mask result;
    result.bit_set(index, true);
    return result;
}

inline void add_parity(Polynomial& p, Mask parity, int coefficient, Budget& budget) {
    for (auto a = parity; !a.is_zero(); a.clear_lowest_bit()) {
        const auto first = bit(a.lowest_bit());
        budget.add(p, first, coefficient);
        auto b = a;
        b.clear_lowest_bit();
        for (; !b.is_zero(); b.clear_lowest_bit()) {
            const auto second = bit(b.lowest_bit());
            budget.add(p, first | second, -2 * coefficient);
            auto c = b;
            c.clear_lowest_bit();
            for (; !c.is_zero(); c.clear_lowest_bit()) {
                budget.add(p, first | second | bit(c.lowest_bit()), 4 * coefficient);
            }
        }
    }
}

inline void insert_row(std::map<uint32_t, Mask>& basis, Mask row) {
    while (!row.is_zero()) {
        const uint32_t pivot =
            row.w[1] ? 127 - std::countl_zero(row.w[1]) : 63 - std::countl_zero(row.w[0]);
        const auto found = basis.find(pivot);
        if (found == basis.end()) {
            basis.emplace(pivot, row);
            return;
        }
        row ^= found->second;
    }
}

inline std::vector<Mask> clifford_kernel(const Polynomial& polynomial, uint32_t width,
                                         Budget& budget) {
    std::map<Mask, Mask, Less> constraints;
    for (uint32_t j = 0; j < width; ++j) {
        Polynomial difference;
        for (const auto& [mask, coefficient] : polynomial) {
            if (mask.bit_get(j)) {
                budget.add(difference, mask ^ bit(j), coefficient);
                budget.add(difference, mask, -2 * coefficient);
            }
        }
        for (const auto& [mask, coefficient] : difference) {
            const int degree = mask.popcount();
            const bool non_pauli = degree == 0   ? (coefficient & 1) != 0
                                   : degree == 1 ? ((coefficient / 2) & 1) != 0
                                                 : ((coefficient / 4) & 1) != 0;
            if (non_pauli) {
                constraints[mask] |= bit(j);
            }
        }
    }
    std::map<uint32_t, Mask> pivots;
    for (const auto& [mask, row] : constraints) {
        (void)mask;
        insert_row(pivots, row);
    }
    std::vector<Mask> kernel;
    for (uint32_t free = 0; free < width; ++free) {
        if (pivots.contains(free)) {
            continue;
        }
        auto vector = bit(free);
        for (const auto& [pivot, row] : pivots) {
            if ((row & vector).popcount() & 1) {
                vector |= bit(pivot);
            }
        }
        kernel.push_back(vector);
    }
    return kernel;
}

struct Basis {
    uint32_t core_width;
    std::vector<Mask> parities;
    std::vector<Mask> core_columns;
};

inline Basis reduce_core(Polynomial& polynomial, uint32_t width, Budget& budget) {
    if (width > 128) {
        throw std::length_error("phase variable budget exhausted");
    }
    std::vector<Mask> generators;
    for (uint32_t j = 0; j < width; ++j) {
        generators.push_back(bit(j));
    }
    const auto kernel = clifford_kernel(polynomial, width, budget);
    std::map<uint32_t, Mask> span;
    for (auto vector : kernel) {
        insert_row(span, vector);
    }
    std::vector<Mask> columns;
    for (uint32_t j = 0; j < width; ++j) {
        const size_t before = span.size();
        insert_row(span, bit(j));
        if (span.size() != before) {
            columns.push_back(bit(j));
        }
    }
    const auto core_width = static_cast<uint32_t>(columns.size());
    // Existing TOHPE synthesis retains its representation and work guards.
    if (core_width > 64) {
        throw std::length_error("non-Clifford core exceeds existing synthesis representation");
    }
    columns.insert(columns.end(), kernel.begin(), kernel.end());
    std::vector<Mask> matrix(width);
    for (uint32_t j = 0; j < width; ++j) {
        for (uint32_t i = 0; i < width; ++i) {
            matrix[i].bit_set(j, columns[j].bit_get(i));
        }
    }
    for (uint32_t j = 0; j < width; ++j) {
        uint32_t pivot = j;
        while (!matrix[pivot].bit_get(j)) {
            ++pivot;
            assert(pivot < width);
        }
        if (pivot != j) {
            std::swap(matrix[j], matrix[pivot]);
            std::swap(generators[j], generators[pivot]);
            Polynomial swapped;
            for (const auto& [mask, coefficient] : polynomial) {
                auto exchanged = mask;
                exchanged.bit_swap(j, pivot);
                budget.add(swapped, exchanged, coefficient);
            }
            polynomial = std::move(swapped);
        }
        for (uint32_t i = 0; i < width; ++i) {
            if (i == j || !matrix[i].bit_get(j)) {
                continue;
            }
            matrix[i] ^= matrix[j];
            Polynomial substituted;
            for (const auto& [mask, coefficient] : polynomial) {
                budget.add(substituted, mask, coefficient);
                if (mask.bit_get(i)) {
                    const auto rest = mask ^ bit(i);
                    budget.add(substituted, rest | bit(j), coefficient);
                    budget.add(substituted, rest | bit(j) | bit(i), -2 * coefficient);
                }
            }
            polynomial = std::move(substituted);
            generators[i] ^= generators[j];
        }
    }
    columns.resize(core_width);
    return {core_width, std::move(generators), std::move(columns)};
}

inline Polynomial synthesize(const Polynomial& terms, const Polynomial& reduced, const Basis& basis,
                             Budget& budget) {
    std::map<uint64_t, uint8_t> projected;
    for (const auto& [parity, coefficient] : terms) {
        uint64_t mask = 0;
        for (uint32_t j = 0; j < basis.core_width; ++j) {
            if ((parity & basis.core_columns[j]).popcount() & 1) {
                mask |= uint64_t{1} << j;
            }
        }
        if (mask) {
            projected[mask] = uint8_t((projected[mask] + coefficient) & 7);
        }
    }
    std::vector<uint64_t> columns;
    for (const auto& [parity, coefficient] : projected) {
        if (coefficient & 1) {
            columns.push_back(parity);
        }
    }
    budget.synthesis_input_t = columns.size();
    Polynomial out;
    auto remainder = reduced;
    for (auto parity : phase_detail::tohpe(std::move(columns), basis.core_width)) {
        budget.add(out, Mask{parity}, 1);
        add_parity(remainder, Mask{parity}, -1, budget);
    }
    for (const auto& [mask, coefficient] : remainder) {
        const auto degree = mask.popcount();
        if (degree == 0) {
            continue;
        }
        if (degree == 1 && coefficient % 2 == 0) {
            budget.add(out, mask, coefficient);
        } else if (degree == 2 && coefficient == 4) {
            const auto first = bit(mask.lowest_bit());
            budget.add(out, first, 2);
            budget.add(out, mask ^ first, 2);
            budget.add(out, mask, -2);
        } else {
            throw std::logic_error("wide synthesis changed the non-Clifford signature");
        }
    }
    return out;
}
}  // namespace clifft::wide_phase_probe
