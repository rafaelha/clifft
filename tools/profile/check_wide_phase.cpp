#include "wide_phase_probe.h"

#include <iostream>
#include <random>
using namespace clifft::wide_phase_probe;
int polynomial_value(const Polynomial& p, Mask x) {
    int v = 0;
    for (auto& [m, c] : p)
        if ((m & x) == m)
            v += c;
    return v & 7;
}
int parity_value(const Polynomial& p, Mask x) {
    int v = 0;
    for (auto& [m, c] : p)
        if ((m & x).popcount() & 1)
            v += c;
    return v & 7;
}
int main() {
    std::mt19937_64 rng(918);
    size_t checks = 0;
    for (uint32_t width : {65, 87, 128})
        for (int trial = 0; trial < 12; ++trial) {
            Budget budget;
            Polynomial original, terms;
            // A nine-variable non-Clifford core embedded across the word boundary.
            std::vector<uint32_t> active{0, 1, 2, 31, 32, 63, width - 3, width - 2, width - 1};
            for (auto q : active) {
                budget.add(terms, bit(q), int(rng() % 8));
            }
            for (int j = 0; j < 12; ++j) {
                Mask mask;
                for (auto q : active)
                    if (rng() & 1)
                        mask |= bit(q);
                budget.add(terms, mask, int(rng() % 8));
            }
            for (uint32_t q = 0; q < width; ++q)
                budget.add(terms, bit(q), 2 * int(rng() % 4));
            for (int j = 0; j < 20; ++j)
                budget.add(terms, bit(rng() % width) | bit(rng() % width), 2 * int(rng() % 4));
            for (auto& [m, c] : terms)
                add_parity(original, m, c, budget);
            auto reduced = original;
            auto basis = reduce_core(reduced, width, budget);
            {
                auto output = synthesize(terms, reduced, basis, budget);
                for (int j = 0; j < 512; ++j) {
                    Mask x;
                    x.w[0] = rng();
                    x.w[1] = rng();
                    Mask y;
                    for (uint32_t k = 0; k < width; ++k)
                        y.bit_set(k, (basis.parities[k] & x).popcount() & 1);
                    if (polynomial_value(original, x) != polynomial_value(reduced, y) ||
                        parity_value(output, y) != polynomial_value(original, x))
                        throw std::runtime_error("polynomial mismatch");
                    ++checks;
                }
            }
        }
    Budget exhausted;
    exhausted.max_updates = 0;
    Polynomial p;
    bool stopped = false;
    try {
        exhausted.add(p, bit(127), 1);
    } catch (const std::length_error&) {
        stopped = true;
    }
    if (!stopped)
        throw std::runtime_error("missing budget stop");
    Budget core_budget;
    Polynomial independent;
    for (uint32_t j = 0; j < 65; ++j)
        core_budget.add(independent, bit(j), 1);
    stopped = false;
    try {
        reduce_core(independent, 65, core_budget);
    } catch (const std::length_error&) {
        stopped = true;
    }
    if (!stopped)
        throw std::runtime_error("missing synthesis representation guard");
    std::cout << checks << " phase evaluations passed at widths 65 87 128\n";
}
