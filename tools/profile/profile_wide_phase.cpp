// Compiler-only experiment for a contiguous commuting phase region.
// Output coordinates are derived from the HIR tableau, never reference metadata.
// Budgets reject the candidate before the ordinary compiler handles the input.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/pauli_axis.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/symbolic_stabilizers.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"
#include "clifft/util/bitmask.h"

#include "wide_phase_probe.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace clifft;
using namespace clifft::wide_phase_probe;
using Poly = Polynomial;
Budget budget;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
void add(Poly& p, Mask m, int c) {
    budget.add(p, m, c);
}
void add_parity(Poly& p, Mask m, int c) {
    wide_phase_probe::add_parity(p, m, c, budget);
}
std::string read(const char* file) {
    std::ifstream in(file);
    return {std::istreambuf_iterator<char>(in), {}};
}
uint32_t pivot(const PauliString& p) {
    auto n = p.x().num_words() * 64;
    auto x = p.x().lowest_bit();
    return x < n ? x : n + p.z().lowest_bit();
}
bool get(const PauliString& p, uint32_t i) {
    auto n = p.x().num_words() * 64;
    return i < n ? p.x().bit_get(i) : p.z().bit_get(i - n);
}
void print_axis(const PauliString& p) {
    std::cout << '"' << (p.sign() ? '-' : '+');
    for (uint32_t q = 0; q < p.num_qubits(); ++q)
        std::cout << "IXZY"[p.x().bit_get(q) + 2 * p.z().bit_get(q)];
    std::cout << '"';
}
void print_mask(Mask m) {
    std::cout << '[' << m.w[0] << ',' << m.w[1] << ']';
}
void print_poly(const Poly& p) {
    std::cout << '[';
    bool comma = false;
    for (auto& [m, c] : p) {
        if (comma)
            std::cout << ',';
        comma = true;
        std::cout << '[';
        print_mask(m);
        std::cout << ',' << int(c) << ']';
    }
    std::cout << ']';
}
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: profile_wide_phase input.stim prefix.stim probes.txt "
                     "baseline|wide|output|raw-output\n";
        return 2;
    }
    const std::string mode = argv[4];
    if (mode != "baseline" && mode != "wide" && mode != "output" && mode != "raw-output")
        return 2;
    if (const auto* cap = std::getenv("CLIFFT_PROBE_UPDATES"))
        budget.max_updates = std::stoull(cap);
    if (const auto* cap = std::getenv("CLIFFT_PROBE_TERMS"))
        budget.max_terms = std::stoull(cap);
    uint32_t max_variables = 128;
    if (const auto* cap = std::getenv("CLIFFT_PROBE_VARIABLES"))
        max_variables = std::min(128UL, std::stoul(cap));
    const auto started = std::chrono::steady_clock::now();
    auto hir = trace(parse(read(argv[1])));
    if (mode == "baseline") {
        auto preparation = hir;
        auto last = std::find_if(preparation.ops.rbegin(), preparation.ops.rend(),
                                 [](const auto& op) { return op.op_type() == OpType::T_GATE; });
        const auto prep_end =
            preparation.ops.size() - std::distance(preparation.ops.rbegin(), last);
        preparation.ops.erase(preparation.ops.begin() + prep_end, preparation.ops.end());
        preparation.source_map.resize(prep_end);
        auto pm = default_hir_pass_manager();
        pm.run(hir);
        const auto optimized = Clock::now();
        const auto plan = sampling::plan_sampling(hir);
        const sampling::ExecutablePlan program(plan);
        const auto done = Clock::now();
        pm.run(preparation);
        std::cout << "{\"mode\":\"baseline\",\"fallback\":false,\"final_t\":" << hir.num_t_gates()
                  << ",\"preparation_width\":" << analyze_active_width(preparation).peak_width
                  << ",\"final_width\":" << analyze_active_width(hir).peak_width
                  << ",\"hir_seconds\":" << seconds(started, optimized)
                  << ",\"compile_seconds\":" << seconds(started, done) << "}\n";
        return 0;
    }
    const auto untouched = hir;
    if (mode != "raw-output")
        PeepholeFusionPass{}.run(hir);
    SymbolicStabilizers known(hir);
    size_t start = 0;
    while (start < hir.ops.size() && hir.ops[start].op_type() != OpType::T_GATE)
        known.advance(hir, hir.ops[start++]);
    const auto fixed = known.fixed_constraints();
    const auto diagnostic_start = Clock::now();
    const auto prefix = trace(parse(read(argv[2])));
    const auto undo = prefix.final_tableau->inverse();
    std::ifstream probes(argv[3]);
    std::string line;
    size_t fixed_count = 0, probed = 0;
    while (std::getline(probes, line)) {
        auto p = PauliString::from_text(line);
        auto e = fixed.eigenvalue(undo.apply(p.view()));
        ++probed;
        if (e && !*e)
            ++fixed_count;
    }
    const double diagnostic_seconds = seconds(diagnostic_start, Clock::now());
    try {
        if (start == hir.ops.size())
            throw std::length_error("no phase region");
        std::vector<PauliString> proofs;
        std::vector<PauliString> phase_axes;
        std::cerr << "fixed " << fixed_count << " / " << probed << "\n";
        struct Row {
            PauliString axis;
            Mask coords;
        };
        std::map<uint32_t, Row> rows;
        std::vector<PauliString> generators;
        Poly terms, p;
        size_t end = start;
        for (; end < hir.ops.size() && hir.ops[end].op_type() == OpType::T_GATE; ++end) {
            auto axis = optimizer_detail::copy_axis(hir.mask_view(hir.ops[end]), hir.num_qubits);
            if (std::ranges::any_of(phase_axes,
                                    [&](const auto& g) { return !axis.view().commutes(g.view()); }))
                throw std::length_error("noncommuting phase region");
            phase_axes.push_back(axis);
            auto reduced = fixed.reduce_body(axis);
            Mask coords;
            for (auto& [key, row] : rows)
                if (get(reduced, key)) {
                    reduced.mut_x().xor_with(row.axis.x());
                    reduced.mut_z().xor_with(row.axis.z());
                    coords ^= row.coords;
                }
            if (!reduced.x().is_zero() || !reduced.z().is_zero()) {
                if (generators.size() == max_variables)
                    throw std::length_error("variable cap");
                auto b = bit(generators.size());
                const auto row_pivot = pivot(reduced);
                rows.emplace(row_pivot, Row{std::move(reduced), coords ^ b});
                coords = b;
                axis.set_sign(false);
                generators.push_back(axis);
            }
            PauliString product(hir.num_qubits);
            for (Mask b = coords; !b.is_zero(); b.clear_lowest_bit())
                product.right_multiply(generators[b.lowest_bit()].view());
            auto original =
                optimizer_detail::copy_axis(hir.mask_view(hir.ops[end]), hir.num_qubits);
            original.set_sign(false);
            product.right_multiply(original.view());
            auto negative = fixed.eigenvalue(product);
            proofs.push_back(product);
            if (!negative)
                throw std::length_error("unresolved phase constraint");
            int c = hir.ops[end].is_dagger() != hir.mask_view(hir.ops[end]).sign() ? -1 : 1;
            c = *negative ? -c : c;
            add(terms, coords, c);
            add_parity(p, coords, c);
        }
        for (const auto& proof : proofs)
            for (const auto& axis : generators)
                if (!proof.view().commutes(axis.view()))
                    throw std::length_error("phase constraint invalidated");
        const auto original_terms = terms;
        const auto original_polynomial = p;
        const auto original_generators = generators;
        if (mode == "output" || mode == "raw-output") {
            auto product_input = [&](const std::vector<PauliString>& axes, Mask mask) {
                PauliString result(hir.num_qubits);
                for (; !mask.is_zero(); mask.clear_lowest_bit())
                    result.right_multiply(axes[mask.lowest_bit()].view());
                return result;
            };
            std::vector<PauliString> preferred;
            struct Change {
                Mask old, new_;
            };
            std::map<uint32_t, Change> conversion;
            const auto final_inverse = hir.final_tableau->inverse();
            for (uint32_t q = 0; q < hir.num_qubits && preferred.size() < generators.size(); ++q) {
                PauliString z(hir.num_qubits);
                z.set_pauli(q, false, true);
                auto axis = final_inverse.apply(z.view());
                axis.set_sign(false);
                if (std::ranges::any_of(
                        phase_axes,
                        [&](const auto& g) { return !axis.view().commutes(g.view()); }) ||
                    std::ranges::any_of(
                        preferred, [&](const auto& g) { return !axis.view().commutes(g.view()); }))
                    continue;
                auto reduced_axis = fixed.reduce_body(axis);
                Mask old;
                for (auto& [key, row] : rows)
                    if (get(reduced_axis, key)) {
                        reduced_axis.mut_x().xor_with(row.axis.x());
                        reduced_axis.mut_z().xor_with(row.axis.z());
                        old ^= row.coords;
                    }
                if (!reduced_axis.x().is_zero() || !reduced_axis.z().is_zero() || old.is_zero())
                    continue;
                Mask new_ = bit(preferred.size());
                for (const auto& [key, row] : conversion)
                    if (old.bit_get(key)) {
                        old ^= row.old;
                        new_ ^= row.new_;
                    }
                if (!old.is_zero()) {
                    conversion.emplace(old.lowest_bit(), Change{old, new_});
                    preferred.push_back(std::move(axis));
                }
            }
            if (preferred.size() != generators.size())
                throw std::length_error("incomplete output basis");
            Poly preferred_terms, preferred_polynomial;
            for (auto& [mask, c] : terms) {
                auto remaining = mask;
                Mask converted;
                for (const auto& [key, row] : conversion)
                    if (remaining.bit_get(key)) {
                        remaining ^= row.old;
                        converted ^= row.new_;
                    }
                if (!remaining.is_zero())
                    throw std::runtime_error("basis conversion failed");
                auto residual = product_input(generators, mask);
                residual.right_multiply(product_input(preferred, converted).view());
                auto negative = fixed.eigenvalue(residual);
                if (!negative)
                    throw std::length_error("unresolved output basis constraint");
                add(preferred_terms, converted, *negative ? -c : c);
                add_parity(preferred_polynomial, converted, *negative ? -c : c);
            }
            generators = std::move(preferred);
            terms = std::move(preferred_terms);
            p = std::move(preferred_polynomial);
        }

        Mask support;
        for (auto& [m, c] : p)
            support |= m;
        const auto collected = std::chrono::steady_clock::now();
        auto reduced = p;
        auto basis = wide_phase_probe::reduce_core(reduced, generators.size(), budget);
        const auto core_ready = std::chrono::steady_clock::now();
        auto synthesis = wide_phase_probe::synthesize(terms, reduced, basis, budget);
        size_t t_count = 0;
        for (auto& [m, c] : synthesis)
            t_count += c & 1;
        std::cerr << "core " << basis.core_width << " terms " << reduced.size() << " synthesis T "
                  << t_count << " input T " << budget.synthesis_input_t << " updates "
                  << budget.updates << "\n";
        const auto synthesized = std::chrono::steady_clock::now();
        auto product_axis = [&](const std::vector<PauliString>& axes, Mask mask) {
            PauliString result(hir.num_qubits);
            for (; !mask.is_zero(); mask.clear_lowest_bit())
                result.right_multiply(axes[mask.lowest_bit()].view());
            return result;
        };
        std::vector<PauliString> changed;
        for (auto mask : basis.parities)
            changed.push_back(product_axis(generators, mask));
        if (t_count > end - start)
            throw std::length_error("T count guard");
        auto candidate = hir;
        candidate.source_map.resize(start);
        std::vector<uint32_t> phase_sources;
        for (size_t i = start; i < end; ++i)
            phase_sources.insert(phase_sources.end(), hir.source_map[i].begin(),
                                 hir.source_map[i].end());
        std::sort(phase_sources.begin(), phase_sources.end());
        phase_sources.erase(std::unique(phase_sources.begin(), phase_sources.end()),
                            phase_sources.end());
        std::vector<HeisenbergOp> operations(hir.ops.begin(), hir.ops.begin() + start);
        optimizer_detail::CliffordAbsorption cliffords;
        size_t written = 0;
        for (auto& [mask, c] : synthesis) {
            auto axis = product_axis(changed, mask);
            int correction = c;
            if (c & 1) {
                bool dagger = c >= 5;
                auto op = hir.ops[start + written++];
                candidate.demote_to_tgate(op, dagger);
                optimizer_detail::write_axis(candidate.mask_at(op), axis);
                operations.push_back(op);
                candidate.source_map.push_back(phase_sources);
                correction = (c - (dagger ? -1 : 1)) & 7;
            }
            if (correction)
                cliffords.absorb(axis, correction);
        }
        const auto preparation_start = Clock::now();
        auto preparation = candidate;
        preparation.ops = operations;
        cliffords.finish(preparation);
        auto pm = default_hir_pass_manager();
        pm.run(preparation);
        auto preparation_width = analyze_active_width(preparation).peak_width;
        const double preparation_seconds = seconds(preparation_start, Clock::now());
        for (size_t i = end; i < hir.ops.size(); ++i) {
            auto op = hir.ops[i];
            cliffords.transform(candidate, op);
            operations.push_back(op);
            candidate.source_map.push_back(hir.source_map[i]);
        }
        candidate.ops = std::move(operations);
        cliffords.finish(candidate);
        const auto emitted = std::chrono::steady_clock::now();
        auto after = default_hir_pass_manager();
        after.run(candidate);
        auto width = analyze_active_width(candidate).peak_width;
        const auto before_width = analyze_active_width(hir);
        const auto after_width = analyze_active_width(candidate);
        if (after_width.peak_width > before_width.peak_width ||
            (after_width.peak_width == before_width.peak_width &&
             estimate_dense_work(after_width) > estimate_dense_work(before_width)))
            throw std::length_error("active width cost guard");
        const auto optimized = Clock::now();
        auto plan = sampling::plan_sampling(candidate);
        sampling::ExecutablePlan program(plan);
        const auto done = Clock::now();
        if (const auto* output = std::getenv("CLIFFT_PROBE_RECORDS")) {
            if (width > 16)
                throw std::length_error("sampling width guard");
            const auto sampled = sampling::sample(program, 65536, 71, 1);
            std::ofstream file(output, std::ios::binary);
            file.write(reinterpret_cast<const char*>(sampled.measurements.data()),
                       sampled.measurements.size());
        }
        std::cout << "{\"mode\":\"" << mode << "\",\"fallback\":false,\"hir_seconds\":"
                  << seconds(started, optimized) - diagnostic_seconds - preparation_seconds
                  << ",\"compile_seconds\":"
                  << seconds(started, done) - diagnostic_seconds - preparation_seconds
                  << ",\"synthesis_input_t\":" << budget.synthesis_input_t << ",";
        std::cerr << "final T " << candidate.num_t_gates() << " prep " << preparation_width
                  << " final " << width << "\n";
        auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        std::cout << "\"core_width\":" << basis.core_width << ",\"synthesis_t\":" << t_count
                  << ",\"final_t\":" << candidate.num_t_gates()
                  << ",\"preparation_width\":" << preparation_width << ",\"final_width\":" << width
                  << ",\"timings\":[" << seconds(started, collected) << ','
                  << seconds(collected, core_ready) << ',' << seconds(core_ready, synthesized)
                  << ',' << seconds(synthesized, emitted) << ',' << seconds(emitted, done) << "],";

        std::cout << "\"start\":" << start << ",\"end\":" << end
                  << ",\"t_count\":" << hir.num_t_gates() << ",\"variables\":" << generators.size()
                  << ",\"nonzero_support\":" << support.popcount()
                  << ",\"fixed_probes\":" << fixed_count << ",\"probes\":" << probed
                  << ",\"work\":" << budget.updates << ",\"peak_terms\":" << budget.peak_terms
                  << ",\"polynomial\":";
        print_poly(original_polynomial);
        std::cout << ",\"terms\":";
        print_poly(original_terms);
        std::cout << ",\"generators\":[";
        bool comma = false;
        for (const auto& g : original_generators) {
            if (comma)
                std::cout << ',';
            comma = true;
            print_axis(prefix.final_tableau->apply(g.view()));
        }
        std::cout << "]}\n";
    } catch (const std::length_error& error) {
        auto fallback = untouched;
        auto pm = default_hir_pass_manager();
        pm.run(fallback);
        const auto hir_done = Clock::now();
        const auto plan = sampling::plan_sampling(fallback);
        const sampling::ExecutablePlan program(plan);
        const auto done = Clock::now();
        std::cout << "{\"mode\":\"" << mode << "\",\"fallback\":true,\"reason\":\"" << error.what()
                  << "\",\"final_t\":" << fallback.num_t_gates()
                  << ",\"final_width\":" << analyze_active_width(fallback).peak_width
                  << ",\"hir_seconds\":" << seconds(started, hir_done) - diagnostic_seconds
                  << ",\"compile_seconds\":" << seconds(started, done) - diagnostic_seconds
                  << ",\"fixed_probes\":" << fixed_count << ",\"work\":" << budget.updates
                  << ",\"peak_terms\":" << budget.peak_terms << "}\n";
    }
}
