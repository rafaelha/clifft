#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/zx_spacetime_pass.h"
#include "clifft/zx/emit.h"
#include "clifft/zx/from_hir.h"
#include "clifft/zx/pbc.h"
#include "clifft/zx/schedule.h"
#include "clifft/zx/simplify.h"

#include "zx_circuit_reference.h"

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using namespace clifft;
using zx_test::Distribution;
using zx_test::Op;
using zx_test::random_ops;
using zx_test::Reference;
using zx_test::TestRng;
using zx_test::to_text;

namespace {

using C = std::complex<double>;
using State = std::vector<C>;

struct Masks {
    uint64_t x = 0;
    uint64_t z = 0;
    bool sign = false;
};

Masks masks_of(PauliMaskView view, uint32_t num_qubits) {
    Masks m;
    for (uint32_t q = 0; q < num_qubits; ++q) {
        m.x |= uint64_t(view.x().bit_get(q)) << q;
        m.z |= uint64_t(view.z().bit_get(q)) << q;
    }
    m.sign = view.sign();
    return m;
}

// P|i> = (-1)^sign i^{#Y} (-1)^{z . i} |i ^ x>, with Y = i X Z.
State apply_pauli(const State& psi, const Masks& m) {
    State out(psi.size());
    const C phase = std::pow(C(0, 1), std::popcount(m.x & m.z)) * (m.sign ? -1.0 : 1.0);
    for (uint64_t i = 0; i < psi.size(); ++i) {
        const double sign = (std::popcount(m.z & i) & 1) ? -1.0 : 1.0;
        out[i ^ m.x] += phase * sign * psi[i];
    }
    return out;
}

// The exact distribution of an HIR module's visible records, branching on
// every noise channel, measurement and readout flip.
class HirSimulator {
  public:
    explicit HirSimulator(const HirModule& hir) : hir_(hir) {}

    Distribution run() {
        State psi(size_t{1} << hir_.num_qubits, 0.0);
        psi[0] = 1.0;
        dist_.clear();
        step(0, std::move(psi), 1.0, 0);
        return dist_;
    }

  private:
    void step(size_t at, State psi, double prob, uint64_t records) {
        if (prob < 1e-15) {
            return;
        }
        for (; at < hir_.ops.size(); ++at) {
            const HeisenbergOp& op = hir_.ops[at];
            switch (op.op_type()) {
                case OpType::T_GATE:
                case OpType::PHASE_ROTATION: {
                    const double alpha = op.op_type() == OpType::T_GATE
                                             ? (op.is_dagger() ? -0.25 : 0.25)
                                             : op.alpha();
                    const State p_psi = apply_pauli(psi, mask(op));
                    const double half = std::numbers::pi * alpha / 2.0;
                    for (size_t i = 0; i < psi.size(); ++i) {
                        psi[i] = std::cos(half) * psi[i] - C(0, std::sin(half)) * p_psi[i];
                    }
                    break;
                }
                case OpType::MEASURE: {
                    const State p_psi = apply_pauli(psi, mask(op));
                    const auto r = static_cast<uint32_t>(op.meas_record_idx());
                    for (int bit = 0; bit < 2; ++bit) {
                        State post(psi.size());
                        double p = 0.0;
                        for (size_t i = 0; i < psi.size(); ++i) {
                            post[i] = (psi[i] + (bit ? -1.0 : 1.0) * p_psi[i]) / 2.0;
                            p += std::norm(post[i]);
                        }
                        if (p < 1e-15) {
                            continue;
                        }
                        for (auto& a : post) {
                            a /= std::sqrt(p);
                        }
                        step(at + 1, std::move(post), prob * p, set_record(records, r, bit));
                    }
                    return;
                }
                case OpType::CONDITIONAL_PAULI: {
                    const auto r = static_cast<uint32_t>(op.controlling_meas());
                    if (record(records, r)) {
                        psi = apply_pauli(psi, mask(op));
                    }
                    break;
                }
                case OpType::NOISE: {
                    const NoiseSite& site = hir_.noise_sites[static_cast<uint32_t>(op.noise_site_idx())];
                    double fired = 0.0;
                    for (const NoiseChannel& c : site.channels) {
                        Masks m = masks_of(hir_.noise_channel_masks.at(c.mask), hir_.num_qubits);
                        m.sign = false;
                        step(at + 1, apply_pauli(psi, m), prob * c.prob, records);
                        fired += c.prob;
                    }
                    prob *= std::max(0.0, 1.0 - fired);
                    if (prob < 1e-15) {
                        return;
                    }
                    break;
                }
                case OpType::READOUT_NOISE: {
                    const ReadoutNoiseEntry& e =
                        hir_.readout_noise[static_cast<uint32_t>(op.readout_noise_idx())];
                    const bool bit = record(records, e.meas_idx);
                    const double flip = bit ? e.prob_one_to_zero : e.prob_zero_to_one;
                    if (flip > 0.0) {
                        step(at + 1, psi, prob * flip, set_record(records, e.meas_idx, !bit));
                    }
                    prob *= 1.0 - flip;
                    if (prob < 1e-15) {
                        return;
                    }
                    break;
                }
                default:
                    break;
            }
        }
        dist_[records & visible_mask()] += prob;
    }

    [[nodiscard]] Masks mask(const HeisenbergOp& op) const {
        return masks_of(hir_.mask_view(op), hir_.num_qubits);
    }
    [[nodiscard]] uint64_t visible_mask() const {
        return hir_.num_measurements >= 64 ? ~uint64_t{0}
                                           : (uint64_t{1} << hir_.num_measurements) - 1;
    }
    // Hidden records are tracked too, since feedback can read them.
    static bool record(uint64_t records, uint32_t r) { return (records >> r) & 1; }
    static uint64_t set_record(uint64_t records, uint32_t r, bool bit) {
        return bit ? records | (uint64_t{1} << r) : records & ~(uint64_t{1} << r);
    }

    const HirModule& hir_;
    Distribution dist_;
};

// The rewritten module, without the pass's width test.
std::optional<HirModule> rewrite(const HirModule& hir, std::string* reason) {
    const auto d = zx::build_single_copy_diagram(hir, reason);
    if (!d) {
        return std::nullopt;
    }
    zx::Graph reduced = d->graph;
    zx::full_reduce(reduced);
    const auto program = zx::read_pbc(*d, reduced, reason);
    if (!program) {
        return std::nullopt;
    }
    std::vector<std::vector<uint32_t>> orders;
    for (const zx::PbcGroup& g : program->groups) {
        orders.push_back(zx::greedy_schedule(g).order);
    }
    return zx::emit_hir(hir, *d, *program, orders, reason);
}

enum class Check { Matched, Rejected, TooWide };

Check check_circuit(const std::vector<Op>& ops, int num_qubits) {
    const std::string text = to_text(ops);
    INFO(text);
    const HirModule hir = trace(parse(text));
    std::string reason;
    auto out = rewrite(hir, &reason);
    if (!out) {
        INFO(reason);
        REQUIRE(reason == "feedback reads a record with asymmetric readout noise");
        return Check::Rejected;
    }
    CHECK_FALSE(out->final_tableau.has_value());
    CHECK(out->noise_sites.size() >= hir.noise_sites.size());
    CHECK(out->readout_noise == hir.readout_noise);
    for (size_t s = 0; s < hir.noise_sites.size(); ++s) {
        CHECK(out->noise_sites[s].total_probability == hir.noise_sites[s].total_probability);
        REQUIRE(out->noise_sites[s].channels.size() == hir.noise_sites[s].channels.size());
    }
    if (out->num_qubits > 14) {
        return Check::TooWide;
    }
    const Distribution want = Reference(num_qubits).run(ops);
    const Distribution got = HirSimulator(*out).run();
    for (uint64_t r = 0; r < (uint64_t{1} << hir.num_measurements); ++r) {
        const double a = want.contains(r) ? want.at(r) : 0.0;
        const double b = got.contains(r) ? got.at(r) : 0.0;
        INFO("records " << r);
        REQUIRE(std::abs(a - b) < 1e-9);
    }
    return Check::Matched;
}

void check_random_circuits(int trials) {
    for (const bool noise_heavy : {false, true}) {
        INFO((noise_heavy ? "noise between basis changes" : "mixed operations"));
        TestRng rng(11);
        int matched = 0, other = 0;
        for (int trial = 0; trial < trials; ++trial) {
            const int num_qubits = 2 + trial % 2;
            const auto ops = random_ops(rng, num_qubits, noise_heavy ? 10 : 14, noise_heavy);
            (check_circuit(ops, num_qubits) == Check::Matched ? matched : other)++;
        }
        CHECK(other * 4 < trials);
    }
}

HirModule fixture(const std::string& name) {
    std::ifstream in(std::string(CLIFFT_FIXTURES_DIR) + "/" + name);
    REQUIRE(in.good());
    std::stringstream text;
    text << in.rdbuf();
    return trace(parse(text.str()));
}

}  // namespace

TEST_CASE("Zx rewritten HIR reproduces record distributions of traced circuits") {
    check_random_circuits(25);
}

TEST_CASE("Zx rewritten HIR reproduces record distributions of many random circuits",
          "[expensive]") {
    check_random_circuits(150);
}

TEST_CASE("ZxSpacetimePass narrows cultivation d5 to width 6") {
    HirModule hir = fixture("cultivation_d5.stim");
    const HirModule input = hir;
    ZxSpacetimePass pass;
    pass.run(hir);
    INFO(pass.reason());
    REQUIRE(pass.applied());
    CHECK(pass.input_peak() == 10);
    CHECK(pass.result_peak() == 6);
    CHECK(pass.reason().empty());

    // Records, detectors and observables keep their indices and targets.
    CHECK(hir.num_measurements == input.num_measurements);
    CHECK(hir.detector_targets == input.detector_targets);
    CHECK(hir.observable_targets == input.observable_targets);
    CHECK(hir.noise_sites.size() >= input.noise_sites.size());
    CHECK_FALSE(hir.final_tableau.has_value());
    CHECK(hir.logical_noise_prefix_matches_schedule());

    default_hir_pass_manager().run(hir);
    CHECK(analyze_active_width(hir).peak_width == 6);
}

TEST_CASE("ZxSpacetimePass leaves circuits it does not narrow unchanged") {
    HirModule hir = fixture("coherent_d5_r5.stim");
    const size_t num_ops = hir.ops.size();
    const uint32_t num_qubits = hir.num_qubits;
    ZxSpacetimePass pass;
    pass.run(hir);
    CHECK_FALSE(pass.applied());
    CHECK(pass.reason() == "the rewrite is not narrower");
    CHECK(pass.result_peak() >= pass.input_peak());
    CHECK(hir.ops.size() == num_ops);
    CHECK(hir.num_qubits == num_qubits);
    CHECK(hir.final_tableau.has_value());
}

TEST_CASE("ZxSpacetimePass rejects operations outside the lowering") {
    const auto reason_for = [](const std::string& text) {
        HirModule hir = trace(parse(text));
        const size_t num_ops = hir.ops.size();
        ZxSpacetimePass pass;
        pass.run(hir);
        CHECK_FALSE(pass.applied());
        CHECK(hir.ops.size() == num_ops);
        return pass.reason();
    };
    CHECK(reason_for("H 0\nT 0\nEXP_VAL X0\nM 0\n") ==
          "expectation-value probes name physical qubits");
    CHECK(reason_for("H 0\nT 0\nM 0\nREADOUT_NOISE(0.1, 0.2) rec[-1]\nCX rec[-1] 1\nH 1\nT 1\nM 1\n") ==
          "feedback reads a record with asymmetric readout noise");
}
