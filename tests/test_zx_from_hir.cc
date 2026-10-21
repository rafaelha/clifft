#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/zx/from_hir.h"
#include "clifft/zx/simplify.h"

#include "zx_fresh_resets.h"
#include "zx_test_helpers.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <numbers>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clifft::zx;
using zx_test::amplitude_table;
using zx_test::TestRng;
using zx_test::with_fresh_qubit_per_reset;

namespace {

// ---------------------------------------------------------------------------
// Circuits in Clifft's syntax, kept as op lists for the reference.

struct Op {
    std::string name;
    std::vector<int> targets;  // qubits; feedback {lookback, qubit}; readout {lookback}
    std::string paulis = {};   // MPP, R_PAULI, E and ELSE letters, parallel to targets
    std::vector<double> args = {};
    bool inverted = false;  // M !q
};

std::string to_text(const std::vector<Op>& ops) {
    std::ostringstream out;
    out.precision(17);
    for (const Op& op : ops) {
        if (op.name == "FEEDBACK_CX" || op.name == "FEEDBACK_CZ") {
            out << (op.name == "FEEDBACK_CX" ? "CX" : "CZ") << " rec[" << op.targets[0] << "] "
                << op.targets[1] << "\n";
            continue;
        }
        out << op.name;
        if (!op.args.empty()) {
            out << "(";
            for (size_t i = 0; i < op.args.size(); ++i) {
                out << (i ? "," : "") << op.args[i];
            }
            out << ")";
        }
        if (op.name == "READOUT_NOISE") {
            out << " rec[" << op.targets[0] << "]\n";
            continue;
        }
        if (op.name == "MPP" || op.name == "R_PAULI") {
            out << " ";
            for (size_t i = 0; i < op.targets.size(); ++i) {
                out << (i ? "*" : "") << op.paulis[i] << op.targets[i];
            }
        } else if (op.name == "E" || op.name == "ELSE_CORRELATED_ERROR") {
            for (size_t i = 0; i < op.targets.size(); ++i) {
                out << " " << op.paulis[i] << op.targets[i];
            }
        } else {
            for (int t : op.targets) {
                out << " " << (op.inverted ? "!" : "") << t;
            }
        }
        out << "\n";
    }
    return out.str();
}

// Three decimals, so the circuit text carries the reference's exact value.
double three_decimals(double x) {
    return std::round(x * 1000.0) / 1000.0;
}

std::vector<Op> random_ops(TestRng& rng, int num_qubits, int depth, bool noise_heavy) {
    auto qubit = [&] { return static_cast<int>(rng.below(static_cast<uint32_t>(num_qubits))); };
    auto distinct = [&](size_t k) {
        std::vector<int> qs;
        while (qs.size() < k) {
            const int q = qubit();
            if (std::find(qs.begin(), qs.end(), q) == qs.end()) {
                qs.push_back(q);
            }
        }
        return qs;
    };
    auto letters = [&](size_t k) {
        std::string s;
        for (size_t i = 0; i < k; ++i) {
            s += "XYZ"[rng.below(3)];
        }
        return s;
    };
    const char* singles[] = {"H", "S", "S_DAG", "T", "T_DAG", "X", "Y", "Z", "SQRT_X"};
    const char* paulis_noise[] = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"};
    std::vector<Op> ops;
    int records = 0;
    // The evaluator enumerates every combination of noise outcomes, so keep
    // their product bounded; a noise op over budget is skipped.
    double combinations = 1.0;
    auto afford = [&](double outcomes) {
        if (combinations * outcomes > 4096.0) {
            return false;
        }
        combinations *= outcomes;
        return true;
    };
    auto measure = [&](const std::string& name) {
        Op op{.name = name, .targets = {qubit()}, .inverted = rng.uniform() < 0.2};
        if (rng.uniform() < 0.25 && afford(2)) {
            op.args = {0.2};
        }
        ops.push_back(op);
        ++records;
        // Feedback right after a measurement is a reset when it anticommutes
        // with the measured Pauli and a look-alike when it commutes.
        if (rng.uniform() < 0.3) {
            ops.push_back(
                {.name = rng.coin() ? "FEEDBACK_CX" : "FEEDBACK_CZ", .targets = {-1, qubit()}});
        }
    };
    for (int i = 0; i < depth; ++i) {
        const double r = rng.uniform();
        if (noise_heavy) {
            // Noise between basis changes, measured at the end: noise bugs show.
            if (r < 0.45) {
                ops.push_back({.name = singles[rng.below(9)], .targets = {qubit()}});
            } else if (r < 0.6) {
                ops.push_back({.name = rng.coin() ? "CX" : "CZ", .targets = distinct(2)});
            } else if (r < 0.75) {
                const uint32_t kind = rng.below(4);
                if (afford(kind == 3 ? 4 : 2)) {
                    ops.push_back(
                        {.name = paulis_noise[kind], .targets = {qubit()}, .args = {0.3}});
                }
            } else if (r < 0.85) {
                if (afford(16)) {
                    ops.push_back({.name = "DEPOLARIZE2", .targets = distinct(2), .args = {0.3}});
                }
            } else if (afford(3)) {
                const auto qs = distinct(1 + rng.below(2));
                ops.push_back(
                    {.name = "E", .targets = qs, .paulis = letters(qs.size()), .args = {0.3}});
                if (rng.coin()) {
                    ops.push_back({.name = "ELSE_CORRELATED_ERROR",
                                   .targets = distinct(1),
                                   .paulis = letters(1),
                                   .args = {0.4}});
                }
            }
            continue;
        }
        if (r < 0.26) {
            ops.push_back({.name = singles[rng.below(9)], .targets = {qubit()}});
        } else if (r < 0.3) {
            ops.push_back({.name = "R_Z",
                           .targets = {qubit()},
                           .args = {three_decimals(0.05 + 0.3 * rng.uniform())}});
        } else if (r < 0.34) {
            const auto qs = distinct(1 + rng.below(2));
            ops.push_back({.name = "R_PAULI",
                           .targets = qs,
                           .paulis = letters(qs.size()),
                           .args = {three_decimals(0.4 * rng.uniform())}});
        } else if (r < 0.46) {
            ops.push_back({.name = rng.coin() ? "CX" : "CZ", .targets = distinct(2)});
        } else if (r < 0.54) {
            const uint32_t kind = rng.below(4);
            if (afford(kind == 3 ? 4 : 2)) {
                ops.push_back({.name = paulis_noise[kind], .targets = {qubit()}, .args = {0.15}});
            }
        } else if (r < 0.57) {
            if (afford(16)) {
                ops.push_back({.name = "DEPOLARIZE2", .targets = distinct(2), .args = {0.2}});
            }
        } else if (r < 0.67) {
            const char* kinds[] = {"M", "MX", "MY"};
            measure(kinds[rng.below(3)]);
        } else if (r < 0.75) {
            const char* kinds[] = {"R", "RX", "RY", "MR", "MRX", "MRY"};
            const std::string name = kinds[rng.below(6)];
            if (name[0] == 'M') {
                measure(name);
            } else {
                ops.push_back({.name = name, .targets = {qubit()}});
            }
        } else if (r < 0.81) {
            ops.push_back({.name = "MPP", .targets = distinct(2), .paulis = letters(2)});
            ++records;
        } else if (r < 0.83) {
            ops.push_back({.name = "MPAD", .targets = {static_cast<int>(rng.below(2))}});
            ++records;
        } else if (records > 0 && r < 0.92) {
            const int back = -1 - static_cast<int>(rng.below(static_cast<uint32_t>(records)));
            ops.push_back(
                {.name = rng.coin() ? "FEEDBACK_CX" : "FEEDBACK_CZ", .targets = {back, qubit()}});
        } else if (records > 0 && afford(2)) {
            const int back = -1 - static_cast<int>(rng.below(static_cast<uint32_t>(records)));
            Op op{.name = "READOUT_NOISE", .targets = {back}, .args = {0.25}};
            if (rng.uniform() < 0.3) {
                op.args = {0.1, 0.35};
            }
            ops.push_back(op);
        }
    }
    if (noise_heavy) {
        const char* kinds[] = {"M", "MX", "MY"};
        for (int q = 0; q < num_qubits; ++q) {
            ops.push_back({.name = kinds[rng.below(3)], .targets = {q}});
        }
    }
    return ops;
}

// ---------------------------------------------------------------------------
// Exact record distribution of a circuit, by enumerating every measurement
// branch and noise outcome on a dense state vector.

using State = std::vector<std::complex<double>>;
using Matrix = std::array<std::complex<double>, 4>;
using Distribution = std::map<uint64_t, double>;

class Reference {
  public:
    explicit Reference(int num_qubits) : n_(num_qubits) {}

    Distribution run(const std::vector<Op>& ops) {
        State psi(size_t{1} << n_, 0.0);
        psi[0] = 1.0;
        dist_.clear();
        branch(ops, 0, std::move(psi), 1.0, 0, 0, false);
        return dist_;
    }

  private:
    using C = std::complex<double>;

    void apply1(State& psi, const Matrix& m, int q) const {
        const size_t bit = size_t{1} << q;
        for (size_t i = 0; i < psi.size(); ++i) {
            if (i & bit) {
                continue;
            }
            const auto a = psi[i], b = psi[i | bit];
            psi[i] = m[0] * a + m[1] * b;
            psi[i | bit] = m[2] * a + m[3] * b;
        }
    }

    static Matrix pauli(char p) {
        if (p == 'X') {
            return {C(0), C(1), C(1), C(0)};
        }
        if (p == 'Y') {
            return {C(0), C(0, -1), C(0, 1), C(0)};
        }
        return {C(1), C(0), C(0), C(-1)};
    }

    State product(const State& psi, const std::vector<int>& qs, const std::string& ps) const {
        State out = psi;
        for (size_t k = 0; k < qs.size(); ++k) {
            apply1(out, pauli(ps[k]), qs[k]);
        }
        return out;
    }

    // (1 + (-1)^outcome P) / 2 applied to psi.
    State project(const State& psi, const std::vector<int>& qs, const std::string& ps,
                  int outcome) const {
        const State p_psi = product(psi, qs, ps);
        State out(psi.size());
        const double s = outcome ? -1.0 : 1.0;
        for (size_t i = 0; i < psi.size(); ++i) {
            out[i] = (psi[i] + s * p_psi[i]) / 2.0;
        }
        return out;
    }

    // exp(-i pi alpha P / 2) = cos(pi alpha / 2) - i sin(pi alpha / 2) P.
    void rotate(State& psi, const std::vector<int>& qs, const std::string& ps, double alpha) const {
        const State p_psi = product(psi, qs, ps);
        const double h = std::numbers::pi * alpha / 2;
        for (size_t i = 0; i < psi.size(); ++i) {
            psi[i] = std::cos(h) * psi[i] - C(0, std::sin(h)) * p_psi[i];
        }
    }

    static bool normalize(State& psi, double& p) {
        p = 0.0;
        for (const auto& a : psi) {
            p += std::norm(a);
        }
        if (p < 1e-15) {
            return false;
        }
        for (auto& a : psi) {
            a /= std::sqrt(p);
        }
        return true;
    }

    // `chain_fired` tracks an E / ELSE_CORRELATED_ERROR chain in progress.
    void branch(const std::vector<Op>& ops, size_t at, State psi, double prob, uint64_t recs,
                int num_recs, bool chain_fired) {
        if (prob < 1e-15) {
            return;
        }
        const double s2 = 1.0 / std::sqrt(2.0);
        for (; at < ops.size(); ++at) {
            const Op& op = ops[at];
            const int q = op.targets.empty() ? 0 : op.targets[0];
            if (op.name != "ELSE_CORRELATED_ERROR") {
                chain_fired = false;
            }
            if (op.name == "H") {
                apply1(psi, {C(s2), C(s2), C(s2), C(-s2)}, q);
            } else if (op.name == "S" || op.name == "S_DAG") {
                apply1(psi, {C(1), C(0), C(0), C(0, op.name == "S" ? 1 : -1)}, q);
            } else if (op.name == "T" || op.name == "T_DAG") {
                const double angle = (op.name == "T" ? 1 : -1) * std::numbers::pi / 4;
                apply1(psi, {C(1), C(0), C(0), std::polar(1.0, angle)}, q);
            } else if (op.name == "X" || op.name == "Y" || op.name == "Z") {
                apply1(psi, pauli(op.name[0]), q);
            } else if (op.name == "SQRT_X") {
                apply1(psi, {C(0.5, 0.5), C(0.5, -0.5), C(0.5, -0.5), C(0.5, 0.5)}, q);
            } else if (op.name == "R_Z") {
                rotate(psi, {q}, "Z", op.args[0]);
            } else if (op.name == "R_PAULI") {
                rotate(psi, op.targets, op.paulis, op.args[0]);
            } else if (op.name == "CX" || op.name == "CZ") {
                const size_t c = size_t{1} << op.targets[0], t = size_t{1} << op.targets[1];
                for (size_t i = 0; i < psi.size(); ++i) {
                    if (!(i & c)) {
                        continue;
                    }
                    if (op.name == "CZ") {
                        if (i & t) {
                            psi[i] = -psi[i];
                        }
                    } else if (!(i & t)) {
                        std::swap(psi[i], psi[i | t]);
                    }
                }
            } else if (op.name == "X_ERROR" || op.name == "Y_ERROR" || op.name == "Z_ERROR" ||
                       op.name == "DEPOLARIZE1") {
                const std::string kinds = op.name == "DEPOLARIZE1" ? "XYZ" : op.name.substr(0, 1);
                const double p = op.args[0] / static_cast<double>(kinds.size());
                for (char l : kinds) {
                    State hit = psi;
                    apply1(hit, pauli(l), q);
                    branch(ops, at + 1, std::move(hit), prob * p, recs, num_recs, false);
                }
                prob *= 1.0 - op.args[0];
            } else if (op.name == "DEPOLARIZE2") {
                for (int k = 1; k < 16; ++k) {
                    State hit = psi;
                    if (k & 3) {
                        apply1(hit, pauli(" XYZ"[k & 3]), op.targets[0]);
                    }
                    if (k >> 2) {
                        apply1(hit, pauli(" XYZ"[k >> 2]), op.targets[1]);
                    }
                    branch(ops, at + 1, std::move(hit), prob * op.args[0] / 15, recs, num_recs,
                           false);
                }
                prob *= 1.0 - op.args[0];
            } else if (op.name == "E" || op.name == "ELSE_CORRELATED_ERROR") {
                if (chain_fired) {
                    continue;
                }
                branch(ops, at + 1, product(psi, op.targets, op.paulis), prob * op.args[0], recs,
                       num_recs, true);
                prob *= 1.0 - op.args[0];
            } else if (op.name == "M" || op.name == "MX" || op.name == "MY" || op.name == "MR" ||
                       op.name == "MRX" || op.name == "MRY" || op.name == "MPP") {
                std::string ps = op.paulis;
                if (op.name != "MPP") {
                    const char last = op.name.back();
                    ps = std::string(1, last == 'X' || last == 'Y' ? last : 'Z');
                }
                const bool resets = op.name.size() >= 2 && op.name[1] == 'R';
                for (int outcome = 0; outcome < 2; ++outcome) {
                    State post = project(psi, op.targets, ps, outcome);
                    double p = 0.0;
                    if (!normalize(post, p)) {
                        continue;
                    }
                    if (resets && outcome) {
                        apply1(post, pauli(ps[0] == 'Z' ? 'X' : 'Z'), q);
                    }
                    const int recorded = outcome ^ static_cast<int>(op.inverted);
                    const uint64_t with = recs | (uint64_t(recorded) << num_recs);
                    const double flip = op.args.empty() ? 0.0 : op.args[0];
                    branch(ops, at + 1, post, prob * p * (1 - flip), with, num_recs + 1, false);
                    if (flip > 0) {
                        branch(ops, at + 1, std::move(post), prob * p * flip,
                               with ^ (uint64_t{1} << num_recs), num_recs + 1, false);
                    }
                }
                return;
            } else if (op.name == "MPAD") {
                recs |= uint64_t(op.targets[0]) << num_recs;
                ++num_recs;
            } else if (op.name == "R" || op.name == "RX" || op.name == "RY") {
                const char basis = op.name == "R" ? 'Z' : op.name[1];
                for (int outcome = 0; outcome < 2; ++outcome) {
                    State post = project(psi, {q}, std::string(1, basis), outcome);
                    double p = 0.0;
                    if (!normalize(post, p)) {
                        continue;
                    }
                    if (outcome) {
                        apply1(post, pauli(basis == 'Z' ? 'X' : 'Z'), q);
                    }
                    branch(ops, at + 1, std::move(post), prob * p, recs, num_recs, false);
                }
                return;
            } else if (op.name == "FEEDBACK_CX" || op.name == "FEEDBACK_CZ") {
                if ((recs >> (num_recs + op.targets[0])) & 1) {
                    apply1(psi, pauli(op.name == "FEEDBACK_CX" ? 'X' : 'Z'), op.targets[1]);
                }
            } else if (op.name == "READOUT_NOISE") {
                const int index = num_recs + op.targets[0];
                const bool bit = ((recs >> index) & 1) != 0;
                const double flip = op.args.size() == 2 && bit ? op.args[1] : op.args[0];
                branch(ops, at + 1, psi, prob * flip, recs ^ (uint64_t{1} << index), num_recs,
                       false);
                prob *= 1.0 - flip;
            } else {
                FAIL("reference does not know " << op.name);
            }
        }
        dist_[recs] += prob;
    }

    int n_;
    Distribution dist_;
};

// ---------------------------------------------------------------------------
// The record distribution the reduced diagram describes.

// Fix the noise parameters to `assignment`, leaving a graph and constraints over
// the outcome parameters only.
Graph bind_noise(const Graph& g, const std::vector<DiagramParam>& params, uint64_t assignment,
                 std::vector<Constraint>& constraints) {
    auto split = [&](const Params& ps, bool& parity) {
        std::vector<uint32_t> kept;
        for (uint32_t p : ps.ids()) {
            if (params[p].kind == DiagramParam::Kind::Noise) {
                parity ^= ((assignment >> p) & 1) != 0;
            } else {
                kept.push_back(p);
            }
        }
        return Params(std::move(kept));
    };
    Graph out = g;
    for (Vertex v : out.vertices()) {
        bool parity = false;
        Params kept = split(out.phase(v).params, parity);
        out.set_phase(v, Phase(out.phase(v).value + (parity ? 1.0 : 0.0), std::move(kept)));
    }
    constraints.clear();
    for (const Constraint& c : g.constraints()) {
        bool parity = false;
        Params kept = split(c.params, parity);
        constraints.push_back(Constraint{.params = std::move(kept), .rhs = c.rhs != parity});
    }
    return out;
}

// The distribution with every classical readout flip applied.
Distribution apply_classical_readout(Distribution dist, const clifft::HirModule& hir,
                                     const std::vector<uint32_t>& entries) {
    for (uint32_t k : entries) {
        const clifft::ReadoutNoiseEntry& e = hir.readout_noise[k];
        Distribution next;
        for (const auto& [recs, p] : dist) {
            const bool bit = ((recs >> e.meas_idx) & 1) != 0;
            const double flip = bit ? e.prob_one_to_zero : e.prob_zero_to_one;
            next[recs] += p * (1 - flip);
            next[recs ^ (uint64_t{1} << e.meas_idx)] += p * flip;
        }
        dist = std::move(next);
    }
    return dist;
}

std::optional<Distribution> diagram_distribution(const SingleCopyDiagram& d, const Graph& reduced,
                                                 const clifft::HirModule& hir) {
    std::vector<uint32_t> outcome_params;
    for (uint32_t p = 0; p < d.params.size(); ++p) {
        if (d.params[p].kind != DiagramParam::Kind::Noise) {
            outcome_params.push_back(p);
        }
    }
    Distribution dist;
    std::vector<size_t> choice(d.noise.size(), 0);
    while (true) {
        double weight = 1.0;
        uint64_t noise_bits = 0;
        for (size_t g = 0; g < d.noise.size(); ++g) {
            const NoiseGroup::Outcome& o = d.noise[g].outcomes[choice[g]];
            weight *= o.probability;
            for (size_t k = 0; k < d.noise[g].bits.size(); ++k) {
                if ((o.pattern >> k) & 1) {
                    noise_bits |= uint64_t{1} << d.noise[g].bits[k];
                }
            }
        }
        if (weight > 0.0) {
            std::vector<Constraint> constraints;
            const Graph bound = bind_noise(reduced, d.params, noise_bits, constraints);
            const auto table = amplitude_table(bound, outcome_params, constraints, 20);
            if (!table) {
                return std::nullopt;
            }
            Distribution given;
            double total = 0.0;
            for (uint64_t a = 0; a < table->size(); ++a) {
                uint64_t records = 0;
                for (size_t k = 0; k < outcome_params.size(); ++k) {
                    const DiagramParam& p = d.params[outcome_params[k]];
                    if (p.kind == DiagramParam::Kind::Record) {
                        records |= ((a >> k) & 1) << p.index;
                    }
                }
                const double p2 = (*table)[a] * (*table)[a];
                given[records] += p2;
                total += p2;
            }
            if (total <= 0.0) {
                return Distribution{};  // a noise outcome the diagram calls impossible
            }
            for (const auto& [records, p2] : given) {
                dist[records] += weight * p2 / total;
            }
        }
        size_t g = 0;
        while (g < choice.size() && ++choice[g] == d.noise[g].outcomes.size()) {
            choice[g++] = 0;
        }
        if (g == choice.size()) {
            break;
        }
    }
    return apply_classical_readout(std::move(dist), hir, d.classical_readout);
}

// The evaluator ignores dropped scalar factors other than constraints, which
// is exact only if each has a parameter-independent modulus.
bool scalar_terms_are_constraints_or_constants(const Graph& g) {
    for (const ScalarTerm& t : g.scalar_terms()) {
        const bool pauli_a = t.alpha == 0.0 || t.alpha == 1.0;
        const bool pauli_b = t.beta == 0.0 || t.beta == 1.0;
        const bool ok = t.kind == ScalarTerm::Kind::Node
                            ? pauli_a || t.a.empty()
                            : (pauli_a && pauli_b) || (t.a.empty() && t.b.empty());
        if (!ok) {
            return false;
        }
    }
    return true;
}

enum class Check { Matched, Rejected };

// Trace the circuit (optionally with fresh qubits per reset), build, reduce,
// and compare the record distribution with the reference.
Check check_circuit(const std::vector<Op>& ops, int num_qubits, bool fresh) {
    const std::string text = to_text(ops);
    INFO(text);
    const clifft::Circuit parsed = clifft::parse(text);
    const clifft::HirModule hir =
        clifft::trace(fresh ? with_fresh_qubit_per_reset(parsed) : parsed);
    std::string reason;
    const std::optional<SingleCopyDiagram> d = build_single_copy_diagram(hir, &reason);
    if (!d) {
        // The one rejection random circuits can trigger.
        REQUIRE(reason == "feedback reads a record with asymmetric readout noise");
        return Check::Rejected;
    }
    Graph reduced = d->graph;
    full_reduce(reduced);
    REQUIRE(scalar_terms_are_constraints_or_constants(reduced));
    const auto got = diagram_distribution(*d, reduced, hir);
    REQUIRE(got);
    const Distribution want = Reference(num_qubits).run(ops);
    for (uint64_t r = 0; r < (uint64_t{1} << hir.num_measurements); ++r) {
        const double a = want.contains(r) ? want.at(r) : 0.0;
        const double b = got->contains(r) ? got->at(r) : 0.0;
        INFO("records " << r);
        REQUIRE(std::abs(a - b) < 1e-9);
    }
    return Check::Matched;
}

// With qubit reuse kept, resets trace as hidden measurements plus corrections,
// which the builder rewrites onto fresh qubits; with fresh qubits per reset in
// the circuit they trace as cut wires. Both must describe the circuit's record
// distribution exactly.
void check_random_circuits(int trials) {
    for (const bool fresh : {false, true}) {
        for (const bool noise_heavy : {false, true}) {
            INFO((fresh ? "fresh qubit per reset" : "qubit reuse kept")
                 << (noise_heavy ? ", noise between basis changes" : ", mixed operations"));
            TestRng rng(4);
            int matched = 0, rejected = 0;
            for (int trial = 0; trial < trials; ++trial) {
                const int num_qubits = 2 + trial % 2;
                const auto ops = random_ops(rng, num_qubits, noise_heavy ? 10 : 14, noise_heavy);
                (check_circuit(ops, num_qubits, fresh) == Check::Matched ? matched : rejected)++;
            }
            CHECK(matched + rejected == trials);
            CHECK(rejected * 10 < trials);
        }
    }
}

}  // namespace

TEST_CASE("Zx diagrams of traced circuits reproduce record distributions") {
    check_random_circuits(25);
}

TEST_CASE("Zx diagrams reproduce record distributions of many random circuits", "[expensive]") {
    check_random_circuits(150);
}

TEST_CASE("Zx diagrams match exact distributions of hand written circuits") {
    // Each case targets one failure: Pauli noise a basis change exposes,
    // dependent channels, Y bases, inverted and padded records, rotations with
    // Y factors, readout noise before and after feedback, and repeated resets.
    const std::vector<std::pair<int, std::vector<Op>>> cases = {
        {1,
         {{.name = "H", .targets = {0}},
          {.name = "DEPOLARIZE1", .targets = {0}, .args = {0.3}},
          {.name = "H", .targets = {0}},
          {.name = "M", .targets = {0}}}},
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "DEPOLARIZE2", .targets = {0, 1}, .args = {0.4}},
          {.name = "H", .targets = {0}},
          {.name = "M", .targets = {0}},
          {.name = "MX", .targets = {1}}}},
        {2,
         {{.name = "E", .targets = {0, 1}, .paulis = "XZ", .args = {0.3}},
          {.name = "ELSE_CORRELATED_ERROR", .targets = {0}, .paulis = "X", .args = {0.5}},
          {.name = "ELSE_CORRELATED_ERROR", .targets = {1}, .paulis = "Y", .args = {0.2}},
          {.name = "M", .targets = {0}},
          {.name = "MX", .targets = {1}}}},
        {1,
         {{.name = "RY", .targets = {0}},
          {.name = "T", .targets = {0}},
          {.name = "MY", .targets = {0}},
          {.name = "MRY", .targets = {0}},
          {.name = "MY", .targets = {0}}}},
        {1,
         {{.name = "H", .targets = {0}},
          {.name = "T", .targets = {0}},
          {.name = "M", .targets = {0}, .inverted = true},
          {.name = "MPAD", .targets = {1}},
          {.name = "MRX", .targets = {0}}}},
        {2,
         {{.name = "R_PAULI", .targets = {0, 1}, .paulis = "XY", .args = {0.3}},
          {.name = "MPP", .targets = {0, 1}, .paulis = "YZ"},
          {.name = "MY", .targets = {0}}}},
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "MR", .targets = {0}, .args = {0.25}},
          {.name = "FEEDBACK_CX", .targets = {-1, 1}},
          {.name = "M", .targets = {1}},
          {.name = "M", .targets = {0}}}},
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "M", .targets = {0}},
          {.name = "FEEDBACK_CX", .targets = {-1, 1}},
          {.name = "READOUT_NOISE", .targets = {-1}, .args = {0.1, 0.4}},
          {.name = "M", .targets = {1}}}},
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "M", .targets = {0}},
          {.name = "READOUT_NOISE", .targets = {-1}, .args = {0.3}},
          {.name = "FEEDBACK_CZ", .targets = {-1, 1}},
          {.name = "H", .targets = {1}},
          {.name = "M", .targets = {1}}}},
        // Feedback on both sides of one flip: the first read sees the record
        // before the flip, the second after it.
        {3,
         {{.name = "H", .targets = {0}},
          {.name = "M", .targets = {0}},
          {.name = "FEEDBACK_CX", .targets = {-1, 1}},
          {.name = "READOUT_NOISE", .targets = {-1}, .args = {0.3}},
          {.name = "FEEDBACK_CX", .targets = {-1, 2}},
          {.name = "M", .targets = {1}},
          {.name = "M", .targets = {2}}}},
        // Feedback right after a measurement that commutes with the measured
        // Pauli is not a reset: on the measured qubit, and on another one.
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "T", .targets = {0}},
          {.name = "MX", .targets = {0}},
          {.name = "FEEDBACK_CX", .targets = {-1, 0}},
          {.name = "H", .targets = {0}},
          {.name = "CX", .targets = {0, 1}},
          {.name = "MY", .targets = {0}},
          {.name = "M", .targets = {1}}}},
        {2,
         {{.name = "H", .targets = {0}},
          {.name = "T", .targets = {0}},
          {.name = "M", .targets = {0}},
          {.name = "FEEDBACK_CX", .targets = {-1, 1}},
          {.name = "H", .targets = {0}},
          {.name = "CX", .targets = {0, 1}},
          {.name = "MX", .targets = {0}},
          {.name = "M", .targets = {1}}}},
        // A Z reset after an X or Y reset of an untouched qubit.
        {1,
         {{.name = "RX", .targets = {0}},
          {.name = "R", .targets = {0}},
          {.name = "M", .targets = {0}}}},
        {1,
         {{.name = "RY", .targets = {0}},
          {.name = "R", .targets = {0}},
          {.name = "M", .targets = {0}}}},
        {1,
         {{.name = "X", .targets = {0}},
          {.name = "R", .targets = {0}},
          {.name = "R", .targets = {0}},
          {.name = "H", .targets = {0}},
          {.name = "R", .targets = {0}},
          {.name = "M", .targets = {0}}}},
    };
    for (const bool fresh : {false, true}) {
        for (const auto& [num_qubits, ops] : cases) {
            INFO((fresh ? "fresh qubit per reset" : "qubit reuse kept"));
            CHECK(check_circuit(ops, num_qubits, fresh) == Check::Matched);
        }
    }
}

TEST_CASE("Zx fresh qubits per reset relabel later targets") {
    const clifft::Circuit c = clifft::parse("H 0\nMR 0\nRX 1\nR 2\nR 2\nCX 0 2\nM 0 1 2\n");
    const clifft::Circuit fresh = with_fresh_qubit_per_reset(c);
    CHECK(fresh.num_measurements == c.num_measurements);
    // MR 0 moves qubit 0 to qubit 3. Qubits 1 and 2 were still |0>, so their
    // resets keep them, and the repeated R 2 allocates nothing.
    CHECK(fresh.num_qubits == 4);
    const auto cx =
        std::find_if(fresh.nodes.begin(), fresh.nodes.end(),
                     [](const clifft::AstNode& n) { return n.gate == clifft::GateType::CX; });
    REQUIRE(cx != fresh.nodes.end());
    REQUIRE(cx->targets.size() == 2);
    CHECK(cx->targets[0].value() == 3);
    CHECK(cx->targets[1].value() == 2);
}

TEST_CASE("Zx fresh qubits per reset reject loss annotations") {
    const clifft::Circuit c = clifft::parse("LOSS(0.1) 0\nR 0\nM 0\n");
    CHECK_THROWS_AS(with_fresh_qubit_per_reset(c), std::invalid_argument);
}

TEST_CASE("Zx diagrams reject noise rescheduled across operations") {
    clifft::HirModule hir = clifft::trace(clifft::parse("H 0\nZ_ERROR(0.3) 0\nT 0\nM 0\n"));
    // Mark the T as having come before the noise, as a scheduler would.
    hir.materialize_logical_noise_prefix();
    bool found = false;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        if (hir.ops[i].op_type() == clifft::OpType::T_GATE) {
            hir.logical_noise_prefix[i] = 0;
            found = true;
        }
    }
    REQUIRE(found);
    std::string reason;
    CHECK_FALSE(build_single_copy_diagram(hir, &reason));
    CHECK(reason == "noise has been rescheduled past other operations");
}
