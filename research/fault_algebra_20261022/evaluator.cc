// Research-only executor: runtime stabilizer algebra is outside Clifft's current
// execution contract. All storage is bounded before the sampling loop.
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using U = uint64_t;
constexpr int MAX_WORDS = 64;
using Bits = std::array<U, MAX_WORDS>;
struct P {
    U x = 0, z = 0;
    int p = 0;  // i^p X^x Z^z, including the Hermitian Y phases.
};
static P mul(P a, P b) {
    return {a.x ^ b.x, a.z ^ b.z, (a.p + b.p + 2 * (std::popcount(a.z & b.x) & 1)) & 3};
}
static bool anti(P a, P b) {
    return std::popcount((a.x & b.z) ^ (a.z & b.x)) & 1;
}
static int sign(P a, P b) {
    assert(a.x == b.x && a.z == b.z && ((a.p ^ b.p) & 1) == 0);
    return ((a.p - b.p) & 2) ? -1 : 1;
}
struct Mask {
    std::vector<std::pair<int, U>> words;
    int parity(const Bits& b) const {
        U v = 0;
        for (auto [i, m] : words)
            v ^= b[i] & m;
        return std::popcount(v) & 1;
    }
    void flip(Bits& b) const {
        for (auto [i, m] : words)
            b[i] ^= m;
    }
};
static Mask read_mask(std::istream& in, int words) {
    Mask result;
    for (int i = 0; i < words; i++) {
        U m;
        in >> m;
        if (m)
            result.words.emplace_back(i, m);
    }
    return result;
}
struct Outcome {
    double cumulative;
    Mask mask;
};
struct Channel {
    double hazard;
    std::vector<Outcome> outcomes;
};
struct Rotation {
    U mask;
    int delta;
    Mask flip;
};
struct Output {
    int index;
    P axis;
    Mask flip;
};
struct Direct {
    int index, constant;
    Mask flip;
};
struct Component {
    int n, width;
    U witness;
    std::array<P, 64> initial;
    std::vector<Mask> initial_flips;
    std::vector<Rotation> rotations;
    std::vector<Output> outputs;
};
struct Plan {
    int nf, params, outputs, detectors, words;
    std::vector<Channel> channels;
    std::vector<double> hazards;
    std::vector<Direct> direct;
    std::vector<Component> components;
    explicit Plan(const char* path) {
        std::ifstream in(path);
        int nc, nd, nk;
        if (!(in >> nf >> params >> outputs >> detectors >> nc >> nd >> nk))
            throw std::runtime_error("invalid plan");
        words = (params + 63) / 64;
        if (words > MAX_WORDS)
            throw std::runtime_error("too many parameters");
        double hazard = 0;
        for (int j = 0; j < nc; j++) {
            int count;
            in >> count;
            Channel c;
            double probability = 0;
            for (int k = 0; k < count; k++) {
                double p;
                in >> p;
                probability += p;
                c.outcomes.push_back({probability, read_mask(in, words)});
            }
            if (!(probability > 0 && probability < 1))
                throw std::runtime_error("invalid noise");
            for (auto& o : c.outcomes)
                o.cumulative /= probability;
            c.outcomes.back().cumulative = 1;
            hazard -= std::log1p(-probability);
            c.hazard = hazard;
            hazards.push_back(hazard);
            channels.push_back(std::move(c));
        }
        for (int j = 0; j < nd; j++) {
            Direct d;
            in >> d.index >> d.constant;
            d.flip = read_mask(in, words);
            direct.push_back(std::move(d));
        }
        for (int j = 0; j < nk; j++) {
            Component c;
            int nr, no;
            in >> c.n >> c.width >> c.witness >> nr >> no;
            if (c.n > 64 || c.width > c.n)
                throw std::runtime_error("unsupported width");
            for (int i = 0; i < c.n; i++) {
                auto& p = c.initial[i];
                in >> p.x >> p.z >> p.p;
                c.initial_flips.push_back(read_mask(in, words));
            }
            for (int i = 0; i < nr; i++) {
                Rotation r;
                in >> r.mask >> r.delta;
                r.flip = read_mask(in, words);
                c.rotations.push_back(std::move(r));
            }
            for (int i = 0; i < no; i++) {
                Output o;
                in >> o.index >> o.axis.x >> o.axis.z >> o.axis.p;
                o.flip = read_mask(in, words);
                c.outputs.push_back(std::move(o));
            }
            components.push_back(std::move(c));
        }
        if (!in)
            throw std::runtime_error("truncated plan");
    }
};
static double uniform(std::mt19937_64& rng) {
    return (rng() >> 11) * 0x1.0p-53;
}

struct State {
    int n, magic;
    std::array<P, 64> g, d;
    void reset(const Component& c, const Bits& f) {
        n = c.n;
        g = c.initial;
        for (int i = 0; i < n; i++) {
            d[i] = i < c.width ? P{0, U{1} << i, 0} : P{U{1} << i, 0, 0};
            g[i].p ^= 2 * c.initial_flips[i].parity(f);
        }
        for (const auto& r : c.rotations) {
            if (!r.flip.parity(f))
                continue;
            U remaining = r.mask;
            while (remaining) {
                int i = std::countr_zero(remaining);
                remaining &= remaining - 1;
                g[i].p = (g[i].p + r.delta) & 3;
                if (r.delta & 1)
                    g[i].z ^= r.mask;
            }
        }
        magic = -1;
        if (c.witness) {
            magic = std::countr_zero(c.witness);
            U remaining = c.witness & (c.witness - 1);
            while (remaining) {
                int i = std::countr_zero(remaining);
                remaining &= remaining - 1;
                g[i] = mul(g[i], g[magic]);
                d[magic] = mul(d[magic], d[i]);
            }
        }
    }
    void random_project(P m, int pivot, int bit) {
        P old = g[pivot];
        for (int i = 0; i < n; i++) {
            if (i != pivot && anti(g[i], m))
                g[i] = mul(g[i], old);
            if (i != pivot && anti(d[i], m))
                d[i] = mul(d[i], old);
        }
        d[pivot] = old;
        m.p ^= 2 * bit;
        g[pivot] = m;
    }
    // Return the normalized probability and optionally project onto forced bit.
    double probability(P m, int bit, bool project) {
        int pivot = -1;
        for (int i = 0; i < n; i++) {
            if (i != magic && anti(g[i], m)) {
                pivot = i;
                break;
            }
        }
        if (pivot >= 0) {
            if (project)
                random_project(m, pivot, bit);
            return 0.5;
        }
        bool a = magic >= 0 && anti(m, g[magic]);
        bool b = magic >= 0 && anti(m, d[magic]);
        P product;
        if (a)
            product = d[magic];
        if (a && b)
            product.p = (product.p + 1) & 3;
        for (int i = 0; i < n; i++) {
            if (anti(m, d[i]))
                product = mul(product, g[i]);
        }
        int s = sign(m, product);
        double expectation = s;
        if (a || b) {
            // exp(-i*pi*X/8)|0> has Bloch vector (0, -1/sqrt(2), 1/sqrt(2)).
            expectation *= a ? (b ? -std::sqrt(0.5) : 0.0) : std::sqrt(0.5);
        }
        double p = (1 + (bit ? -expectation : expectation)) * 0.5;
        if (project && p > 0 && (a || b)) {
            if (a)
                random_project(m, magic, bit);
            else if ((s < 0) != bool(bit))
                g[magic].p ^= 2;
            magic = -1;
        }
        return p;
    }
    int measure(P m, std::mt19937_64& rng) {
        double p0 = probability(m, 0, false);
        int bit = p0 == 1 ? 0 : p0 == 0 ? 1 : int(uniform(rng) >= p0);
        probability(m, bit, true);
        return bit;
    }
};
static void noise(const Plan& p, Bits& f, std::mt19937_64& rng) {
    f.fill(0);
    // Cumulative Bernoulli hazards skip zero outcomes exactly. Each successful
    // channel starts a fresh exponential clock, preserving independent channels.
    double position = 0;
    size_t begin = 0;
    while (begin < p.channels.size()) {
        position -= std::log1p(-uniform(rng));
        auto it = std::upper_bound(p.hazards.begin() + begin, p.hazards.end(), position);
        if (it == p.hazards.end())
            break;
        size_t j = it - p.hazards.begin();
        const auto& c = p.channels[j];
        double r = c.outcomes.size() == 1 ? 0 : uniform(rng);
        for (const auto& o : c.outcomes) {
            if (r < o.cumulative) {
                o.mask.flip(f);
                break;
            }
        }
        position = c.hazard;
        begin = j + 1;
    }
    for (int i = p.nf; i < p.params; i++) {
        if (rng() & 1)
            f[i / 64] ^= U{1} << (i % 64);
    }
}

extern "C" {
void* make_plan(const char* path) {
    return new Plan(path);
}
void free_plan(void* pointer) {
    delete static_cast<Plan*>(pointer);
}
// Output arrays and parameter arrays are caller-owned and allocated before entry.
uint64_t sample(void* pointer, uint64_t shots, uint64_t seed, int postselect, unsigned char* out,
                const uint64_t* fixed, uint64_t* observable_counts, int noise_only) {
    auto& p = *static_cast<Plan*>(pointer);
    std::mt19937_64 rng(seed);
    State state;
    Bits f;
    std::vector<unsigned char> observations(p.outputs - p.detectors);
    uint64_t passed = 0;
    for (uint64_t shot = 0; shot < shots; shot++) {
        if (fixed)
            std::copy_n(fixed + shot * p.words, p.words, f.begin());
        else
            noise(p, f, rng);
        if (noise_only) {
            passed += std::popcount(f[0]);
            continue;
        }
        bool accepted = true;
        for (const auto& o : p.direct) {
            int b = o.constant ^ o.flip.parity(f);
            if (out)
                out[shot * p.outputs + o.index] = b;
            if (postselect && o.index < p.detectors && b) {
                accepted = false;
                break;
            }
            if (observable_counts && o.index >= p.detectors)
                observations[o.index - p.detectors] = b;
        }
        if (!accepted)
            continue;
        for (const auto& c : p.components) {
            state.reset(c, f);
            for (const auto& o : c.outputs) {
                P m = o.axis;
                m.p ^= 2 * o.flip.parity(f);
                int b = state.measure(m, rng);
                if (out)
                    out[shot * p.outputs + o.index] = b;
                if (postselect && o.index < p.detectors && b) {
                    accepted = false;
                    break;
                }
                if (observable_counts && o.index >= p.detectors)
                    observations[o.index - p.detectors] = b;
            }
            if (!accepted)
                break;
        }
        passed += accepted;
        if (accepted && observable_counts) {
            for (size_t i = 0; i < observations.size(); i++)
                observable_counts[i] += observations[i];
        }
    }
    return passed;
}
// Exact conditional probability of a complete output string, for validation.
double probability(void* pointer, const uint64_t* parameters, const unsigned char* bits) {
    auto& p = *static_cast<Plan*>(pointer);
    Bits f{};
    std::copy_n(parameters, p.words, f.begin());
    for (const auto& o : p.direct) {
        if ((o.constant ^ o.flip.parity(f)) != bits[o.index])
            return 0;
    }
    State state;
    double result = 1;
    for (const auto& c : p.components) {
        state.reset(c, f);
        for (const auto& o : c.outputs) {
            P m = o.axis;
            m.p ^= 2 * o.flip.parity(f);
            result *= state.probability(m, bits[o.index], true);
            if (result == 0)
                return 0;
        }
    }
    return result;
}
double expectation(void* pointer, const uint64_t* parameters, const unsigned char* selection) {
    auto& p = *static_cast<Plan*>(pointer);
    Bits f{};
    std::copy_n(parameters, p.words, f.begin());
    double result = 1;
    for (const auto& o : p.direct) {
        if (selection[o.index] && (o.constant ^ o.flip.parity(f)))
            result = -result;
    }
    State state;
    for (const auto& c : p.components) {
        state.reset(c, f);
        P m;
        for (const auto& o : c.outputs) {
            if (!selection[o.index])
                continue;
            P axis = o.axis;
            axis.p ^= 2 * o.flip.parity(f);
            m = mul(m, axis);
        }
        result *= 2 * state.probability(m, 0, false) - 1;
    }
    return result;
}
}
