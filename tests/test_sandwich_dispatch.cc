// Checks per-shot sandwich dispatch: the key map that classifies a shot's
// presampled noise, program families of specialized plans, and the samplers
// that run each shot on the plan specialized for its class. Keys are checked
// against the feedback-aware reference of the specialization tests.

#include "clifft/circuit/circuit.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/sandwich_blocks.h"
#include "clifft/optimizer/sandwich_specialization_pass.h"
#include "clifft/sampling/program_family.h"
#include "clifft/sampling/sampler.h"
#include "clifft/sampling/sandwich_keys.h"
#include "clifft/util/xoshiro.h"

#include "sandwich_test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace clifft;
using namespace clifft::sandwich_test;
using Catch::Matchers::ContainsSubstring;
using sampling::compute_sandwich_key_map;
using sampling::FamilyMember;
using sampling::FamilySamplingStats;
using sampling::ProgramFamily;
using sampling::SamplingPlanOptions;
using sampling::SamplingSurvivorResult;
using sampling::SandwichKeyMap;
using sampling::SandwichKeyRows;

namespace {

// A closed toy block on data qubits 0-2 with ancilla 3 and depolarizing
// noise before, inside and after it.
constexpr const char* kNoisyBlock =
    "RX 3\nDEPOLARIZE1(0.1) 0\nT 0 1 2\nDEPOLARIZE1(0.1) 2 3\nCX 3 0 3 1 3 2\nMX 3\n"
    "RX 3\nDEPOLARIZE1(0.1) 1\nCX 3 0 3 1 3 2\nT_DAG 0 1 2\nDEPOLARIZE1(0.1) 0\n";

// Two toy blocks linked by feedback: the X measurement of data qubit 0
// after the first block conditions an X on qubit 4, which the second block
// rotates about Z. An interior X on qubit 0 reaches the second block only
// through the first block's correction, which turns that measurement into Y.
constexpr const char* kFeedbackBlocks =
    "RX 3\nDEPOLARIZE1(0.1) 0\nT 0 1 2\nDEPOLARIZE1(0.1) 0 3\nCX 3 0 3 1 3 2\nMX 3\n"
    "RX 3\nCX 3 0 3 1 3 2\nT_DAG 0 1 2\nMX 0\nCX rec[-1] 4\n"
    "RX 7\nT 4 5 6\nCX 7 4 7 5 7 6\nMX 7\nRX 7\nCX 7 4 7 5 7 6\nT_DAG 4 5 6\nMX 4 5 6\n";

std::vector<uint32_t> fired_ordinals(std::span<const uint8_t> values) {
    std::vector<uint32_t> fired;
    for (uint32_t ordinal = 0; ordinal < values.size(); ++ordinal) {
        if (values[ordinal] != 0) {
            fired.push_back(ordinal);
        }
    }
    return fired;
}

// Requires the key map to classify `fired` as the reference does, through
// both of its entry points. Returns the keys.
std::vector<SandwichKey> check_keys_match_reference(const HirModule& hir, const SamplingPlan& plan,
                                                    const SandwichKeyMap& key_map,
                                                    const std::vector<FiredChannel>& fired) {
    const std::vector<uint8_t> values = presampled_values(hir, plan, fired);
    const std::vector<SandwichKey> keys = key_map.keys_for(values);
    CHECK(keys == keys_for(hir, fired));
    CHECK(key_map.keys_for_fired(fired_ordinals(values)) == keys);
    return keys;
}

// Up to `max_fired` channels on distinct sites, each with nonzero
// probability, drawn uniformly over sites and then over channels.
std::vector<FiredChannel> random_realization(const HirModule& hir, Xoshiro256PlusPlus& rng,
                                             uint32_t max_fired) {
    const auto count = static_cast<uint32_t>(rng() % (max_fired + 1));
    std::vector<FiredChannel> fired;
    while (fired.size() < count) {
        const auto site = static_cast<uint32_t>(rng() % hir.noise_sites.size());
        if (std::ranges::any_of(fired, [&](const FiredChannel& c) { return c.site == site; })) {
            continue;
        }
        const std::vector<NoiseChannel>& channels = hir.noise_sites[site].channels;
        const auto channel = static_cast<uint32_t>(rng() % channels.size());
        if (channels[channel].prob != 0.0) {
            fired.push_back(FiredChannel{site, channel});
        }
    }
    return fired;
}

bool any_flag(const std::vector<SandwichKey>& keys, bool internal) {
    return std::ranges::any_of(keys, [&](const SandwichKey& key) {
        const std::vector<uint8_t>& flags = internal ? key.internal_support : key.axis_flip;
        return std::ranges::count(flags, 1) > 0;
    });
}

// The same map without the cross-block corrections of supported axes.
SandwichKeyMap without_support_targets(const SandwichKeyMap& key_map) {
    std::vector<SandwichKeyRows> rows = key_map.blocks();
    for (SandwichKeyRows& block : rows) {
        for (std::vector<sampling::SandwichAxisRef>& targets : block.support_flip_targets) {
            targets.clear();
        }
    }
    return SandwichKeyMap(std::move(rows), key_map.num_presampled());
}

// Data qubits of a block: the targets of the T gates on its opening lines.
std::vector<uint32_t> opening_qubits(const Circuit& circuit, const HirModule& hir,
                                     const SandwichBlock& block) {
    std::set<uint32_t> qubits;
    for (const size_t op : block.opening_ops) {
        for (const uint32_t line : hir.source_map.at(op)) {
            for (const AstNode& node : circuit.nodes) {
                if (node.source_line == line &&
                    (node.gate == GateType::T || node.gate == GateType::T_DAG)) {
                    for (const Target& target : node.targets) {
                        qubits.insert(target.value());
                    }
                }
            }
        }
    }
    return {qubits.begin(), qubits.end()};
}

// Distinct keys of the noiseless realization and of each single error of
// the given kinds on a data qubit just before a block opens.
std::vector<std::vector<SandwichKey>> single_error_keys(const Circuit& circuit,
                                                        const HirModule& hir,
                                                        const std::vector<Pauli1>& paulis) {
    const SamplingPlan plan = sampling::plan_sampling(hir);
    const SandwichKeyMap key_map = compute_sandwich_key_map(hir);
    std::vector<std::vector<SandwichKey>> members{
        key_map.keys_for(presampled_values(hir, plan, {}))};
    for (const SandwichBlock& block : find_sandwich_blocks(hir)) {
        for (const uint32_t qubit : opening_qubits(circuit, hir, block)) {
            for (const Pauli1 pauli : paulis) {
                const FiredChannel fired =
                    find_channel(circuit, hir, 0, block.opening_ops.front(), qubit, pauli, true);
                std::vector<SandwichKey> keys =
                    key_map.keys_for(presampled_values(hir, plan, {fired}));
                if (std::ranges::find(members, keys) == members.end()) {
                    members.push_back(std::move(keys));
                }
            }
        }
    }
    return members;
}

ProgramFamily bench_family(const Circuit& circuit, const HirModule& hir,
                           const std::vector<Pauli1>& paulis,
                           const SamplingPlanOptions& options = {}) {
    const std::vector<std::vector<SandwichKey>> members = single_error_keys(circuit, hir, paulis);
    return sampling::build_program_family(hir, members, options);
}

uint64_t member_shots(const FamilySamplingStats& stats) {
    uint64_t total = 0;
    for (const uint64_t count : stats.shots_per_member) {
        total += count;
    }
    return total;
}

std::vector<uint8_t> all_detectors(const HirModule& hir) {
    return std::vector<uint8_t>(hir.num_detectors, 1);
}

}  // namespace

TEST_CASE("The sandwich key map matches the reference keys on toy blocks", "[sandwich]") {
    const auto [name, text] = GENERATE(table<std::string, std::string>({
        {"closed T block", std::string(kNoisyBlock) + "MX 0 1 2\n"},
        {"closed mixed block",
         "RX 3\nDEPOLARIZE1(0.1) 1\nT_DAG 0 1\nT 2\nDEPOLARIZE1(0.1) 0 3\nCX 3 0 3 1 3 2\n"
         "MX 3\nRX 3\nDEPOLARIZE1(0.1) 2\nCX 3 0 3 1 3 2\nT 0 1\nT_DAG 2\nM 0 1 2\n"},
        {"terminal block",
         "RX 3\nDEPOLARIZE1(0.1) 0\nT 0 1 2\nCX 3 0 3 1 3 2\nDEPOLARIZE1(0.1) 1 3\nMX 3\n"
         "RX 3\nDEPOLARIZE1(0.1) 2\n"},
        {"two blocks around a data reset",
         std::string(kNoisyBlock) + "R 0 1 2\n" + kNoisyBlock + "MX 0 1 2\n"},
        {"two blocks linked by feedback", kFeedbackBlocks},
    }));
    CAPTURE(name);
    const HirModule hir = hir_of(text);
    const SamplingPlan plan = sampling::plan_sampling(hir);
    const SandwichKeyMap key_map = compute_sandwich_key_map(hir);
    REQUIRE(key_map.num_presampled() == ExecutablePlan(plan).num_presampled_symbols());
    REQUIRE(key_map.blocks().size() == find_sandwich_blocks(hir).size());

    size_t flipped = 0;
    size_t supported = 0;
    for (const std::vector<FiredChannel>& fired : small_realizations(hir)) {
        const std::vector<SandwichKey> keys = check_keys_match_reference(hir, plan, key_map, fired);
        flipped += any_flag(keys, false) ? 1 : 0;
        supported += any_flag(keys, true) ? 1 : 0;
    }
    CHECK(flipped > 0);
    CHECK(supported > 0);
}

TEST_CASE("A supported axis carried by feedback reverses a later block axis", "[sandwich]") {
    const HirModule hir = hir_of(kFeedbackBlocks);
    const std::vector<SandwichBlock> blocks = find_sandwich_blocks(hir);
    REQUIRE(blocks.size() == 2);
    REQUIRE(blocks[0].supported);
    REQUIRE(blocks[1].supported);
    const SamplingPlan plan = sampling::plan_sampling(hir);
    const SandwichKeyMap key_map = compute_sandwich_key_map(hir);
    // Z0 reaches the second block's Z4 axis through the conditional X.
    CHECK(key_map.blocks()[0].support_flip_targets[0] ==
          std::vector<sampling::SandwichAxisRef>{{1, 0}});
    const SandwichKeyMap uncorrected = without_support_targets(key_map);

    // The correction turns the X measurement random, so these classes have
    // no specialized plan; the keys must still name them exactly so that no
    // shot of theirs reaches a member built for another class.
    size_t corrected = 0;
    for (const std::vector<FiredChannel>& fired : small_realizations(hir)) {
        const std::vector<uint8_t> values = presampled_values(hir, plan, fired);
        const std::vector<SandwichKey> keys = key_map.keys_for(values);
        if (keys == uncorrected.keys_for(values)) {
            continue;
        }
        ++corrected;
        CHECK(keys == keys_for(hir, fired));
    }
    CHECK(corrected > 0);
}

TEST_CASE("The sandwich key map matches the reference keys on the cultivation benches",
          "[sandwich]") {
    const auto [name, trials] = GENERATE(table<std::string, int>({
        {"cultivation_d3_bench.stim", 200},
        {"cultivation_d5_bench.stim", 100},
    }));
    CAPTURE(name);
    const HirModule hir = trace(parse_file(fixture_path(name)));
    const SamplingPlan plan = sampling::plan_sampling(hir);
    const SandwichKeyMap key_map = compute_sandwich_key_map(hir);
    Xoshiro256PlusPlus rng(20261020);
    size_t flipped = 0;
    size_t supported = 0;
    for (int trial = 0; trial < trials; ++trial) {
        const std::vector<FiredChannel> fired = random_realization(hir, rng, 3);
        INFO("trial " << trial);
        const std::vector<SandwichKey> keys = check_keys_match_reference(hir, plan, key_map, fired);
        flipped += any_flag(keys, false) ? 1 : 0;
        supported += any_flag(keys, true) ? 1 : 0;
    }
    CHECK(flipped > 0);
    CHECK(supported > 0);
}

TEST_CASE("Plans specialized for key map classes replay d3 bench records", "[sandwich]") {
    const HirModule hir =
        without_readout_noise(trace(parse_file(fixture_path("cultivation_d3_bench.stim"))));
    const SamplingPlan plan = sampling::plan_sampling(hir);
    const SandwichKeyMap key_map = compute_sandwich_key_map(hir);
    Xoshiro256PlusPlus rng(5);
    for (int trial = 0; trial < 30; ++trial) {
        const std::vector<FiredChannel> fired = random_realization(hir, rng, 3);
        INFO("trial " << trial);
        PlanPair plans(hir, fired, key_map.keys_for(presampled_values(hir, plan, fired)));
        check_sampled_rows_agree(plans, 10);
    }
}

TEST_CASE("The sandwich key map rejects inputs it cannot classify", "[sandwich]") {
    const HirModule hir = hir_of(std::string(kNoisyBlock) + "MX 0 1 2\n");
    const SandwichKeyMap key_map = compute_sandwich_key_map(hir);

    SECTION("an HIR that already holds sandwich ops") {
        const HirModule specialized = specialize(hir, key_map.zero_keys());
        CHECK_THROWS_WITH(compute_sandwich_key_map(specialized),
                          ContainsSubstring("without sandwich operations"));
    }

    SECTION("values and ordinals of the wrong shape") {
        CHECK_THROWS_AS(key_map.keys_for(std::vector<uint8_t>(key_map.num_presampled() + 1, 0)),
                        std::invalid_argument);
        CHECK_THROWS_AS(key_map.keys_for_fired(std::vector<uint32_t>{key_map.num_presampled()}),
                        std::invalid_argument);
        CHECK_THROWS_AS(key_map.keys_for_fired(std::vector<uint32_t>{0, 0}), std::invalid_argument);
    }

    SECTION("rows that are ragged or name earlier blocks") {
        std::vector<SandwichKeyRows> rows = key_map.blocks();
        rows[0].internal_support_terms.pop_back();
        CHECK_THROWS_WITH(SandwichKeyMap(rows, key_map.num_presampled()),
                          ContainsSubstring("different widths"));
        rows = key_map.blocks();
        rows[0].support_flip_targets[0].push_back({0, 1});
        CHECK_THROWS_WITH(SandwichKeyMap(rows, key_map.num_presampled()),
                          ContainsSubstring("later blocks"));
    }
}

TEST_CASE("Program families reject members they cannot dispatch", "[sandwich]") {
    const HirModule hir = hir_of(std::string(kNoisyBlock) + "MX 0 1 2\n");
    const SandwichKeyMap key_map = compute_sandwich_key_map(hir);
    const std::vector<SandwichKey> noiseless = key_map.zero_keys();
    const auto member = [&](const HirModule& source, const std::vector<SandwichKey>& keys) {
        return FamilyMember{keys,
                            ExecutablePlan(sampling::plan_sampling(specialize(source, keys)))};
    };
    const auto family = [&](std::vector<FamilyMember> members) {
        return ProgramFamily(ExecutablePlan(sampling::plan_sampling(hir)), key_map,
                             std::move(members));
    };

    SECTION("a valid family finds its members") {
        std::vector<SandwichKey> flipped = noiseless;
        flipped[0].axis_flip[1] = 1;
        std::vector<FamilyMember> members;
        members.push_back(member(hir, noiseless));
        members.push_back(member(hir, flipped));
        const ProgramFamily valid = family(std::move(members));
        CHECK(valid.find(noiseless) == std::optional<size_t>{0});
        CHECK(valid.find(flipped) == std::optional<size_t>{1});
        flipped[0].internal_support[0] = 1;
        CHECK_FALSE(valid.find(flipped).has_value());
    }

    SECTION("repeated keys") {
        std::vector<FamilyMember> members;
        members.push_back(member(hir, noiseless));
        members.push_back(member(hir, noiseless));
        CHECK_THROWS_WITH(family(std::move(members)), ContainsSubstring("repeats the keys"));
    }

    SECTION("keys that do not match the blocks") {
        std::vector<FamilyMember> members;
        members.push_back(member(hir, noiseless));
        members[0].keys[0].axis_flip.pop_back();
        CHECK_THROWS_WITH(family(std::move(members)), ContainsSubstring("one 0 or 1 flag"));
    }

    SECTION("a plan of another circuit") {
        const HirModule other = hir_of(std::string(kNoisyBlock) + "MX 0 1\n");
        std::vector<FamilyMember> members;
        members.push_back(member(other, noiseless));
        CHECK_THROWS_WITH(family(std::move(members)), ContainsSubstring("outputs that differ"));
    }

    SECTION("a plan with different noise") {
        std::string text = std::string(kNoisyBlock) + "MX 0 1 2\n";
        text.replace(text.find("DEPOLARIZE1(0.1) 1"), 18, "DEPOLARIZE1(0.2) 1");
        std::vector<FamilyMember> members;
        members.push_back(member(hir_of(text), noiseless));
        CHECK_THROWS_WITH(family(std::move(members)), ContainsSubstring("presampled noise"));
    }
}

TEST_CASE("Family survivor sampling keeps only accepted rows", "[sandwich]") {
    const Circuit circuit = parse_file(fixture_path("cultivation_d3_bench.stim"));
    const HirModule hir = trace(circuit);
    const std::vector<uint8_t> postselected = all_detectors(hir);
    SamplingPlanOptions options;
    options.postselection_mask = postselected;
    const ProgramFamily family = bench_family(circuit, hir, {Pauli1::X}, options);
    constexpr uint32_t kShots = 500;

    CHECK_THROWS_WITH(sampling::sample_family(family, 1, 0), ContainsSubstring("postselection"));
    FamilySamplingStats stats;
    const SamplingSurvivorResult kept =
        sampling::sample_family_survivors(family, kShots, 41, true, &stats);
    const SamplingSurvivorResult counted = sampling::sample_family_survivors(family, kShots, 41);
    CHECK(member_shots(stats) + stats.fallback_shots == kShots);
    CHECK(kept.total_shots == kShots);
    REQUIRE(kept.passed_shots > 0);
    CHECK(kept.passed_shots < kShots);
    CHECK(counted.passed_shots == kept.passed_shots);
    CHECK(counted.logical_errors == kept.logical_errors);
    CHECK(counted.observable_ones == kept.observable_ones);
    CHECK(counted.measurements.empty());
    REQUIRE(kept.detectors.size() ==
            static_cast<size_t>(kept.passed_shots) * family.generic().num_detectors());
    CHECK(kept.measurements.size() ==
          static_cast<size_t>(kept.passed_shots) * family.generic().num_visible_records());
    CHECK(std::ranges::all_of(kept.detectors, [](uint8_t bit) { return bit == 0; }));
}
