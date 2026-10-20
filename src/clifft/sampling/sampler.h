#pragma once

#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/results.h"
#include "clifft/util/config.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clifft::sampling {

class ProgramFamily;

inline constexpr double kDefaultBatchTuningBudgetSeconds = 0.25;

// Sampling pipeline:
//   optimized HirModule -> SamplingPlan -> ExecutablePlan -> Executor -> results
// Planning produces semantic actions, lowering prepares fixed CPU descriptors,
// Executor owns mutable state for one shot, BatchExecutor owns a packed lane
// group, and the functions below select a prepared path and collect outputs.

// Explicitly partitions sampling workers across independent shots and the
// coefficient kernels within each shot. When supplied, this layout overrides
// the automatic policy selected by threads, whose value is then ignored. The
// minimum active width is evaluated during executor setup and at coefficient
// actions; it does not introduce work inside coefficient loops.
struct ThreadLayout {
    uint32_t shot_workers = 1;
    uint32_t intra_shot_workers = 1;
    uint32_t intra_shot_min_active_width = kDefaultIntraShotMinActiveWidth;
};

// Samples a fixed number of shots into row-major visible-record storage. The
// plan and executor are prepared once, and all output is allocated before the
// first shot enters hot execution. Plans with presampled symbols are rejected
// until their sampling distribution is part of the executable contract.
// threads is a total worker budget. threads=0 selects the implementation-
// reported hardware concurrency; the public Python API spells this as
// threads="auto". Automatic scheduling uses either cross-shot or intra-shot
// workers, not a hybrid layout. batch_size is empty for adaptive selection,
// one for scalar execution, or an explicit packed lane-capacity limit. A seed
// replays within one execution configuration; scalar and packed modes, or two
// packed capacities, are statistically equivalent but need not return the same
// individual rows. A tuning budget opts into calibration before production,
// requires an empty batch_size, and bounds extra trial work with a soft deadline.
// Trial shots are excluded from production results and survivor counts.
[[nodiscard]] std::vector<uint8_t> sample_records(
    const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
    uint32_t threads = 1, std::optional<ThreadLayout> thread_layout = std::nullopt,
    std::optional<uint32_t> batch_size = std::nullopt,
    std::optional<double> tuning_budget_seconds = std::nullopt);

// Replays each row-major visible record and returns its joint log probability.
// Unreachable records map to the lowest finite double because release builds
// assume finite arithmetic. Plans with presampled symbols or hidden records
// are rejected because this API does not yet marginalize over either source
// of hidden stochastic state.
[[nodiscard]] std::vector<double> record_log_probabilities(const ExecutablePlan& plan,
                                                           std::span<const uint8_t> forced_records,
                                                           size_t num_records);

[[nodiscard]] SamplingResult sample(const ExecutablePlan& plan, uint32_t shots,
                                    std::optional<uint64_t> seed = std::nullopt,
                                    uint32_t threads = 1,
                                    std::optional<ThreadLayout> thread_layout = std::nullopt,
                                    std::optional<uint32_t> batch_size = std::nullopt,
                                    std::optional<double> tuning_budget_seconds = std::nullopt);

[[nodiscard]] SamplingSurvivorResult sample_survivors(
    const ExecutablePlan& plan, uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
    bool keep_records = false, uint32_t threads = 1,
    std::optional<ThreadLayout> thread_layout = std::nullopt,
    std::optional<uint32_t> batch_size = std::nullopt,
    std::optional<double> tuning_budget_seconds = std::nullopt);

[[nodiscard]] SamplingResult sample_k(const ExecutablePlan& plan, uint32_t shots, uint32_t k,
                                      std::optional<uint64_t> seed = std::nullopt,
                                      uint32_t threads = 1,
                                      std::optional<ThreadLayout> thread_layout = std::nullopt,
                                      std::optional<uint32_t> batch_size = std::nullopt,
                                      std::optional<double> tuning_budget_seconds = std::nullopt);

[[nodiscard]] SamplingSurvivorResult sample_k_survivors(
    const ExecutablePlan& plan, uint32_t shots, uint32_t k,
    std::optional<uint64_t> seed = std::nullopt, bool keep_records = false, uint32_t threads = 1,
    std::optional<ThreadLayout> thread_layout = std::nullopt,
    std::optional<uint32_t> batch_size = std::nullopt,
    std::optional<double> tuning_budget_seconds = std::nullopt);

// Dispatch counts from one family sampling call. The call overwrites them.
struct FamilySamplingStats {
    // Shots run on each member, indexed like ProgramFamily::member.
    std::vector<uint64_t> shots_per_member;
    // Shots whose keys matched no member and ran on the generic plan.
    uint64_t fallback_shots = 0;
    // When set on entry, `dispatched` receives one entry per shot: the member
    // the shot ran on, or nullopt for the generic plan. This lets a caller
    // audit dispatch against the drawn noise.
    bool record_dispatch = false;
    std::vector<std::optional<uint32_t>> dispatched;
};

// Samples with per-shot sandwich dispatch. Each shot draws its presampled
// noise from the generic plan's distributions, computes its keys with the
// family's key map, and runs the matching member's plan, or the generic plan
// when none matches, on those presampled values. Outputs follow sample():
// row-major visible records, detectors, observables and expectation values.
// Plans that postselect must use sample_family_survivors.
//
// Execution is single-threaded and uses the scalar executor, so this takes
// no thread or batch options. Executors for every plan are built before the
// first shot. Noise and execution draw from separate per-shot streams
// derived from the seed, so a seed reproduces this function's rows, but not
// the rows sample() returns for the same seed.
[[nodiscard]] SamplingResult sample_family(const ProgramFamily& family, uint32_t shots,
                                           std::optional<uint64_t> seed = std::nullopt,
                                           FamilySamplingStats* stats = nullptr);

// Survivor sampling with per-shot sandwich dispatch. Counts and kept rows
// follow sample_survivors(); dispatch and seeding follow sample_family().
[[nodiscard]] SamplingSurvivorResult sample_family_survivors(
    const ProgramFamily& family, uint32_t shots, std::optional<uint64_t> seed = std::nullopt,
    bool keep_records = false, FamilySamplingStats* stats = nullptr);

// The presampled values, by ordinal, that sample_family and
// sample_family_survivors draw for `shot` under `seed`.
[[nodiscard]] std::vector<uint8_t> family_presampled_values(const ProgramFamily& family,
                                                            uint64_t seed, uint32_t shot);

}  // namespace clifft::sampling
