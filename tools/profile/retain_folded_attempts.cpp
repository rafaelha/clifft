// Observe the pinned research implementation without changing its state updates.
#define main original_folded_benchmark_main
#include "sample_folded_protocol.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc != 4)
        throw std::invalid_argument("usage: retain_folded_attempts bundle shots seed");
    const std::filesystem::path directory(argv[1]);
    const auto shots = std::stoull(argv[2]), seed = std::stoull(argv[3]);
    const auto start = std::chrono::steady_clock::now();
    Protocol sampler(directory, seed);
    const auto plan =
        compile(read_text(directory / "prefix.stim"), "off", sampler.prefix_detectors(), false);
    sampler.certify_prefix(plan);
    ExecutablePlan executable(plan);
    Executor prefix(executable, seed);
    std::cout << std::setprecision(17);
    std::cout << "{\"kind\":\"setup\",\"seconds\":"
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << ",\"prefix_width\":" << plan.peak_active_width
              << ",\"lookup_bytes\":" << sampler.lookup_bytes()
              << ",\"workspace_bytes\":" << sampler.workspace_bytes() << "}\n";
    for (size_t shot = 0; shot < shots; ++shot) {
        const auto begun = std::chrono::steady_clock::now();
        sampler.run(prefix, false);
        const auto executed = std::chrono::steady_clock::now();
        std::cout << "{\"kind\":\"attempt\",\"index\":" << shot << ",\"records\":";
        print_bits(sampler.result().records);
        std::cout << ",\"detectors\":";
        print_bits(sampler.result().detectors);
        std::cout << ",\"observable\":" << unsigned(sampler.result().observable)
                  << ",\"accepted\":" << (sampler.result().accepted ? "true" : "false")
                  << ",\"conditional_log_probability\":"
                  << sampler.result().conditional_log_probability << ",\"prefix_faults\":[";
        bool first = true;
        for (size_t i = 0; i < plan.presampled_noise_sites.size(); ++i)
            for (size_t j = 0; j < plan.presampled_noise_sites[i].outcomes.size(); ++j)
                if (prefix.symbols()[index(plan.presampled_noise_sites[i].outcomes[j].symbol)]) {
                    std::cout << (first ? "" : ",") << '[' << i << ',' << j + 1 << ']';
                    first = false;
                }
        std::cout << ']';
        const auto faults = [&](const char* name, std::span<const size_t> choices) {
            std::cout << ",\"" << name << "\":[";
            bool first_choice = true;
            for (size_t i = 0; i < choices.size(); ++i)
                if (choices[i]) {
                    std::cout << (first_choice ? "" : ",") << '[' << i << ',' << choices[i] << ']';
                    first_choice = false;
                }
            std::cout << ']';
        };
        faults("suffix_faults", sampler.result().fault_choices);
        faults("earlier_faults", sampler.earlier_faults());
        faults("growth_faults", sampler.growth_faults());
        std::cout << ",\"execution_seconds\":"
                  << std::chrono::duration<double>(executed - begun).count() << ",\"seconds\":"
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count()
                  << "}\n";
    }
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "{\"kind\":\"end\",\"native_peak_rss_bytes\":" << usage.ru_maxrss * 1024
              << ",\"normalization_error\":" << sampler.normalization_error()
              << ",\"continuation_error\":" << sampler.continuation_error() << "}\n";
}
