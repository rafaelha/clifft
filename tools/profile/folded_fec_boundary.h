// Included inside the pinned research sampler, at its resolved syndrome boundary.
void recover_folded_fec(Boundary& state, std::span<const uint8_t> records,
                        std::span<const Mask> x_rows, std::span<const Mask> z_rows, Mask logical_x,
                        Mask logical_z, size_t width) {
    const auto bits = [](Mask value, size_t count) {
        std::cout << '"';
        for (size_t i = 0; i < count; ++i)
            std::cout << unsigned((value >> i) & 1);
        std::cout << '"';
    };
    const auto bloch = [&]() {
        const auto xy = std::conj(state.logical[0]) * state.logical[1];
        std::cout << '[' << 2 * xy.real() << ',' << 2 * xy.imag() << ','
                  << std::norm(state.logical[0]) - std::norm(state.logical[1]) << ']';
    };
    std::cout << "{\"kind\":\"recovery\",\"records\":\"";
    for (auto value : records)
        std::cout << unsigned(value);
    std::cout << "\",\"x_syndrome\":";
    bits(state.syndrome, x_rows.size());
    std::cout << ",\"z_syndrome\":\"";
    for (auto row : z_rows)
        std::cout << parity(state.offset & row);
    std::cout << "\",\"logical_x\":";
    bits(logical_x, width);
    std::cout << ",\"logical_z\":";
    bits(logical_z, width);
    std::cout << ",\"bloch_before\":";
    bloch();
    std::cout << "}\n" << std::flush;

    // This offline host boundary deliberately includes decoder and IPC cost.
    std::string x_bits, z_bits;
    if (!(std::cin >> x_bits >> z_bits) || x_bits.size() != width || z_bits.size() != width)
        throw std::invalid_argument("missing FEC recovery");
    Mask x = 0, z = 0;
    for (size_t i = 0; i < width; ++i) {
        if ((x_bits[i] != '0' && x_bits[i] != '1') || (z_bits[i] != '0' && z_bits[i] != '1'))
            throw std::invalid_argument("nonbinary FEC recovery");
        x |= Mask(x_bits[i] == '1') << i;
        z |= Mask(z_bits[i] == '1') << i;
    }
    for (size_t i = 0; i < x_rows.size(); ++i)
        if (parity(z & x_rows[i]) != bool((state.syndrome >> i) & 1) ||
            parity((state.offset ^ x) & z_rows[i]))
            throw std::invalid_argument("FEC recovery does not clear the measured syndrome");
    state.offset ^= x;
    state.syndrome = 0;
    if (parity(x & logical_z))
        std::swap(state.logical[0], state.logical[1]);
    if (parity(z & logical_x))
        state.logical[1] *= -1;
    std::cout << "{\"kind\":\"recovered\",\"bloch_after\":";
    bloch();
    std::cout << "}\n";
}
