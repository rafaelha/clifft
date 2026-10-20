// Checks recognition of T-sandwich blocks and the dormant-subspace closure
// test, using stim-text circuits traced through the frontend and a few
// hand-built HIR modules for ops the frontend cannot place between gates.

#include "clifft/circuit/circuit.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/frontend/hir.h"
#include "clifft/optimizer/sandwich_blocks.h"
#include "clifft/util/symplectic.h"

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using namespace clifft;

namespace {

#ifndef CLIFFT_FIXTURES_DIR
#define CLIFFT_FIXTURES_DIR "tests/fixtures"
#endif

// A three-axis T sandwich around an ancilla check, assembled from replaceable
// pieces so each negative case changes exactly one of them. Qubits 0-2 carry
// the T layers. Qubit 3 is the ancilla: its X measurement picks up the data X
// parity through the ladder, so the check anticommutes with each Z axis.
struct SandwichText {
    std::string prep = "RX 3\n";
    std::string opening = "T 0 1 2\n";
    std::string ladder = "CX 3 0 3 1 3 2\n";
    std::string before_check;
    std::string check = "MX 3\n";
    std::string after_check = "RX 3\n";
    std::string closing = "T_DAG 0 1 2\n";
    std::string tail = "MX 0 1 2\n";

    [[nodiscard]] std::string str() const {
        return prep + opening + ladder + before_check + check + after_check + ladder + closing +
               tail;
    }
};

HirModule hir_of(const std::string& text) {
    return trace(parse(text));
}

HirModule hir_of(const SandwichText& text) {
    return hir_of(text.str());
}

size_t count_ops(const HirModule& hir, OpType type) {
    return static_cast<size_t>(
        std::count_if(hir.ops.begin(), hir.ops.end(),
                      [type](const HeisenbergOp& op) { return op.op_type() == type; }));
}

std::vector<size_t> t_gate_indices(const HirModule& hir) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < hir.ops.size(); ++i) {
        if (hir.ops[i].op_type() == OpType::T_GATE) {
            indices.push_back(i);
        }
    }
    return indices;
}

bool same_unsigned_axis(const HirModule& hir, size_t a, size_t b) {
    return hir.destab_mask(hir.ops[a]) == hir.destab_mask(hir.ops[b]) &&
           hir.stab_mask(hir.ops[a]) == hir.stab_mask(hir.ops[b]);
}

// Structural invariants every reported block must satisfy, independent of the
// circuit that produced it. A block with no closing ops is terminal and its
// interior runs to the end of the HIR.
void require_well_formed(const HirModule& hir, const SandwichBlock& block) {
    REQUIRE_FALSE(block.opening_ops.empty());
    const bool terminal = block.closing_ops.empty();
    REQUIRE((terminal || block.closing_ops.size() == block.opening_ops.size()));
    REQUIRE(block.axis_anticommutes_with_check.size() == block.opening_ops.size());
    REQUIRE(std::is_sorted(block.opening_ops.begin(), block.opening_ops.end()));
    REQUIRE(block.opening_ops.back() < block.check_op);
    for (const size_t op : block.opening_ops) {
        REQUIRE(hir.ops[op].op_type() == OpType::T_GATE);
    }
    for (size_t i = 0; i < block.closing_ops.size(); ++i) {
        const HeisenbergOp& open = hir.ops[block.opening_ops[i]];
        const HeisenbergOp& close = hir.ops[block.closing_ops[i]];
        REQUIRE(close.op_type() == OpType::T_GATE);
        REQUIRE(block.check_op < block.closing_ops[i]);
        REQUIRE(same_unsigned_axis(hir, block.opening_ops[i], block.closing_ops[i]));
        REQUIRE(hir.sign(open) == hir.sign(close));
        REQUIRE(open.is_dagger() != close.is_dagger());
    }
    for (size_t i = 0; i < block.opening_ops.size(); ++i) {
        for (size_t j = i + 1; j < block.opening_ops.size(); ++j) {
            const HeisenbergOp& a = hir.ops[block.opening_ops[i]];
            const HeisenbergOp& b = hir.ops[block.opening_ops[j]];
            REQUIRE_FALSE(anti_commute(hir.destab_mask(a), hir.stab_mask(a), hir.destab_mask(b),
                                       hir.stab_mask(b)));
        }
    }
    const HeisenbergOp& check = hir.ops[block.check_op];
    REQUIRE(check.op_type() == OpType::MEASURE);
    REQUIRE_FALSE(check.is_hidden());
    for (const size_t echo : block.echo_ops) {
        REQUIRE(hir.ops[echo].op_type() == OpType::MEASURE);
        REQUIRE(hir.ops[echo].is_hidden());
        REQUIRE(same_unsigned_axis(hir, echo, block.check_op));
    }
    for (const size_t noise : block.interior_noise_ops) {
        REQUIRE(hir.ops[noise].op_type() == OpType::NOISE);
    }
    for (const size_t cond : block.interior_conditional_ops) {
        REQUIRE(hir.ops[cond].op_type() == OpType::CONDITIONAL_PAULI);
    }
    REQUIRE(block.axis_anticommutes_with_check ==
            axis_anticommutation(hir, block, hir.mask_view(check)));
    REQUIRE(std::count(block.axis_anticommutes_with_check.begin(),
                       block.axis_anticommutes_with_check.end(), uint8_t{1}) > 0);

    // Recount the interior directly from the ops.
    const size_t interior_end =
        terminal ? hir.ops.size()
                 : *std::min_element(block.closing_ops.begin(), block.closing_ops.end());
    size_t visible = 0;
    size_t noise = 0;
    size_t conditional = 0;
    size_t hidden = 0;
    for (size_t i = block.opening_ops.back() + 1; i < interior_end; ++i) {
        const HeisenbergOp& op = hir.ops[i];
        REQUIRE(op.op_type() != OpType::T_GATE);
        REQUIRE(op.op_type() != OpType::PHASE_ROTATION);
        REQUIRE(op.op_type() != OpType::INSTRUMENT);
        REQUIRE(op.op_type() != OpType::EXP_VAL);
        if (op.op_type() == OpType::MEASURE) {
            if (op.is_hidden()) {
                ++hidden;
            } else {
                ++visible;
            }
        } else if (op.op_type() == OpType::NOISE) {
            ++noise;
        } else if (op.op_type() == OpType::CONDITIONAL_PAULI) {
            ++conditional;
            for (const size_t axis : block.opening_ops) {
                REQUIRE_FALSE(anti_commute(hir.destab_mask(op), hir.stab_mask(op),
                                           hir.destab_mask(hir.ops[axis]),
                                           hir.stab_mask(hir.ops[axis])));
            }
        }
    }
    REQUIRE(visible == 1);
    CHECK(hidden == block.echo_ops.size());
    CHECK(noise == block.interior_noise_ops.size());
    CHECK(conditional == block.interior_conditional_ops.size());
}

std::vector<SandwichBlock> blocks_of(const HirModule& hir) {
    std::vector<SandwichBlock> blocks = find_sandwich_blocks(hir);
    for (const SandwichBlock& block : blocks) {
        require_well_formed(hir, block);
    }
    return blocks;
}

std::string fixture_path(const std::string& name) {
    return std::string(CLIFFT_FIXTURES_DIR) + "/" + name;
}

// The circuit instruction that produced an HIR op, so fixture assertions can
// name a check by what it measures rather than by an op index.
const AstNode& source_node(const Circuit& circuit, const HirModule& hir, size_t op_index) {
    REQUIRE(hir.source_map[op_index].size() == 1);
    const uint32_t line = hir.source_map[op_index][0];
    const auto it = std::find_if(circuit.nodes.begin(), circuit.nodes.end(),
                                 [line](const AstNode& node) { return node.source_line == line; });
    REQUIRE(it != circuit.nodes.end());
    return *it;
}

void append_hidden_measure(HirModule& hir, uint64_t destab, uint64_t stab, MeasRecordIdx idx) {
    test::append_measure(hir, destab, stab, false, idx).set_hidden(true);
}

// Four qubits in |0000>: T and T_DAG layers on Z0 and Z1 around a visible
// X0X1 measurement, which anticommutes with both axes. `add_interior` inserts
// ops between the check and the closing layer.
HirModule manual_block(const std::function<void(HirModule&)>& add_interior = {}) {
    HirModule hir(4, 8);
    test::append_tgate(hir, 0, test::Z(0), false, false);
    test::append_tgate(hir, 0, test::Z(1), false, false);
    test::append_measure(hir, test::X(0) | test::X(1), 0, false, MeasRecordIdx{0});
    hir.num_measurements = 1;
    if (add_interior) {
        add_interior(hir);
    }
    test::append_tgate(hir, 0, test::Z(0), false, true);
    test::append_tgate(hir, 0, test::Z(1), false, true);
    return hir;
}

}  // namespace

TEST_CASE("Sandwich blocks recognize a three axis block around an ancilla check", "[sandwich]") {
    // RX 3 contributes a hidden measurement and a correction before the layer,
    // so the T ops start at index 2. Data qubits stay in |0>, which every
    // opening axis commutes with.
    const HirModule hir = hir_of(SandwichText{});
    REQUIRE(hir.ops.size() == 14);

    const std::vector<SandwichBlock> blocks = blocks_of(hir);
    REQUIRE(blocks.size() == 1);
    const SandwichBlock& block = blocks[0];

    CHECK(block.opening_ops == std::vector<size_t>{2, 3, 4});
    CHECK(block.closing_ops == std::vector<size_t>{8, 9, 10});
    CHECK(block.check_op == 5);
    CHECK(block.echo_ops == std::vector<size_t>{6});
    CHECK(block.interior_noise_ops.empty());
    // The only conditional between the layers is the reset correction.
    CHECK(block.interior_conditional_ops == std::vector<size_t>{7});
    CHECK(block.axis_anticommutes_with_check == std::vector<uint8_t>{1, 1, 1});
    CHECK(block.supported);
    CHECK(block.active_width_at_opening == 0);

    SECTION("axis_anticommutation compares any mask against every opening axis") {
        // The hidden reset measurement shares the check Pauli.
        CHECK(axis_anticommutation(hir, block, hir.mask_view(hir.ops[block.echo_ops[0]])) ==
              std::vector<uint8_t>{1, 1, 1});
        // A closing gate repeats an opening axis, which commutes with all of them.
        CHECK(axis_anticommutation(hir, block, hir.mask_view(hir.ops[block.closing_ops[0]])) ==
              std::vector<uint8_t>{0, 0, 0});
    }
}

TEST_CASE("Sandwich blocks flag only the axes that anticommute with the check", "[sandwich]") {
    SandwichText text;
    text.ladder = "CX 3 0 3 1\n";

    const std::vector<SandwichBlock> blocks = blocks_of(hir_of(text));
    REQUIRE(blocks.size() == 1);
    CHECK(blocks[0].opening_ops.size() == 3);
    CHECK(blocks[0].axis_anticommutes_with_check == std::vector<uint8_t>{1, 1, 0});
    CHECK(blocks[0].supported);
}

TEST_CASE("Sandwich blocks pair a reordered closing layer by axis", "[sandwich]") {
    SandwichText text;
    text.closing = "T_DAG 2 1 0\n";

    const std::vector<SandwichBlock> blocks = blocks_of(hir_of(text));
    REQUIRE(blocks.size() == 1);
    // closing_ops follows the pairing, so it runs against the closing layer's order.
    CHECK(blocks[0].opening_ops == std::vector<size_t>{2, 3, 4});
    CHECK(blocks[0].closing_ops == std::vector<size_t>{10, 9, 8});
}

TEST_CASE("Sandwich blocks report interior noise and feedback", "[sandwich]") {
    SandwichText text;
    // Noise between the gates of a layer belongs to the layer and is not
    // reported. Noise between the layers is.
    text.opening = "T 0\nX_ERROR(0.01) 0\nT 1 2\nX_ERROR(0.02) 3\n";
    text.before_check = "DEPOLARIZE1(0.03) 3\n";
    text.check = "MX(0.1) 3\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
    // A Z-type correction commutes with every Z axis.
    text.after_check = "CZ rec[-1] 0\nRX 3\n";
    text.closing = "T_DAG 0\nDETECTOR rec[-1]\nX_ERROR(0.04) 1\nT_DAG 1 2\n";
    const HirModule hir = hir_of(text);
    REQUIRE(count_ops(hir, OpType::NOISE) == 4);
    REQUIRE(count_ops(hir, OpType::READOUT_NOISE) == 1);
    REQUIRE(count_ops(hir, OpType::DETECTOR) == 2);
    REQUIRE(count_ops(hir, OpType::OBSERVABLE) == 1);

    const std::vector<SandwichBlock> blocks = blocks_of(hir);
    REQUIRE(blocks.size() == 1);
    const SandwichBlock& block = blocks[0];
    CHECK(block.opening_ops.size() == 3);
    CHECK(block.interior_noise_ops.size() == 2);
    CHECK(block.interior_conditional_ops.size() == 2);
    CHECK(block.echo_ops.size() == 1);
    CHECK(block.supported);
    for (const size_t noise : block.interior_noise_ops) {
        CHECK(noise > block.opening_ops.back());
        CHECK(noise < *std::min_element(block.closing_ops.begin(), block.closing_ops.end()));
    }
}

TEST_CASE("Sandwich blocks do not split layers at noise or bookkeeping ops", "[sandwich]") {
    // Built by hand because the frontend never emits a detector or readout
    // noise before the first measurement they refer to.
    HirModule hir(4, 8);
    hir.noise_sites.push_back(NoiseSite{});
    hir.readout_noise.push_back(ReadoutNoiseEntry{0, 0.1, 0.1});
    hir.detector_targets.push_back({0});
    hir.observable_targets.push_back({0});
    hir.num_measurements = 1;

    test::append_tgate(hir, 0, test::Z(0), false, false);                            // 0
    hir.append_noise(NoiseSiteIdx{0});                                               // 1
    test::append_tgate(hir, 0, test::Z(1), false, false);                            // 2
    hir.append_readout_noise(ReadoutNoiseIdx{0});                                    // 3
    hir.append_detector(DetectorIdx{0});                                             // 4
    hir.append_observable(ObservableIdx{0}, 0);                                      // 5
    test::append_measure(hir, test::X(0) | test::X(1), 0, false, MeasRecordIdx{0});  // 6
    hir.append_detector(DetectorIdx{0});                                             // 7
    hir.append_observable(ObservableIdx{0}, 0);                                      // 8
    test::append_tgate(hir, 0, test::Z(0), false, true);                             // 9
    hir.append_readout_noise(ReadoutNoiseIdx{0});                                    // 10
    hir.append_detector(DetectorIdx{0});                                             // 11
    hir.append_observable(ObservableIdx{0}, 0);                                      // 12
    hir.append_noise(NoiseSiteIdx{0});                                               // 13
    test::append_tgate(hir, 0, test::Z(1), false, true);                             // 14

    const std::vector<SandwichBlock> blocks = blocks_of(hir);
    REQUIRE(blocks.size() == 1);
    CHECK(blocks[0].opening_ops == std::vector<size_t>{0, 2});
    CHECK(blocks[0].closing_ops == std::vector<size_t>{9, 14});
    CHECK(blocks[0].check_op == 6);
    // The noise op at index 1 sits inside the opening layer.
    CHECK(blocks[0].interior_noise_ops.empty());
    CHECK(blocks[0].supported);
}

TEST_CASE("Sandwich blocks reject disallowed ops between the layers", "[sandwich]") {
    REQUIRE(blocks_of(manual_block()).size() == 1);

    SECTION("a phase rotation") {
        const HirModule hir = manual_block(
            [](HirModule& h) { test::append_phase_rotation(h, 0, test::Z(2), false, 0.3); });
        REQUIRE(count_ops(hir, OpType::PHASE_ROTATION) == 1);
        CHECK(find_sandwich_blocks(hir).empty());
    }

    SECTION("an expectation value probe") {
        const HirModule hir = manual_block(
            [](HirModule& h) { test::append_exp_val(h, 0, test::Z(2), false, ExpValIdx{0}); });
        REQUIRE(count_ops(hir, OpType::EXP_VAL) == 1);
        CHECK(find_sandwich_blocks(hir).empty());
    }

    SECTION("an instrument") {
        const HirModule hir = manual_block([](HirModule& h) {
            h.instrument_sites.push_back(InstrumentSite{});
            h.append_instrument(InstrumentSiteIdx{0},
                                [](MutablePauliMaskView slot) { slot.z().words[0] = test::Z(2); });
        });
        REQUIRE(count_ops(hir, OpType::INSTRUMENT) == 1);
        CHECK(find_sandwich_blocks(hir).empty());
    }

    SECTION("a second visible measurement") {
        const HirModule hir = manual_block([](HirModule& h) {
            test::append_measure(h, test::X(0) | test::X(1), 0, false, MeasRecordIdx{1});
        });
        CHECK(find_sandwich_blocks(hir).empty());
    }

    SECTION("a hidden measurement of another Pauli") {
        const HirModule hir = manual_block(
            [](HirModule& h) { append_hidden_measure(h, test::X(2), 0, MeasRecordIdx{1}); });
        CHECK(find_sandwich_blocks(hir).empty());
    }

    SECTION("a conditional Pauli that anticommutes with an axis") {
        const HirModule hir = manual_block([](HirModule& h) {
            test::append_conditional(h, test::X(1), 0, false, ControllingMeasIdx{0});
        });
        CHECK(find_sandwich_blocks(hir).empty());
    }

    SECTION("a conditional Pauli that commutes with every axis") {
        const HirModule hir = manual_block([](HirModule& h) {
            test::append_conditional(h, 0, test::Z(0), false, ControllingMeasIdx{0});
        });
        const std::vector<SandwichBlock> blocks = blocks_of(hir);
        REQUIRE(blocks.size() == 1);
        CHECK(blocks[0].interior_conditional_ops == std::vector<size_t>{3});
    }

    SECTION("a hidden measurement of the check Pauli is an echo") {
        const HirModule hir = manual_block([](HirModule& h) {
            append_hidden_measure(h, test::X(0) | test::X(1), 0, MeasRecordIdx{1});
        });
        const std::vector<SandwichBlock> blocks = blocks_of(hir);
        REQUIRE(blocks.size() == 1);
        CHECK(blocks[0].echo_ops == std::vector<size_t>{3});
    }
}

TEST_CASE("Sandwich blocks reject layer pairs that are not inverses", "[sandwich]") {
    SandwichText text;
    // Runs on the traced module to confirm the case exercises the rule it names.
    std::function<void(const HirModule&)> precondition;

    SECTION("closing gates repeat the opening dagger flags") {
        text.closing = "T 0 1 2\n";
    }

    SECTION("closing layer has fewer gates") {
        text.closing = "T_DAG 0 1\n";
    }

    SECTION("closing layer acts on a different axis") {
        text.closing = "T_DAG 0 1 3\n";
    }

    SECTION("two opening gates share an axis") {
        text.opening = "T 0 0 1\n";
        text.closing = "T_DAG 0 0 1\n";
    }

    SECTION("a Pauli between the layers flips a closing mask sign") {
        text.after_check = "RX 3\nX 0\n";
        precondition = [](const HirModule& hir) {
            const std::vector<size_t> t_ops = t_gate_indices(hir);
            REQUIRE(t_ops.size() == 6);
            REQUIRE(hir.sign(hir.ops[t_ops[0]]) != hir.sign(hir.ops[t_ops[3]]));
        };
    }

    SECTION("opening axes anticommute") {
        text.opening = "T 0\nH 0\nT 0\nH 0\n";
        text.ladder = "CX 3 0\n";
        text.closing = "T_DAG 0\nH 0\nT_DAG 0\nH 0\n";
        text.tail = "MX 0\n";
        precondition = [](const HirModule& hir) {
            const std::vector<size_t> t_ops = t_gate_indices(hir);
            REQUIRE(t_ops.size() == 4);
            REQUIRE(
                anti_commute(hir.destab_mask(hir.ops[t_ops[0]]), hir.stab_mask(hir.ops[t_ops[0]]),
                             hir.destab_mask(hir.ops[t_ops[1]]), hir.stab_mask(hir.ops[t_ops[1]])));
        };
    }

    const HirModule hir = hir_of(text);
    if (precondition) {
        precondition(hir);
    }
    CHECK(find_sandwich_blocks(hir).empty());
}

TEST_CASE("Sandwich blocks reject stim circuits whose interior breaks the rules", "[sandwich]") {
    SandwichText text;
    OpType expected_op = OpType::NUM_OP_TYPES;
    size_t expected_count = 0;

    SECTION("a phase rotation between the layers") {
        text.before_check = "R_Z(0.3) 3\n";
        expected_op = OpType::PHASE_ROTATION;
        expected_count = 1;
    }

    SECTION("an expectation value between the layers") {
        text.before_check = "EXP_VAL Z3\n";
        expected_op = OpType::EXP_VAL;
        expected_count = 1;
    }

    SECTION("two visible measurements between the layers") {
        text.before_check = "MX 3\n";
    }

    SECTION("a feedback Pauli that anticommutes with an axis") {
        text.after_check = "RX 3\nCX rec[-1] 0\n";
        // Preparation correction, reset correction and the feedback.
        expected_op = OpType::CONDITIONAL_PAULI;
        expected_count = 3;
    }

    SECTION("a reset in another basis than the check") {
        text.after_check = "R 3\n";
    }

    SECTION("a check that commutes with every axis") {
        // Data qubits control the ladder, so X on the ancilla stays X3.
        text.ladder = "CX 0 3 1 3 2 3\n";
    }

    const HirModule hir = hir_of(text);
    if (expected_op != OpType::NUM_OP_TYPES) {
        REQUIRE(count_ops(hir, expected_op) == expected_count);
    }
    CHECK(find_sandwich_blocks(hir).empty());
}

TEST_CASE("Sandwich blocks skip a lone layer and resume after each closing layer", "[sandwich]") {
    SECTION("a lone injection layer before a block") {
        SandwichText text;
        text.prep = "RX 3\nT 3\nRX 3\n";

        const std::vector<SandwichBlock> blocks = blocks_of(hir_of(text));
        REQUIRE(blocks.size() == 1);
        CHECK(blocks[0].opening_ops.size() == 3);
    }

    SECTION("a closing layer is not reused as an opening layer") {
        // The second T layer would invert the first closing layer across a
        // second check, but the scan resumes after the first block.
        const std::string ladder = "CX 3 0 3 1 3 2\n";
        const std::string text = "RX 3\nT 0 1 2\n" + ladder + "MX 3\nRX 3\n" + ladder +
                                 "T_DAG 0 1 2\n" + ladder + "MX 3\nRX 3\n" + ladder +
                                 "T 0 1 2\nMX 0 1 2\n";
        const HirModule hir = hir_of(text);
        REQUIRE(count_ops(hir, OpType::T_GATE) == 9);

        const std::vector<SandwichBlock> blocks = blocks_of(hir);
        REQUIRE(blocks.size() == 1);
        CHECK(blocks[0].opening_ops == std::vector<size_t>{2, 3, 4});
        CHECK(blocks[0].closing_ops == std::vector<size_t>{8, 9, 10});
    }

    SECTION("consecutive blocks are reported in op order without overlap") {
        const std::string ladder = "CX 3 0 3 1 3 2\n";
        const std::string body = "T 0 1 2\n" + ladder + "MX 3\nRX 3\n" + ladder + "T_DAG 0 1 2\n";
        const HirModule hir = hir_of("RX 3\n" + body + "MX 0\n" + body + "MX 0 1 2\n");

        const std::vector<SandwichBlock> blocks = blocks_of(hir);
        REQUIRE(blocks.size() == 2);
        CHECK(blocks[0].supported);
        const size_t first_end =
            *std::max_element(blocks[0].closing_ops.begin(), blocks[0].closing_ops.end());
        CHECK(first_end < blocks[1].opening_ops.front());
    }

    SECTION("a terminal block can follow a closed block") {
        const std::string ladder = "CX 3 0 3 1 3 2\n";
        const std::string body = "T 0 1 2\n" + ladder + "MX 3\nRX 3\n" + ladder + "T_DAG 0 1 2\n";
        const HirModule hir = hir_of("RX 3\n" + body + "MX 0\nT 0 1 2\n" + ladder + "MX 3\n");

        const std::vector<SandwichBlock> blocks = blocks_of(hir);
        REQUIRE(blocks.size() == 2);
        CHECK_FALSE(blocks[0].closing_ops.empty());
        CHECK(blocks[1].closing_ops.empty());
        CHECK(blocks[0].closing_ops.back() < blocks[1].opening_ops.front());
    }
}

namespace {

// An opening layer, the ancilla ladder and a check with nothing quantum after
// it. `after_check` holds whatever follows the check.
struct TerminalText {
    std::string prep = "RX 3\n";
    std::string opening = "T 0 1 2\n";
    std::string ladder = "CX 3 0 3 1 3 2\n";
    std::string check = "MX 3\n";
    std::string after_check;

    [[nodiscard]] std::string str() const { return prep + opening + ladder + check + after_check; }
};

}  // namespace

TEST_CASE("Sandwich blocks recognize a terminal block with no closing layer", "[sandwich]") {
    TerminalText text;
    text.check = "MX(0.1) 3\n";
    text.after_check =
        "X_ERROR(0.02) 3\nDETECTOR rec[-1]\nCZ rec[-1] 0\nRX 3\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
    const HirModule hir = hir_of(text.str());
    REQUIRE(count_ops(hir, OpType::READOUT_NOISE) == 1);
    REQUIRE(count_ops(hir, OpType::NOISE) == 1);
    REQUIRE(count_ops(hir, OpType::DETECTOR) == 1);
    REQUIRE(count_ops(hir, OpType::OBSERVABLE) == 1);

    const std::vector<SandwichBlock> blocks = blocks_of(hir);
    REQUIRE(blocks.size() == 1);
    const SandwichBlock& block = blocks[0];
    CHECK(block.opening_ops == std::vector<size_t>{2, 3, 4});
    CHECK(block.closing_ops.empty());
    CHECK(block.check_op == 5);
    CHECK(block.axis_anticommutes_with_check == std::vector<uint8_t>{1, 1, 1});
    // The interior runs to the end of the HIR, so noise, feedback and the reset
    // after the check are all reported.
    CHECK(block.interior_noise_ops.size() == 1);
    CHECK(block.interior_conditional_ops.size() == 2);
    CHECK(block.echo_ops.size() == 1);
    CHECK(block.supported);
    CHECK(block.active_width_at_opening == 0);
}

TEST_CASE("Sandwich blocks apply the closure check to terminal blocks", "[sandwich]") {
    TerminalText text;
    text.prep = "RX 0 1 2 3\n";

    const std::vector<SandwichBlock> blocks = blocks_of(hir_of(text.str()));
    REQUIRE(blocks.size() == 1);
    CHECK(blocks[0].closing_ops.empty());
    CHECK_FALSE(blocks[0].supported);
}

TEST_CASE("Sandwich blocks accept the smallest terminal block", "[sandwich]") {
    // The X measurement anticommutes with the Z rotation axis.
    const std::vector<SandwichBlock> blocks = blocks_of(hir_of("T 0\nMX 0\n"));
    REQUIRE(blocks.size() == 1);
    CHECK(blocks[0].opening_ops == std::vector<size_t>{0});
    CHECK(blocks[0].check_op == 1);
    CHECK(blocks[0].closing_ops.empty());
    CHECK(blocks[0].supported);
}

TEST_CASE("Sandwich blocks reject terminal candidates that break the rules", "[sandwich]") {
    TerminalText text;
    OpType expected_op = OpType::NUM_OP_TYPES;
    size_t expected_count = 0;

    SECTION("a second visible measurement of another Pauli follows the check") {
        text.after_check = "M 0\n";
    }

    SECTION("a second visible measurement of the same Pauli follows the check") {
        text.after_check = "MX 3\n";
    }

    SECTION("a T gate on another qubit follows the check") {
        text.after_check = "T 4\n";
        expected_op = OpType::T_GATE;
        expected_count = 4;
    }

    SECTION("a phase rotation follows the check") {
        text.after_check = "R_Z(0.3) 4\n";
        expected_op = OpType::PHASE_ROTATION;
        expected_count = 1;
    }

    SECTION("an expectation value follows the check") {
        text.after_check = "EXP_VAL Z4\n";
        expected_op = OpType::EXP_VAL;
        expected_count = 1;
    }

    SECTION("feedback that anticommutes with an axis follows the check") {
        text.after_check = "CX rec[-1] 0\n";
        // Preparation correction and the feedback.
        expected_op = OpType::CONDITIONAL_PAULI;
        expected_count = 2;
    }

    SECTION("a hidden measurement of another Pauli follows the check") {
        text.after_check = "R 3\n";
    }

    SECTION("the check commutes with every axis") {
        text.ladder = "CX 0 3 1 3 2 3\n";
    }

    SECTION("no measurement follows the opening layer") {
        text.check = "";
    }

    const HirModule hir = hir_of(text.str());
    if (expected_op != OpType::NUM_OP_TYPES) {
        REQUIRE(count_ops(hir, expected_op) == expected_count);
    }
    CHECK(find_sandwich_blocks(hir).empty());
}

TEST_CASE("Sandwich blocks check closure against the dormant subspace", "[sandwich]") {
    SECTION("a Bell pair of data qubits keeps its image inside the subspace") {
        // The opening axes X0, X0Z1 and Z2 in the initial frame. A generator
        // that anticommutes with two of them maps to a product that is still
        // a stabilizer, so the check is supported without commuting with it.
        SandwichText text;
        text.prep = "H 0\nCX 0 1\nRX 3\n";

        const std::vector<SandwichBlock> blocks = blocks_of(hir_of(text));
        REQUIRE(blocks.size() == 1);
        CHECK(blocks[0].axis_anticommutes_with_check == std::vector<uint8_t>{1, 1, 1});
        CHECK(blocks[0].supported);
        CHECK(blocks[0].active_width_at_opening == 0);
    }

    SECTION("rotating one half of a Bell pair leaves the subspace") {
        SandwichText text;
        text.prep = "H 0\nCX 0 1\nRX 3\n";
        text.opening = "T 0 2\n";
        text.ladder = "CX 3 0 3 2\n";
        text.closing = "T_DAG 0 2\n";

        const std::vector<SandwichBlock> blocks = blocks_of(hir_of(text));
        REQUIRE(blocks.size() == 1);
        CHECK(blocks[0].axis_anticommutes_with_check == std::vector<uint8_t>{1, 1});
        CHECK_FALSE(blocks[0].supported);
    }

    SECTION("an axis that commutes with the check stays out of the image") {
        // Z2 anticommutes with the stabilizer X of the |+> on qubit 2, but the
        // check does not touch qubit 2, so only Z0 and Z1 act on the image.
        SandwichText text;
        text.prep = "RX 2 3\n";
        text.ladder = "CX 3 0 3 1\n";

        const std::vector<SandwichBlock> blocks = blocks_of(hir_of(text));
        REQUIRE(blocks.size() == 1);
        CHECK(blocks[0].axis_anticommutes_with_check == std::vector<uint8_t>{1, 1, 0});
        CHECK(blocks[0].supported);
    }

    SECTION("injected magic states on the data qubits are not supported") {
        // |+> data qubits are stabilized by X, which a Z-axis rotation maps to Y.
        SandwichText text;
        text.prep = "RX 0 1 2 3\n";

        const std::vector<SandwichBlock> blocks = blocks_of(hir_of(text));
        REQUIRE(blocks.size() == 1);
        CHECK(blocks[0].axis_anticommutes_with_check == std::vector<uint8_t>{1, 1, 1});
        CHECK_FALSE(blocks[0].supported);
    }

    SECTION("the width before the opening layer counts earlier injections") {
        SandwichText text;
        text.prep = "H 4\nT 4\nRX 3\n";

        const std::vector<SandwichBlock> blocks = blocks_of(hir_of(text));
        REQUIRE(blocks.size() == 1);
        CHECK(blocks[0].active_width_at_opening == 1);
        CHECK(blocks[0].supported);
    }
}

TEST_CASE("Sandwich blocks find the three cultivation checks", "[sandwich]") {
    const Circuit circuit = parse_file(fixture_path("cultivation_d5.stim"));
    const HirModule hir = trace(circuit);

    const std::vector<SandwichBlock> blocks = blocks_of(hir);
    // The d3 double cat check and the d5 cat check are closed sandwiches. The
    // T layer before the final destructive MPP has no inverse layer and forms
    // a terminal block.
    REQUIRE(blocks.size() == 3);

    const AstNode& d3_check = source_node(circuit, hir, blocks[0].check_op);
    CHECK(d3_check.gate == GateType::MX);
    REQUIRE(d3_check.targets.size() == 1);
    CHECK(d3_check.targets[0].value() == 7);
    CHECK(blocks[0].opening_ops.size() == 7);
    CHECK(blocks[0].closing_ops.size() == 7);

    const AstNode& d5_check = source_node(circuit, hir, blocks[1].check_op);
    CHECK(d5_check.gate == GateType::MX);
    REQUIRE(d5_check.targets.size() == 1);
    CHECK(d5_check.targets[0].value() == 25);
    CHECK(blocks[1].opening_ops.size() == 19);
    CHECK(blocks[1].closing_ops.size() == 19);

    const AstNode& final_check = source_node(circuit, hir, blocks[2].check_op);
    CHECK(final_check.gate == GateType::MPP);
    CHECK(blocks[2].opening_ops.size() == 19);
    CHECK(blocks[2].closing_ops.empty());
    CHECK(blocks[2].echo_ops.empty());
    CHECK(blocks[2].interior_conditional_ops.empty());
    // Only the observable accumulation follows the final measurement.
    CHECK(blocks[2].check_op + 2 == hir.ops.size());

    for (const SandwichBlock& block : blocks) {
        CHECK(block.supported);
        CHECK(block.active_width_at_opening == 1);
    }
    for (size_t i = 0; i < 2; ++i) {
        CHECK(blocks[i].echo_ops.size() == 1);
        CHECK(blocks[i].interior_conditional_ops.size() == 1);
        CHECK_FALSE(blocks[i].interior_noise_ops.empty());
    }

    // The only T gate outside a block is the lone T_DAG injection.
    size_t block_gates = 0;
    for (const SandwichBlock& block : blocks) {
        block_gates += block.opening_ops.size() + block.closing_ops.size();
    }
    CHECK(hir.num_t_gates() - block_gates == 1);
}

TEST_CASE("Sandwich blocks are absent from circuits without T sandwiches", "[sandwich]") {
    for (const char* name : {"surface_d5_r5_p05.stim", "surface_d7_r7_p001.stim"}) {
        CAPTURE(name);
        const HirModule hir = trace(parse_file(fixture_path(name)));
        CHECK(find_sandwich_blocks(hir).empty());
    }

    CHECK(find_sandwich_blocks(HirModule{}).empty());
    CHECK(find_sandwich_blocks(hir_of("T 0\nM 0\n")).empty());
}
