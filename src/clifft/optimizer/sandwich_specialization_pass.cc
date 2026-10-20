#include "clifft/optimizer/sandwich_specialization_pass.h"

#include "clifft/optimizer/clifford_absorption.h"
#include "clifft/optimizer/pauli_axis.h"
#include "clifft/optimizer/sandwich_blocks.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clifft {

namespace {

using optimizer_detail::CliffordAbsorption;
using optimizer_detail::copy_axis;

constexpr uint32_t kNoBlock = std::numeric_limits<uint32_t>::max();

// Moving a Pauli error across a T gate reverses that gate, so noise between
// the gates of one layer has no single position relative to the projection.
void require_noise_free_layer(const HirModule& hir, const std::vector<size_t>& layer) {
    if (layer.empty()) {
        return;
    }
    const auto [first, last] = std::ranges::minmax(layer);
    for (size_t i = first + 1; i < last; ++i) {
        if (hir.ops[i].op_type() == OpType::NOISE) {
            throw std::invalid_argument(
                "SandwichSpecializationPass: a sandwich layer interleaves noise with its gates at "
                "operation " +
                std::to_string(i));
        }
    }
}

std::vector<uint32_t> merged_sources(const HirModule& hir, const SandwichBlock& block) {
    std::vector<uint32_t> lines;
    for (const auto* layer : {&block.opening_ops, &block.closing_ops}) {
        for (const size_t op : *layer) {
            lines.insert(lines.end(), hir.source_map[op].begin(), hir.source_map[op].end());
        }
    }
    std::ranges::sort(lines);
    lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
    return lines;
}

// Per-op role in the rewrite. Ops outside specialized blocks are copied.
enum class Role : uint8_t { Copy, Drop, Projection, Record };

class Rewriter {
  public:
    Rewriter(const HirModule& input, const std::vector<SandwichBlock>& blocks,
             const std::vector<SandwichKey>& keys, const std::vector<size_t>& selected)
        : input_(input),
          blocks_(blocks),
          keys_(keys),
          selected_(selected),
          output_module_(input.with_additional_pauli_masks(selected.size())),
          role_(input.ops.size(), Role::Copy),
          owner_(input.ops.size(), kNoBlock),
          absorb_at_(input.ops.size(), kNoBlock) {
        output_module_.logical_noise_prefix.clear();
        for (uint32_t ordinal = 0; ordinal < selected_.size(); ++ordinal) {
            const SandwichBlock& block = blocks_[selected_[ordinal]];
            for (const size_t op : block.opening_ops) {
                role_[op] = Role::Drop;
            }
            role_[block.opening_ops.front()] = Role::Projection;
            for (const size_t op : block.closing_ops) {
                role_[op] = Role::Drop;
            }
            role_[block.check_op] = Role::Record;
            for (const size_t op : block.echo_ops) {
                role_[op] = Role::Record;
            }
            owner_[block.opening_ops.front()] = ordinal;
            owner_[block.check_op] = ordinal;
            for (const size_t op : block.echo_ops) {
                owner_[op] = ordinal;
            }
            if (!block.closing_ops.empty()) {
                absorb_at_[*std::ranges::min_element(block.closing_ops)] = ordinal;
            } else {
                // A terminal block has nothing quantum after its check. Its
                // correction can only reach the final frame.
                terminal_absorbs_.push_back(ordinal);
            }
        }
    }

    HirModule run() {
        HirModule& out = output_module_;
        const bool has_sources = !input_.source_map.empty();
        std::vector<HeisenbergOp> ops;
        std::vector<std::vector<uint32_t>> sources;
        ops.reserve(input_.ops.size());
        if (has_sources) {
            sources.reserve(input_.ops.size());
        }
        site_of_.assign(selected_.size(), 0);

        for (size_t i = 0; i < input_.ops.size(); ++i) {
            if (absorb_at_[i] != kNoBlock) {
                absorb_correction(absorb_at_[i]);
            }
            const HeisenbergOp op = out.ops[i];
            switch (role_[i]) {
                case Role::Copy:
                    cliffords_.transform(out, op);
                    ops.push_back(op);
                    if (has_sources) {
                        sources.push_back(input_.source_map[i]);
                    }
                    break;
                case Role::Drop:
                    break;
                case Role::Projection: {
                    const HeisenbergOp projection = make_projection(owner_[i]);
                    cliffords_.transform(out, projection);
                    ops.push_back(projection);
                    if (has_sources) {
                        sources.push_back(merged_sources(input_, blocks_[selected_[owner_[i]]]));
                    }
                    break;
                }
                case Role::Record: {
                    HeisenbergOp record =
                        HeisenbergOp::make_sandwich_record(op.mask_handle(), op.meas_record_idx(),
                                                           SandwichSiteIdx{site_of_[owner_[i]]});
                    record.set_hidden(op.is_hidden());
                    cliffords_.transform(out, record);
                    ops.push_back(record);
                    if (has_sources) {
                        sources.push_back(input_.source_map[i]);
                    }
                    break;
                }
            }
        }
        for (const uint32_t ordinal : terminal_absorbs_) {
            absorb_correction(ordinal);
        }

        out.ops = std::move(ops);
        out.source_map = std::move(sources);
        cliffords_.finish(out);
        return std::move(output_module_);
    }

  private:
    // The opening ops are dropped, so the site adopts their mask slots. The
    // check keeps its own slot for its record, so the projection copies it.
    HeisenbergOp make_projection(uint32_t ordinal) {
        HirModule& out = output_module_;
        const SandwichBlock& block = blocks_[selected_[ordinal]];
        const SandwichKey& key = keys_[selected_[ordinal]];
        SandwichSite site;
        site.axes.reserve(block.opening_ops.size());
        site.axis_dagger.reserve(block.opening_ops.size());
        for (const size_t op : block.opening_ops) {
            site.axes.push_back(out.ops[op].mask_handle());
            site.axis_dagger.push_back(out.ops[op].is_dagger() ? 1 : 0);
        }
        site.axis_flip.reserve(key.axis_flip.size());
        for (const uint8_t flip : key.axis_flip) {
            site.axis_flip.push_back(flip != 0 ? 1 : 0);
        }
        const HeisenbergOp& check = out.ops[block.check_op];
        site.check_record = static_cast<uint32_t>(check.meas_record_idx());

        const auto site_index = static_cast<uint32_t>(out.sandwich_sites.size());
        out.sandwich_sites.push_back(std::move(site));
        site_of_[ordinal] = site_index;

        const PauliMaskView check_mask = out.mask_view(check);
        const PauliMaskHandle handle = out.claim_side_mask([&](MutablePauliMaskView slot) {
            slot.x().xor_with(check_mask.x());
            slot.z().xor_with(check_mask.z());
            slot.set_sign(check_mask.sign());
        });
        return HeisenbergOp::make_sandwich_projection(handle, SandwichSiteIdx{site_index});
    }

    // Folds K = prod exp(+i pi/4 s_j P_j) over the internally supported axes.
    // The site axes were brought into the current frame when the projection
    // was emitted, and no correction is absorbed inside a block.
    void absorb_correction(uint32_t ordinal) {
        const HirModule& out = output_module_;
        const SandwichKey& key = keys_[selected_[ordinal]];
        const SandwichSite& site = out.sandwich_sites[site_of_[ordinal]];
        for (size_t j = 0; j < site.axes.size(); ++j) {
            if (key.internal_support[j] == 0) {
                continue;
            }
            const PauliString axis = copy_axis(out.pauli_masks.at(site.axes[j]), out.num_qubits);
            cliffords_.absorb(axis, site.axis_dagger[j] != 0 ? 2 : 6);
        }
    }

    const HirModule& input_;
    const std::vector<SandwichBlock>& blocks_;
    const std::vector<SandwichKey>& keys_;
    const std::vector<size_t>& selected_;
    HirModule output_module_;
    std::vector<Role> role_;
    // Ordinal into selected_ of the block an op belongs to, for the ops whose
    // role needs it.
    std::vector<uint32_t> owner_;
    // Ordinal of the block whose correction is absorbed before this op.
    std::vector<uint32_t> absorb_at_;
    std::vector<uint32_t> terminal_absorbs_;
    std::vector<uint32_t> site_of_;
    CliffordAbsorption cliffords_;
};

}  // namespace

size_t SandwichKeysHash::operator()(const std::vector<SandwichKey>& keys) const noexcept {
    // FNV-1a over the flags, with each vector's length mixed in so keys
    // of different shapes stay distinct.
    uint64_t hash = 0xcbf29ce484222325ULL;
    const auto mix = [&hash](uint64_t value) {
        hash ^= value;
        hash *= 0x100000001b3ULL;
    };
    for (const SandwichKey& key : keys) {
        for (const auto* flags : {&key.axis_flip, &key.internal_support}) {
            mix(flags->size());
            for (const uint8_t flag : *flags) {
                mix(flag);
            }
        }
    }
    return static_cast<size_t>(hash);
}

SandwichSpecializationPass::SandwichSpecializationPass(std::vector<SandwichKey> keys)
    : keys_(std::move(keys)) {}

void SandwichSpecializationPass::run(HirModule& hir) {
    blocks_found_ = 0;
    blocks_specialized_ = 0;
    // A projection replaces ops that may have crossed noise; their recorded
    // positions would have to move with it.
    if (!hir.logical_noise_prefix_matches_schedule()) {
        throw std::invalid_argument(
            "SandwichSpecializationPass requires noise operations at their logical positions");
    }
    if (!hir.source_map.empty() && hir.source_map.size() != hir.ops.size()) {
        throw std::invalid_argument("HIR source map size does not match the operation count");
    }

    const std::vector<SandwichBlock> blocks = find_sandwich_blocks(hir);
    blocks_found_ = blocks.size();
    if (keys_.size() > blocks.size()) {
        throw std::invalid_argument("SandwichSpecializationPass received " +
                                    std::to_string(keys_.size()) + " keys for " +
                                    std::to_string(blocks.size()) + " recognized blocks");
    }

    std::vector<size_t> selected;
    for (size_t b = 0; b < keys_.size(); ++b) {
        const SandwichKey& key = keys_[b];
        if (key.empty()) {
            continue;
        }
        const SandwichBlock& block = blocks[b];
        const size_t width = block.opening_ops.size();
        if (key.axis_flip.size() != width || key.internal_support.size() != width) {
            throw std::invalid_argument(
                "SandwichSpecializationPass key " + std::to_string(b) + " has " +
                std::to_string(key.axis_flip.size()) + " axis flips and " +
                std::to_string(key.internal_support.size()) +
                " internal-support flags for a block with " + std::to_string(width) + " axes");
        }
        if (!block.supported) {
            throw std::invalid_argument("SandwichSpecializationPass key " + std::to_string(b) +
                                        " names a block whose projection is not supported");
        }
        require_noise_free_layer(hir, block.opening_ops);
        require_noise_free_layer(hir, block.closing_ops);
        selected.push_back(b);
    }
    if (selected.empty()) {
        return;
    }

    Rewriter rewriter(hir, blocks, keys_, selected);
    hir = rewriter.run();
    blocks_specialized_ = selected.size();
}

}  // namespace clifft
