"""Tests for SandwichSpecializationPass on the noiseless cultivation workloads."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import binomial_tolerance

import clifft

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_WORKLOADS = ["cultivation_d3_bench.stim", "cultivation_d5_bench.stim"]

_NOISE = re.compile(r"^(DEPOLARIZE[12]|[XYZ]_ERROR|PAULI_CHANNEL_[12])\(")
_READOUT = re.compile(r"^(M[A-Z]*|MPP)\([^)]*\)")

Key = tuple[list[bool], list[bool]]


@lru_cache(maxsize=None)
def _noiseless(name: str) -> str:
    """The fixture with its Pauli channels and readout flips removed."""
    lines = []
    for line in (_FIXTURES / name).read_text().splitlines():
        if _NOISE.match(line):
            continue
        lines.append(_READOUT.sub(r"\1", line))
    return "\n".join(lines) + "\n"


def _noiseless_keys(hir: clifft.HirModule) -> list[Key]:
    """One key per recognized block for the realization with no error."""
    blocks = clifft.find_sandwich_blocks(hir)
    assert blocks and all(block["supported"] for block in blocks)
    return [
        ([False] * len(block["opening_ops"]), [False] * len(block["opening_ops"]))
        for block in blocks
    ]


def _run(pass_: clifft.HirPass, hir: clifft.HirModule) -> None:
    manager = clifft.HirPassManager()
    manager.add(pass_)
    manager.run(hir)


def _check_same_outputs(generic: clifft.Program, specialized: clifft.Program) -> None:
    """Compare two programs of one noiseless circuit through sampled outputs.

    Replaying records needs the hidden reset records, which sampling does not
    return, so the programs are compared through what it does return. Every
    detector and observable of the noiseless circuit is deterministic.
    """
    assert specialized.num_measurements == generic.num_measurements
    assert specialized.num_detectors == generic.num_detectors
    shots = 400
    generic_result = clifft.sample(generic, shots, seed=11)
    specialized_result = clifft.sample(specialized, shots, seed=12)
    assert np.all(generic_result.detectors == generic_result.detectors[0])
    assert np.all(specialized_result.detectors == generic_result.detectors[0])
    assert np.all(specialized_result.observables == generic_result.observables[0])

    generic_means = generic_result.measurements.mean(axis=0)
    specialized_means = specialized_result.measurements.mean(axis=0)
    for column, (expected, actual) in enumerate(zip(generic_means, specialized_means)):
        tolerance = 2 * binomial_tolerance(float(expected), shots)
        assert abs(actual - expected) < tolerance, f"record column {column}"


@pytest.mark.parametrize("name", _WORKLOADS)
def test_specializing_every_block_leaves_one_active_qubit(name: str) -> None:
    text = _noiseless(name)
    keys = _noiseless_keys(clifft.trace(clifft.parse(text)))
    manager = clifft.HirPassManager()
    manager.add(clifft.SandwichSpecializationPass(keys))
    specialized = clifft.compile(text, hir_passes=manager)
    generic = clifft.compile(text)

    # Only the injection T gate remains outside the blocks.
    assert specialized.peak_active_width == 1
    assert generic.peak_active_width > 1
    _check_same_outputs(generic, specialized)


@pytest.mark.parametrize("name", _WORKLOADS)
def test_specialization_after_the_default_pipeline(name: str) -> None:
    text = _noiseless(name)
    hir = clifft.trace(clifft.parse(text))
    clifft.default_hir_pass_manager().run(hir)
    keys = _noiseless_keys(hir)
    manager = clifft.default_hir_pass_manager()
    manager.add(clifft.SandwichSpecializationPass(keys))
    specialized = clifft.compile(text, hir_passes=manager)
    generic = clifft.compile(text)

    # Without noise to separate them, the default passes merge the gates of
    # the earlier checks, leaving a terminal block at the final check.
    assert len(keys) == 1
    assert specialized.peak_active_width <= generic.peak_active_width
    _check_same_outputs(generic, specialized)


def test_pass_reports_found_and_specialized_blocks() -> None:
    hir = clifft.trace(clifft.parse(_noiseless("cultivation_d5_bench.stim")))
    t_gates = hir.num_t_gates
    blocks = clifft.find_sandwich_blocks(hir)
    assert len(blocks) == 3

    width = len(blocks[1]["opening_ops"])
    pass_ = clifft.SandwichSpecializationPass([([], []), ([False] * width, [False] * width)])
    _run(pass_, hir)

    assert pass_.blocks_found == 3
    assert pass_.blocks_specialized == 1
    assert hir.num_t_gates == t_gates - 2 * width
    op_types = [op.op_type for op in hir]
    assert op_types.count(clifft.OpType.SANDWICH_PROJECTION) == 1
    assert op_types.count(clifft.OpType.SANDWICH_RECORD) == len(blocks[1]["echo_ops"]) + 1
    projection = next(op for op in hir if op.op_type == clifft.OpType.SANDWICH_PROJECTION)
    assert projection.as_dict()["sandwich_site_idx"] == 0
    assert "blocks_specialized=1" in repr(pass_)


def test_pass_rejects_malformed_keys() -> None:
    hir = clifft.trace(clifft.parse(_noiseless("cultivation_d3_bench.stim")))
    with pytest.raises(ValueError, match="axis flips"):
        _run(clifft.SandwichSpecializationPass([([False], [False])]), hir)
    missing_support: Any = [([False],)]
    with pytest.raises(TypeError):
        clifft.SandwichSpecializationPass(missing_support)
