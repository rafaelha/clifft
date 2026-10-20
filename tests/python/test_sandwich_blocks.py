"""Tests for find_sandwich_blocks, the T-sandwich block recognizer."""

from pathlib import Path
from typing import Any

import clifft

_FIXTURES = Path(__file__).parents[1] / "fixtures"

# Data qubits 0-2 in |0> carry the T layers around an ancilla X check.
_SANDWICH = (
    "RX 3\n"
    "T 0 1 2\n"
    "CX 3 0 3 1 3 2\n"
    "MX 3\n"
    "RX 3\n"
    "CX 3 0 3 1 3 2\n"
    "T_DAG 0 1 2\n"
    "MX 0 1 2\n"
)


def _blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = clifft.find_sandwich_blocks(clifft.trace(clifft.parse(text)))
    return blocks


def test_synthetic_block_reports_every_field() -> None:
    (block,) = _blocks(_SANDWICH)

    assert block == {
        "opening_ops": [2, 3, 4],
        "closing_ops": [8, 9, 10],
        "check_op": 5,
        "echo_ops": [6],
        "interior_noise_ops": [],
        "interior_conditional_ops": [7],
        "axis_anticommutes_with_check": [True, True, True],
        "supported": True,
        "active_width_at_opening": 0,
    }
    assert isinstance(block["supported"], bool)
    assert all(isinstance(flag, bool) for flag in block["axis_anticommutes_with_check"])


def test_unsupported_block_is_reported_with_supported_false() -> None:
    # Injected |+> data qubits are stabilized by X, which a Z-axis rotation leaves.
    (block,) = _blocks(_SANDWICH.replace("RX 3\nT 0 1 2", "RX 0 1 2 3\nT 0 1 2", 1))

    assert block["supported"] is False
    assert len(block["opening_ops"]) == 3


def test_terminal_block_has_no_closing_layer() -> None:
    text = "RX 3\nT 0 1 2\nCX 3 0 3 1 3 2\nMX 3\nRX 3\n"

    (block,) = _blocks(text)

    assert block["opening_ops"] == [2, 3, 4]
    assert block["closing_ops"] == []
    assert block["check_op"] == 5
    assert block["echo_ops"] == [6]
    assert block["supported"] is True


def test_cultivation_d5_has_three_supported_blocks() -> None:
    text = (_FIXTURES / "cultivation_d5.stim").read_text()
    hir = clifft.trace(clifft.parse(text))
    lines = text.splitlines()

    blocks = clifft.find_sandwich_blocks(hir)

    assert len(blocks) == 3
    assert [len(block["opening_ops"]) for block in blocks] == [7, 19, 19]
    # The T layer before the final destructive MPP has no inverse layer.
    assert [len(block["closing_ops"]) for block in blocks] == [7, 19, 0]
    for block in blocks:
        assert block["supported"] is True
        assert block["active_width_at_opening"] == 1
    check_lines = [lines[hir.source_map[block["check_op"]][0] - 1] for block in blocks]
    assert check_lines[:2] == ["MX(0.005) 7", "MX(0.005) 25"]
    assert check_lines[2].startswith("MPP Y0*Y3*Y7*")
    for block in blocks:
        for opening, closing in zip(block["opening_ops"], block["closing_ops"], strict=False):
            assert hir[opening].op_type == clifft.OpType.T_GATE
            assert hir[closing].op_type == clifft.OpType.T_GATE
            assert hir[opening].is_dagger != hir[closing].is_dagger


def test_circuit_without_t_sandwiches_has_no_blocks() -> None:
    text = (_FIXTURES / "surface_d5_r5_p05.stim").read_text()

    assert clifft.find_sandwich_blocks(clifft.trace(clifft.parse(text))) == []
