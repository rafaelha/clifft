"""Reference proofs and normalization against independent parity expectations."""

from pathlib import Path

import numpy as np
import pytest
import stim
from utils_conformance import SamplingMode

import clifft


@pytest.mark.parametrize("preparation", ["H 0 1", "RX 0 1\nR_Z(0.75) 0 1"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_random_reference_rejected_in_either_measurement_order(
    preparation: str, reverse: bool, optimize: bool
) -> None:
    measurement = "M" if preparation.startswith("H") else "MX"
    order = [1, 0] if reverse else [0, 1]
    records = [-1, -2] if reverse else [-2, -1]
    source = preparation + "\n" + "\n".join(f"{measurement} {q}" for q in order)
    source += "\n" + "\n".join(f"DETECTOR rec[{r}]" for r in records)
    with pytest.raises(ValueError, match="Cannot prove noiseless detector 0 deterministic"):
        clifft.compute_reference_syndrome(clifft.trace(clifft.parse(source)))
    passes = clifft.default_hir_pass_manager() if optimize else None
    with pytest.raises(ValueError, match="expected_detectors.*normalize_syndromes=False"):
        clifft.compile(source, hir_passes=passes, normalize_syndromes=True)
    if preparation.startswith("H"):
        with pytest.raises(ValueError, match="non-deterministic detectors"):
            stim.Circuit(source).detector_error_model()


@pytest.mark.parametrize(
    ("annotations", "offender"),
    [
        ("DETECTOR rec[-1] rec[-1]\nDETECTOR rec[-1]", "detector 1"),
        ("DETECTOR rec[-1] rec[-1]\nOBSERVABLE_INCLUDE(3) rec[-1]", "observable 3"),
    ],
)
def test_reference_reports_first_unproven_output(annotations: str, offender: str) -> None:
    source = "H 0\nM 0\n" + annotations
    with pytest.raises(ValueError, match=f"Cannot prove noiseless {offender} deterministic"):
        clifft.compute_reference_syndrome(clifft.trace(clifft.parse(source)))
    with pytest.raises(ValueError, match=offender):
        clifft.compile(source, normalize_syndromes=True)


@pytest.mark.parametrize("preparation", ["R 0", "RX 0\nR_Z(0.75) 0"])
def test_random_records_cancel_in_detectors_and_complete_observables(
    sampling_mode: SamplingMode, preparation: str
) -> None:
    checks = (
        "MX 0\nZ 0\nMX 0\nDETECTOR rec[-2] rec[-1]\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    source = preparation + "\n" + checks
    reference = clifft.compute_reference_syndrome(clifft.trace(clifft.parse(source)))
    assert reference == {"detectors": [1], "observables": [1]}
    program = sampling_mode.compile(source, normalize_syndromes=True, postselection_mask=[1])
    result = sampling_mode.sample_survivors(program, 65, seed=7, keep_records=True)
    assert result.passed_shots == 65
    assert result.logical_errors == 0
    # Both preparations have the same certain parity, despite different first-shot biases.
    expected = (
        stim.Circuit(checks).compile_detector_sampler(seed=7).sample(65, separate_observables=True)
    )
    np.testing.assert_array_equal(result.detectors, expected[0])
    np.testing.assert_array_equal(result.observables, expected[1])


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("H 0\nT 0\nX_ERROR(0.2) 0\nT_DAG 0\nMX(0.1) 0", 0),
        ("H 0\nT 0\nT 0\nT 0\nT 0\nMX 0", 1),
    ],
)
def test_reference_optimizes_noise_stripped_copy(source: str, expected: int) -> None:
    source += "\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    hir = clifft.trace(clifft.parse(source))
    original_ops = hir.num_ops
    assert clifft.compute_reference_syndrome(hir) == {
        "detectors": [expected],
        "observables": [expected],
    }
    assert hir.num_ops == original_ops
    clifft.compile(source, hir_passes=None, normalize_syndromes=True)


def test_unproven_deterministic_reference_accepts_explicit_override(
    sampling_mode: SamplingMode,
) -> None:
    # The probes prevent fusion across four T gates, whose product is Z.
    source = (
        "H 0\n" + "T 0\nEXP_VAL X0\n" * 4 + "MX 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    with pytest.raises(ValueError, match="Cannot prove noiseless detector 0 deterministic"):
        clifft.compute_reference_syndrome(clifft.trace(clifft.parse(source)))
    with pytest.raises(ValueError, match="detector 0"):
        sampling_mode.compile(source, normalize_syndromes=True)
    raw = sampling_mode.sample(sampling_mode.compile(source), 65, seed=11)
    np.testing.assert_array_equal(raw.detectors, 1)
    program = sampling_mode.compile(
        source, expected_detectors=[1], expected_observables=[1], postselection_mask=[1]
    )
    result = sampling_mode.sample_survivors(program, 65, seed=11, keep_records=True)
    assert result.passed_shots == 65
    assert result.logical_errors == 0
    np.testing.assert_array_equal(result.detectors, 0)
    np.testing.assert_array_equal(result.observables, 0)


def test_explicit_reference_preserves_random_output_distribution(
    sampling_mode: SamplingMode,
) -> None:
    source = "RX 0\nR_Z(0.75) 0\nMX 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    raw = sampling_mode.sample(sampling_mode.compile(source), 257, seed=13)
    flipped = sampling_mode.sample(
        sampling_mode.compile(source, expected_detectors=[1], expected_observables=[1]),
        257,
        seed=13,
    )
    np.testing.assert_array_equal(raw.measurements, flipped.measurements)
    np.testing.assert_array_equal(raw.detectors ^ 1, flipped.detectors)
    np.testing.assert_array_equal(raw.observables ^ 1, flipped.observables)


@pytest.mark.parametrize("source", ["", "H 0\nM 0", "H 0\nT 0\nM 0"])
def test_random_measurements_without_syndromes_need_no_reference(source: str) -> None:
    assert clifft.compute_reference_syndrome(clifft.trace(clifft.parse(source))) == {
        "detectors": [],
        "observables": [],
    }
    clifft.compile(source, normalize_syndromes=True)


@pytest.mark.parametrize("fixture", ["coherent_d5_r5.stim", "cultivation_d5.stim"])
def test_fixture_references_require_explicit_parities(fixture: str) -> None:
    source = (Path(__file__).parents[1] / "fixtures" / fixture).read_text()
    with pytest.raises(ValueError, match="Cannot prove noiseless detector"):
        clifft.compute_reference_syndrome(clifft.trace(clifft.parse(source)))
    with pytest.raises(ValueError, match="Cannot prove noiseless detector"):
        clifft.compile(source, normalize_syndromes=True)


def test_cultivation_accepts_its_explicit_ideal_reference(sampling_mode: SamplingMode) -> None:
    source = (Path(__file__).parents[1] / "fixtures" / "cultivation_d5.stim").read_text()
    program = sampling_mode.compile(source)
    # Cultivation's intended ideal checks are all even, including those the
    # symbolic frame cannot prove. A supplied reference must still compile.
    normalized = sampling_mode.compile(
        source,
        expected_detectors=[0] * program.num_detectors,
        expected_observables=[0] * program.num_observables,
    )
    raw = sampling_mode.sample(program, 65, seed=731)
    explicit = sampling_mode.sample(normalized, 65, seed=731)
    np.testing.assert_array_equal(raw.detectors, explicit.detectors)
    np.testing.assert_array_equal(raw.observables, explicit.observables)
