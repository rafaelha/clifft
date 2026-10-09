"""Export complete factory diagnostics and fixed-fault oracle queries.

Circuit files contain the original physical gates, never the shared-phase
replacement. Selected exact probabilities are correctness probes, not a full
distribution certificate or a noise-weighted protocol-rate calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from automatic_specialization import FaultModel
from profile_external_feedback import MERLIN_REVISION
from study_quadcycle_factory import factory, load_css
from validate_steane_continuation import Extraction
from validate_theory_ccz_faults import Workload

import clifft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2610109)
    args = parser.parse_args()
    if args.draws < 0:
        parser.error("Draw count must be nonnegative")
    if args.output_dir.exists():
        parser.error("Output directory must be new to preserve earlier benchmark inputs")
    css = load_css(args.merlin_checkout)
    workload = Workload(css)
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []

    def save(name: str, source: str, **metadata: Any) -> dict[str, Any]:
        path = args.output_dir / (name + ".stim")
        path.write_text(source)
        parsed = clifft.parse(source)
        row = {
            "file": path.name,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "physical_qubits": parsed.num_qubits,
            "measurement_records": parsed.num_measurements,
            **metadata,
        }
        rows.append(row)
        return row

    def panel(name: str, source: str, extraction: Extraction | None = None) -> None:
        model = FaultModel(source)
        save(name, source, mode="live_noise", fault_sites=len(model.sites))
        fault_rng = random.Random(args.seed)
        query_rng = random.Random(args.seed + 1)
        histories = [()] + [model.draw(fault_rng) for _ in range(args.draws)]
        for index, history in enumerate(histories):
            row = save(
                f"{name}-fixed-{index}",
                model.render(history),
                mode="fixed_faults",
                parent=name + ".stim",
                history=history,
                selection="identity" if index == 0 else "noise_draw",
            )
            if extraction is not None:
                boundary = len(workload.shared.model.sites)
                region = tuple((site, outcome) for site, outcome in history if site < boundary)
                tail = tuple(
                    (site - boundary, outcome) for site, outcome in history if site >= boundary
                )
                reference = extraction.reference(region, tail)
                records = [extraction.draw_record(reference, query_rng) for _ in range(4)]
                invalid = records[0].copy()
                invalid[0] ^= 1
                records.append(invalid)
                row["record_probability_queries"] = [
                    {
                        "record": record,
                        "probability": str(extraction.record_probability(record, reference)),
                    }
                    for record in records
                ]

    for p, name in ((0.0, "quadcycle-ideal"), (0.001, "quadcycle-noisy")):
        region, tail, _ = factory(css, p)
        if p:
            panel(name, region + tail)
        else:
            save(name, region + tail, mode="ideal")
    for protocol in ("D", "E"):
        extraction = Extraction(workload, protocol)
        data_probes = "\n".join(workload.checks + workload.z_checks) + "\n"
        panel(
            f"steane-{protocol}",
            workload.source + extraction.model.source + data_probes,
            extraction,
        )
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "draws_per_noisy_source": args.draws,
        "merlin_helper_revision": MERLIN_REVISION,
        "initial_state": "all_zero",
        "scope": (
            "Complete diagnostic circuits with original physical gates; no region hints. "
            "D/E include ideal data X/Z probes after raw ancilla readout. "
            "Fixed-fault queries are selected exact full-record probabilities only. "
            "D/E noise strengths are diagnostic location weights, not published rate models."
        ),
        "cases": rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(rows)} circuits and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
