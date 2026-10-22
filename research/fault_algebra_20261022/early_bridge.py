"""Control: expose known output checks at their earliest record boundary.

Detector ordering changes, so this export is used only for all-check aggregate
postselection, where the permutation does not change acceptance or observables.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
STUDY = ROOT.parent / "global_reconstruction_20261022"
sys.path.insert(0, str(STUDY))
from study_io import read_circuit  # type: ignore[import-not-found]  # noqa: E402


def early(name):
    operations = []
    outputs = []
    records = 0
    detector = 0
    for line in read_circuit(STUDY / f"{name}-detectors-compressed.clifft").splitlines():
        gate = line.split()[0]
        if gate == "DETECTOR" or gate.startswith("OBSERVABLE_INCLUDE("):
            dependencies = [records + int(i) for i in re.findall(r"rec\[(-?\d+)\]", line)]
            outputs.append((gate, dependencies, detector if gate == "DETECTOR" else -1))
            detector += gate == "DETECTOR"
            continue
        count = len(line.split()) - 1 if gate in ("M", "MPP", "MPAD") else 0
        operations.append((line, count))
        records += count
    text = []
    mapping = []
    available = 0
    remaining = outputs[:]

    def flush():
        nonlocal remaining
        pending = []
        for gate, deps, index in remaining:
            if all(d < available for d in deps):
                text.append(gate + " " + " ".join(f"rec[{d - available}]" for d in deps))
                if index >= 0:
                    mapping.append(index)
            else:
                pending.append((gate, deps, index))
        remaining = pending

    flush()
    for line, count in operations:
        text.append(line)
        available += count
        if count:
            flush()
    assert not remaining
    return "\n".join(text) + "\n", mapping


if __name__ == "__main__":
    for name in ["d5", "d7"]:
        text, mapping = early(name)
        (ROOT / f"{name}-early.clifft").write_text(text)
        (ROOT / f"{name}-early-map.json").write_text(json.dumps(mapping) + "\n")
