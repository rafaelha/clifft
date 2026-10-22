"""Locate the record-width gap without implementing further graph rewrites."""

import json

import numpy as np
from frontend import Dilation, Operation, mask  # type: ignore[import-not-found]
from planning import Group, hidden_measurements, schedule  # type: ignore[import-not-found]
from validate import OLD, ROOT  # type: ignore[import-not-found]


def peak(n, ops, order):
    g = Group(n)
    p = 0
    for j in order:
        g.apply(ops[j])
        p = max(p, g.width())
    return p


def check(label, n, ops):
    order, p = schedule(n, ops)
    completed = hidden_measurements(n, ops)
    ordered, q = schedule(n, completed)
    return dict(
        label=label,
        lines=n,
        operations=len(ops),
        hidden=len(completed) - len(ops),
        source_order_peak=peak(n, ops, range(len(ops))),
        scheduled_without_hidden=p,
        scheduled_with_hidden=q,
    )


def run():
    rows = []
    for name in ("d5", "d7"):
        d = Dilation.parse((OLD / "inputs" / f"{name}.stim").read_text())
        for mode in ("detectors", "records"):
            n, ops = d.reconstruct(mode)
            rows.append(check(f"native-{name}-{mode}", n, ops))
            paths = list(OLD.glob(f"{name}-{mode}-component*.npz"))
            a = max((dict(np.load(path)) for path in paths), key=lambda a: a["x"].shape[1])
            ops = [
                Operation(mask(x), mask(z), 0, float(angle), int(slot))
                for x, z, angle, slot in zip(a["x"], a["z"], a["angle"], a["slot"])
            ]
            # Strip the reference's hidden measurements before recompleting them.
            ops = [o for o in ops if o.angle or o.slot >= 0]
            rows.append(check(f"zx-reference-{name}-{mode}", a["x"].shape[1], ops))
            if mode == "records":
                sorted_ops = sorted(
                    ops, key=lambda o: -1 if o.angle else a["output_indices"][o.slot]
                )
                rows.append(check(f"zx-source-ordered-{name}-{mode}", a["x"].shape[1], sorted_ops))
    print(json.dumps(rows, indent=2))
    (ROOT / "ablation.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    run()
