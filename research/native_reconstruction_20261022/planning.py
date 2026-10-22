"""Compile-time commutant completion and dependency scheduling.

The commutant construction and measurement-first scheduling follow the method
used by Tsim's PBC reader/scheduler, expressed here on packed binary Paulis.
This module consumes only the independently reconstructed operations.
"""

from frontend import Basis, Operation, bits  # type: ignore[import-not-found]


def anti(a, b):
    return ((a.x & b.z) ^ (a.z & b.x)).bit_count() & 1


def hidden_measurements(n, operations):
    columns = Basis()
    null = []
    for i in range(2 * n):
        column = sum(
            (((o.z if i < n else o.x) >> (i % n)) & 1) << j for j, o in enumerate(operations)
        )
        dependency = columns.insert(column, 1 << i)
        if dependency is not None:
            null.append(dependency)
    selected: list = []
    xparts = Basis()
    for m in sorted(null, key=int.bit_count):
        p = Operation(m & ((1 << n) - 1), m >> n, 0, hidden=True)
        if xparts.solve(p.x) is not None or any(anti(p, q) for q in selected):
            continue
        xparts.insert(p.x, 1 << len(selected))
        selected.append(p)
    return selected + operations


class Group:
    def __init__(self, n, rows=None):
        self.n = n
        self.rows = [Operation(0, 1 << i, 0) for i in range(n)] if rows is None else rows[:]

    def clone(self):
        return Group(self.n, self.rows)

    def width(self):
        return self.n - len(self.rows)

    def keeps(self, p):
        return not any(anti(p, g) for g in self.rows)

    def apply(self, p):
        pivots = [i for i, g in enumerate(self.rows) if anti(p, g)]
        if pivots:
            pivot = pivots[0]
            g = self.rows[pivot]
            for i in pivots[1:]:
                r = self.rows[i]
                self.rows[i] = Operation(r.x ^ g.x, r.z ^ g.z, 0)
            if p.angle:
                self.rows.pop(pivot)
            else:
                self.rows[pivot] = p
        elif not p.angle:
            b = Basis()
            for r in self.rows:
                b.insert(r.x | (r.z << self.n), 0)
            if b.solve(p.x | (p.z << self.n)) is None:
                self.rows.append(p)


def schedule(n, operations):
    count = len(operations)
    predecessors = [
        sum(int(anti(operations[i], operations[j])) << i for i in range(j)) for j in range(count)
    ]
    ancestors: list = []
    for pred in predecessors:
        transitive = pred
        for i in bits(pred):
            transitive |= ancestors[i]
        ancestors.append(transitive)
    pending = (1 << count) - 1
    group = Group(n)
    order = []
    peak = 0

    def run(j):
        nonlocal pending, peak
        group.apply(operations[j])
        pending ^= 1 << j
        order.append(j)
        peak = max(peak, group.width())

    while pending:
        progressed = True
        while progressed:
            progressed = False
            for j in list(bits(pending)):
                if predecessors[j] & pending:
                    continue
                if not operations[j].angle or group.keeps(operations[j]):
                    run(j)
                    progressed = True
        measured = [j for j in bits(pending) if not operations[j].angle]
        if not measured:
            for j in list(bits(pending)):
                run(j)
            break
        best = None
        lowest = float("inf")
        for j in measured:
            chain = list(bits(ancestors[j] & pending))
            trial = group.clone()
            highest = trial.width()
            for k in chain:
                trial.apply(operations[k])
                highest = max(highest, trial.width())
                if highest >= lowest:
                    break
            if highest < lowest:
                lowest = highest
                best = (j, chain)
        assert best is not None
        j, chain = best
        for k in chain:
            run(k)
        run(j)
    return order, peak
