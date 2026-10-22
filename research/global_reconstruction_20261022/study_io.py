"""Read generated circuit artifacts without keeping their expanded text on disk."""

import gzip


def read_circuit(path):
    if path.exists():
        return path.read_text()
    with gzip.open(str(path) + ".gz", "rt") as source:
        return source.read()
