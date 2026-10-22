"""Small compile-time binary linear algebra helpers for the research scripts."""

import numpy as np


class Basis:
    def __init__(self):
        self.rows = {}

    def reduce(self, vector, label=0):
        for pivot in sorted(self.rows, reverse=True):
            if vector >> pivot & 1:
                row, provenance = self.rows[pivot]
                vector ^= row
                label ^= provenance
        return vector, label

    def insert(self, vector, label):
        remainder, provenance = self.reduce(int(vector), int(label))
        if remainder == 0:
            return provenance
        self.rows[remainder.bit_length() - 1] = (remainder, provenance)
        return 0

    def solve(self, vector):
        remainder, provenance = self.reduce(int(vector))
        return provenance if remainder == 0 else None


def mask(bits):
    return int.from_bytes(np.packbits(bits, bitorder="little").tobytes(), "little")


def unmask(value, length):
    return np.array([bool(value >> i & 1) for i in range(length)])
