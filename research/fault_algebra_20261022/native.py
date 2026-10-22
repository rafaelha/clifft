"""ctypes interface to the isolated native experiment."""

import ctypes as C
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
LIB = C.CDLL(str(ROOT / "evaluator.so"))
LIB.make_plan.argtypes = [C.c_char_p]
LIB.make_plan.restype = C.c_void_p
LIB.free_plan.argtypes = [C.c_void_p]
LIB.sample.argtypes = [
    C.c_void_p,
    C.c_uint64,
    C.c_uint64,
    C.c_int,
    C.c_void_p,
    C.c_void_p,
    C.c_void_p,
    C.c_int,
]
LIB.sample.restype = C.c_uint64
LIB.probability.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
LIB.probability.restype = C.c_double
LIB.expectation.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
LIB.expectation.restype = C.c_double


class Native:
    def __init__(self, path):
        self.path = Path(path)
        header = list(map(int, self.path.read_text().splitlines()[0].split()))
        self.nf, self.params, self.outputs, self.detectors = header[:4]
        self.words = (self.params + 63) // 64
        self.ptr = LIB.make_plan(str(path).encode())

    def sample(
        self, shots, seed=42, postselect=False, retain=False, parameters=None, noise_only=False
    ):
        output = np.zeros((shots, self.outputs), dtype=np.uint8) if retain else None
        counts = np.zeros(self.outputs - self.detectors, dtype=np.uint64)
        if parameters is not None:
            assert parameters.dtype == np.uint64 and parameters.shape == (shots, self.words)
            assert parameters.flags.c_contiguous
        passed = LIB.sample(
            self.ptr,
            shots,
            seed,
            int(postselect),
            None if output is None else output.ctypes.data,
            None if parameters is None else parameters.ctypes.data,
            counts.ctypes.data,
            int(noise_only),
        )
        return passed, output, counts

    def probability(self, parameters, bits):
        p = np.asarray(parameters, dtype=np.uint64)
        b = np.asarray(bits, dtype=np.uint8)
        assert p.shape == (self.words,) and b.shape == (self.outputs,)
        return LIB.probability(self.ptr, p.ctypes.data, b.ctypes.data)

    def __del__(self):
        if hasattr(self, "ptr"):
            LIB.free_plan(self.ptr)

    def expectation(self, parameters, selection):
        p = np.asarray(parameters, dtype=np.uint64)
        b = np.asarray(selection, dtype=np.uint8)
        assert p.shape == (self.words,) and b.shape == (self.outputs,)
        return LIB.expectation(self.ptr, p.ctypes.data, b.ctypes.data)
