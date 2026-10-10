"""Bounded offline Clifford-instrument reduction on an encoded small state.

This research host does tableau algebra before handing a reconstructed state
to ordinary Clifft. It is not an executor or a production continuation API.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass

import numpy as np
import stim


class Refused(ValueError):
    """The exact bounded representation has no certificate for this operation."""


def masks(p: stim.PauliString) -> tuple[int, int]:
    return (
        sum(1 << q for q, a in enumerate(p) if a in (1, 2)),
        sum(1 << q for q, a in enumerate(p) if a in (2, 3)),
    )


def body(p: stim.PauliString) -> int:
    x, z = masks(p)
    return x | (z << len(p))


def apply(p: stim.PauliString, vector: np.ndarray) -> np.ndarray:
    x, z = masks(p)
    result = np.empty_like(vector)
    phase = p.sign * 1j ** (x & z).bit_count()
    for i, value in enumerate(vector):
        result[i ^ x] = value * phase * (-1) ** (i & z).bit_count()
    return result


def independent(rows: list[stim.PauliString]) -> list[stim.PauliString]:
    basis = {}
    result = []
    for p in rows:
        word = body(p)
        while word:
            pivot = word.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = word
                result.append(p)
                break
            word ^= basis[pivot]
    return result


def intersection(
    left: list[stim.PauliString], right: list[stim.PauliString]
) -> list[stim.PauliString]:
    """Intersect signed stabilizer groups without enumerating their elements."""
    if not left or not right:
        return []
    basis: dict[int, tuple[int, int]] = {}
    relations = []
    for i, row in enumerate(left + right):
        word, choice = body(row), 1 << i
        while word:
            pivot = word.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = word, choice
                break
            other, selected = basis[pivot]
            word ^= other
            choice ^= selected
        if not word:
            relations.append(choice)
    result = []
    negative = None
    for choice in relations:
        a, b = stim.PauliString(len(left[0])), stim.PauliString(len(left[0]))
        for i, p in enumerate(left):
            if choice >> i & 1:
                a *= p
        for i, p in enumerate(right, len(left)):
            if choice >> i & 1:
                b *= p
        if a == b:
            result.append(a)
        elif a == -b:
            if negative is None:
                negative = a
            else:
                result.append(a * negative)
        else:
            raise AssertionError("Invalid stabilizer intersection relation")
    return independent(result)


class StabilizerAmplitudes:
    """A canonical stabilizer ray with a positive amplitude at one support point."""

    def __init__(self, frame: stim.Tableau):
        self.n = len(frame)
        self.x_basis: dict[int, stim.PauliString] = {}
        equations = []
        for i in range(self.n):
            row = frame.z_output(i)
            x, _ = masks(row)
            while x:
                pivot = x.bit_length() - 1
                if pivot not in self.x_basis:
                    self.x_basis[pivot] = row
                    break
                row *= self.x_basis[pivot]
                x, _ = masks(row)
            if not x:
                equations.append((masks(row)[1], int(row.sign.real < 0)))
        basis = {}
        for word, value in equations:
            while word:
                pivot = word.bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = word, value
                    break
                other, sign = basis[pivot]
                word ^= other
                value ^= sign
            if not word and value:
                raise AssertionError("Inconsistent stabilizer support")
        self.anchor = 0
        for pivot, (word, value) in sorted(basis.items()):
            if ((word & self.anchor).bit_count() & 1) != value:
                self.anchor ^= 1 << pivot
        self.scale = 2 ** (-len(self.x_basis) / 2)

    def amplitude(self, index: int) -> complex:
        remaining = index ^ self.anchor
        p = stim.PauliString(self.n)
        while remaining:
            pivot = remaining.bit_length() - 1
            if pivot not in self.x_basis:
                return 0j
            row = self.x_basis[pivot]
            remaining ^= masks(row)[0]
            p *= row
        x, z = masks(p)
        return complex(
            self.scale * p.sign * 1j ** (x & z).bit_count() * (-1) ** (z & self.anchor).bit_count()
        )

    def pauli_amplitude(self, p: stim.PauliString, index: int) -> complex:
        x, z = masks(p)
        source = index ^ x
        return complex(
            p.sign
            * 1j ** (x & z).bit_count()
            * (-1) ** (z & source).bit_count()
            * self.amplitude(source)
        )


@dataclass
class MonomialClifford:
    """The exact operator D-dagger P D, including its physical global phase."""

    pauli: stim.PauliString
    signs: dict[int, int]

    def __post_init__(self) -> None:
        if not len(self.pauli) or any(
            not 0 <= q < len(self.pauli) or sign not in (-1, 1) for q, sign in self.signs.items()
        ):
            raise ValueError("Invalid signed single-T layer")

    def tableau(self) -> stim.Tableau:
        circuit = stim.Circuit(f"I {len(self.pauli) - 1}")
        for q, axis in enumerate(self.pauli):
            if axis:
                circuit.append("IXYZ"[axis], [q])
            if axis in (1, 2) and q in self.signs:
                circuit.append("S_DAG" if self.signs[q] == 1 else "S", [q])
        return circuit.to_tableau()

    def amplitude(self, state: StabilizerAmplitudes, index: int) -> complex:
        x = masks(self.pauli)[0]
        source = index ^ x
        exponent = sum(
            sign * (((source >> q) & 1) - ((index >> q) & 1)) for q, sign in self.signs.items()
        )
        return cmath.exp(1j * np.pi * exponent / 4) * state.pauli_amplitude(self.pauli, index)


class SmallState:
    """A Clifford encoder of a bounded vector on its last coordinates."""

    def __init__(self, n: int):
        self.frame = stim.Tableau(n)
        self.vector = np.array([1.0 + 0j])

    @property
    def width(self) -> int:
        return len(self.vector).bit_length() - 1

    @property
    def n(self) -> int:
        return len(self.frame)

    def clifford(self, circuit: stim.Circuit) -> None:
        self.frame = self.frame.then((circuit + stim.Circuit(f"I {self.n - 1}")).to_tableau())

    def decode(self, p: stim.PauliString) -> stim.PauliString:
        return self.frame.inverse()(p)

    def expectation(self, p: stim.PauliString) -> float:
        decoded = self.decode(p)
        inactive = self.n - self.width
        if any(decoded[q] in (1, 2) for q in range(inactive)):
            return 0.0
        reduced = decoded[inactive:]
        reduced.sign = decoded.sign
        return float(np.vdot(self.vector, apply(reduced, self.vector)).real)

    def compress(self, limit: int = 1) -> None:
        r = self.width
        if not r:
            return
        # Enumeration is restricted to the explicitly bounded quantum core.
        rows = []
        for p in stim.PauliString.iter_all(r):
            if not body(p):
                continue
            value = float(np.vdot(self.vector, apply(p, self.vector)).real)
            if abs(abs(value) - 1) < 1e-9:
                p.sign = 1 if value > 0 else -1
                rows.append(p)
        rows = independent(rows)
        residual = r - len(rows)
        if residual > limit:
            raise Refused(f"Output requires {residual} active coordinates; limit is {limit}")
        if not rows:
            return
        local = stim.Tableau.from_stabilizers(rows, allow_underconstrained=True)
        if len(local) < r:
            raise AssertionError("Lost the bounded core dimension")
        matrix = local.inverse().to_unitary_matrix(endian="little").astype(complex)
        matrix /= np.linalg.norm(matrix[:, 0])
        decoded = matrix @ self.vector
        indices = np.arange(1 << residual) << len(rows)
        result = decoded[indices]
        if abs(float(np.vdot(result, result).real) - 1) > 1e-8:
            raise AssertionError("Compression lost state norm")
        lift = stim.Tableau(self.n)
        lift.append(local, list(range(self.n - r, self.n)))
        self.frame = lift.then(self.frame)
        self.vector = result / np.linalg.norm(result)

    def measure(self, p: stim.PauliString, outcome: int) -> float:
        expectation = self.expectation(p)
        probability = (1 + (-1) ** outcome * expectation) / 2
        if probability < 1e-12:
            return 0.0
        decoded = self.decode(p)
        inactive = self.n - self.width
        pivot = next((q for q in range(inactive) if decoded[q] in (1, 2)), None)
        if pivot is not None:
            z = stim.PauliString(self.n)
            z[pivot] = "Z"
            rotation = decoded * z
            rotation.sign *= -1j * (-1) ** outcome
            targets = "*".join("IXYZ"[a] + str(q) for q, a in enumerate(rotation) if a)
            if rotation.sign.real < 0:
                targets = "!" + targets
            # On the dormant +Z coordinate this quarter turn is the normalized projector.
            update = stim.Circuit(f"SPP_DAG {targets}\nI {self.n - 1}").to_tableau()
            self.frame = update.then(self.frame)
        else:
            local = decoded[inactive:]
            local.sign = decoded.sign
            self.vector = (self.vector + (-1) ** outcome * apply(local, self.vector)) / (
                2 * np.sqrt(probability)
            )
            self.compress()
        return float(probability)

    def rotate_t(self, p: stim.PauliString, sign: int) -> None:
        """Inject one coordinate or rotate within the existing certified core."""
        decoded = self.decode(p)
        if self.width:
            inactive = self.n - self.width
            if any(decoded[q] in (1, 2) for q in range(inactive)):
                raise Refused("An ungrouped T gate expands the existing quantum core")
            local = decoded[inactive:]
            local.sign = decoded.sign
            self.vector = np.cos(np.pi / 8) * self.vector - (
                1j * sign * np.sin(np.pi / 8) * apply(local, self.vector)
            )
            self.compress()
            return
        pivot = next((q for q, axis in enumerate(decoded) if axis in (1, 2)), None)
        if pivot is None:
            return
        rows = [self.frame.z_output(i) for i in range(self.n)]
        reference = rows[pivot]
        fixed = []
        for i, row in enumerate(rows):
            if i != pivot:
                fixed.append(row if row.commutes(p) else row * reference)
        frame = stim.Tableau.from_stabilizers(fixed + [reference])
        local = frame.inverse()(p)[-1:]
        local.sign = frame.inverse()(p).sign
        self.frame = frame
        self.vector = np.cos(np.pi / 8) * np.array([1, 0], complex) - (
            1j * sign * np.sin(np.pi / 8) * apply(local, np.array([1, 0], complex))
        )

    def in_frame(self, frame: stim.Tableau, width: int) -> np.ndarray:
        conversion = self.frame.then(frame.inverse())
        inactive = self.n - width
        rows = []
        for i in range(self.n):
            p = conversion.z_output(i)
            if any(p[q] in (1, 2) for q in range(inactive)):
                raise Refused("The common subspace does not contain the input")
            local = p[inactive:]
            local.sign = p.sign
            if body(local):
                rows.append(local)
            elif local.sign != 1:
                raise Refused("The input has a conflicting dormant sign")
        zero = (
            stim.Tableau.from_stabilizers(rows, allow_redundant=True)
            .to_state_vector(endian="little")
            .astype(complex)
        )
        result = np.zeros(1 << width, complex)
        for i, coefficient in enumerate(self.vector):
            p = stim.PauliString(self.n)
            for j in range(self.width):
                if i >> j & 1:
                    p *= conversion.x_output(self.n - self.width + j)
            if any(p[q] in (1, 2) for q in range(inactive)):
                raise Refused("A logical input changes the common dormant state")
            local = p[inactive:]
            local.sign = p.sign
            result += coefficient * apply(local, zero)
        result /= np.linalg.norm(result)
        return result

    def clifford_projector(
        self, operator: MonomialClifford, outcome: int | None, *, max_core: int = 3
    ) -> tuple[float, dict]:
        """Certify an invariant small subspace and evaluate the full coherent projector.

        With outcome=None this is a read-only probability query. Relative
        Clifford phases are fixed by physical amplitudes, not tableau phases.
        """
        action = operator.tableau()
        rows = [self.frame.z_output(i) for i in range(self.n - self.width)]
        common = intersection(rows, [action(p) for p in rows])
        width = self.n - len(common)
        if width > max_core:
            raise Refused(f"Common-subspace core {width} exceeds bound {max_core}")
        frame = stim.Tableau.from_stabilizers(common, allow_underconstrained=True)
        if len(frame) != self.n:
            if common:
                raise AssertionError("Wrong common-subspace dimension")
            frame = stim.Tableau(self.n)
        inactive = self.n - width
        restricted = frame.then(action).then(frame.inverse())
        for i in range(inactive):
            p = restricted.z_output(i)
            if (
                p.sign != 1
                or any(p[q] not in (0, 3) for q in range(inactive))
                or any(p[q] for q in range(inactive, self.n))
            ):
                raise Refused("The signed common subspace is not invariant")
        xs: list[stim.PauliString] = []
        zs: list[stim.PauliString] = []
        for q in range(inactive, self.n):
            for values, p in ((xs, restricted.x_output(q)), (zs, restricted.z_output(q))):
                if any(p[j] in (1, 2) for j in range(inactive)):
                    raise Refused("The Clifford leaves the certified subspace")
                local = p[inactive:]
                local.sign = p.sign
                values.append(local)
        logical = stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)
        matrix = logical.to_unitary_matrix(endian="little").astype(complex)
        amplitude = StabilizerAmplitudes(frame)
        anchor = amplitude.anchor ^ masks(operator.pauli)[0]
        denominator = 0j
        for i in range(1 << width):
            p = stim.PauliString(self.n)
            for j in range(width):
                if i >> j & 1:
                    p *= frame.x_output(inactive + j)
            denominator += matrix[i, 0] * amplitude.pauli_amplitude(p, anchor)
        numerator = operator.amplitude(amplitude, anchor)
        phase = numerator / denominator
        if abs(abs(phase) - 1) > 1e-6:
            raise AssertionError("Invalid restricted Clifford phase")
        matrix *= phase / abs(phase)
        # Stim's matrix conversion uses complex64; normalize before probability arithmetic.
        matrix /= np.linalg.norm(matrix[:, 0])
        if not np.allclose(matrix, matrix.conj().T, atol=1e-7):
            raise Refused("The proposed check is not Hermitian")
        vector = self.in_frame(frame, width)
        expectation = float(np.vdot(vector, matrix @ vector).real)
        probability = min(1.0, max(0.0, (1 + (-1) ** (outcome or 0) * expectation) / 2))
        metadata = dict(input_width=self.width, instrument_width=width, expectation=expectation)
        if outcome is not None and probability > 1e-12:
            self.vector = (vector + (-1) ** outcome * (matrix @ vector)) / (
                2 * np.sqrt(probability)
            )
            self.vector /= np.linalg.norm(self.vector)
            self.frame = frame
            self.compress()
            metadata["output_width"] = self.width
        return probability, metadata

    def source(self) -> str:
        if self.width > 1:
            raise Refused("Only a one-qubit core is exportable")
        prefix = ""
        if self.width:
            # The bounded-coordinate certificate alone does not certify a T-state orbit.
            for gate in ("", "T"):
                v = (
                    np.array([1, 0], complex)
                    if not gate
                    else np.array([1, cmath.exp(1j * np.pi / 4)]) / np.sqrt(2)
                )
                for local in stim.Tableau.iter_all(1):
                    candidate = local.to_unitary_matrix(endian="little") @ v
                    if abs(abs(np.vdot(candidate, self.vector)) - 1) < 1e-7:
                        q = self.n - 1
                        prefix = f"H {q}\nT {q}\n" if gate else ""
                        for line in str(local.to_circuit()).splitlines():
                            name, *targets = line.split()
                            prefix += name + " " + " ".join(str(q) for _ in targets) + "\n"
                        return prefix + str(self.frame.to_circuit())
            raise Refused("The remaining qubit is outside the stabilizer-or-T orbit")
        return prefix + str(self.frame.to_circuit())
