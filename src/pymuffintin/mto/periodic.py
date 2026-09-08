"""Bloch-periodic spherical waves from a reference-subtracted layer operator.

The free Hamiltonian is ``-laplacian/2`` (Hartree energies, Bohr lengths).
The negative reference energy is a numerical resolvent-splitting parameter,
not an interstitial potential or a change of the physical energy mesh.
Real-space bare lattice sums are formed before the primitive-cell solve.
No outgoing/standing-wave branch is selected at the requested energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import pi, sqrt

import numpy as np
from numpy.typing import NDArray

from ..tensor import contract, solve
from .usw import (
    RealHarmonic,
    _decaying_hankel,
    _radial_at_sphere,
    _real_gaunt,
    _regular_bessel,
    _regular_bessel_with_energy_derivative,
    _translation_channels,
    real_spherical_harmonics,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _vectors_within(basis: FloatArray, shift: FloatArray, radius: float) -> FloatArray:
    inverse = np.linalg.inv(basis)
    bounds = np.ceil(
        radius * np.linalg.norm(inverse, axis=0) + np.abs(shift @ inverse)
    ).astype(int)
    integers = np.asarray(
        tuple(product(*(range(-int(bound), int(bound) + 1) for bound in bounds))),
        dtype=float,
    )
    vectors = integers @ basis + shift
    return vectors[np.linalg.norm(vectors, axis=1) <= radius]


def _spherical_bessel(l: int, arguments: FloatArray) -> FloatArray:
    """Standard spherical Bessel, using its series below the recurrence regime."""

    result = np.empty_like(arguments)
    zero = arguments == 0.0
    result[zero] = 1.0 if l == 0 else 0.0
    small = (arguments > 0.0) & (arguments < l + 1.0)
    if np.any(small):
        result[small], _ = _regular_bessel(l, 0.5, arguments[small])
    large = arguments >= l + 1.0
    x = arguments[large]
    lower = np.sin(x) / x
    if l == 0:
        result[large] = lower
        return result
    current = (lower - np.cos(x)) / x
    for degree in range(1, l):
        lower, current = current, (2 * degree + 1) * current / x - lower
    result[large] = current
    return result


def _angular_at_vectors(
    vectors: FloatArray, channels: tuple[RealHarmonic, ...]
) -> tuple[FloatArray, FloatArray]:
    distance = np.linalg.norm(vectors, axis=1)
    directions = vectors.copy()
    directions[distance == 0.0] = (0.0, 0.0, 1.0)
    return distance, real_spherical_harmonics(directions, channels)


@dataclass(frozen=True)
class PeriodicUswGeometry:
    """Geometry and reusable sums for one Cartesian Bloch vector.

    ``sites`` are Cartesian primitive-cell sites; lattice vectors are rows.
    ``g_cutoff`` bounds ``|k+G|`` and ``lattice_radius`` bounds ``|T|``.
    Both cutoffs are explicit numerical convergence parameters. A typical
    reference is ``reference_energy=-1.0`` Hartree; it must be negative.
    """

    lattice: FloatArray
    sites: FloatArray
    radii: FloatArray
    channels: tuple[RealHarmonic, ...]
    k: FloatArray
    g_cutoff: float
    reference_energy: float
    lattice_radius: float
    volume: float = field(init=False)
    translations: FloatArray = field(init=False, repr=False)
    wave_vectors: FloatArray = field(init=False, repr=False)
    reciprocal_indices: NDArray[np.int64] = field(init=False, repr=False)
    kinetic_energies: FloatArray = field(init=False, repr=False)
    form_factors: ComplexArray = field(init=False, repr=False)
    reference_boundary: ComplexArray = field(init=False, repr=False)
    reference_regular: FloatArray = field(init=False, repr=False)
    reference_hankel: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.reference_energy >= 0.0:
            raise ValueError("periodic USW reference energy must be negative")
        if self.g_cutoff <= 0.0 or self.lattice_radius <= 0.0:
            raise ValueError("periodic USW cutoffs must be positive")
        for name in ("lattice", "sites", "radii", "k"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "volume", float(abs(np.linalg.det(self.lattice))))
        reciprocal = 2.0 * pi * np.linalg.inv(self.lattice).T
        translations = _vectors_within(self.lattice, np.zeros(3), self.lattice_radius)
        wave_vectors = _vectors_within(reciprocal, self.k, self.g_cutoff)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "wave_vectors", wave_vectors)
        object.__setattr__(
            self, "reciprocal_indices",
            np.rint((wave_vectors - self.k) @ np.linalg.inv(reciprocal)).astype(np.int64),
        )
        distance, angular = _angular_at_vectors(wave_vectors, self.channels)
        object.__setattr__(self, "kinetic_energies", 0.5 * distance * distance)
        count = len(self.channels)
        form = np.empty((len(wave_vectors), len(self.sites) * count), dtype=complex)
        regular = np.empty(len(self.sites) * count)
        neumann = np.empty_like(regular)
        hankel = np.empty_like(regular)
        for site, (center, radius) in enumerate(zip(self.sites, self.radii, strict=True)):
            phase = np.exp(-1j * (wave_vectors @ center))
            for local, channel in enumerate(self.channels):
                index = site * count + local
                form[:, index] = (
                    4.0 * pi * (-1j) ** channel.l * phase * angular[:, local]
                    * _spherical_bessel(channel.l, distance * radius)
                )
                sphere = np.asarray([radius])
                j, _, n = _radial_at_sphere(channel.l, self.reference_energy, sphere)
                h, _ = _decaying_hankel(channel.l, self.reference_energy, sphere)
                regular[index], neumann[index], hankel[index] = j[0], n[0], h[0]
        object.__setattr__(self, "form_factors", form)
        object.__setattr__(self, "reference_regular", regular)
        object.__setattr__(self, "reference_hankel", hankel)
        bare = self._reference_bare()
        boundary = -2.0 * regular[:, None] * bare * regular[None, :]
        boundary -= np.diag(2.0 * regular * neumann)
        object.__setattr__(self, "reference_boundary", boundary)

    def _reference_bare(self) -> ComplexArray:
        """Sum central-to-translated bare blocks, without a cluster inversion."""

        count = len(self.channels)
        size = len(self.sites) * count
        result = np.zeros((size, size), dtype=complex)
        q = sqrt(-2.0 * self.reference_energy)
        for site in range(len(self.sites)):
            for local, channel in enumerate(self.channels):
                index = site * count + local
                result[index, index] = (-1) ** channel.l * q ** (2 * channel.l + 1)
        translated_channels = _translation_channels(max(c.l for c in self.channels))
        gaunt = _real_gaunt(self.channels, translated_channels)
        kernel = np.zeros_like(gaunt)
        for a, row in enumerate(self.channels):
            for b, column in enumerate(self.channels):
                for c, translated in enumerate(translated_channels):
                    exponent = row.l + column.l - translated.l
                    if exponent < 0 or exponent % 2:
                        continue
                    # Displacement points from the source (column) to the
                    # target (row), as in the Fourier surface-source factors.
                    kernel[a, b, c] = (
                        4.0 * pi * (-1) ** (column.l - translated.l)
                        * (-2.0 * self.reference_energy) ** (exponent // 2)
                        * gaunt[a, b, c]
                    )
        for a, row in enumerate(self.sites):
            for b, column in enumerate(self.sites):
                displacement = row - column - self.translations
                selected = np.any(displacement != 0.0, axis=1)
                displacement = displacement[selected]
                if len(displacement) == 0:
                    continue
                distance, angular = _angular_at_vectors(displacement, translated_channels)
                for c, channel in enumerate(translated_channels):
                    radial, _ = _decaying_hankel(channel.l, self.reference_energy, distance)
                    angular[:, c] *= radial
                phase = np.exp(1j * (self.translations[selected] @ self.k))
                result[a * count : (a + 1) * count, b * count : (b + 1) * count] += (
                    contract("abc,tc,t->ab", kernel, angular, phase)
                )
        return result

    def sample(self, energy: float) -> PeriodicUswSample:
        """Solve one periodic boundary problem and its analytic energy derivative."""

        denominator = self.kinetic_energies - energy
        if np.any(denominator == 0.0):
            raise ValueError(f"periodic free-resolvent pole at energy {energy}")
        reference_denominator = self.kinetic_energies - self.reference_energy
        correction = (energy - self.reference_energy) / (
            self.volume * denominator * reference_denominator
        )
        form = self.form_factors
        boundary = self.reference_boundary + contract("pi,p,pj->ij", form.conj(), correction, form)
        boundary_dot = contract(
            "pi,p,pj->ij", form.conj(), 1.0 / (self.volume * denominator**2), form
        )
        inverse = solve(boundary, np.eye(len(boundary), dtype=complex))
        radius = np.repeat(self.radii, len(self.channels))
        angular_l = np.tile([channel.l for channel in self.channels], len(self.sites))
        diagonal = np.empty_like(radius)
        diagonal_dot = np.empty_like(radius)
        for l in sorted(set(angular_l)):
            selected = angular_l == l
            j, jr, je, jre = _regular_bessel_with_energy_derivative(int(l), energy, radius[selected])
            diagonal[selected] = radius[selected] * jr / j
            diagonal_dot[selected] = radius[selected] * (jre * j - jr * je) / j**2
        slope = np.diag(diagonal) - 2.0 * inverse / radius[:, None]
        slope_dot = np.diag(diagonal_dot) + 2.0 / radius[:, None] * contract(
            "ij,jk,kl->il", inverse, boundary_dot, inverse
        )
        fourier = contract("pi,ij->pj", form, inverse) / (self.volume * denominator[:, None])
        return PeriodicUswSample(self, float(energy), slope, slope_dot, inverse, fourier)

    def reference_values(self, points: FloatArray) -> ComplexArray:
        """Evaluate the negative-reference source layers, including shell interiors."""

        points = np.asarray(points, dtype=float)
        count = len(self.channels)
        values = np.zeros((len(points), len(self.sites) * count), dtype=complex)
        phase = np.exp(1j * (self.translations @ self.k))
        for site, (center, radius) in enumerate(zip(self.sites, self.radii, strict=True)):
            for translation, bloch_phase in zip(self.translations, phase, strict=True):
                distance, angular = _angular_at_vectors(points - center - translation, self.channels)
                inside = distance < radius
                outside = ~inside
                for local, channel in enumerate(self.channels):
                    index = site * count + local
                    radial = np.empty(len(points))
                    if np.any(outside):
                        h, _ = _decaying_hankel(channel.l, self.reference_energy, distance[outside])
                        radial[outside] = -2.0 * self.reference_regular[index] * h
                    nonzero_inside = inside & (distance > 0.0)
                    if np.any(nonzero_inside):
                        j, _ = _regular_bessel(channel.l, self.reference_energy, distance[nonzero_inside])
                        radial[nonzero_inside] = -2.0 * self.reference_hankel[index] * j
                    radial[distance == 0.0] = (
                        -2.0 * self.reference_hankel[index] if channel.l == 0 else 0.0
                    )
                    values[:, index] += bloch_phase * radial * angular[:, local]
        return values


@dataclass(frozen=True)
class PeriodicUswSample:
    """One periodic unit-boundary basis and its consistent slope matrices.

    ``fourier_coefficients`` are the full Bloch coefficients, not the smooth
    reference correction. ``evaluate`` applies the resolvent subtraction
    before a batched type-2 NUFFT, with tolerance 1e-12 and one FFT thread.
    """

    geometry: PeriodicUswGeometry
    energy: float
    slope: ComplexArray
    slope_derivative: ComplexArray
    boundary_inverse: ComplexArray
    fourier_coefficients: ComplexArray

    def evaluate(
        self, points: FloatArray, *, reference_values: ComplexArray | None = None
    ) -> ComplexArray:
        """Evaluate full envelopes; a reusable reference layer may be supplied."""

        import finufft

        geometry = self.geometry
        reference = geometry.reference_values(points) if reference_values is None else reference_values
        coefficients = self.fourier_coefficients * (
            (self.energy - geometry.reference_energy)
            / (geometry.kinetic_energies - geometry.reference_energy)
        )[:, None]
        indices = geometry.reciprocal_indices
        half_width = np.max(np.abs(indices), axis=0)
        grid = np.zeros(
            (coefficients.shape[1], *(2 * half_width + 1)), dtype=complex
        )
        shifted = indices + half_width
        grid[:, shifted[:, 0], shifted[:, 1], shifted[:, 2]] = coefficients.T
        points = np.asarray(points, dtype=float)
        theta = np.mod(2.0 * pi * (points @ np.linalg.inv(geometry.lattice)) + pi, 2.0 * pi) - pi
        smooth = finufft.nufft3d2(
            np.ascontiguousarray(theta[:, 0]),
            np.ascontiguousarray(theta[:, 1]),
            np.ascontiguousarray(theta[:, 2]),
            grid,
            eps=1.0e-12,
            isign=1,
            nthreads=1,
        )
        return contract("pi,ij->pj", reference, self.boundary_inverse) + (
            smooth.T * np.exp(1j * (points @ geometry.k))[:, None]
        )
