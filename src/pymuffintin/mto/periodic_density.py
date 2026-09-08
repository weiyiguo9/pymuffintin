"""Bloch-periodic augmented NMTO orbitals and their scalar density.

The periodic free envelope is retained inside every sphere.  Only its active
regular partial waves are replaced by current-potential radial solutions;
inactive angular components are not discarded.  Koelling--Harmon orbitals
carry both radial and tangential small-component contributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial, pi, sqrt
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.special import spherical_in, spherical_jn

from ..symmetry import SymmetryDataset
from ..tensor import contract
from .density import ScalarRadialSamples
from .electrons import NmtoBands, NmtoOccupations, interpolate_nmto_basis, nmto_density_matrices
from .nmto import NmtoResult
from .periodic import PeriodicUswSample
from .usw import RealHarmonic


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _angular_values_and_gradients(
    displacements: FloatArray, channels: tuple[RealHarmonic, ...]
) -> tuple[FloatArray, FloatArray]:
    """Return real harmonics and pole-safe Cartesian tangential gradients.

    Homogeneous regular solid harmonics are differentiated as polynomials,
    then their radial derivative is removed on the unit sphere.  At the
    origin an arbitrary common direction represents the regular radial
    limit; density contractions do not depend on that direction.
    """

    radii = np.linalg.norm(displacements, axis=1)
    directions = np.zeros_like(displacements)
    directions[:, 2] = 1.0
    np.divide(displacements, radii[:, None], out=directions, where=radii[:, None] > 0.0)
    x_plus_iy = directions[:, 0] + 1j * directions[:, 1]
    z = directions[:, 2]
    z_gradient = np.array([0.0, 0.0, 1.0])
    xy_gradient = np.array([1.0, 1j, 0.0])
    polynomials: dict[tuple[int, int], tuple[ComplexArray, ComplexArray]] = {}
    maximum_l = max(channel.l for channel in channels)
    for m in range(maximum_l + 1):
        if m == 0:
            value = np.ones(len(displacements), dtype=np.complex128)
            gradient = np.zeros((len(displacements), 3), dtype=np.complex128)
        else:
            previous, previous_gradient = polynomials[(m - 1, m - 1)]
            value = -(2 * m - 1) * x_plus_iy * previous
            gradient = -(2 * m - 1) * (
                x_plus_iy[:, None] * previous_gradient
                + previous[:, None] * xy_gradient
            )
        polynomials[(m, m)] = value, gradient
        for l in range(m + 1, maximum_l + 1):
            previous, previous_gradient = polynomials[(l - 1, m)]
            value = (2 * l - 1) * z * previous
            gradient = (2 * l - 1) * (
                z[:, None] * previous_gradient + previous[:, None] * z_gradient
            )
            if l > m + 1:
                earlier, earlier_gradient = polynomials[(l - 2, m)]
                value -= (l + m - 1) * earlier
                gradient -= (l + m - 1) * (
                    earlier_gradient + 2.0 * directions * earlier[:, None]
                )
            polynomials[(l, m)] = value / (l - m), gradient / (l - m)

    values = np.empty((len(displacements), len(channels)))
    gradients = np.empty((len(displacements), 3, len(channels)))
    for column, channel in enumerate(channels):
        l, m = channel.l, channel.m
        normalization = sqrt(
            (2 * l + 1) * factorial(l - abs(m)) / (4.0 * pi * factorial(l + abs(m)))
        )
        solid, solid_gradient = polynomials[(l, abs(m))]
        angular = normalization * solid
        angular_gradient = normalization * (
            solid_gradient - l * directions * solid[:, None]
        )
        if m > 0:
            factor = sqrt(2.0) * (-1) ** m
            values[:, column] = factor * angular.real
            gradients[:, :, column] = factor * angular_gradient.real
        elif m < 0:
            values[:, column] = -sqrt(2.0) * angular.imag
            gradients[:, :, column] = -sqrt(2.0) * angular_gradient.imag
        else:
            values[:, column] = angular.real
            gradients[:, :, column] = angular_gradient.real
    return values, gradients


def _free_radial_ratio(l: int, energy: float, radii: FloatArray, boundary: float) -> FloatArray:
    """Regular free radial solution with unit value at the sphere boundary."""

    if energy > 0.0:
        wave_number = sqrt(2.0 * energy)
        return spherical_jn(l, wave_number * radii) / spherical_jn(l, wave_number * boundary)
    if energy < 0.0:
        wave_number = sqrt(-2.0 * energy)
        return spherical_in(l, wave_number * radii) / spherical_in(l, wave_number * boundary)
    return (radii / boundary) ** l


@dataclass(frozen=True)
class PeriodicNmtoBasisEvaluator:
    """One common periodic orbital representation for bands and density.

    ``periodic_samples`` is k-major, then energy-major.  The small-component
    representation has four components: its scalar radial amplitude and
    three Cartesian tangential-gradient amplitudes.  Their squared norm is
    the spin-averaged Koelling--Harmon metric, including cross-channel terms.
    """

    direct_lattice: FloatArray
    site_fractional: FloatArray
    muffin_tin_radii: FloatArray
    channels: tuple[RealHarmonic, ...]
    energies: FloatArray
    interstitial_energies: FloatArray
    k_cartesian: FloatArray
    k_weights: FloatArray
    results: tuple[NmtoResult, ...]
    bands: NmtoBands
    occupations: NmtoOccupations
    periodic_samples: tuple[tuple[PeriodicUswSample, ...], ...]
    radial_samples: Mapping[tuple[int, int], ScalarRadialSamples]
    symmetry: SymmetryDataset | None
    symmetry_operation_indices: NDArray[np.int64] | None = None

    def density(self, points: FloatArray) -> FloatArray:
        """Evaluate the symmetry-projected scalar valence density."""

        sample_points = np.asarray(points, dtype=np.float64)
        if self.symmetry is None:
            return self._raw_density(sample_points)
        fractional = np.mod(sample_points @ np.linalg.inv(self.direct_lattice), 1.0)
        result = np.zeros(len(sample_points), dtype=np.float64)
        operations = self._symmetry_operations()
        for operation in operations:
            rotation = self.symmetry.rotations[operation]
            translation = self.symmetry.translations[operation]
            transformed = (
                np.mod((fractional - translation) @ np.linalg.inv(rotation).T, 1.0)
                @ self.direct_lattice
            )
            result += self._raw_density(transformed)
        return result / len(operations)

    def _symmetry_operations(self) -> NDArray[np.int64]:
        if self.symmetry is None:
            return np.empty(0, dtype=np.int64)
        if self.symmetry_operation_indices is None:
            return np.arange(len(self.symmetry.rotations), dtype=np.int64)
        return np.unique(self.symmetry_operation_indices)

    def _raw_density(self, points: FloatArray) -> FloatArray:
        density_matrices = nmto_density_matrices(self.bands, self.occupations)
        density = np.zeros(len(points), dtype=np.float64)
        for k_index, weight in enumerate(self.k_weights):
            large, small = self._basis_values(points, k_index)
            values = contract(
                "pa,ab,pb->p", large, density_matrices[k_index], large.conj()
            ) + contract(
                "pca,ab,pcb->p", small, density_matrices[k_index], small.conj()
            )
            density += weight * values.real
        return density

    def _basis_values(self, points: FloatArray, k_index: int) -> tuple[ComplexArray, ComplexArray]:
        """Evaluate augmented energy-node orbitals before NMTO interpolation."""

        sample_points = np.asarray(points, dtype=np.float64)
        samples = self.periodic_samples[k_index]
        reference_values = samples[0].geometry.reference_values(sample_points)
        large_nodes = np.stack(
            tuple(sample.evaluate(sample_points, reference_values=reference_values) for sample in samples)
        )
        small_nodes = np.zeros(
            (len(samples), len(sample_points), 4, large_nodes.shape[-1]),
            dtype=np.complex128,
        )
        displacements, sphere_sites = self._nearest_sites(sample_points)
        site_cartesian = self.site_fractional @ self.direct_lattice
        for site in range(len(self.site_fractional)):
            selected = sphere_sites == site
            if not np.any(selected):
                continue
            local_displacements = displacements[selected]
            radii = np.linalg.norm(local_displacements, axis=1)
            angular, angular_gradients = _angular_values_and_gradients(local_displacements, self.channels)
            image_translations = sample_points[selected] - local_displacements - site_cartesian[site]
            phase = np.exp(1j * (image_translations @ self.k_cartesian[k_index]))
            for local, channel in enumerate(self.channels):
                radial = self.radial_samples[(site, channel.l)]
                column = site * len(self.channels) + local
                interpolation_radii = np.clip(radii, radial.mesh_radii[0], radial.mesh_radii[-1])
                atomic_large_rows = radial.large_interpolant(interpolation_radii)
                atomic_small_rows = radial.small_interpolant(interpolation_radii)
                tangential_rows = radial.tangential_interpolant(interpolation_radii)
                for node, energy in enumerate(self.interstitial_energies):
                    boundary = radial.boundary_values[node]
                    atomic_large = atomic_large_rows[node] / boundary
                    atomic_small = atomic_small_rows[node] / boundary
                    tangential = tangential_rows[node] / boundary
                    if channel.l > 0:
                        atomic_large[radii == 0.0] = 0.0
                    if channel.l != 1:
                        atomic_small[radii == 0.0] = 0.0
                        tangential[radii == 0.0] = 0.0
                    regular = _free_radial_ratio(
                        channel.l, float(energy), radii, float(self.muffin_tin_radii[site])
                    )
                    large_nodes[node, selected, column] += phase * angular[:, local] * (atomic_large - regular)
                    small_nodes[node, selected, 0, column] = phase * angular[:, local] * atomic_small
                    small_nodes[node, selected, 1:, column] = (
                        (phase * tangential)[:, None] * angular_gradients[:, :, local]
                    )
        lagrange = self.results[k_index].lagrange_matrices
        large = interpolate_nmto_basis(large_nodes, lagrange)
        small = contract("epca,eab->pcb", small_nodes, lagrange)
        return large, small

    def _nearest_sites(self, points: FloatArray) -> tuple[FloatArray, NDArray[np.int64]]:
        """Locate the containing sphere in the periodic primitive cell."""

        inverse_lattice = np.linalg.inv(self.direct_lattice)
        fractional = np.mod(points @ inverse_lattice, 1.0)
        bounds = np.ceil(
            np.max(self.muffin_tin_radii) * np.linalg.norm(inverse_lattice, axis=0)
        ).astype(int)
        translations = np.asarray(
            tuple(product(*(range(-bound, bound + 1) for bound in bounds))), dtype=float
        )
        best_distance = np.full(len(points), np.inf)
        best_displacement = np.zeros_like(points)
        best_site = np.full(len(points), -1, dtype=np.int64)
        for site, position in enumerate(self.site_fractional):
            candidates = (
                fractional[:, None, :] - np.mod(position, 1.0)[None, None, :]
                - translations[None, :, :]
            ) @ self.direct_lattice
            distances = np.linalg.norm(candidates, axis=2)
            images = np.argmin(distances, axis=1)
            rows = np.arange(len(points))
            site_distances = distances[rows, images]
            replace = (site_distances <= self.muffin_tin_radii[site]) & (
                site_distances < best_distance
            )
            best_distance[replace] = site_distances[replace]
            best_displacement[replace] = candidates[rows[replace], images[replace]]
            best_site[replace] = site
        return best_displacement, best_site
