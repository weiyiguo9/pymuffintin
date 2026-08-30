"""Regional density synthesis from occupied scalar NMTO states."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import ModuleType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..regional import _complex_spherical_harmonics
from ..symmetry import SymmetryDataset
from ..tensor import contract
from .electrons import NmtoBands, NmtoOccupations, interpolate_nmto_basis, nmto_density_matrices
from .nmto import NmtoResult
from .usw import RealHarmonic, evaluate_folded_usw, real_spherical_harmonics


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ScalarRadialSamples:
    """Energy-major large and small physical radial functions for one site/l."""

    mesh_radii: FloatArray
    large: FloatArray
    small: FloatArray
    boundary_values: FloatArray

    @classmethod
    def from_export(cls, exported: Mapping[str, object]) -> ScalarRadialSamples:
        boundary = np.asarray(exported["boundary_radial"], dtype=np.float64)
        large = np.asarray(exported["radial_samples"], dtype=np.float64)
        small = np.asarray(exported["small_radial_samples"], dtype=np.float64)
        return cls(
            mesh_radii=np.asarray(exported["mesh_radii"], dtype=np.float64),
            large=large,
            small=small,
            boundary_values=boundary[:, 0],
        )


@dataclass(frozen=True)
class NmtoBasisEvaluator:
    """Evaluate occupied NMTO density in the primitive periodic cell."""

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
    translations: FloatArray
    centers: FloatArray
    folded_coefficients: ComplexArray
    radial_samples: Mapping[tuple[int, int], ScalarRadialSamples]
    symmetry: SymmetryDataset | None

    def __post_init__(self) -> None:
        site_count = len(self.site_fractional)
        primitive_size = site_count * len(self.channels)
        if len(self.results) != len(self.k_cartesian):
            raise ValueError("one NMTO result is required per k point")
        expected = (
            len(self.k_cartesian),
            len(self.energies),
            len(self.centers) * len(self.channels),
            primitive_size,
        )
        if self.folded_coefficients.shape != expected:
            raise ValueError(f"folded_coefficients must have shape {expected}")
        if self.k_weights.shape != (len(self.k_cartesian),):
            raise ValueError("k_weights must contain one value per k point")

    def density(self, points: FloatArray) -> FloatArray:
        """Evaluate the symmetry-projected scalar valence density."""

        sample_points = np.asarray(points, dtype=np.float64)
        if self.symmetry is None:
            return self._raw_density(sample_points)
        inverse_direct = np.linalg.inv(self.direct_lattice)
        fractional = np.mod(sample_points @ inverse_direct, 1.0)
        transformed = []
        for rotation, translation in zip(
            self.symmetry.rotations, self.symmetry.translations, strict=True
        ):
            inverse = np.linalg.inv(rotation)
            transformed.append(
                np.mod((fractional - translation) @ inverse.T, 1.0)
                @ self.direct_lattice
            )
        values = self._raw_density(np.concatenate(transformed))
        return values.reshape((len(transformed), len(sample_points))).mean(axis=0)

    def _raw_density(self, points: FloatArray) -> FloatArray:
        density_matrices = nmto_density_matrices(self.bands, self.occupations)
        density = np.zeros(len(points), dtype=np.float64)
        for k_index, weight in enumerate(self.k_weights):
            large, small = self._basis_values(points, k_index)
            values = contract(
                "pa,ab,pb->p",
                large,
                density_matrices[k_index],
                large.conj(),
            ) + contract(
                "pa,ab,pb->p",
                small,
                density_matrices[k_index],
                small.conj(),
            )
            density += weight * np.real_if_close(values).real
        return density

    def _basis_values(
        self, points: FloatArray, k_index: int
    ) -> tuple[ComplexArray, ComplexArray]:
        displacements, sphere_sites = self._nearest_sites(points)
        primitive_size = len(self.site_fractional) * len(self.channels)
        large = np.zeros((len(points), primitive_size), dtype=np.complex128)
        small = np.zeros_like(large)
        interstitial = sphere_sites < 0
        if np.any(interstitial):
            node_values = np.stack(
                tuple(
                    evaluate_folded_usw(
                        float(energy),
                        points[interstitial],
                        self.centers,
                        self.channels,
                        self.folded_coefficients[k_index, energy_index],
                    )
                    for energy_index, energy in enumerate(self.interstitial_energies)
                )
            )
            large[interstitial] = interpolate_nmto_basis(
                node_values, self.results[k_index].lagrange_matrices
            )
        for site in range(len(self.site_fractional)):
            selected = sphere_sites == site
            if not np.any(selected):
                continue
            radii = np.linalg.norm(displacements[selected], axis=1)
            angular = real_spherical_harmonics(displacements[selected], self.channels)
            large_nodes = np.zeros(
                (len(self.energies), np.count_nonzero(selected), primitive_size),
                dtype=np.complex128,
            )
            small_nodes = np.zeros_like(large_nodes)
            for local, channel in enumerate(self.channels):
                radial = self.radial_samples[(site, channel.l)]
                column = site * len(self.channels) + local
                for energy_index in range(len(self.energies)):
                    scale = angular[:, local] / radial.boundary_values[energy_index]
                    large_nodes[energy_index, :, column] = scale * np.interp(
                        radii,
                        radial.mesh_radii,
                        radial.large[energy_index],
                    )
                    small_nodes[energy_index, :, column] = scale * np.interp(
                        radii,
                        radial.mesh_radii,
                        radial.small[energy_index],
                    )
            large[selected] = interpolate_nmto_basis(
                large_nodes, self.results[k_index].lagrange_matrices
            )
            small[selected] = interpolate_nmto_basis(
                small_nodes, self.results[k_index].lagrange_matrices
            )
        return large, small

    def _nearest_sites(self, points: FloatArray) -> tuple[FloatArray, NDArray[np.int64]]:
        fractional = np.mod(points @ np.linalg.inv(self.direct_lattice), 1.0)
        translations = np.asarray(tuple(product((-1.0, 0.0, 1.0), repeat=3)))
        best_distance = np.full(len(points), np.inf)
        best_displacement = np.zeros_like(points)
        best_site = np.full(len(points), -1, dtype=np.int64)
        for site, position in enumerate(self.site_fractional):
            candidates = (
                fractional[:, None, :] - position[None, None, :] - translations[None, :, :]
            ) @ self.direct_lattice
            distances = np.linalg.norm(candidates, axis=2)
            images = np.argmin(distances, axis=1)
            rows = np.arange(len(points))
            site_distances = distances[rows, images]
            replace = site_distances < best_distance
            best_distance[replace] = site_distances[replace]
            best_displacement[replace] = candidates[rows[replace], images[replace]]
            best_site[replace] = site
        best_site[best_distance > self.muffin_tin_radii[best_site]] = -1
        return best_displacement, best_site


def assemble_nmto_regional_density(
    native: ModuleType,
    structure: object,
    field_layout: object,
    g_vectors: NDArray[np.int64],
    density_l_max: int,
    evaluator: NmtoBasisEvaluator,
) -> object:
    """Project occupied NMTO states into the native regional density layout."""

    interstitial = _fit_interstitial_density(g_vectors, evaluator)
    labels, offsets, muffin_tins = _project_muffin_tin_density(
        density_l_max, evaluator
    )
    interstitial_components = np.zeros((4, len(g_vectors)), dtype=np.complex128)
    interstitial_components[0] = interstitial
    mt_components = np.zeros((4, len(muffin_tins)), dtype=np.complex128)
    mt_components[0] = muffin_tins
    return native.RegionalDensity(
        structure,
        field_layout,
        "complex-condon-shortley",
        interstitial_components,
        labels,
        offsets,
        mt_components,
    )


def _fit_interstitial_density(
    g_vectors: NDArray[np.int64], evaluator: NmtoBasisEvaluator
) -> ComplexArray:
    vectors = np.asarray(g_vectors, dtype=np.int64)
    counts = 2 * np.max(np.abs(vectors), axis=0) + 3
    axes = [(np.arange(count) + 0.5) / count for count in counts]
    fractional = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    points = fractional @ evaluator.direct_lattice
    _, sphere_sites = evaluator._nearest_sites(points)
    fractional = fractional[sphere_sites < 0]
    points = points[sphere_sites < 0]
    density = evaluator.density(points)
    design = np.exp(2j * np.pi * (fractional @ vectors.T))
    coefficients = np.linalg.lstsq(design, density, rcond=None)[0]
    by_vector = {tuple(vector): index for index, vector in enumerate(vectors)}
    for vector, index in by_vector.items():
        opposite = by_vector[tuple(-np.asarray(vector))]
        average = 0.5 * (coefficients[index] + coefficients[opposite].conj())
        coefficients[index] = average
        coefficients[opposite] = average.conj()
    zero = by_vector[(0, 0, 0)]
    coefficients[zero] = coefficients[zero].real
    return coefficients


def _project_muffin_tin_density(
    density_l_max: int, evaluator: NmtoBasisEvaluator
) -> tuple[NDArray[np.int64], NDArray[np.int64], ComplexArray]:
    density_channels = tuple(
        RealHarmonic(l, m)
        for l in range(density_l_max + 1)
        for m in range(-l, l + 1)
    )
    theta_nodes, theta_weights = np.polynomial.legendre.leggauss(density_l_max + 2)
    phi_count = 2 * density_l_max + 3
    phi = 2.0 * np.pi * np.arange(phi_count) / phi_count
    directions = np.asarray(
        [
            [np.sqrt(1.0 - z * z) * np.cos(angle), np.sqrt(1.0 - z * z) * np.sin(angle), z]
            for z in theta_nodes
            for angle in phi
        ]
    )
    weights = np.repeat(theta_weights, phi_count) * (2.0 * np.pi / phi_count)
    harmonics = _complex_spherical_harmonics(directions, density_channels)
    labels = []
    offsets = [0]
    samples = []
    site_cartesian = evaluator.site_fractional @ evaluator.direct_lattice
    for site, center in enumerate(site_cartesian):
        mesh = evaluator.radial_samples[(site, 0)].mesh_radii
        points = (
            center[None, None, :] + mesh[:, None, None] * directions[None, :, :]
        ).reshape((-1, 3))
        density = evaluator.density(points).reshape((len(mesh), len(directions)))
        coefficients = contract(
            "ra,a,aL->rL", density, weights, harmonics.conj()
        )
        for channel, values in zip(density_channels, coefficients.T, strict=True):
            labels.append((site, channel.l, channel.m))
            samples.extend(values)
            offsets.append(len(samples))
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(samples, dtype=np.complex128),
    )
