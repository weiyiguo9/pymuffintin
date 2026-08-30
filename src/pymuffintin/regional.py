"""Sampling of exported muffin-tin regional scalar fields.

The sampler consumes the array dictionary exported by ``libmuffintin`` and
evaluates one Pauli component in Cartesian coordinates.  Interstitial Fourier
coefficients and muffin-tin radial harmonics retain their native conventions;
no new density representation is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial, pi, sqrt
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .mto.usw import RealHarmonic, real_spherical_harmonics


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _associated_legendre(l: int, m: int, x: FloatArray) -> FloatArray:
    value = np.ones_like(x)
    root = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    for order in range(1, m + 1):
        value *= -(2 * order - 1) * root
    if l == m:
        return value
    next_value = (2 * m + 1) * x * value
    if l == m + 1:
        return next_value
    previous = value
    current = next_value
    for degree in range(m + 2, l + 1):
        following = (
            (2 * degree - 1) * x * current - (degree + m - 1) * previous
        ) / (degree - m)
        previous, current = current, following
    return current


def _complex_spherical_harmonics(
    directions: FloatArray, channels: tuple[RealHarmonic, ...]
) -> ComplexArray:
    radius = np.linalg.norm(directions, axis=1)
    unit = np.zeros_like(directions)
    nonzero = radius > 0.0
    unit[nonzero] = directions[nonzero] / radius[nonzero, None]
    unit[~nonzero, 2] = 1.0
    cos_theta = unit[:, 2]
    phi = np.arctan2(unit[:, 1], unit[:, 0])
    values = np.empty((len(unit), len(channels)), dtype=np.complex128)
    for column, channel in enumerate(channels):
        l = channel.l
        m = channel.m
        absolute_m = abs(m)
        norm = sqrt(
            (2 * l + 1)
            / (4 * pi)
            * factorial(l - absolute_m)
            / factorial(l + absolute_m)
        )
        positive = (
            norm
            * _associated_legendre(l, absolute_m, cos_theta)
            * np.exp(1j * absolute_m * phi)
        )
        values[:, column] = (
            positive if m >= 0 else ((-1) ** absolute_m) * positive.conj()
        )
    return values


@dataclass(frozen=True)
class RegionalScalarSampler:
    """Evaluate one exported regional scalar-field component.

    ``direct_lattice`` uses row lattice vectors and ``site_fractional`` uses
    the same primitive-cell fractional convention.  Points may lie in any
    periodic image of the primitive cell.
    """

    direct_lattice: FloatArray
    site_fractional: FloatArray
    muffin_tin_radii: FloatArray
    angular_basis: str
    g_vectors: NDArray[np.int32]
    fourier_coefficients: ComplexArray
    mt_mesh_offsets: NDArray[np.int64]
    mt_mesh_radii: FloatArray
    mt_channel_labels: NDArray[np.int64]
    mt_sample_offsets: NDArray[np.int64]
    mt_coefficients: ComplexArray

    @classmethod
    def from_export(
        cls,
        exported: Mapping[str, object],
        direct_lattice: FloatArray,
        site_fractional: FloatArray,
        muffin_tin_radii: FloatArray,
        *,
        component: int = 0,
    ) -> RegionalScalarSampler:
        components = np.asarray(exported["components"], dtype=np.complex128)
        mt_components = np.asarray(exported["mt_components"], dtype=np.complex128)
        if type(component) is not int or not 0 <= component < 4:
            raise ValueError("component must be an integer in [0, 4)")
        return cls(
            direct_lattice=np.asarray(direct_lattice, dtype=np.float64),
            site_fractional=np.asarray(site_fractional, dtype=np.float64),
            muffin_tin_radii=np.asarray(muffin_tin_radii, dtype=np.float64),
            angular_basis=str(exported["angular_basis"]),
            g_vectors=np.asarray(exported["g_vectors"], dtype=np.int32),
            fourier_coefficients=components[component],
            mt_mesh_offsets=np.asarray(exported["mt_mesh_offsets"], dtype=np.int64),
            mt_mesh_radii=np.asarray(exported["mt_mesh_radii"], dtype=np.float64),
            mt_channel_labels=np.asarray(exported["mt_channel_labels"], dtype=np.int64),
            mt_sample_offsets=np.asarray(exported["mt_sample_offsets"], dtype=np.int64),
            mt_coefficients=mt_components[component],
        )

    def __post_init__(self) -> None:
        site_count = len(self.site_fractional)
        if self.direct_lattice.shape != (3, 3):
            raise ValueError("direct_lattice must have shape (3, 3)")
        if self.site_fractional.shape != (site_count, 3):
            raise ValueError("site_fractional must have shape (site, 3)")
        if self.muffin_tin_radii.shape != (site_count,):
            raise ValueError("muffin_tin_radii must contain one radius per site")
        if self.angular_basis not in (
            "complex-condon-shortley",
            "real-tesseral-condon-shortley",
        ):
            raise ValueError(f"unsupported angular basis {self.angular_basis!r}")
        if self.g_vectors.shape != (len(self.fourier_coefficients), 3):
            raise ValueError("g_vectors and fourier_coefficients do not match")
        if self.mt_mesh_offsets.shape != (site_count + 1,):
            raise ValueError("mt_mesh_offsets must contain one range per site")
        if self.mt_channel_labels.ndim != 2 or self.mt_channel_labels.shape[1] != 3:
            raise ValueError("mt_channel_labels must have shape (channel, 3)")
        if self.mt_sample_offsets.shape != (len(self.mt_channel_labels) + 1,):
            raise ValueError("mt_sample_offsets must contain one range per channel")
        if self.mt_sample_offsets[-1] != len(self.mt_coefficients):
            raise ValueError("mt sample offsets do not cover mt_coefficients")

    def __call__(self, points: FloatArray) -> ComplexArray:
        sample_points = np.asarray(points, dtype=np.float64)
        if sample_points.ndim != 2 or sample_points.shape[1] != 3:
            raise ValueError("points must have shape (point, 3)")
        inverse_direct = np.linalg.inv(self.direct_lattice)
        fractional = np.mod(sample_points @ inverse_direct, 1.0)
        displacements, sphere_sites = self._nearest_sites(fractional)
        inside = sphere_sites >= 0
        values = np.empty(len(sample_points), dtype=np.complex128)
        if np.any(~inside):
            reciprocal = 2.0 * pi * inverse_direct.T
            g_cartesian = self.g_vectors @ reciprocal
            phases = np.exp(1j * (sample_points[~inside] @ g_cartesian.T))
            values[~inside] = phases @ self.fourier_coefficients
        for site in range(len(self.site_fractional)):
            selected = sphere_sites == site
            if np.any(selected):
                values[selected] = self._sample_muffin_tin(
                    site, displacements[selected]
                )
        return values

    def _nearest_sites(self, fractional: FloatArray) -> tuple[FloatArray, NDArray[np.int64]]:
        translations = np.asarray(tuple(product((-1.0, 0.0, 1.0), repeat=3)))
        best_distance = np.full(len(fractional), np.inf)
        best_displacement = np.zeros_like(fractional)
        best_site = np.full(len(fractional), -1, dtype=np.int64)
        for site, position in enumerate(self.site_fractional):
            candidates = (
                fractional[:, None, :] - position[None, None, :] - translations[None, :, :]
            ) @ self.direct_lattice
            distances = np.linalg.norm(candidates, axis=2)
            images = np.argmin(distances, axis=1)
            rows = np.arange(len(fractional))
            site_distances = distances[rows, images]
            replace = site_distances < best_distance
            best_distance[replace] = site_distances[replace]
            best_displacement[replace] = candidates[rows[replace], images[replace]]
            best_site[replace] = site
        outside = best_distance > self.muffin_tin_radii[best_site]
        best_site[outside] = -1
        return best_displacement, best_site

    def _sample_muffin_tin(self, site: int, displacements: FloatArray) -> ComplexArray:
        radius = np.linalg.norm(displacements, axis=1)
        channel_indices = np.flatnonzero(self.mt_channel_labels[:, 0] == site)
        channels = tuple(
            RealHarmonic(int(self.mt_channel_labels[index, 1]), int(self.mt_channel_labels[index, 2]))
            for index in channel_indices
        )
        if self.angular_basis == "complex-condon-shortley":
            harmonics = _complex_spherical_harmonics(displacements, channels)
        else:
            harmonics = np.asarray(
                real_spherical_harmonics(displacements, channels), dtype=np.complex128
            )
        mesh_start = int(self.mt_mesh_offsets[site])
        mesh_stop = int(self.mt_mesh_offsets[site + 1])
        mesh = self.mt_mesh_radii[mesh_start:mesh_stop]
        radial_values = np.empty_like(harmonics)
        for column, channel_index in enumerate(channel_indices):
            start = int(self.mt_sample_offsets[channel_index])
            stop = int(self.mt_sample_offsets[channel_index + 1])
            samples = self.mt_coefficients[start:stop]
            radial_values[:, column] = np.interp(radius, mesh, samples.real) + 1j * np.interp(
                radius, mesh, samples.imag
            )
        return np.sum(radial_values * harmonics, axis=1)
