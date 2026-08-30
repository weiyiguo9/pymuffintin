"""Unitary spherical waves (USWs) and their screened slope matrix.

This module follows Nohara--Andersen, Phys. Rev. B 94, 085148 (2016),
Secs. II B and IV A.  The wave equation is ``(-nabla^2 / 2 - energy) psi=0``
outside non-overlapping hard spheres.  For ``epsilon <= 0`` the bare
spherical waves are screened by real-space cluster inversion, and the
dimensionless slope matrix is formed from Eq. (31) of that paper.
Lengths are in Bohr and ``energy`` is in Hartree throughout, following
libmuffintin.  Hence the evanescent wave number is ``sqrt(-2 * energy)``.

The angular basis is the orthonormal real tesseral ("cubic harmonic") basis
obtained from Condon--Shortley complex spherical harmonics as

``m > 0: sqrt(2) (-1)^m Re Y_l^m`` and
``m < 0: -sqrt(2) Im Y_l^|m|``.

This is the same real-harmonic normalization used by the SPEX-facing
libmuffintin convention; the only ordering chosen here is explicit ``(l,m)``
rather than a packed ``L`` index.  Sites are the slow index and harmonics the
fast index in all matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial, pi, sqrt
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..tensor import contract, solve


@dataclass(frozen=True, order=True)
class RealHarmonic:
    """A real tesseral harmonic identified by ``l`` and ``-l <= m <= l``."""

    l: int
    m: int


IDENTITY_HARMONICS_L4 = (
    RealHarmonic(0, 0),
    RealHarmonic(2, 0),
    RealHarmonic(3, -2),
    RealHarmonic(4, 0),
    RealHarmonic(4, 4),
)
"""Five symmetry-adapted density channels listed in Eq. (82) of the paper.

Using this reduction in a screening inversion additionally requires the
site-local symmetry projectors.  The unsymmetrized Table I finite-cluster
inversion therefore uses :data:`HARMONICS_L4` rather than treating this tuple
as an index subset.
"""

HARMONICS_L4 = tuple(
    RealHarmonic(l, m) for l in range(5) for m in range(-l, l + 1)
)
"""All 25 real harmonics through ``l_max=4`` in SPEX packed-L order."""


def _double_factorial(n: int) -> int:
    value = 1
    for factor in range(n, 0, -2):
        value *= factor
    return value


def _associated_legendre(l: int, m: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
    p_mm = np.ones_like(x)
    root = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    for order in range(1, m + 1):
        p_mm *= -(2 * order - 1) * root
    if l == m:
        return p_mm
    p_m1m = (2 * m + 1) * x * p_mm
    if l == m + 1:
        return p_m1m
    p_lm2 = p_mm
    p_lm1 = p_m1m
    for degree in range(m + 2, l + 1):
        p_lm = (
            (2 * degree - 1) * x * p_lm1 - (degree + m - 1) * p_lm2
        ) / (degree - m)
        p_lm2, p_lm1 = p_lm1, p_lm
    return p_lm1


def real_spherical_harmonics(
    directions: NDArray[np.float64], channels: Sequence[RealHarmonic]
) -> NDArray[np.float64]:
    """Evaluate normalized real harmonics for unit ``directions``.

    Returns an array with shape ``(n_direction, n_channel)``.
    """

    xyz = np.asarray(directions, dtype=float)
    radius = np.linalg.norm(xyz, axis=1)
    unit = xyz / radius[:, None]
    cos_theta = unit[:, 2]
    phi = np.arctan2(unit[:, 1], unit[:, 0])
    values = np.empty((len(unit), len(channels)), dtype=float)
    for column, channel in enumerate(channels):
        l = channel.l
        m = channel.m
        abs_m = abs(m)
        norm = sqrt(
            (2 * l + 1)
            / (4 * pi)
            * factorial(l - abs_m)
            / factorial(l + abs_m)
        )
        base = norm * _associated_legendre(l, abs_m, cos_theta)
        if m == 0:
            values[:, column] = base
        elif m > 0:
            values[:, column] = sqrt(2.0) * ((-1) ** m) * base * np.cos(m * phi)
        else:
            values[:, column] = -sqrt(2.0) * base * np.sin(abs_m * phi)
    return values


def _regular_bessel(
    l: int, energy: float, radius: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the Hartree-energy-analytic ``j_l`` and radial derivative."""

    r = np.asarray(radius, dtype=float)
    term = r**l / _double_factorial(2 * l + 1)
    value = term.copy()
    derivative = term * l / r
    for k in range(1, 80):
        term = term * ((-2.0 * energy) * r * r) / (
            2 * k * (2 * l + 2 * k + 1)
        )
        value += term
        derivative += term * (l + 2 * k) / r
        if np.max(np.abs(term)) <= 2e-16 * max(1.0, float(np.max(np.abs(value)))):
            break
    return value, derivative


def _regular_bessel_with_energy_derivative(
    l: int, energy: float, radius: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r = np.asarray(radius, dtype=float)
    term = r**l / _double_factorial(2 * l + 1)
    term_energy = np.zeros_like(r)
    value = term.copy()
    radial = term * l / r
    value_energy = np.zeros_like(r)
    radial_energy = np.zeros_like(r)
    for k in range(1, 80):
        scale = r * r / (k * (2 * l + 2 * k + 1))
        factor = -energy * scale
        previous = term
        term = previous * factor
        term_energy = term_energy * factor - previous * scale
        value += term
        value_energy += term_energy
        radial_factor = (l + 2 * k) / r
        radial += term * radial_factor
        radial_energy += term_energy * radial_factor
        if np.max(np.abs(term)) <= 2e-16 * max(1.0, float(np.max(np.abs(value)))):
            break
    return value, radial, value_energy, radial_energy


def _decaying_hankel(
    l: int, energy: float, radius: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the paper's renormalized decaying ``h_l`` and derivative."""

    r = np.asarray(radius, dtype=float)
    q = sqrt(-2.0 * energy)
    h0 = -np.exp(-q * r) / r
    if l == 0:
        return h0, np.exp(-q * r) * (q / r + 1.0 / (r * r))
    h1 = h0 * (q + 1.0 / r)
    values = [h0, h1]
    for degree in range(1, l):
        values.append(q * q * values[-2] + (2 * degree + 1) * values[-1] / r)
    value = values[l]
    derivative = -(q * q) * values[l - 1] - (l + 1) * value / r
    return value, derivative


def _decaying_hankel_with_energy_derivative(
    l: int, energy: float, radius: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r = np.asarray(radius, dtype=float)
    q = sqrt(-2.0 * energy)
    q_energy = -1.0 / q
    exponential = np.exp(-q * r)
    h0 = -exponential / r
    h0_energy = -exponential / q
    if l == 0:
        radial = exponential * (q / r + 1.0 / (r * r))
        return h0, radial, h0_energy, exponential
    h1 = h0 * (q + 1.0 / r)
    h1_energy = h0_energy * (q + 1.0 / r) + h0 * q_energy
    values = [h0, h1]
    energy_values = [h0_energy, h1_energy]
    for degree in range(1, l):
        values.append(q * q * values[-2] + (2 * degree + 1) * values[-1] / r)
        energy_values.append(
            -2.0 * values[-3]
            + q * q * energy_values[-2]
            + (2 * degree + 1) * energy_values[-1] / r
        )
    value = values[l]
    value_energy = energy_values[l]
    radial = -(q * q) * values[l - 1] - (l + 1) * value / r
    radial_energy = (
        2.0 * values[l - 1]
        - q * q * energy_values[l - 1]
        - (l + 1) * value_energy / r
    )
    return value, radial, value_energy, radial_energy


def _standing_neumann_with_energy_derivative(
    l: int, energy: float, radius: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return the real positive-energy Neumann envelope and derivatives."""

    r = np.asarray(radius, dtype=float)
    wave_number = sqrt(2.0 * energy)
    n_minus_one, n_minus_one_radial = _regular_bessel(0, energy, r)
    values = [-np.cos(wave_number * r) / r]
    previous = n_minus_one
    for degree in range(l):
        following = (2 * degree + 1) * values[-1] / r - 2.0 * energy * previous
        previous = values[-1]
        values.append(following)
    value = values[l]
    lower = n_minus_one if l == 0 else values[l - 1]
    lower_radial = n_minus_one_radial
    if l > 0:
        lower_lower = n_minus_one if l == 1 else values[l - 2]
        lower_radial = 2.0 * energy * lower_lower - l * lower / r
    radial = 2.0 * energy * lower - (l + 1) * value / r
    value_energy = r * lower
    radial_energy = lower + r * lower_radial
    return value, radial, value_energy, radial_energy


def _radial_at_sphere(
    l: int, energy: float, radius: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    j, j_prime = _regular_bessel(l, energy, radius)
    if energy > 0.0:
        n, _, _, _ = _standing_neumann_with_energy_derivative(l, energy, radius)
        return j, j_prime, n
    h, _ = _decaying_hankel(l, energy, radius)
    q = sqrt(-2.0 * energy)
    power = ((-1) ** l) * q ** (2 * l + 1)
    return j, j_prime, h - power * j


def _radial_at_sphere_with_energy_derivative(
    l: int, energy: float, radius: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    j, j_radial, j_energy, j_radial_energy = _regular_bessel_with_energy_derivative(
        l, energy, radius
    )
    if energy > 0.0:
        n, _, n_energy, _ = _standing_neumann_with_energy_derivative(
            l, energy, radius
        )
        return j, j_radial, n, j_energy, j_radial_energy, n_energy
    h, _, h_energy, _ = _decaying_hankel_with_energy_derivative(l, energy, radius)
    q = sqrt(-2.0 * energy)
    exponent = 2 * l + 1
    sign = (-1) ** l
    power = sign * q**exponent
    power_energy = -sign * exponent * q ** (exponent - 2)
    n = h - power * j
    n_energy = h_energy - power_energy * j - power * j_energy
    return j, j_radial, n, j_energy, j_radial_energy, n_energy


def _translation_channels(l_max: int) -> tuple[RealHarmonic, ...]:
    return tuple(RealHarmonic(l, m) for l in range(2 * l_max + 1) for m in range(-l, l + 1))


def _real_gaunt(
    channels: Sequence[RealHarmonic], translations: Sequence[RealHarmonic]
) -> NDArray[np.float64]:
    l_max = max(channel.l for channel in translations)
    x, theta_weights = np.polynomial.legendre.leggauss(2 * l_max + 2)
    n_phi = 4 * l_max + 1
    phi = 2 * pi * np.arange(n_phi) / n_phi
    sin_theta = np.sqrt(1.0 - x * x)
    directions = np.stack(
        (
            np.repeat(sin_theta, n_phi) * np.tile(np.cos(phi), len(x)),
            np.repeat(sin_theta, n_phi) * np.tile(np.sin(phi), len(x)),
            np.repeat(x, n_phi),
        ),
        axis=1,
    )
    weights = np.repeat(theta_weights, n_phi) * (2 * pi / n_phi)
    angular = real_spherical_harmonics(directions, channels)
    translated = real_spherical_harmonics(directions, translations)
    return contract("qa,qb,qc,q->abc", angular, angular, translated, weights)


def bare_structure_matrix(
    energy: float,
    centers: NDArray[np.float64],
    channels: Sequence[RealHarmonic],
    *,
    standing_wave: bool = False,
) -> NDArray[np.float64]:
    """Build the analytical bare structure matrix ``B(epsilon)``.

    Every site carries the same ordered real-harmonic channel set.  Positive
    energy selects the real Neumann branch.  ``standing_wave=True`` retains the analytic
    Neumann part used to evaluate ``dS/depsilon`` at zero; ordinary negative-
    energy USWs use the default decaying Hankel branch.
    """

    sites = np.asarray(centers, dtype=float)
    n_site = len(sites)
    n_channel = len(channels)
    size = n_site * n_channel
    result = np.zeros((size, size), dtype=float)
    standing = standing_wave or energy > 0.0
    if not standing:
        q = sqrt(-2.0 * energy)
        for site in range(n_site):
            offset = site * n_channel
            for local, channel in enumerate(channels):
                result[offset + local, offset + local] = ((-1) ** channel.l) * q ** (
                    2 * channel.l + 1
                )

    row_site, column_site = np.nonzero(~np.eye(n_site, dtype=bool))
    displacement = sites[row_site] - sites[column_site]
    distance = np.linalg.norm(displacement, axis=1)
    translation_channels = _translation_channels(max(channel.l for channel in channels))
    translated = real_spherical_harmonics(displacement, translation_channels)
    gaunt = _real_gaunt(channels, translation_channels)
    kernel = np.zeros_like(gaunt)
    for a, row_channel in enumerate(channels):
        for b, column_channel in enumerate(channels):
            for c, translation in enumerate(translation_channels):
                power = row_channel.l + column_channel.l - translation.l
                if power < 0 or power % 2:
                    continue
                phase = (-1) ** (row_channel.l - translation.l)
                kernel[a, b, c] = (
                    4
                    * pi
                    * phase
                    * (-2.0 * energy) ** (power // 2)
                    * gaunt[a, b, c]
                )
    for c, translation in enumerate(translation_channels):
        if energy > 0.0:
            radial, _, _, _ = _standing_neumann_with_energy_derivative(
                translation.l, energy, distance
            )
        else:
            radial, _ = _decaying_hankel(translation.l, energy, distance)
        if standing_wave and energy <= 0.0:
            q = sqrt(-2.0 * energy)
            regular, _ = _regular_bessel(translation.l, energy, distance)
            radial = radial - ((-1) ** translation.l) * q ** (
                2 * translation.l + 1
            ) * regular
        translated[:, c] *= radial
    blocks = contract("abc,pc->pab", kernel, translated)
    for pair, (row, column) in enumerate(zip(row_site, column_site, strict=True)):
        row_slice = slice(row * n_channel, (row + 1) * n_channel)
        column_slice = slice(column * n_channel, (column + 1) * n_channel)
        result[row_slice, column_slice] = blocks[pair]
    return result


def _bare_structure_matrix_with_energy_derivative(
    energy: float,
    centers: NDArray[np.float64],
    channels: Sequence[RealHarmonic],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    sites = np.asarray(centers, dtype=float)
    n_site = len(sites)
    n_channel = len(channels)
    size = n_site * n_channel
    result = np.zeros((size, size), dtype=float)
    derivative = np.zeros_like(result)
    if energy < 0.0:
        q = sqrt(-2.0 * energy)
        for site in range(n_site):
            offset = site * n_channel
            for local, channel in enumerate(channels):
                exponent = 2 * channel.l + 1
                sign = (-1) ** channel.l
                result[offset + local, offset + local] = sign * q**exponent
                derivative[offset + local, offset + local] = (
                    -sign * exponent * q ** (exponent - 2)
                )

    row_site, column_site = np.nonzero(~np.eye(n_site, dtype=bool))
    displacement = sites[row_site] - sites[column_site]
    distance = np.linalg.norm(displacement, axis=1)
    translation_channels = _translation_channels(max(channel.l for channel in channels))
    angular = real_spherical_harmonics(displacement, translation_channels)
    gaunt = _real_gaunt(channels, translation_channels)
    kernel = np.zeros_like(gaunt)
    kernel_energy = np.zeros_like(gaunt)
    for a, row_channel in enumerate(channels):
        for b, column_channel in enumerate(channels):
            for c, translation in enumerate(translation_channels):
                exponent = row_channel.l + column_channel.l - translation.l
                if exponent < 0 or exponent % 2:
                    continue
                phase = (-1) ** (row_channel.l - translation.l)
                prefactor = 4 * pi * phase * gaunt[a, b, c]
                kernel[a, b, c] = prefactor * (-2.0 * energy) ** (exponent // 2)
                if exponent:
                    kernel_energy[a, b, c] = (
                        -prefactor
                        * exponent
                        * (-2.0 * energy) ** (exponent // 2 - 1)
                    )
    radial_angular = angular.copy()
    radial_energy_angular = angular.copy()
    for c, translation in enumerate(translation_channels):
        if energy > 0.0:
            radial, _, radial_energy, _ = _standing_neumann_with_energy_derivative(
                translation.l, energy, distance
            )
        else:
            radial, _, radial_energy, _ = _decaying_hankel_with_energy_derivative(
                translation.l, energy, distance
            )
        radial_angular[:, c] *= radial
        radial_energy_angular[:, c] *= radial_energy
    blocks = contract("abc,pc->pab", kernel, radial_angular)
    derivative_blocks = contract(
        "abc,pc->pab", kernel_energy, radial_angular
    ) + contract("abc,pc->pab", kernel, radial_energy_angular)
    for pair, (row, column) in enumerate(zip(row_site, column_site, strict=True)):
        row_slice = slice(row * n_channel, (row + 1) * n_channel)
        column_slice = slice(column * n_channel, (column + 1) * n_channel)
        result[row_slice, column_slice] = blocks[pair]
        derivative[row_slice, column_slice] = derivative_blocks[pair]
    return result, derivative


def usw_matrices(
    energy: float,
    centers: NDArray[np.float64],
    radii: NDArray[np.float64],
    channels: Sequence[RealHarmonic],
    *,
    standing_wave: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Hankel coefficients ``M`` and dimensionless USW slopes ``S``."""

    sites = np.asarray(centers, dtype=float)
    sphere_radii = np.asarray(radii, dtype=float)
    repeated_radii = np.repeat(sphere_radii, len(channels))
    repeated_l = np.tile(np.array([channel.l for channel in channels]), len(sites))
    j = np.empty_like(repeated_radii)
    j_prime = np.empty_like(repeated_radii)
    n = np.empty_like(repeated_radii)
    for l in sorted(set(repeated_l)):
        selected = repeated_l == l
        j[selected], j_prime[selected], n[selected] = _radial_at_sphere(
            int(l), energy, repeated_radii[selected]
        )
    bare = bare_structure_matrix(energy, sites, channels, standing_wave=standing_wave)
    screened = bare + np.diag(n / j)
    inverse_times_j = solve(screened, np.diag(1.0 / j))
    hankel_coefficients = inverse_times_j
    slope = np.diag(repeated_radii * j_prime / j) + (
        1.0 / (repeated_radii * j)
    )[:, None] * inverse_times_j
    return hankel_coefficients, slope


def usw_matrices_with_energy_derivative(
    energy: float,
    centers: NDArray[np.float64],
    radii: NDArray[np.float64],
    channels: Sequence[RealHarmonic],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``M(E)``, ``S(E)``, and the exact Hartree derivative ``dS/dE``.

    The derivative is obtained by differentiating the radial series, bare
    structure matrix, and screening solve analytically.  Positive energy uses
    the real Neumann continuation; the zero-energy analytic derivative retains
    its separate standing-wave limit.
    """

    sites = np.asarray(centers, dtype=float)
    sphere_radii = np.asarray(radii, dtype=float)
    repeated_radii = np.repeat(sphere_radii, len(channels))
    repeated_l = np.tile(np.array([channel.l for channel in channels]), len(sites))
    j = np.empty_like(repeated_radii)
    j_radial = np.empty_like(repeated_radii)
    n = np.empty_like(repeated_radii)
    j_energy = np.empty_like(repeated_radii)
    j_radial_energy = np.empty_like(repeated_radii)
    n_energy = np.empty_like(repeated_radii)
    for l in sorted(set(repeated_l)):
        selected = repeated_l == l
        (
            j[selected],
            j_radial[selected],
            n[selected],
            j_energy[selected],
            j_radial_energy[selected],
            n_energy[selected],
        ) = _radial_at_sphere_with_energy_derivative(
            int(l), energy, repeated_radii[selected]
        )
    bare, bare_energy = _bare_structure_matrix_with_energy_derivative(
        energy, sites, channels
    )
    screened = bare + np.diag(n / j)
    screened_energy = bare_energy + np.diag(
        (n_energy * j - n * j_energy) / (j * j)
    )
    inverse_j = np.diag(1.0 / j)
    hankel_coefficients = solve(screened, inverse_j)
    right_energy = np.diag(-j_energy / (j * j)) - contract(
        "ij,jk->ik", screened_energy, hankel_coefficients
    )
    hankel_energy = solve(screened, right_energy)
    row_scale = 1.0 / (repeated_radii * j)
    row_scale_energy = -j_energy / (repeated_radii * j * j)
    slope = np.diag(repeated_radii * j_radial / j) + row_scale[:, None] * (
        hankel_coefficients
    )
    slope_energy = np.diag(
        repeated_radii
        * (j_radial_energy * j - j_radial * j_energy)
        / (j * j)
    ) + row_scale_energy[:, None] * hankel_coefficients + row_scale[:, None] * (
        hankel_energy
    )
    return hankel_coefficients, slope, slope_energy


def evaluate_usw(
    energy: float,
    points: NDArray[np.float64],
    centers: NDArray[np.float64],
    radii: NDArray[np.float64],
    channels: Sequence[RealHarmonic],
) -> NDArray[np.float64]:
    """Evaluate every cluster USW at interstitial ``points``."""

    sites = np.asarray(centers, dtype=float)
    sample_points = np.asarray(points, dtype=float)
    coefficients, _ = usw_matrices(energy, sites, radii, channels)
    envelopes = np.empty((len(sample_points), len(sites) * len(channels)))
    for site, center in enumerate(sites):
        displacement = sample_points - center
        distance = np.linalg.norm(displacement, axis=1)
        angular = real_spherical_harmonics(displacement, channels)
        for local, channel in enumerate(channels):
            if energy > 0.0:
                radial, _, _, _ = _standing_neumann_with_energy_derivative(
                    channel.l, energy, distance
                )
            else:
                radial, _ = _decaying_hankel(channel.l, energy, distance)
            envelopes[:, site * len(channels) + local] = radial * angular[:, local]
    return contract("pi,ij->pj", envelopes, coefficients)


def bloch_sum(
    translated_matrices: NDArray[np.float64],
    translations: NDArray[np.float64],
    k_point: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    """Bloch-sum translated screened matrices; ``k_point=None`` means ``k=0``."""

    k = np.zeros(3) if k_point is None else np.asarray(k_point, dtype=float)
    phase = np.exp(1j * (np.asarray(translations, dtype=float) @ k))
    return contract("t,tij->ij", phase, np.asarray(translated_matrices))


def cluster_bloch_sum(
    cluster_matrix: NDArray[np.float64],
    centers: NDArray[np.float64],
    n_channels: int,
    k_cartesian: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    """Fold the center-row blocks of a screened real-space cluster.

    The cluster uses site-major/channel-minor ordering.  This helper is the
    one-site primitive-cell form needed after real-space screening; general
    multi-site cells should provide translated blocks directly to
    :func:`bloch_sum`.
    """

    sites = np.asarray(centers, dtype=float)
    center = int(np.argmin(np.linalg.norm(sites, axis=1)))
    row = np.asarray(cluster_matrix)[
        center * n_channels : (center + 1) * n_channels
    ].reshape((n_channels, len(sites), n_channels))
    k = np.zeros(3) if k_cartesian is None else np.asarray(k_cartesian, dtype=float)
    phase = np.exp(1j * ((sites - sites[center]) @ k))
    return contract("t,itj->ij", phase, row)


def lattice_cluster(
    structure: str, n_sites: int, touching_radius: float = 1.0
) -> NDArray[np.float64]:
    """Return the cluster used by the Table I constant-density regression."""

    if structure == "bcc":
        lattice_constant = 4 * touching_radius / sqrt(3.0)
        primitive = lattice_constant / 2 * np.array(
            [[1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]
        )
        basis = (np.zeros(3),)
    elif structure == "diamond":
        lattice_constant = 8 * touching_radius / sqrt(3.0)
        primitive = lattice_constant / 2 * np.array(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        )
        basis = (np.zeros(3), np.full(3, lattice_constant / 4))
    else:
        raise ValueError(f"unknown structure {structure!r}")
    extent = 2
    while True:
        candidates = []
        for index in product(range(-extent, extent + 1), repeat=3):
            origin = np.asarray(index) @ primitive
            candidates.extend(origin + offset for offset in basis)
        sites = np.unique(np.round(np.asarray(candidates), decimals=13), axis=0)
        order = np.argsort(np.linalg.norm(sites, axis=1), kind="stable")
        sites = sites[order]
        cutoff = np.linalg.norm(sites[n_sites - 1])
        complete = np.count_nonzero(np.isclose(np.linalg.norm(sites, axis=1), cutoff))
        selected = sites[np.linalg.norm(sites, axis=1) <= cutoff + 1e-11]
        if len(selected) == n_sites and complete and cutoff < extent * touching_radius:
            return selected
        extent += 1


def interstitial_volume(structure: str, hard_sphere_radius: float) -> float:
    """Exact primitive-cell interstitial volume for ``touching_radius=1``."""

    if structure == "bcc":
        cell_volume = 32.0 / (3.0 * sqrt(3.0))
        spheres = 1
    elif structure == "diamond":
        cell_volume = 128.0 / (3.0 * sqrt(3.0))
        spheres = 2
    else:
        raise ValueError(f"unknown structure {structure!r}")
    return cell_volume - spheres * 4 * pi * hard_sphere_radius**3 / 3


def constant_interstitial_volume(
    structure: str,
    n_sites: int,
    hard_sphere_radius: float,
    *,
    energy_step: float = 5e-6,
    channels: Sequence[RealHarmonic] = HARMONICS_L4,
) -> float:
    """Evaluate the paper's constant-density volume measure ``4 pi sum a Sdot``."""

    centers = lattice_cluster(structure, n_sites)
    radii = np.full(len(centers), hard_sphere_radius)
    _, slope_zero = usw_matrices(0.0, centers, radii, channels, standing_wave=True)
    _, slope_below = usw_matrices(
        -energy_step, centers, radii, channels, standing_wave=True
    )
    slope_derivative = (slope_zero - slope_below) / (2.0 * energy_step)
    center = int(np.argmin(np.linalg.norm(centers, axis=1)))
    s_local = channels.index(RealHarmonic(0, 0))
    center_s = center * len(channels) + s_local
    all_s = np.arange(len(centers)) * len(channels) + s_local
    sites_per_cell = 1 if structure == "bcc" else 2
    return float(
        sites_per_cell
        * 4
        * pi
        * hard_sphere_radius
        * np.sum(slope_derivative[center_s, all_s])
    )
