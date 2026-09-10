"""Potential-sphere shells: mesh extension and free back-extrapolation.

An overlapping-muffin-tin (OMT) reference keeps the non-overlapping hard
spheres of radius ``a`` (kink spheres, augmentation partition, and regional
field boundary) inside potential spheres of radius ``s >= a`` that may
overlap.  The radial solution ``u`` of the spherical well is integrated out
to ``s``.  Between ``a`` and ``s`` it is continued inward by the free
nonrelativistic solution ``u0`` that matches ``u`` in value and
mass-weighted flux at ``s``.  The kink matrix at ``a`` then uses ``u0``, and
the augmented orbital is ``u/u0(a)`` inside ``a``, ``psi + (u-u0)/u0(a)`` in
the shell, and ``psi`` outside ``s``.  With ``s == a`` every quantity reduces
exactly to the non-overlapping construction.

Energies are Hartree and lengths Bohr.  ``kinetic_energies`` are the free
interstitial energies ``E - V0`` of the envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, exp, log
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..tensor import solve
from .kink import BoundaryJets
from .usw import (
    _decaying_hankel_with_energy_derivative,
    _regular_bessel_with_energy_derivative,
    _standing_neumann_with_energy_derivative,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def exponential_mesh(first: float, increment: float, count: int) -> FloatArray:
    """Return the native exponential mesh ``first * exp(increment * i)``."""

    return first * np.exp(increment * np.arange(count))


def snap_to_exponential_mesh(first: float, increment: float, radius: float) -> float:
    """Return the smallest exponential-mesh point that is not below ``radius``."""

    if radius <= 0.0 or first <= 0.0 or increment <= 0.0:
        raise ValueError("mesh parameters and radius must be positive")
    index = ceil(log(radius / first) / increment - 1.0e-10)
    return first * exp(increment * index)


def exponential_mesh_count(
    first: float, increment: float, radius: float, *, tolerance: float = 1.0e-8
) -> int:
    """Return the point count whose last mesh radius equals ``radius``."""

    count = int(round(log(radius / first) / increment)) + 1
    last = first * exp(increment * (count - 1))
    if abs(last - radius) > tolerance * max(1.0, radius):
        raise ValueError(
            f"radius {radius!r} is not a point of the exponential mesh "
            f"(first={first!r}, increment={increment!r}); snap it first"
        )
    return count


def sphere_images(
    points: FloatArray,
    lattice: FloatArray,
    centers: FloatArray,
    inner_radii: FloatArray,
    outer_radii: FloatArray,
) -> list[tuple[int, IntArray, FloatArray]]:
    """Return every periodic sphere image containing a point in its shell.

    For each site the result lists the point indices and Cartesian
    displacements ``r - R - T`` with ``inner < |r - R - T| <= outer``.  A
    point may appear several times when several images or sites contain it;
    sites without any such point are omitted.  Lattice vectors are rows.
    """

    sample_points = np.asarray(points, dtype=np.float64)
    cell = np.asarray(lattice, dtype=np.float64)
    sites = np.asarray(centers, dtype=np.float64)
    inner = np.asarray(inner_radii, dtype=np.float64)
    outer = np.asarray(outer_radii, dtype=np.float64)
    inverse = np.linalg.inv(cell)
    fractional = np.mod(sample_points @ inverse, 1.0)
    site_fractional = np.mod(sites @ inverse, 1.0)
    bounds = np.ceil(np.max(outer) * np.linalg.norm(inverse, axis=0)).astype(int)
    translations = np.asarray(
        tuple(product(*(range(-bound, bound + 1) for bound in bounds))), dtype=float
    )
    result = []
    for site in range(len(sites)):
        if outer[site] <= inner[site]:
            continue
        candidates = (
            fractional[:, None, :] - site_fractional[site][None, None, :] - translations[None, :, :]
        ) @ cell
        distances = np.linalg.norm(candidates, axis=2)
        rows, columns = np.nonzero(
            (distances > inner[site]) & (distances <= outer[site])
        )
        if len(rows) == 0:
            continue
        result.append((site, rows.astype(np.int64), candidates[rows, columns]))
    return result


@dataclass(frozen=True)
class FreeContinuation:
    """Free continuation ``u0`` of one radial solution from ``s`` down to ``a``.

    ``values`` has shape ``(n_energy, n_shell)`` on the shell mesh radii,
    which run from the hard radius to the potential radius inclusive.  The
    boundary arrays are ``u0(a)``, ``u0'(a)``, and their exact energy
    derivatives, one entry per energy.
    """

    hard_radius: float
    potential_radius: float
    radii: FloatArray
    values: FloatArray
    boundary_values: FloatArray
    boundary_radial: FloatArray
    boundary_energy: FloatArray
    boundary_energy_radial: FloatArray


def _free_pair(l: int, kinetic_energy: float, radii: FloatArray) -> tuple[FloatArray, ...]:
    if kinetic_energy == 0.0:
        raise ValueError("free shell continuation requires a nonzero interstitial energy")
    regular = _regular_bessel_with_energy_derivative(l, kinetic_energy, radii)
    if kinetic_energy < 0.0:
        irregular = _decaying_hankel_with_energy_derivative(l, kinetic_energy, radii)
    else:
        irregular = _standing_neumann_with_energy_derivative(l, kinetic_energy, radii)
    return (*regular, *irregular)


def free_continuation(
    l: int,
    kinetic_energies: Sequence[float] | FloatArray,
    hard_radius: float,
    potential_radius: float,
    shell_radii: FloatArray,
    boundary_jets: FloatArray,
) -> FreeContinuation:
    """Continue ``u`` from the potential radius inward with free waves.

    ``boundary_jets`` has one row per energy:
    ``[u(s), u'(s), du/dE(s), du'/dE(s), m^-1(s), dm^-1/dE(s)]`` with the
    native mass-weighted flux ``m^-1 u'``.  The continuation is the
    nonrelativistic free solution ``u0 = A J + B N`` with
    ``u0(s) = u(s)`` and ``u0'(s)/2 = m^-1(s) u'(s)``, the same flux bridge
    the kink matrix uses.  Its energy derivatives follow by differentiating
    the two matching conditions exactly.  For ``s == a`` this returns
    ``u0(a) = u(a)`` and ``u0'(a) = 2 m^-1(a) u'(a)``.
    """

    energies = np.asarray(kinetic_energies, dtype=np.float64)
    radii = np.asarray(shell_radii, dtype=np.float64)
    jets = np.asarray(boundary_jets, dtype=np.float64)
    if jets.shape != (len(energies), 6):
        raise ValueError("boundary jets must have shape (n_energy, 6)")
    if radii.ndim != 1 or len(radii) == 0:
        raise ValueError("shell radii must be a nonempty 1D array")
    if abs(radii[0] - hard_radius) > 1.0e-12 * max(1.0, hard_radius) or abs(
        radii[-1] - potential_radius
    ) > 1.0e-12 * max(1.0, potential_radius):
        raise ValueError("shell radii must run from the hard radius to the potential radius")
    values = np.empty((len(energies), len(radii)))
    boundary = np.empty((len(energies), 4))
    edge = np.asarray([hard_radius, potential_radius])
    for index, energy in enumerate(energies):
        j, j_r, j_e, j_re, n, n_r, n_e, n_re = _free_pair(l, float(energy), edge)
        u, u_r, u_e, u_re, mass, mass_e = jets[index]
        matrix = np.array([[j[1], n[1]], [j_r[1], n_r[1]]])
        rhs = np.array([u, 2.0 * mass * u_r])
        coefficients = solve(matrix, rhs)
        matrix_e = np.array([[j_e[1], n_e[1]], [j_re[1], n_re[1]]])
        rhs_e = np.array([u_e, 2.0 * (mass_e * u_r + mass * u_re)])
        coefficients_e = solve(matrix, rhs_e - matrix_e @ coefficients)
        boundary[index] = (
            coefficients @ (j[0], n[0]),
            coefficients @ (j_r[0], n_r[0]),
            coefficients_e @ (j[0], n[0]) + coefficients @ (j_e[0], n_e[0]),
            coefficients_e @ (j_r[0], n_r[0]) + coefficients @ (j_re[0], n_re[0]),
        )
        j_all, _, _, _, n_all, _, _, _ = _free_pair(l, float(energy), radii)
        values[index] = coefficients[0] * j_all + coefficients[1] * n_all
    return FreeContinuation(
        hard_radius=float(hard_radius),
        potential_radius=float(potential_radius),
        radii=radii,
        values=values,
        boundary_values=boundary[:, 0],
        boundary_radial=boundary[:, 1],
        boundary_energy=boundary[:, 2],
        boundary_energy_radial=boundary[:, 3],
    )


def continuation_jets(continuations: Sequence[FreeContinuation]) -> BoundaryJets:
    """Assemble hard-sphere boundary jets, one continuation per channel column.

    The continued function is a nonrelativistic free wave, so its flux uses
    the constant inverse mass ``1/2`` with zero energy derivative.
    """

    if len(continuations) == 0:
        raise ValueError("at least one continuation is required")
    shape = (len(continuations[0].boundary_values), len(continuations))
    return BoundaryJets(
        potential_radii=np.asarray([item.hard_radius for item in continuations]),
        values=np.stack([item.boundary_values for item in continuations], axis=1),
        radial_derivatives=np.stack([item.boundary_radial for item in continuations], axis=1),
        energy_derivatives=np.stack([item.boundary_energy for item in continuations], axis=1),
        energy_radial_derivatives=np.stack(
            [item.boundary_energy_radial for item in continuations], axis=1
        ),
        inverse_masses=np.full(shape, 0.5),
        energy_inverse_masses=np.zeros(shape),
    )
