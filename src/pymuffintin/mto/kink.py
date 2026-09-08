"""Kink matrices, exact energy derivatives, and active-channel downfolding.

The screened slopes are dimensionless.  Potential radii are in Bohr and all
energies and energy derivatives use Hartree.  Boundary radial functions and
their derivatives must be supplied by the radial solver; this module never
reconstructs an energy derivative by finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..tensor import contract, solve


NumericArray: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class BoundaryJets:
    """Radial boundary values on an energy mesh.

    Every value array has shape ``(n_energy, n_channel)``.
    ``energy_radial_derivatives`` is the mixed derivative ``d/dE (d f/dr)``;
    ``inverse_masses`` and ``energy_inverse_masses`` specify the physical
    radial flux and its energy derivative.  ``potential_radii`` names the
    radius role deliberately: screening or augmentation radii are not valid
    substitutes in the kink formula.
    """

    potential_radii: FloatArray
    values: NumericArray
    radial_derivatives: NumericArray
    energy_derivatives: NumericArray
    energy_radial_derivatives: NumericArray
    inverse_masses: FloatArray
    energy_inverse_masses: FloatArray

    def __post_init__(self) -> None:
        shape = self.values.shape
        if self.values.ndim != 2:
            raise ValueError("boundary-jet values must have shape (energy, channel)")
        if any(
            array.shape != shape
            for array in (
                self.radial_derivatives,
                self.energy_derivatives,
                self.energy_radial_derivatives,
                self.inverse_masses,
                self.energy_inverse_masses,
            )
        ):
            raise ValueError("all boundary-jet arrays must have the same shape")
        if self.potential_radii.shape != (shape[1],):
            raise ValueError("potential radii must contain one value per channel")


@dataclass(frozen=True)
class KinkMesh:
    """Kink matrices and their exact first energy derivatives."""

    energies: FloatArray
    potential_radii: FloatArray
    values: NumericArray
    derivatives: NumericArray

    def __post_init__(self) -> None:
        size = len(self.potential_radii)
        expected = (len(self.energies), size, size)
        if self.values.shape != expected or self.derivatives.shape != expected:
            raise ValueError("kink values and derivatives must have shape (energy, channel, channel)")


@dataclass(frozen=True)
class DownfoldedKink:
    """One active-space Schur complement and its passive reconstruction."""

    values: NumericArray
    derivatives: NumericArray | None
    reconstruction: NumericArray
    reconstruction_derivative: NumericArray | None
    residual_norm: float


def build_kink_mesh(
    energies: FloatArray,
    slope_matrices: NumericArray,
    slope_derivatives: NumericArray,
    boundary_jets: BoundaryJets,
    potential_radii: FloatArray,
) -> KinkMesh:
    r"""Build ``K`` and ``Kdot`` on an energy mesh.

    The convention is exactly

    .. math:: K = \tfrac12\operatorname{diag}(a) S
       - \operatorname{diag}(a^2 m^{-1} f'/f).

    The interstitial kinetic operator is ``-nabla^2/2``.  The radial flux
    uses ``m^-1=1/2`` for the nonrelativistic equation, and
    ``m^-1=[2+(E-V)/c^2]^-1`` for the scalar Koelling--Harmon equation.
    Both the logarithmic derivative and inverse mass are differentiated.
    """

    mesh = np.asarray(energies, dtype=float)
    slopes = np.asarray(slope_matrices)
    slope_dots = np.asarray(slope_derivatives)
    radii = np.asarray(potential_radii, dtype=float)
    if not np.array_equal(radii, boundary_jets.potential_radii):
        raise ValueError("kink radii must equal the boundary jets' proper potential radii")
    size = len(radii)
    expected = (len(mesh), size, size)
    if slopes.shape != expected or slope_dots.shape != expected:
        raise ValueError("slope values and derivatives must have shape (energy, channel, channel)")
    if boundary_jets.values.shape != (len(mesh), size):
        raise ValueError("boundary jets must use the same energy mesh and channels as the slopes")

    logarithmic_derivatives = boundary_jets.radial_derivatives / boundary_jets.values
    logarithmic_derivative_dots = (
        boundary_jets.energy_radial_derivatives / boundary_jets.values
        - boundary_jets.radial_derivatives
        * boundary_jets.energy_derivatives
        / (boundary_jets.values * boundary_jets.values)
    )
    values = 0.5 * radii[None, :, None] * slopes
    derivatives = 0.5 * radii[None, :, None] * slope_dots
    diagonal = np.arange(size)
    values[:, diagonal, diagonal] -= (
        radii[None, :] ** 2
        * boundary_jets.inverse_masses
        * logarithmic_derivatives
    )
    derivatives[:, diagonal, diagonal] -= (
        radii[None, :] ** 2
        * (
            boundary_jets.inverse_masses * logarithmic_derivative_dots
            + boundary_jets.energy_inverse_masses * logarithmic_derivatives
        )
    )
    return KinkMesh(
        energies=mesh,
        potential_radii=radii,
        values=values,
        derivatives=derivatives,
    )


def downfold_kink(
    values: NumericArray,
    active: NDArray[np.int64],
    passive: NDArray[np.int64],
    derivatives: NumericArray | None = None,
) -> DownfoldedKink:
    r"""Schur-downfold one kink matrix and return its reconstruction map.

    For active coefficients ``c_a``, the returned full map produces
    ``(c_a, c_p)`` in the original channel ordering with
    ``c_p = -K_pp^-1 K_pa c_a``.  The residual is the norm of the passive
    block equation ``K_pa + K_pp R_p``.
    """

    matrix = np.asarray(values)
    active_indices = np.asarray(active, dtype=np.int64)
    passive_indices = np.asarray(passive, dtype=np.int64)
    kaa = matrix[np.ix_(active_indices, active_indices)]
    kap = matrix[np.ix_(active_indices, passive_indices)]
    kpa = matrix[np.ix_(passive_indices, active_indices)]
    kpp = matrix[np.ix_(passive_indices, passive_indices)]
    passive_solution = solve(kpp, kpa)
    schur = kaa - contract("ij,jk->ik", kap, passive_solution)

    reconstruction = np.zeros(
        (matrix.shape[0], len(active_indices)), dtype=np.result_type(matrix.dtype)
    )
    reconstruction[active_indices, np.arange(len(active_indices))] = 1
    reconstruction[passive_indices] = -passive_solution
    passive_residual = kpa + contract(
        "ij,jk->ik", kpp, reconstruction[passive_indices]
    )

    schur_dot = None
    reconstruction_dot = None
    if derivatives is not None:
        matrix_dot = np.asarray(derivatives)
        kaa_dot = matrix_dot[np.ix_(active_indices, active_indices)]
        kap_dot = matrix_dot[np.ix_(active_indices, passive_indices)]
        kpa_dot = matrix_dot[np.ix_(passive_indices, active_indices)]
        kpp_dot = matrix_dot[np.ix_(passive_indices, passive_indices)]
        passive_solution_dot = solve(
            kpp,
            kpa_dot - contract("ij,jk->ik", kpp_dot, passive_solution),
        )
        schur_dot = (
            kaa_dot
            - contract("ij,jk->ik", kap_dot, passive_solution)
            - contract("ij,jk->ik", kap, passive_solution_dot)
        )
        reconstruction_dot = np.zeros_like(reconstruction)
        reconstruction_dot[passive_indices] = -passive_solution_dot

    return DownfoldedKink(
        values=schur,
        derivatives=schur_dot,
        reconstruction=reconstruction,
        reconstruction_derivative=reconstruction_dot,
        residual_norm=float(np.linalg.norm(passive_residual)),
    )


def downfold_kink_mesh(
    mesh: KinkMesh,
    active: NDArray[np.int64],
    passive: NDArray[np.int64],
) -> tuple[KinkMesh, tuple[DownfoldedKink, ...]]:
    """Downfold every matrix in a mesh, retaining per-energy reconstructions."""

    results = tuple(
        downfold_kink(value, active, passive, derivative)
        for value, derivative in zip(mesh.values, mesh.derivatives, strict=True)
    )
    active_indices = np.asarray(active, dtype=np.int64)
    reduced = KinkMesh(
        energies=mesh.energies,
        potential_radii=mesh.potential_radii[active_indices],
        values=np.stack(tuple(result.values for result in results)),
        derivatives=np.stack(tuple(result.derivatives for result in results)),
    )
    return reduced, results
