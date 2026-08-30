"""Nth-order muffin-tin-orbital matrix transforms.

This module implements the matrix part of the NMTO construction: ordinary
and confluent divided differences of ``G(E)=K(E)^-1``, matrix-valued Lagrange
coefficients, nonorthogonal Hamiltonian/overlap matrices, and a strict
positive-definite Löwdin transform.  Energies are in Hartree throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..tensor import contract, eigh, inv, solve
from .kink import KinkMesh


NumericArray: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class GreenMesh:
    """Green matrices and exact energy derivatives on an NMTO mesh."""

    energies: FloatArray
    values: NumericArray
    derivatives: NumericArray


@dataclass(frozen=True)
class LowdinResult:
    """Strict symmetric Löwdin orthogonalization of one matrix pair."""

    transformation: NumericArray
    hamiltonian: NumericArray
    overlap_eigenvalues: FloatArray


@dataclass(frozen=True)
class NmtoResult:
    """The energy-mesh matrix data defining an Nth-order NMTO set."""

    energies: FloatArray
    green: NumericArray
    green_derivatives: NumericArray
    lagrange_matrices: NumericArray
    hamiltonian: NumericArray
    overlap: NumericArray
    lowdin: LowdinResult

    @property
    def order(self) -> int:
        return len(self.energies) - 1


def ordinary_divided_differences(
    energies: FloatArray,
    values: NumericArray,
) -> tuple[NumericArray, ...]:
    """Return the triangular ordinary matrix divided-difference table."""

    mesh = np.asarray(energies, dtype=float)
    current = np.asarray(values)
    table: list[NumericArray] = [current]
    trailing = (1,) * (current.ndim - 1)
    for order in range(1, len(mesh)):
        denominator = (mesh[order:] - mesh[:-order]).reshape((-1, *trailing))
        current = (current[1:] - current[:-1]) / denominator
        table.append(current)
    return tuple(table)


def green_mesh(kinks: KinkMesh) -> GreenMesh:
    r"""Invert ``K`` and form exact ``Gdot=-G Kdot G`` at every mesh point."""

    green = np.stack(tuple(inv(matrix) for matrix in kinks.values))
    derivatives = -contract(
        "eij,ejk,ekl->eil", green, kinks.derivatives, green
    )
    return GreenMesh(kinks.energies, green, derivatives)


def lagrange_matrices(
    energies: FloatArray,
    green: NumericArray,
) -> NumericArray:
    r"""Return the matrix Lagrange coefficients ``L_n^(N)``.

    If ``D=G[0,...,N]`` is the highest ordinary divided difference, then
    ``L_n = G(E_n) / prod_(m!=n)(E_n-E_m) D^-1``.
    """

    mesh = np.asarray(energies, dtype=float)
    values = np.asarray(green)
    highest = ordinary_divided_differences(mesh, values)[-1][0]
    result = np.empty_like(values)
    for node in range(len(mesh)):
        denominator = float(np.prod(mesh[node] - np.delete(mesh, node)))
        numerator = values[node] / denominator
        result[node] = solve(highest.T, numerator.T).T
    return result


def confluent_divided_difference(
    energies: FloatArray,
    values: NumericArray,
    derivatives: NumericArray,
    multiplicities: NDArray[np.int64],
) -> NumericArray:
    """Return one confluent Hermite matrix divided difference.

    Each energy may occur once or twice.  A doubled node consumes the supplied
    exact first derivative; no finite-difference estimate is made.
    """

    mesh = np.asarray(energies, dtype=float)
    counts = np.asarray(multiplicities, dtype=np.int64)
    if counts.shape != mesh.shape or np.any((counts < 1) | (counts > 2)):
        raise ValueError("Hermite multiplicities must contain one or two for every energy")
    repeated_nodes = np.repeat(mesh, counts)
    repeated_indices = np.repeat(np.arange(len(mesh)), counts)
    current = np.asarray(values)[repeated_indices]
    for order in range(1, len(repeated_nodes)):
        next_values = np.empty_like(current[:-1])
        for start in range(len(next_values)):
            if repeated_nodes[start + order] == repeated_nodes[start]:
                if order != 1:
                    raise ValueError("only first radial energy derivatives are supplied")
                next_values[start] = derivatives[repeated_indices[start]]
            else:
                next_values[start] = (current[start + 1] - current[start]) / (
                    repeated_nodes[start + order] - repeated_nodes[start]
                )
        current = next_values
    return current[0]


def _inverse_sandwich(matrix: NumericArray, middle: NumericArray) -> NumericArray:
    """Return ``matrix^-dagger middle matrix^-1`` using a strict solve."""

    inverse = solve(matrix, np.eye(matrix.shape[0], dtype=matrix.dtype))
    return contract("ji,jk,kl->il", inverse.conj(), middle, inverse)


def nmto_hamiltonian_overlap(
    green: GreenMesh,
) -> tuple[NumericArray, NumericArray]:
    r"""Form the Nth-order nonorthogonal NMTO Hamiltonian and overlap.

    With ``D=G[0,...,N]``, ``A=G[[0,...,N-1]N]`` and
    ``B=G[[0,...,N]]``, the formulas are

    ``S = -D^-1 B D^-1`` and
    ``H = E_N S - D^-1 A D^-1``.
    """

    mesh = green.energies
    order = len(mesh) - 1
    highest = ordinary_divided_differences(mesh, green.values)[-1][0]
    a_multiplicities = np.full(len(mesh), 2, dtype=np.int64)
    a_multiplicities[-1] = 1
    b_multiplicities = np.full(len(mesh), 2, dtype=np.int64)
    hermite_a = confluent_divided_difference(
        mesh, green.values, green.derivatives, a_multiplicities
    )
    hermite_b = confluent_divided_difference(
        mesh, green.values, green.derivatives, b_multiplicities
    )
    if order == 0:
        # The multiplicity pattern above already produces A=G(E0); the branch
        # only documents that no lower-order mesh is required.
        hermite_a = green.values[0]
    overlap = -_inverse_sandwich(highest, hermite_b)
    hamiltonian = mesh[-1] * overlap - _inverse_sandwich(highest, hermite_a)
    overlap = 0.5 * (overlap + overlap.conj().T)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    return hamiltonian, overlap


def lowdin_orthogonalize(
    hamiltonian: NumericArray,
    overlap: NumericArray,
) -> LowdinResult:
    """Apply symmetric Löwdin orthogonalization; reject every nonpositive mode."""

    eigenvalues, eigenvectors = eigh(overlap)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("NMTO overlap is not strictly positive definite")
    inverse_square_roots = 1.0 / np.sqrt(eigenvalues)
    transformation = contract(
        "ia,a,ja->ij", eigenvectors, inverse_square_roots, eigenvectors.conj()
    )
    orthogonal_hamiltonian = contract(
        "ij,jk,kl->il",
        transformation.conj().T,
        hamiltonian,
        transformation,
    )
    return LowdinResult(
        transformation=transformation,
        hamiltonian=orthogonal_hamiltonian,
        overlap_eigenvalues=eigenvalues,
    )


def build_nmto(kinks: KinkMesh) -> NmtoResult:
    """Build all Nth-order NMTO matrices from a kink mesh."""

    green = green_mesh(kinks)
    lagrange = lagrange_matrices(green.energies, green.values)
    hamiltonian, overlap = nmto_hamiltonian_overlap(green)
    lowdin = lowdin_orthogonalize(hamiltonian, overlap)
    return NmtoResult(
        energies=green.energies,
        green=green.values,
        green_derivatives=green.derivatives,
        lagrange_matrices=lagrange,
        hamiltonian=hamiltonian,
        overlap=overlap,
        lowdin=lowdin,
    )
