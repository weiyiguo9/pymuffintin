import numpy as np
import pytest

from pymuffintin.mto.kink import (
    BoundaryJets,
    KinkMesh,
    build_kink_mesh,
    downfold_kink,
)
from pymuffintin.mto.nmto import (
    GreenMesh,
    build_nmto,
    confluent_divided_difference,
    green_mesh,
    lagrange_matrices,
    lowdin_orthogonalize,
)


def test_kink_matrix_uses_exact_radial_energy_jets() -> None:
    energies = np.array([-0.4, 0.2])
    radii = np.array([0.7, 1.1])
    slopes_0 = np.array([[1.5, 0.2], [0.2 * radii[0] / radii[1], 0.8]])
    slope_dot = np.array([[0.3, -0.1], [-0.1 * radii[0] / radii[1], 0.4]])
    slopes = slopes_0[None, :, :] + energies[:, None, None] * slope_dot
    values = 1.2 + energies[:, None] * np.array([0.2, -0.1])
    radial = 0.4 + energies[:, None] * np.array([0.3, 0.5])
    jets = BoundaryJets(
        potential_radii=radii,
        values=values,
        radial_derivatives=radial,
        energy_derivatives=np.broadcast_to(np.array([0.2, -0.1]), values.shape),
        energy_radial_derivatives=np.broadcast_to(np.array([0.3, 0.5]), values.shape),
    )
    mesh = build_kink_mesh(
        energies,
        slopes,
        np.broadcast_to(slope_dot, slopes.shape),
        jets,
        radii,
    )

    log_dot = jets.energy_radial_derivatives / values - radial * jets.energy_derivatives / values**2
    expected_dot = radii[None, :, None] * np.broadcast_to(slope_dot, slopes.shape)
    diagonal = np.arange(2)
    expected_dot[:, diagonal, diagonal] -= radii[None, :] ** 2 * log_dot
    np.testing.assert_allclose(mesh.derivatives, expected_dot)


def test_schur_downfold_reconstructs_the_passive_block_exactly() -> None:
    matrix = np.array(
        [[4.0, 1.0, 2.0], [1.0, 3.0, -1.0], [2.0, -1.0, 5.0]]
    )
    result = downfold_kink(
        matrix,
        np.array([0, 2], dtype=np.int64),
        np.array([1], dtype=np.int64),
    )
    expected = matrix[np.ix_([0, 2], [0, 2])] - np.outer(
        matrix[[0, 2], 1], matrix[1, [0, 2]]
    ) / matrix[1, 1]
    np.testing.assert_allclose(result.values, expected)
    np.testing.assert_allclose(result.residual_norm, 0.0, atol=1.0e-15)


def test_confluent_divided_difference_is_exact_for_a_cubic_matrix_polynomial() -> None:
    energies = np.array([-1.0, 2.0])
    cubic = np.array([[2.0, -0.3], [-0.3, 1.0]])
    quadratic = np.array([[0.4, 0.2], [0.2, -0.7]])
    linear = np.eye(2)
    constant = np.array([[3.0, 0.1], [0.1, 2.0]])
    values = np.stack(
        tuple(cubic * e**3 + quadratic * e**2 + linear * e + constant for e in energies)
    )
    derivatives = np.stack(
        tuple(3.0 * cubic * e**2 + 2.0 * quadratic * e + linear for e in energies)
    )
    difference = confluent_divided_difference(
        energies, values, derivatives, np.array([2, 2], dtype=np.int64)
    )
    np.testing.assert_allclose(difference, cubic, atol=1.0e-14)


def test_n0_nmto_recovers_an_exact_linear_kink_pencil() -> None:
    energy = -0.2
    hamiltonian = np.array([[-0.8, 0.15], [0.15, 0.6]])
    identity = np.eye(2)
    kinks = KinkMesh(
        energies=np.array([energy]),
        potential_radii=np.ones(2),
        values=np.array([energy * identity - hamiltonian]),
        derivatives=np.array([identity]),
    )
    result = build_nmto(kinks)

    np.testing.assert_allclose(result.overlap, identity, atol=1.0e-13)
    np.testing.assert_allclose(result.hamiltonian, hamiltonian, atol=1.0e-13)
    np.testing.assert_allclose(result.lowdin.hamiltonian, hamiltonian, atol=1.0e-13)


def test_matrix_lagrange_coefficients_sum_to_identity() -> None:
    energies = np.array([-1.0, -0.1, 0.7])
    hamiltonian = np.array([[-1.8, 0.2], [0.2, 1.4]])
    identity = np.eye(2)
    kinks = KinkMesh(
        energies=energies,
        potential_radii=np.ones(2),
        values=np.stack(tuple(e * identity - hamiltonian for e in energies)),
        derivatives=np.broadcast_to(identity, (len(energies), 2, 2)),
    )
    green = green_mesh(kinks)
    matrices = lagrange_matrices(energies, green.values)
    np.testing.assert_allclose(np.sum(matrices, axis=0), identity, atol=2.0e-13)


def test_lowdin_requires_a_strictly_positive_overlap() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        lowdin_orthogonalize(np.eye(2), np.diag([1.0, 0.0]))
