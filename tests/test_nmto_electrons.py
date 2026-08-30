import numpy as np

from pymuffintin.mto import (
    LowdinResult,
    NmtoBands,
    NmtoOccupations,
    NmtoResult,
    fermi_dirac_occupations,
    interpolate_nmto_basis,
    nmto_density_matrices,
    sample_nmto_density,
    solve_nmto_bands,
)


def _nmto_result(hamiltonian: np.ndarray, transformation: np.ndarray) -> NmtoResult:
    basis_count = hamiltonian.shape[0]
    return NmtoResult(
        energies=np.array([0.0]),
        green=np.zeros((1, basis_count, basis_count)),
        green_derivatives=np.zeros((1, basis_count, basis_count)),
        lagrange_matrices=np.eye(basis_count)[None, :, :],
        hamiltonian=hamiltonian,
        overlap=np.eye(basis_count),
        lowdin=LowdinResult(
            transformation=transformation,
            hamiltonian=hamiltonian,
            overlap_eigenvalues=np.ones(basis_count),
        ),
    )


def test_solve_nmto_bands_maps_lowdin_eigenvectors_to_the_nmto_basis() -> None:
    hamiltonian = np.array([[1.0, 0.0], [0.0, -0.5]])
    transformation = np.diag([2.0, 0.5])

    bands = solve_nmto_bands((_nmto_result(hamiltonian, transformation),))

    np.testing.assert_allclose(bands.energies, [[-0.5, 1.0]])
    np.testing.assert_allclose(
        bands.coefficients,
        np.einsum(
            "kab,kbc->kac",
            transformation[None, :, :],
            bands.orthonormal_coefficients,
        ),
    )


def test_fermi_dirac_occupations_reproduce_count_and_band_energy() -> None:
    energies = np.array([[-1.0, 1.0], [-0.5, 0.5]])
    weights = np.array([0.25, 0.75])

    occupations = fermi_dirac_occupations(
        energies,
        weights,
        electron_count=2.0,
        temperature=0.2,
    )

    np.testing.assert_allclose(occupations.chemical_potential, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        occupations.electron_count,
        np.einsum("k,kb->", weights, occupations.values),
    )
    np.testing.assert_allclose(
        occupations.band_energy,
        np.einsum("k,kb,kb->", weights, occupations.values, energies),
    )
    assert occupations.minus_temperature_entropy < 0.0


def test_nmto_basis_interpolation_and_sampled_density() -> None:
    node_values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    lagrange = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.0], [0.0, 0.5]],
        ]
    )
    np.testing.assert_allclose(
        interpolate_nmto_basis(node_values, lagrange),
        np.einsum("epc,ecb->pb", node_values, lagrange),
    )

    coefficients = np.array([[[1.0, 0.0], [0.0, 0.5]]])
    bands = NmtoBands(
        energies=np.array([[-1.0, 1.0]]),
        orthonormal_coefficients=np.eye(2)[None, :, :],
        coefficients=coefficients,
    )
    occupations = NmtoOccupations(
        chemical_potential=0.0,
        values=np.array([[2.0, 1.0]]),
        electron_count=3.0,
        band_energy=-1.0,
    )
    density_matrices = nmto_density_matrices(bands, occupations)
    np.testing.assert_allclose(density_matrices, [[[2.0, 0.0], [0.0, 0.25]]])

    basis_values = np.array([[[1.0, 2.0], [0.5, -1.0]]])
    np.testing.assert_allclose(
        sample_nmto_density(basis_values, bands, occupations, np.ones(1)),
        [3.0, 0.75],
    )
