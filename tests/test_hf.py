import numpy as np

from pymuffintin.contracts import (
    AuxiliaryRepresentation,
    CoulombBlock,
    FixedOccupation,
    PairLayout,
    RegionalChargeExpansion,
)
from pymuffintin.mbpt import fixed_orbital_exchange


def _brute_force_sigma(
    coefficients: np.ndarray,
    matrix: np.ndarray,
    occupation_values: np.ndarray,
    k_minus_q_indices: np.ndarray,
    q_weight: float,
    n_k: int,
    n_orb: int,
    n_aux: int,
) -> np.ndarray:
    """Independent scalar-loop reference: sigma[k,j,j'] = -q_weight *
    sum_i occ[k-q,i] * sum_{mu,nu} conj(C[k,i,j,mu]) V[mu,nu] C[k,i,j',nu].

    Indexes the flat (n_columns, n_aux) coefficient array by the raw
    PairColumnLayout formula column = k*n_orb^2 + i*n_orb + j directly,
    rather than the implementation's `.reshape`, so a reshape-order bug
    would not be masked by reusing the same indexing path.
    """
    sigma = np.zeros((n_k, n_orb, n_orb), dtype=np.complex128)
    for k in range(n_k):
        kq = int(k_minus_q_indices[k])
        for j in range(n_orb):
            for j_prime in range(n_orb):
                total = 0.0 + 0.0j
                for i in range(n_orb):
                    column = k * n_orb * n_orb + i * n_orb + j
                    column_prime = k * n_orb * n_orb + i * n_orb + j_prime
                    inner = 0.0 + 0.0j
                    for mu in range(n_aux):
                        for nu in range(n_aux):
                            inner += (
                                np.conj(coefficients[column, mu])
                                * matrix[mu, nu]
                                * coefficients[column_prime, nu]
                            )
                    total += occupation_values[kq, i] * inner
                sigma[k, j, j_prime] = -q_weight * total
    return sigma


def test_fixed_orbital_exchange_matches_brute_force_reference_with_asymmetric_occupations() -> None:
    rng = np.random.default_rng(20260830)
    n_k, n_orb, n_aux = 2, 3, 4
    layout = PairLayout(n_k=n_k, n_orb=n_orb, n_columns=n_k * n_orb * n_orb)

    coefficients = (
        rng.standard_normal((layout.n_columns, n_aux))
        + 1j * rng.standard_normal((layout.n_columns, n_aux))
    ).astype(np.complex128)
    representation = AuxiliaryRepresentation(
        q_index=0,
        layout=layout,
        expansions=(RegionalChargeExpansion(region="synthetic", coefficients=coefficients),),
        residual_norm=0.0,
    )

    generator = rng.standard_normal((n_aux, n_aux)) + 1j * rng.standard_normal((n_aux, n_aux))
    matrix = generator @ generator.conj().T + n_aux * np.eye(n_aux)  # Hermitian positive-definite
    coulomb = CoulombBlock(q_index=0, matrix=matrix.astype(np.complex128), gamma_policy="test")

    # Asymmetric across both bands and k: no permutation symmetry that could
    # accidentally hide an i/j axis swap.
    occupation_values = np.array([[0.9, 0.2, 0.05], [0.1, 0.7, 0.4]], dtype=np.float64)
    k_minus_q_indices = np.array([[1, 0]], dtype=np.int64)  # k=0 -> k-q=1, k=1 -> k-q=0
    occupation = FixedOccupation(
        values=occupation_values,
        k_weights=np.array([0.6, 0.4], dtype=np.float64),
        q_weights=np.array([1.3], dtype=np.float64),
        k_minus_q_indices=k_minus_q_indices,
    )

    result = fixed_orbital_exchange((representation,), (coulomb,), occupation)

    expected = _brute_force_sigma(
        coefficients,
        matrix,
        occupation_values,
        k_minus_q_indices[0],
        float(occupation.q_weights[0]),
        n_k,
        n_orb,
        n_aux,
    )
    np.testing.assert_allclose(result.sigma_x, expected, rtol=1.0e-10, atol=1.0e-12)

    for k_index in range(n_k):
        np.testing.assert_allclose(
            result.sigma_x[k_index],
            result.sigma_x[k_index].conj().T,
            rtol=0.0,
            atol=1.0e-12,
        )

    expected_energy = 0.0
    for k_index in range(n_k):
        expected_energy += 0.5 * occupation.k_weights[k_index] * float(
            np.sum(occupation_values[k_index] * np.real(np.diag(expected[k_index])))
        )
    np.testing.assert_allclose(
        result.exchange_energy, expected_energy, rtol=1.0e-10, atol=1.0e-12
    )
