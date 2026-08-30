import numpy as np
import pytest

from pymuffintin.auxiliary import (
    build_hybrid_coulomb,
    interstitial_thc,
    muffin_tin_lri,
    weighted_isdf,
)
from pymuffintin.contracts import (
    AuxiliaryRepresentation,
    CoulombBlock,
    FixedOccupation,
    PairLayout,
    PairSamples,
    RegionalChargeExpansion,
)
from pymuffintin.mbpt import compare_exchange, fixed_orbital_exchange


def _layout() -> PairLayout:
    return PairLayout(n_k=1, n_orb=2, n_columns=4)


def test_contracts_reject_implicit_dtype_conversion() -> None:
    with pytest.raises(TypeError, match="dtype complex128"):
        RegionalChargeExpansion(
            region="interstitial", coefficients=np.ones((4, 1), dtype=np.float64)
        )


def test_muffin_tin_lri_uses_a_separate_overlap_evd_per_site() -> None:
    layout = _layout()
    values = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )
    samples = PairSamples(
        q_index=0,
        layout=layout,
        points=np.zeros((4, 3), dtype=np.float64),
        weights=np.ones(4, dtype=np.float64),
        site_indices=np.array([0, 0, 1, 1], dtype=np.int64),
        values=values,
    )
    result = muffin_tin_lri(samples, cutoff=0.0)

    assert [block.region for block in result.expansions] == ["muffin_tin:0", "muffin_tin:1"]
    assert [block.n_auxiliary for block in result.expansions] == [2, 2]
    np.testing.assert_allclose(
        result.coefficients @ result.coefficients.conj().T,
        values.conj().T @ values,
        atol=1.0e-14,
    )


def test_weighted_qrcp_isdf_is_deterministic_and_reports_fit_residual() -> None:
    values = np.array([[2, 0], [0, 1], [1, 1]], dtype=np.complex128)
    weights = np.ones(3, dtype=np.float64)
    fit = weighted_isdf(values, weights, rank=2)

    np.testing.assert_array_equal(fit.point_indices, np.array([0, 1], dtype=np.int64))
    np.testing.assert_allclose(fit.zeta @ values[fit.point_indices], values, atol=1.0e-14)
    assert fit.residual_norm < 1.0e-14


def test_interstitial_thc_preserves_global_point_indices() -> None:
    layout = PairLayout(n_k=1, n_orb=1, n_columns=1)
    samples = PairSamples(
        q_index=0,
        layout=layout,
        points=np.zeros((3, 3), dtype=np.float64),
        weights=np.ones(3, dtype=np.float64),
        site_indices=np.array([0, -1, -1], dtype=np.int64),
        values=np.array([[3], [2], [1]], dtype=np.complex128),
    )
    representation, selection = interstitial_thc(samples, rank=1)

    np.testing.assert_array_equal(selection.point_indices, np.array([1], dtype=np.int64))
    np.testing.assert_allclose(representation.coefficients, np.array([[2]], dtype=np.complex128))


def test_hybrid_requests_one_full_coulomb_matrix_with_cross_blocks() -> None:
    layout = PairLayout(n_k=1, n_orb=1, n_columns=1)
    muffin_tin = AuxiliaryRepresentation(
        q_index=0,
        layout=layout,
        expansions=(
            RegionalChargeExpansion(
                "muffin_tin:0", np.array([[1]], dtype=np.complex128)
            ),
        ),
        residual_norm=0.0,
    )
    interstitial = AuxiliaryRepresentation(
        q_index=0,
        layout=layout,
        expansions=(
            RegionalChargeExpansion(
                "interstitial", np.array([[2]], dtype=np.complex128)
            ),
        ),
        residual_norm=0.0,
    )

    class FullMatrixProvider:
        received: AuxiliaryRepresentation | None = None

        def coulomb(self, representation, *, gamma_policy, **request):
            self.received = representation
            return CoulombBlock(
                q_index=0,
                matrix=np.array([[3, 0.5], [0.5, 4]], dtype=np.complex128),
                gamma_policy=gamma_policy,
            )

    provider = FullMatrixProvider()
    hybrid, block = build_hybrid_coulomb(
        muffin_tin, interstitial, provider, gamma_policy="finite_body"
    )
    assert provider.received is hybrid
    assert block.matrix[0, 1] == 0.5


def test_fixed_orbital_exchange_and_ablation_use_explicit_weights() -> None:
    layout = PairLayout(n_k=1, n_orb=1, n_columns=1)
    reference_representation = AuxiliaryRepresentation(
        q_index=0,
        layout=layout,
        expansions=(
            RegionalChargeExpansion(
                "muffin_tin:0", np.array([[2]], dtype=np.complex128)
            ),
        ),
        residual_norm=0.0,
    )
    trial_representation = AuxiliaryRepresentation(
        q_index=0,
        layout=layout,
        expansions=(
            RegionalChargeExpansion(
                "interstitial", np.array([[1]], dtype=np.complex128)
            ),
        ),
        residual_norm=0.0,
    )
    coulomb = CoulombBlock(
        q_index=0, matrix=np.array([[3]], dtype=np.complex128), gamma_policy="finite_body"
    )
    occupation = FixedOccupation(
        values=np.ones((1, 1), dtype=np.float64),
        k_weights=np.ones(1, dtype=np.float64),
        q_weights=np.ones(1, dtype=np.float64),
        k_minus_q_indices=np.zeros((1, 1), dtype=np.int64),
    )
    reference = fixed_orbital_exchange((reference_representation,), (coulomb,), occupation)
    trial = fixed_orbital_exchange((trial_representation,), (coulomb,), occupation)
    ablation = compare_exchange(reference, trial)

    np.testing.assert_allclose(reference.sigma_x, [[[-12.0]]])
    assert reference.exchange_energy == -6.0
    np.testing.assert_allclose(ablation.sigma_difference, [[[9.0]]])
    assert ablation.exchange_energy_difference == 4.5
