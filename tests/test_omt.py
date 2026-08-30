import numpy as np

from pymuffintin.mto.omt import (
    evaluate_omt,
    fit_omt,
    overlap_fractions,
    radial_hat_matrix,
)


def test_radial_hats_are_nodal_and_join_zero_at_the_potential_radius() -> None:
    knots = np.array([0.0, 0.5, 1.0])
    values = radial_hat_matrix(np.array([0.0, 0.5, 0.75, 1.0, 1.2]), knots)
    np.testing.assert_allclose(
        values,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.5],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_periodic_omt_weighted_fit_recovers_a_constant_and_radial_hats() -> None:
    lattice = 4.0 * np.eye(3)
    centers = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([1.0])
    knots = (np.array([0.0, 0.5, 1.0]),)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [1.25, 0.0, 0.0],
        ]
    )
    constant = -0.7
    coefficients = np.array([1.4, -0.2])
    values = constant + radial_hat_matrix(np.linalg.norm(points, axis=1), knots[0]) @ coefficients
    fit = fit_omt(
        points,
        values,
        np.array([1.0, 2.0, 1.5, 0.7, 3.0]),
        lattice,
        centers,
        radii,
        knots,
    )

    np.testing.assert_allclose(fit.constant, constant, atol=1.0e-14)
    np.testing.assert_allclose(fit.radial_coefficients[0], coefficients, atol=1.0e-14)
    np.testing.assert_allclose(fit.diagnostics.weighted_residual_norm, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        evaluate_omt(fit, points + lattice[0]), values, atol=1.0e-14
    )


def test_overlap_fraction_uses_the_exact_nearest_periodic_image_formula() -> None:
    lattice = np.diag([4.0, 5.0, 6.0])
    centers = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    radii = np.array([0.8, 0.7])
    fractions = overlap_fractions(lattice, centers, radii)

    np.testing.assert_allclose(fractions[0, 1], (0.8 + 0.7) / 1.0 - 1.0)
    np.testing.assert_allclose(fractions[0, 0], 2.0 * 0.8 / 4.0 - 1.0)
    np.testing.assert_allclose(fractions[1, 1], 2.0 * 0.7 / 4.0 - 1.0)
