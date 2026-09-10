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


def test_shell_hat_matrix_is_nodal_and_vanishes_off_the_shell() -> None:
    from pymuffintin.mto.omt import shell_hat_matrix

    knots = np.array([1.0, 1.2, 1.6])
    hats = shell_hat_matrix(np.array([0.9, 1.0, 1.1, 1.2, 1.5, 1.6, 1.7]), knots)
    np.testing.assert_allclose(
        hats,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.25, 0.75],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
    )
    assert shell_hat_matrix(np.array([1.0, 2.0]), np.array([1.0])).shape == (2, 1)
    np.testing.assert_array_equal(shell_hat_matrix(np.array([1.0, 2.0]), np.array([1.0])), 0.0)


def _shell_fixture():
    from pymuffintin.mto.omt import OmtFitDiagnostics, OmtShellReference

    lattice = 6.0 * np.eye(3)
    centers = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    hard = np.array([1.0, 1.0])
    outer = np.array([1.6, 1.6])
    knots = (np.array([1.0, 1.2, 1.4, 1.6]), np.array([1.0, 1.25, 1.6]))
    constant = -0.3
    pinned = np.array([-0.8, -0.6])
    values = (
        np.array([pinned[0] - constant, -0.35, -0.1, 0.0]),
        np.array([pinned[1] - constant, -0.25, 0.0]),
    )
    reference = OmtShellReference(
        lattice=lattice,
        centers=centers,
        hard_radii=hard,
        potential_radii=outer,
        shell_knots=knots,
        shell_values=values,
        constant=constant,
        diagnostics=OmtFitDiagnostics(0.0, 0.0, 0.0, np.zeros((2, 2))),
    )
    return reference, pinned


def test_fit_omt_shells_recovers_a_pinned_constant_plus_tail_reference() -> None:
    from pymuffintin.mto.omt import fit_omt_shells, periodic_distances

    reference, pinned = _shell_fixture()
    rng = np.random.default_rng(3)
    points = rng.uniform(0.0, 6.0, size=(4000, 3))
    distances = periodic_distances(points, reference.centers, reference.lattice)
    interstitial = np.all(distances > reference.hard_radii[None, :], axis=1)
    inside = ~interstitial
    truth = reference.tail_field(points)
    # Inside a hard sphere the model residual V - V_sph is the penetrating tails only.
    fit = fit_omt_shells(
        lattice=reference.lattice,
        centers=reference.centers,
        hard_radii=reference.hard_radii,
        potential_radii=reference.potential_radii,
        shell_knots=reference.shell_knots,
        hard_sphere_values=pinned,
        interstitial_points=points[interstitial],
        interstitial_values=truth[interstitial],
        interstitial_weights=np.full(int(np.count_nonzero(interstitial)), 0.01),
        sphere_points=points[inside],
        sphere_residuals=truth[inside] - reference.constant,
        sphere_weights=np.full(int(np.count_nonzero(inside)), 0.02),
    )
    assert fit.has_any_shell
    np.testing.assert_allclose(fit.constant, reference.constant, atol=1e-11)
    for site in range(2):
        np.testing.assert_allclose(fit.shell_values[site], reference.shell_values[site], atol=1e-11)
        np.testing.assert_allclose(fit.shell_potential(site, np.array([0.5, 1.7])), 0.0)
        np.testing.assert_allclose(
            fit.shell_potential(site, reference.shell_knots[site]), reference.shell_values[site]
        )
    np.testing.assert_allclose(fit.diagnostics.weighted_rms, 0.0, atol=1e-11)
    np.testing.assert_allclose(fit.diagnostics.maximum_overlap_fraction, 3.2 / 3.0 - 1.0)
    np.testing.assert_allclose(fit.tail_field(points), truth, atol=1e-10)


def test_fit_omt_shells_without_shells_returns_the_weighted_interstitial_mean() -> None:
    from pymuffintin.mto.omt import fit_omt_shells

    lattice = 5.0 * np.eye(3)
    centers = np.array([[2.5, 2.5, 2.5]])
    values = np.array([-0.1, -0.3, 0.2])
    weights = np.array([1.0, 2.0, 1.0])
    fit = fit_omt_shells(
        lattice=lattice,
        centers=centers,
        hard_radii=np.array([1.0]),
        potential_radii=np.array([1.0]),
        shell_knots=(np.array([1.0]),),
        hard_sphere_values=np.array([-0.9]),
        interstitial_points=np.array([[0.1, 0.1, 0.1], [4.0, 0.2, 0.3], [0.5, 4.0, 4.0]]),
        interstitial_values=values,
        interstitial_weights=weights,
    )
    assert not fit.has_any_shell
    np.testing.assert_allclose(fit.constant, np.sum(values * weights) / np.sum(weights))
    np.testing.assert_allclose(fit.shell_potential(0, np.array([0.5, 1.0, 1.5])), 0.0)


def test_nearest_neighbor_distance_uses_exact_periodic_images() -> None:
    from pymuffintin.mto.omt import nearest_neighbor_distance

    a = 6.742
    lattice = a * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    centers = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]) @ lattice
    np.testing.assert_allclose(nearest_neighbor_distance(lattice, centers), a * np.sqrt(3.0) / 4.0)
    np.testing.assert_allclose(nearest_neighbor_distance(4.0 * np.eye(3), np.zeros((1, 3))), 4.0)
