import numpy as np

from pymuffintin.mto import (
    RealHarmonic,
    build_vd_coefficients,
    combine_usw,
    constant_interstitial_volume,
    constrained_vd_weights,
    evaluate_usw,
    interstitial_volume,
    usw_matrices,
    usw_matrices_with_energy_derivative,
)


def test_single_sphere_usw_and_vd_super_unitary_jets() -> None:
    radius = 0.8
    energies = np.array([-0.5, -1.0, -2.0, -4.0])
    slopes = np.empty((4, 1, 1))
    scripted_slopes = np.empty_like(slopes)
    boundary_jets = np.empty((4, 4, 1, 1))

    for index, energy in enumerate(energies):
        coefficients, slope = usw_matrices(
            energy,
            np.zeros((1, 3)),
            np.array([radius]),
            (RealHarmonic(0, 0),),
        )
        np.testing.assert_allclose(
            slope[0, 0], -1.0 - radius * np.sqrt(-2.0 * energy)
        )
        assert coefficients.shape == (1, 1)
        slopes[index] = slope
        scripted_slopes[index] = slope + np.eye(1)
        boundary_jets[index, 0, 0, 0] = radius
        boundary_jets[index, 1, 0, 0] = scripted_slopes[index, 0, 0]
        boundary_jets[index, 2, 0, 0] = -2.0 * radius * energy
        boundary_jets[index, 3, 0, 0] = (
            -2.0 * energy * scripted_slopes[index, 0, 0]
        )

    transformation = build_vd_coefficients(
        energies,
        scripted_slopes,
        np.array([radius]),
        np.array([0]),
    )
    transformed_jets = combine_usw(boundary_jets, transformation)[:, :, 0, 0]
    np.testing.assert_allclose(transformed_jets, np.eye(4), atol=2.0e-12)


def test_screened_slope_radius_symmetry_and_energy_order_invariance() -> None:
    centers = np.array([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]])
    radii = np.array([0.7, 0.9])
    channels = (RealHarmonic(0, 0),)
    energies = np.array([-0.4, -0.8, -1.5, -2.5])
    points = np.array([[0.0, 1.4, 0.0], [1.2, 1.1, 0.3]])
    slopes = []
    values = []
    for energy in energies:
        _, slope = usw_matrices(energy, centers, radii, channels)
        np.testing.assert_allclose(radii[:, None] * slope, (radii[:, None] * slope).T)
        slopes.append(slope + np.eye(2))
        values.append(evaluate_usw(energy, points, centers, radii, channels))
    slopes_array = np.stack(slopes)
    values_array = np.stack(values)
    angular_momenta = np.zeros(2, dtype=int)
    direct = combine_usw(
        values_array,
        build_vd_coefficients(energies, slopes_array, radii, angular_momenta),
    )

    permutation = np.array([2, 0, 3, 1])
    permuted = combine_usw(
        values_array[permutation],
        build_vd_coefficients(
            energies[permutation],
            slopes_array[permutation],
            radii,
            angular_momenta,
        ),
    )
    np.testing.assert_allclose(permuted, direct, rtol=2.0e-11, atol=2.0e-12)

    energy = -0.8
    _, slope, slope_energy = usw_matrices_with_energy_derivative(
        energy, centers, radii, channels
    )
    step = 1.0e-6
    _, slope_plus = usw_matrices(energy + step, centers, radii, channels)
    _, slope_minus = usw_matrices(energy - step, centers, radii, channels)
    np.testing.assert_allclose(
        slope_energy,
        (slope_plus - slope_minus) / (2.0 * step),
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_positive_energy_standing_wave_slope_derivative() -> None:
    centers = np.array([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]])
    radii = np.array([0.7, 0.9])
    channels = (RealHarmonic(0, 0),)
    energy = 0.08
    _, slope, slope_energy = usw_matrices_with_energy_derivative(
        energy, centers, radii, channels
    )
    step = 1.0e-6
    _, slope_plus = usw_matrices(energy + step, centers, radii, channels)
    _, slope_minus = usw_matrices(energy - step, centers, radii, channels)
    np.testing.assert_allclose(radii[:, None] * slope, (radii[:, None] * slope).T)
    np.testing.assert_allclose(
        slope_energy,
        (slope_plus - slope_minus) / (2.0 * step),
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_open_structure_weights_are_the_strict_minimum_norm_solution() -> None:
    localized = np.zeros((2, 4, 1))
    extended = np.array(
        [
            [[1.0], [0.0], [1.0], [0.0]],
            [[0.0], [1.0], [0.0], [1.0]],
        ]
    )
    target = np.array([2.0, -1.0])

    weights = constrained_vd_weights(localized, extended, target)

    np.testing.assert_allclose(weights[:, 0], np.array([1.0, -0.5, 1.0, -0.5]))


def test_gate_a_reproduces_table_i_constant_density_volume_errors() -> None:
    cases = (
        ("bcc", 51, 0.8, 0.14),
        ("diamond", 159, 0.8, -1.17),
    )
    for structure, n_sites, radius, published_error in cases:
        measured = constant_interstitial_volume(structure, n_sites, radius)
        exact = interstitial_volume(structure, radius)
        error_times_1000 = 1.0e3 * (measured - exact) / exact
        np.testing.assert_allclose(error_times_1000, published_error, atol=0.02)
