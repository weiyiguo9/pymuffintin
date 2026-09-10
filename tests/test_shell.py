import numpy as np
import pytest

from pymuffintin.mto.shell import (
    continuation_jets,
    exponential_mesh,
    exponential_mesh_count,
    free_continuation,
    snap_to_exponential_mesh,
    sphere_images,
)
from pymuffintin.mto.usw import (
    _decaying_hankel_with_energy_derivative,
    _regular_bessel_with_energy_derivative,
    _standing_neumann_with_energy_derivative,
)


FIRST = 4.58863382e-05
INCREMENT = 0.022


def test_exponential_mesh_snapping_and_counting_round_trip() -> None:
    mesh = exponential_mesh(FIRST, INCREMENT, 471)
    np.testing.assert_allclose(mesh[-1], 1.42, rtol=1.0e-9)
    assert exponential_mesh_count(FIRST, INCREMENT, mesh[-1]) == 471
    snapped = snap_to_exponential_mesh(FIRST, INCREMENT, 1.2 * 1.42)
    assert snapped >= 1.2 * 1.42
    assert snapped < 1.2 * 1.42 * np.exp(INCREMENT)
    assert exponential_mesh_count(FIRST, INCREMENT, snapped) == 480
    assert snap_to_exponential_mesh(FIRST, INCREMENT, mesh[300]) == pytest.approx(mesh[300], rel=1e-12)
    with pytest.raises(ValueError):
        exponential_mesh_count(FIRST, INCREMENT, 1.5)


@pytest.mark.parametrize("l", [0, 1, 2])
@pytest.mark.parametrize("kinetic", [-0.3, 0.17])
def test_free_continuation_without_shell_returns_the_flux_bridged_native_jets(l, kinetic) -> None:
    rng = np.random.default_rng(7 + l)
    energies = np.array([kinetic, kinetic + 0.05])
    jets = rng.normal(size=(2, 6))
    jets[:, 4] = 0.4999  # inverse mass
    radius = 1.3
    continuation = free_continuation(l, energies, radius, radius, np.array([radius]), jets)

    np.testing.assert_allclose(continuation.boundary_values, jets[:, 0], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        continuation.boundary_radial, 2.0 * jets[:, 4] * jets[:, 1], rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(continuation.boundary_energy, jets[:, 2], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        continuation.boundary_energy_radial,
        2.0 * (jets[:, 5] * jets[:, 1] + jets[:, 4] * jets[:, 3]),
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(continuation.values[:, 0], jets[:, 0], rtol=1e-12)


@pytest.mark.parametrize("l", [0, 1, 3])
@pytest.mark.parametrize("kinetic", [-0.25, 0.4])
def test_free_continuation_reproduces_a_free_regular_wave(l, kinetic) -> None:
    hard, outer = 1.1, 1.7
    radii = np.linspace(hard, outer, 9)
    energies = np.array([kinetic])
    j, j_r, j_e, j_re = _regular_bessel_with_energy_derivative(l, kinetic, np.array([outer]))
    jets = np.array([[j[0], j_r[0], j_e[0], j_re[0], 0.5, 0.0]])
    continuation = free_continuation(l, energies, hard, outer, radii, jets)

    expected, expected_r, expected_e, expected_re = _regular_bessel_with_energy_derivative(
        l, kinetic, radii
    )
    np.testing.assert_allclose(continuation.values[0], expected, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(continuation.boundary_values[0], expected[0], rtol=1e-10)
    np.testing.assert_allclose(continuation.boundary_radial[0], expected_r[0], rtol=1e-10)
    np.testing.assert_allclose(continuation.boundary_energy[0], expected_e[0], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(
        continuation.boundary_energy_radial[0], expected_re[0], rtol=1e-8, atol=1e-12
    )


def test_free_continuation_energy_derivative_matches_finite_differences() -> None:
    l, hard, outer = 1, 1.0, 1.5
    radii = np.array([hard, 1.25, outer])
    kinetic = -0.2
    step = 1.0e-6
    _, _, _, _ = _decaying_hankel_with_energy_derivative(l, kinetic, radii)
    _, _, _, _ = _standing_neumann_with_energy_derivative(l, 0.3, radii)

    def jets_at(energy: float) -> np.ndarray:
        # A smooth, energy-dependent but otherwise arbitrary native boundary trace.
        u = 0.7 + 0.3 * energy
        du = -0.4 + 0.1 * energy
        mass = 0.5 - 0.01 * energy
        return np.array([[u, du, 0.3, 0.1, mass, -0.01]])

    center = free_continuation(l, [kinetic], hard, outer, radii, jets_at(kinetic))
    plus = free_continuation(l, [kinetic + step], hard, outer, radii, jets_at(kinetic + step))
    minus = free_continuation(l, [kinetic - step], hard, outer, radii, jets_at(kinetic - step))
    np.testing.assert_allclose(
        center.boundary_energy[0],
        (plus.boundary_values[0] - minus.boundary_values[0]) / (2.0 * step),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        center.boundary_energy_radial[0],
        (plus.boundary_radial[0] - minus.boundary_radial[0]) / (2.0 * step),
        rtol=1e-6,
        atol=1e-8,
    )


def test_free_continuation_rejects_zero_kinetic_energy_and_bad_radii() -> None:
    jets = np.array([[1.0, 0.1, 0.0, 0.0, 0.5, 0.0]])
    with pytest.raises(ValueError):
        free_continuation(0, [0.0], 1.0, 1.2, np.array([1.0, 1.2]), jets)
    with pytest.raises(ValueError):
        free_continuation(0, [-0.1], 1.0, 1.2, np.array([1.05, 1.2]), jets)


def test_continuation_jets_use_the_free_flux_convention() -> None:
    jets = np.array([[0.9, -0.1, -0.3, -1.0, 0.5, -1e-5], [0.8, -0.2, -0.2, -0.9, 0.5, -1e-5]])
    left = free_continuation(0, [-0.2, 0.1], 1.0, 1.0, np.array([1.0]), jets)
    right = free_continuation(1, [-0.2, 0.1], 1.2, 1.2, np.array([1.2]), jets)
    assembled = continuation_jets([left, right, right])
    np.testing.assert_array_equal(assembled.potential_radii, [1.0, 1.2, 1.2])
    assert assembled.values.shape == (2, 3)
    np.testing.assert_array_equal(assembled.inverse_masses, 0.5)
    np.testing.assert_array_equal(assembled.energy_inverse_masses, 0.0)
    np.testing.assert_allclose(assembled.values[:, 0], jets[:, 0])
    np.testing.assert_allclose(assembled.radial_derivatives[:, 1], jets[:, 1])


def test_sphere_images_lists_every_periodic_shell_image() -> None:
    lattice = 4.0 * np.eye(3)
    centers = np.zeros((1, 3))
    points = np.array(
        [
            [3.5, 0.0, 0.0],  # in the shell of the image at (4,0,0)
            [0.1, 0.0, 0.0],  # inside the hard sphere: excluded
            [2.0, 0.0, 0.0],  # equidistant from two images
            [1.0, 1.0, 1.0],  # inside the home potential sphere at sqrt(3)
            [2.0, 2.0, 2.0],  # outside every potential sphere at 2 sqrt(3)
        ]
    )
    result = sphere_images(points, lattice, centers, np.array([0.2]), np.array([2.0]))
    assert len(result) == 1
    site, indices, displacements = result[0]
    assert site == 0
    assert sorted(indices.tolist()) == [0, 2, 2, 3]
    distances = np.linalg.norm(displacements, axis=1)
    np.testing.assert_allclose(sorted(distances), [0.5, np.sqrt(3.0), 2.0, 2.0])
    assert sphere_images(points, lattice, centers, np.array([2.0]), np.array([2.0])) == []
