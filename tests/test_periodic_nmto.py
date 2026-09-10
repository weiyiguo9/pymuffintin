import numpy as np
import pytest

from pymuffintin.mto.kink import BoundaryJets, build_kink_mesh
from pymuffintin.mto.nmto import build_nmto
from pymuffintin.mto.periodic import PeriodicUswGeometry
from pymuffintin.mto.usw import (
    RealHarmonic,
    _regular_bessel_with_energy_derivative,
    real_spherical_harmonics,
)


@pytest.fixture(scope="module")
def geometry() -> PeriodicUswGeometry:
    return PeriodicUswGeometry(
        lattice=5.0 * np.eye(3),
        sites=np.array([[0.8, 0.9, 1.0], [2.5, 2.1, 2.3]]),
        radii=np.array([0.6, 0.7]),
        channels=tuple(RealHarmonic(l, m) for l in range(2) for m in range(-l, l + 1)),
        k=np.array([0.21, 0.12, 0.09]),
        g_cutoff=12.0,
        reference_energy=-1.0,
        lattice_radius=24.0,
    )


def test_periodic_slope_derivative_and_resolvent_difference(geometry) -> None:
    energies = (-0.2, 0.17)
    samples = tuple(geometry.sample(energy) for energy in energies)
    radius = np.repeat(geometry.radii, len(geometry.channels))
    step = 1.0e-6
    for sample in samples:
        plus = geometry.sample(sample.energy + step)
        minus = geometry.sample(sample.energy - step)
        np.testing.assert_allclose(
            sample.slope_derivative,
            (plus.slope - minus.slope) / (2.0 * step),
            rtol=2.0e-6,
            atol=3.0e-7,
        )
        weighted = radius[:, None] * sample.slope
        np.testing.assert_allclose(weighted, weighted.conj().T, atol=2.0e-12)

    # The unknown reference sum cancels: this independently checks the
    # spectral energy difference, its sign, and the Hartree denominator.
    kinetic = 0.5 * np.sum(geometry.wave_vectors**2, axis=1)
    weight = (energies[0] - energies[1]) / (
        geometry.volume * (kinetic - energies[0]) * (kinetic - energies[1])
    )
    expected = geometry.form_factors.conj().T @ (weight[:, None] * geometry.form_factors)
    identity = np.eye(len(samples[0].boundary_inverse))
    actual = np.linalg.solve(samples[0].boundary_inverse, identity) - np.linalg.solve(
        samples[1].boundary_inverse, identity
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)


def test_reference_boundary_projection_and_bloch_covariance(geometry) -> None:
    pytest.importorskip("finufft")
    cosine, theta_weights = np.polynomial.legendre.leggauss(12)
    phi = 2.0 * np.pi * np.arange(25) / 25
    sine = np.sqrt(1.0 - cosine**2)
    directions = np.stack(
        (
            np.repeat(sine, len(phi)) * np.tile(np.cos(phi), len(cosine)),
            np.repeat(sine, len(phi)) * np.tile(np.sin(phi), len(cosine)),
            np.repeat(cosine, len(phi)),
        ),
        axis=1,
    )
    weights = np.repeat(theta_weights, len(phi)) * 2.0 * np.pi / len(phi)
    angular = real_spherical_harmonics(directions, geometry.channels)
    projected = []
    for center, radius in zip(geometry.sites, geometry.radii, strict=True):
        layers = geometry.reference_values(center + radius * directions)
        projected.append(angular.T @ (weights[:, None] * layers))
    # Smooth nonoverlapping-sphere angular projection; the same finite
    # real-space lattice sum enters both sides, with no reciprocal tail.
    np.testing.assert_allclose(
        np.concatenate(projected), geometry.reference_boundary,
        rtol=2.0e-8, atol=2.0e-9,
    )

    points = np.array([[1.7, 0.8, 1.1], [2.8, 3.0, 2.4], [0.4, 2.7, 3.1]])
    sample = geometry.sample(0.17)
    translation = geometry.lattice[0]
    values = sample.evaluate(points)
    translated = sample.evaluate(points + translation)
    # The reciprocal part is exactly Bloch-periodic; this tolerance covers
    # the exponentially small omitted reference images at radius 24 Bohr.
    np.testing.assert_allclose(
        translated, np.exp(1j * (translation @ geometry.k)) * values,
        rtol=2.0e-8, atol=2.0e-9,
    )


def test_n1_constant_potential_has_positive_overlap_and_free_eigenvalue() -> None:
    geometry = PeriodicUswGeometry(
        lattice=4.0 * np.eye(3),
        sites=np.array([[1.0, 1.0, 1.0]]),
        radii=np.array([0.8]),
        channels=(RealHarmonic(0, 0),),
        k=np.array([0.15, 0.11, 0.07]),
        g_cutoff=12.0,
        reference_energy=-1.0,
        lattice_radius=20.0,
    )
    free_energy = 0.5 * float(geometry.k @ geometry.k)
    energies = free_energy + np.array([-0.04, 0.04])
    samples = tuple(geometry.sample(float(energy)) for energy in energies)
    radial = np.asarray(
        [_regular_bessel_with_energy_derivative(0, float(energy), geometry.radii) for energy in energies]
    )
    jets = BoundaryJets(
        potential_radii=geometry.radii,
        values=radial[:, 0],
        radial_derivatives=radial[:, 1],
        energy_derivatives=radial[:, 2],
        energy_radial_derivatives=radial[:, 3],
        inverse_masses=np.full((2, 1), 0.5),
        energy_inverse_masses=np.zeros((2, 1)),
    )
    result = build_nmto(
        build_kink_mesh(
            energies,
            np.stack([sample.slope for sample in samples]),
            np.stack([sample.slope_derivative for sample in samples]),
            jets,
            geometry.radii,
        )
    )
    assert result.order == 1
    assert result.lowdin.overlap_eigenvalues[0] > 0.0
    # One s channel and an N=1 mesh approximate, rather than span exactly,
    # the lowest plane wave; allow 1e-4 Ha for finite-mesh admixture.
    np.testing.assert_allclose(result.lowdin.hamiltonian[0, 0], free_energy, atol=1.0e-4, rtol=0.0)


def _free_shell_evaluator(shell_difference):
    """A free-space s-channel evaluator with an extended potential shell.

    ``shell_difference(r)`` is added to ``u - u0`` on the shell; zero makes
    the shell construction exactly equivalent to the hard-sphere one.
    """

    from pymuffintin.mto.density import ScalarRadialSamples
    from pymuffintin.mto.periodic_density import PeriodicNmtoBasisEvaluator
    from pymuffintin.mto.shell import continuation_jets, exponential_mesh, free_continuation

    geometry = PeriodicUswGeometry(
        lattice=4.0 * np.eye(3),
        sites=np.array([[1.0, 1.0, 1.0]]),
        radii=np.array([0.8]),
        channels=(RealHarmonic(0, 0),),
        k=np.array([0.15, 0.11, 0.07]),
        g_cutoff=12.0,
        reference_energy=-1.0,
        lattice_radius=20.0,
    )
    free_energy = 0.5 * float(geometry.k @ geometry.k)
    energies = free_energy + np.array([-0.04, 0.04])
    samples = tuple(geometry.sample(float(energy)) for energy in energies)
    hard = 0.8
    count = 200
    first = 1.0e-4
    increment = np.log(hard / first) / (count - 1)
    extended = count + 15
    mesh = exponential_mesh(first, increment, extended)
    outer = float(mesh[-1])
    large = np.stack(
        [_regular_bessel_with_energy_derivative(0, float(energy), mesh)[0] for energy in energies]
    )
    jets = np.stack(
        [
            np.array([*(_regular_bessel_with_energy_derivative(0, float(energy), np.array([outer]))[i][0] for i in range(4)), 0.5, 0.0])
            for energy in energies
        ]
    )
    continuation = free_continuation(0, energies, hard, outer, mesh[count - 1 :], jets)
    np.testing.assert_allclose(continuation.values, large[:, count - 1 :], rtol=1e-10, atol=1e-14)
    shell_free = continuation.values - shell_difference(mesh[count - 1 :])[None, :]

    def radial(with_shell: bool) -> ScalarRadialSamples:
        stop = extended if with_shell else count
        return ScalarRadialSamples(
            mesh_radii=mesh[:stop],
            large=large[:, :stop],
            small=np.zeros((2, stop)),
            boundary_values=continuation.boundary_values,
            inverse_mass=np.full((2, stop), 0.5),
            inverse_speed_of_light=0.0,
            hard_radius=hard if with_shell else None,
            shell_start=count - 1 if with_shell else None,
            shell_free_large=shell_free if with_shell else None,
        )

    result = build_nmto(
        build_kink_mesh(
            energies,
            np.stack([sample.slope for sample in samples]),
            np.stack([sample.slope_derivative for sample in samples]),
            continuation_jets([continuation]),
            geometry.radii,
        )
    )

    def evaluator(with_shell: bool) -> PeriodicNmtoBasisEvaluator:
        return PeriodicNmtoBasisEvaluator(
            direct_lattice=geometry.lattice,
            site_fractional=geometry.sites @ np.linalg.inv(geometry.lattice),
            muffin_tin_radii=geometry.radii,
            channels=geometry.channels,
            energies=energies,
            interstitial_energies=energies,
            k_cartesian=geometry.k[None, :],
            k_weights=np.ones(1),
            results=(result,),
            bands=None,
            occupations=None,
            periodic_samples=(samples,),
            radial_samples={(0, 0): radial(with_shell)},
            symmetry=None,
            potential_sphere_radii=np.array([outer]) if with_shell else None,
        )

    return evaluator(False), evaluator(True), geometry, result, hard, outer, mesh, count, energies


def test_shell_augmentation_is_inert_for_a_free_well_and_adds_the_shell_difference() -> None:
    pytest.importorskip("finufft")
    from pymuffintin.mto.electrons import interpolate_nmto_basis

    plain, shelled, geometry, result, hard, outer, mesh, count, energies = _free_shell_evaluator(
        lambda r: np.zeros_like(r)
    )
    center = geometry.sites[0]
    direction = np.array([0.6, 0.0, 0.8])
    points = np.array(
        [
            center + 0.5 * direction,  # inside the hard sphere
            center + 0.9 * direction,  # in the shell
            center + 0.5 * (hard + outer) * direction,  # in the shell
            center + 1.5 * outer * direction,  # outside the potential sphere
            center + 0.9 * direction + geometry.lattice[0],  # shell of a periodic image
        ]
    )
    large_plain, small_plain = plain._basis_values(points, 0)
    large_shell, small_shell, pieces = shelled._basis_values_and_shell_pieces(points, 0)
    np.testing.assert_allclose(large_shell, large_plain, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(small_shell, small_plain, atol=1e-15)
    assert len(pieces) == 1
    site, indices, distances, piece_large, piece_small = pieces[0]
    assert site == 0
    assert sorted(indices.tolist()) == [1, 2, 4]
    np.testing.assert_allclose(sorted(distances), sorted([0.9, 0.5 * (hard + outer), 0.9]))
    radial = shelled.radial_samples[(0, 0)]
    expected_nodes = np.stack(
        [
            radial.large_interpolant(distances)[node]
            / radial.boundary_values[node]
            * (1.0 / np.sqrt(4.0 * np.pi))
            for node in range(2)
        ]
    )[:, :, None]
    # Bloch phase of the containing image relative to the home cell.
    image_translation = np.rint((points[indices] - center) @ np.linalg.inv(geometry.lattice)) @ geometry.lattice
    phase = np.exp(1j * (image_translation @ geometry.k))
    expected = interpolate_nmto_basis(expected_nodes * phase[None, :, None], result.lagrange_matrices)
    np.testing.assert_allclose(piece_large, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(piece_small, 0.0, atol=1e-15)

    bump = lambda r: 0.05 * (r - outer) ** 2
    plain2, shelled2, *_ = _free_shell_evaluator(bump)
    large_plain2, _ = plain2._basis_values(points, 0)
    large_shell2, _, pieces2 = shelled2._basis_values_and_shell_pieces(points, 0)
    difference = large_shell2 - large_plain2
    np.testing.assert_allclose(difference[[0, 3]], 0.0, atol=1e-13)
    radial2 = shelled2.radial_samples[(0, 0)]
    expected_nodes = np.stack(
        [bump(distances) / radial2.boundary_values[node] / np.sqrt(4.0 * np.pi) for node in range(2)]
    )[:, :, None]
    expected = interpolate_nmto_basis(expected_nodes * phase[None, :, None], result.lagrange_matrices)
    np.testing.assert_allclose(difference[indices], expected, rtol=1e-8, atol=1e-12)
