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
