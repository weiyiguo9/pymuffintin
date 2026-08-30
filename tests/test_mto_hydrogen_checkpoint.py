from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

from pymuffintin.mto import (
    BoundaryJets,
    RealHarmonic,
    build_kink_mesh,
    build_nmto,
    cluster_bloch_sum,
    fit_omt,
    usw_matrices_with_energy_derivative,
)


LIBMUFFINTIN = Path(__file__).resolve().parents[2] / "libmuffintin"
FIXTURES = LIBMUFFINTIN / "python" / "tests" / "fixtures"


def _checkpoint_physics():
    sys.path.insert(0, str(LIBMUFFINTIN / "python"))
    mt = pytest.importorskip("libmuffintin")
    checkpoint = mt.load_checkpoint(FIXTURES / "hydrogen_checkpoint.toml")
    return mt.CheckpointPhysics(checkpoint)


def test_frozen_hydrogen_radial_omt_overlap_fractions_and_rms_trend() -> None:
    physics = _checkpoint_physics()
    frozen = physics.export_frozen_potential()
    mesh = frozen["mt_mesh_radii"]
    radial_potential = frozen["mt_components"][0].real / np.sqrt(4.0 * np.pi)
    np.testing.assert_allclose(radial_potential[[0, -1]], [-10000.0, -1.0])

    lattice = 8.0 * np.eye(3)
    centers = np.array([[2.0, 4.0, 4.0]])
    axis = (np.arange(16) + 0.5) * 0.5
    points = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    displacement = points - centers[0]
    displacement -= np.rint(displacement / 8.0) * 8.0
    distances = np.linalg.norm(displacement, axis=1)
    values = np.zeros(len(points))
    inside = distances <= mesh[-1]
    values[inside] = np.interp(distances[inside], mesh, radial_potential)

    potential_radii = (3.2, 4.0, 4.8)
    fits = tuple(
        fit_omt(
            points,
            values,
            np.ones(len(points)),
            lattice,
            centers,
            np.array([radius]),
            (np.linspace(0.0, radius, 5),),
        )
        for radius in potential_radii
    )
    np.testing.assert_allclose(
        [fit.diagnostics.maximum_overlap_fraction for fit in fits],
        [-0.2, 0.0, 0.2],
    )
    weighted_rms = np.array([fit.diagnostics.weighted_rms for fit in fits])
    assert np.all(np.diff(weighted_rms) < 0.0)


def test_frozen_hydrogen_n2_s_nmto_agrees_with_gamma_lapw_within_2_millihartree() -> None:
    physics = _checkpoint_physics()
    energies = np.array([-0.18, -0.14, -0.10])
    radial = physics.sample_frozen_scalar_radials("H-1", 0, energies)

    centers = 8.0 * np.array(list(product(range(-2, 3), repeat=3)), dtype=float)
    radii = np.ones(len(centers))
    slopes = []
    slope_derivatives = []
    for energy in energies:
        _, slope, slope_derivative = usw_matrices_with_energy_derivative(
            energy, centers, radii, (RealHarmonic(0, 0),)
        )
        slopes.append(cluster_bloch_sum(slope, centers, 1))
        slope_derivatives.append(cluster_bloch_sum(slope_derivative, centers, 1))

    boundary = radial["boundary_radial"]
    boundary_energy = radial["energy_derivative_boundary_radial"]
    jets = BoundaryJets(
        potential_radii=np.ones(1),
        values=boundary[:, 0, None],
        radial_derivatives=boundary[:, 1, None],
        energy_derivatives=boundary_energy[:, 0, None],
        energy_radial_derivatives=boundary_energy[:, 1, None],
    )
    kinks = build_kink_mesh(
        energies,
        np.stack(slopes),
        np.stack(slope_derivatives),
        jets,
        np.ones(1),
    )
    nmto_energy = float(build_nmto(kinks).lowdin.hamiltonian[0, 0].real)

    product_input = physics.scalar_product_input(
        FIXTURES / "hydrogen_nmto_input.toml", q=[0.0, 0.0, 0.0]
    )
    lapw_energy = float(product_input.export_orbitals()["channels"][0]["energies"][0, 0])
    np.testing.assert_allclose(nmto_energy, -0.13703203, atol=2.0e-7)
    np.testing.assert_allclose(lapw_energy, -0.13573124, atol=2.0e-7)
    assert abs(nmto_energy - lapw_energy) < 2.0e-3
