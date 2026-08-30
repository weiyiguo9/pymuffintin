from types import SimpleNamespace

import numpy as np

from pymuffintin.mto.density import (
    ScalarRadialSamples,
    assemble_nmto_regional_density,
)
from pymuffintin.mto.usw import (
    RealHarmonic,
    evaluate_bloch_usw,
    evaluate_usw,
)


def test_single_cell_bloch_usw_matches_cluster_evaluation_at_gamma() -> None:
    points = np.array([[1.4, 0.2, 0.1], [1.8, -0.3, 0.4]])
    centers = np.zeros((1, 3))
    radii = np.ones(1)
    channels = (RealHarmonic(0, 0),)

    expected = evaluate_usw(-0.1, points, centers, radii, channels)
    actual = evaluate_bloch_usw(
        -0.1,
        points,
        centers,
        radii,
        channels,
        translations=np.zeros((1, 3)),
        site_count=1,
        k_cartesian=np.zeros(3),
    )

    np.testing.assert_allclose(actual, expected)


def test_constant_density_projects_to_g0_and_muffin_tin_monopole() -> None:
    density_value = 0.25
    radial = ScalarRadialSamples(
        mesh_radii=np.array([0.2, 0.8]),
        large=np.ones((2, 2)),
        small=np.zeros((2, 2)),
        boundary_values=np.ones(2),
    )

    class Evaluator:
        direct_lattice = 4.0 * np.eye(3)
        site_fractional = np.array([[0.5, 0.5, 0.5]])
        muffin_tin_radii = np.array([0.8])
        radial_samples = {(0, 0): radial}

        @staticmethod
        def _nearest_sites(points):
            return np.zeros_like(points), np.full(len(points), -1, dtype=np.int64)

        @staticmethod
        def density(points):
            return np.full(len(points), density_value)

    captured = {}

    def regional_density(*args):
        captured["args"] = args
        return SimpleNamespace()

    native = SimpleNamespace(RegionalDensity=regional_density)
    g_vectors = np.array(
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int64
    )

    assemble_nmto_regional_density(
        native,
        structure=object(),
        field_layout=object(),
        g_vectors=g_vectors,
        density_l_max=0,
        evaluator=Evaluator(),
    )

    interstitial = captured["args"][3][0]
    np.testing.assert_allclose(interstitial, [0.0, density_value, 0.0], atol=1.0e-14)
    labels = captured["args"][4]
    offsets = captured["args"][5]
    muffin_tin = captured["args"][6][0]
    np.testing.assert_array_equal(labels, [[0, 0, 0]])
    np.testing.assert_array_equal(offsets, [0, 2])
    np.testing.assert_allclose(
        muffin_tin,
        np.full(2, density_value * np.sqrt(4.0 * np.pi)),
        atol=1.0e-14,
    )
