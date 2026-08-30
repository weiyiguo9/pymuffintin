import numpy as np

from pymuffintin.coulomb import FastPeriodicDmk, LeafBox, project_density


def test_project_density_recovers_tensor_polynomial_on_ordered_uniform_leaves() -> None:
    root = LeafBox(center=np.array([0.25, -0.5, 0.75], dtype=np.float64), width=2.0)

    def polynomial(points: np.ndarray) -> np.ndarray:
        x, y, z = points.T
        return np.asarray(1.5 + 0.2 * x - 0.3 * y * z + 0.4 * x * x, dtype=np.float64)

    densities, removed_mean = project_density(
        root,
        polynomial,
        level=1,
        degree_count=3,
        quadrature_order=4,
    )

    assert removed_mean == 0.0
    assert len(densities) == 8
    leaf_width = root.width / 2
    expected_centers = tuple(
        root.lower + leaf_width * (np.asarray(index, dtype=np.float64) + 0.5)
        for index in np.ndindex(2, 2, 2)
    )
    for density, expected_center in zip(densities, expected_centers, strict=True):
        np.testing.assert_array_equal(density.box.center, expected_center)
        assert density.coefficients.dtype == np.dtype(np.float64)
        points = np.array(
            [density.box.center, density.box.lower + 0.37 * density.box.width],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            density.evaluate(points), polynomial(points), rtol=0.0, atol=1.0e-14
        )


def test_project_density_removes_complex_volume_weighted_mean() -> None:
    root = LeafBox(center=np.array([0.4, -0.2, 0.3], dtype=np.float64), width=1.5)

    def affine_density(points: np.ndarray) -> np.ndarray:
        return np.asarray(
            2.0 + 0.7j + (0.5 - 0.25j) * points[:, 0] - 0.3j * points[:, 2],
            dtype=np.complex128,
        )

    densities, removed_mean = project_density(
        root,
        affine_density,
        level=1,
        degree_count=2,
        quadrature_order=3,
        remove_mean=True,
    )

    expected_mean = affine_density(root.center[np.newaxis, :])[0]
    np.testing.assert_allclose(removed_mean, expected_mean, rtol=0.0, atol=5.0e-16)
    charges = np.asarray(
        [density.box.width**3 * density.coefficients[0, 0, 0] for density in densities],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(np.sum(charges), 0.0, rtol=0.0, atol=2.0e-15)
    assert all(density.coefficients.dtype == np.dtype(np.complex128) for density in densities)
    FastPeriodicDmk(root=root, densities=densities)
