import numpy as np
import pytest

from pymuffintin.coulomb import (
    ContinuousDmk,
    CoulombKernelSplit,
    LeafBox,
    LeafDensity,
    PeriodicDmk,
)


def _constant_cube_potential(target: np.ndarray) -> float:
    """Independent rectangular-prism antiderivative for unit density."""
    boundaries = [(-0.5 - target[axis], 0.5 - target[axis]) for axis in range(3)]
    result = 0.0
    for i, x in enumerate(boundaries[0]):
        for j, y in enumerate(boundaries[1]):
            for k, z in enumerate(boundaries[2]):
                radius = np.sqrt(x * x + y * y + z * z)
                primitive = (
                    x * y * np.log(z + radius)
                    + y * z * np.log(x + radius)
                    + z * x * np.log(y + radius)
                    - 0.5 * x * x * np.arctan((y * z) / (x * radius))
                    - 0.5 * y * y * np.arctan((z * x) / (y * radius))
                    - 0.5 * z * z * np.arctan((x * y) / (z * radius))
                )
                result += (-1.0 if (i + j + k) % 2 == 0 else 1.0) * primitive
    return float(result)


def test_coulomb_split_reconstructs_inverse_distance() -> None:
    split = CoulombKernelSplit(
        root_width=1.0,
        max_level=2,
        tolerance=1.0e-10,
        gaussian_order=18,
    )
    distances = np.array([0.03, 0.2, 0.7, 1.5, 1.0e3], dtype=np.float64)

    np.testing.assert_allclose(
        split.reconstruct(distances),
        1.0 / distances,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_continuous_dmk_constant_leaf_matches_direct_cubature() -> None:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    density = LeafDensity(
        box=root,
        coefficients=np.ones((1, 1, 1), dtype=np.float64),
    )
    solver = ContinuousDmk(
        root=root,
        densities=(density,),
        tolerance=1.0e-10,
        gaussian_order=18,
        source_quadrature_order=18,
        local_quadrature_order=10,
    )
    targets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.13, -0.07, 0.21],
            [1.25, 0.1, -0.2],
            [-1.25, -0.1, 0.2],
        ],
        dtype=np.float64,
    )

    potential = solver.apply(targets)
    expected = np.array([_constant_cube_potential(target) for target in targets])

    assert np.all(np.isfinite(potential))
    np.testing.assert_allclose(potential, expected, rtol=2.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(potential[2], potential[3], rtol=0.0, atol=1.0e-12)


def test_adaptive_tree_requires_a_complete_leaf_partition() -> None:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    incomplete = LeafDensity(
        box=LeafBox(
            center=np.array([-0.25, -0.25, -0.25], dtype=np.float64),
            width=0.5,
        ),
        coefficients=np.zeros((1, 1, 1), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="complete partition"):
        ContinuousDmk(root=root, densities=(incomplete,))


def test_continuous_dmk_balanced_multilevel_partition_matches_full_cube() -> None:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    densities = []
    refined_center = np.array([0.25, 0.25, 0.25], dtype=np.float64)
    for signs in np.ndindex(2, 2, 2):
        center = np.array([(-0.25, 0.25)[index] for index in signs], dtype=np.float64)
        if np.array_equal(center, refined_center):
            continue
        densities.append(
            LeafDensity(
                LeafBox(center=center, width=0.5),
                np.ones((1, 1, 1), dtype=np.float64),
            )
        )
    for signs in np.ndindex(2, 2, 2):
        center = np.array([(0.125, 0.375)[index] for index in signs], dtype=np.float64)
        densities.append(
            LeafDensity(
                LeafBox(center=center, width=0.25),
                np.ones((1, 1, 1), dtype=np.float64),
            )
        )
    solver = ContinuousDmk(
        root=root,
        densities=tuple(densities),
        tolerance=1.0e-8,
        gaussian_order=14,
        source_quadrature_order=14,
        local_quadrature_order=10,
    )
    targets = np.array([[0.0, 0.0, 0.0], [0.31, 0.27, 0.19]], dtype=np.float64)

    potential = solver.apply(targets)
    expected = np.array([_constant_cube_potential(target) for target in targets])

    np.testing.assert_allclose(potential, expected, rtol=5.0e-7, atol=2.0e-9)


def test_dmk_rejects_underresolved_polynomial_and_nonneutral_periodic_density() -> None:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    linear = np.zeros((2, 1, 1), dtype=np.complex128)
    linear[1, 0, 0] = 1j
    with pytest.raises(ValueError, match="source_quadrature_order"):
        ContinuousDmk(
            root=root,
            densities=(LeafDensity(root, linear),),
            source_quadrature_order=1,
        )

    net_charge = np.array([[[1.0e-12]]], dtype=np.float64)
    with pytest.raises(ValueError, match="zero total charge"):
        PeriodicDmk(root=root, densities=(LeafDensity(root, net_charge),), tolerance=1.0e-8)


def test_periodic_dmk_matches_sawtooth_density_fourier_series() -> None:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    coefficients = np.zeros((2, 1, 1), dtype=np.float64)
    coefficients[1, 0, 0] = 1.0
    density = LeafDensity(box=root, coefficients=coefficients)
    solver = PeriodicDmk(
        root=root,
        densities=(density,),
        tolerance=1.0e-8,
        source_quadrature_order=16,
        local_quadrature_order=10,
    )
    targets = np.array(
        [
            [0.13, 0.11, -0.07],
            [0.13, -0.31, 0.22],
            [-0.27, 0.04, 0.19],
            [1.13, 0.11, -0.07],
            [0.5 - 1.0e-7, -0.16, 0.23],
            [-0.5 + 1.0e-7, 0.28, -0.09],
        ],
        dtype=np.float64,
    )

    potential = solver.apply(targets)
    modes = np.arange(1, 20_001, dtype=np.float64)
    expected = np.array(
        [
            -2.0
            / np.pi**2
            * np.sum(
                np.power(-1.0, modes)
                * np.sin(2.0 * np.pi * modes * target[0])
                / modes**3
            )
            for target in targets
        ]
    )

    np.testing.assert_allclose(potential.real, expected, rtol=2.0e-6, atol=2.0e-8)
    np.testing.assert_allclose(potential.imag, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(potential[0], potential[1], rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(potential[0], potential[3], rtol=0.0, atol=2.0e-8)
