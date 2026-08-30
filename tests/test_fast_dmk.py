from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pymuffintin.coulomb import ContinuousDmk, FastDmk, LeafBox, LeafDensity


def _uniform_densities(
    coefficients: np.ndarray,
) -> tuple[LeafBox, tuple[LeafDensity, ...]]:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    densities = tuple(
        LeafDensity(
            box=LeafBox(
                center=np.array([(-0.25, 0.25)[axis] for axis in bits], dtype=np.float64),
                width=0.5,
            ),
            coefficients=coefficients.copy(),
        )
        for bits in np.ndindex(2, 2, 2)
    )
    return root, densities


def _uniform_constant_densities(level: int) -> tuple[LeafBox, tuple[LeafDensity, ...]]:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    width = 1.0 / 2**level
    densities = tuple(
        LeafDensity(
            box=LeafBox(
                center=-0.5 + width * (np.asarray(cell, dtype=np.float64) + 0.5),
                width=width,
            ),
            coefficients=np.ones((1, 1, 1), dtype=np.float64),
        )
        for cell in np.ndindex(*(2**level,) * 3)
    )
    return root, densities


def _adaptive_constant_densities() -> tuple[LeafBox, tuple[LeafDensity, ...]]:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    densities: list[LeafDensity] = []
    for bits in np.ndindex(2, 2, 2):
        center = np.array([(-0.25, 0.25)[axis] for axis in bits], dtype=np.float64)
        if bits == (1, 1, 1):
            continue
        densities.append(
            LeafDensity(
                LeafBox(center=center, width=0.5),
                np.ones((1, 1, 1), dtype=np.float64),
            )
        )
    for bits in np.ndindex(2, 2, 2):
        densities.append(
            LeafDensity(
                LeafBox(
                    center=np.array([(0.125, 0.375)[axis] for axis in bits], dtype=np.float64),
                    width=0.25,
                ),
                np.ones((1, 1, 1), dtype=np.float64),
            )
        )
    return root, tuple(densities)


@pytest.mark.parametrize("complex_density", [False, True])
def test_fast_dmk_matches_continuous_for_low_order_inside_and_outside(
    complex_density: bool,
) -> None:
    dtype = np.complex128 if complex_density else np.float64
    coefficients = np.zeros((2, 2, 1), dtype=dtype)
    coefficients[0, 0, 0] = 0.8 + (0.25j if complex_density else 0.0)
    coefficients[1, 0, 0] = -0.17 + (0.11j if complex_density else 0.0)
    coefficients[0, 1, 0] = 0.09 - (0.08j if complex_density else 0.0)
    root, densities = _uniform_densities(coefficients)
    settings = dict(
        root=root,
        densities=densities,
        tolerance=1.0e-8,
        gaussian_order=14,
        source_quadrature_order=14,
        local_quadrature_order=9,
    )
    reference = ContinuousDmk(**settings)
    solver = FastDmk(**settings, interpolation_order=8)
    targets = np.array(
        [
            [-0.31, -0.27, -0.19],
            [0.0, 0.0, 0.0],
            [0.29, 0.21, 0.33],
            [1.17, -0.14, 0.08],
            [-1.31, 0.22, -0.17],
        ],
        dtype=np.float64,
    )

    potential, work = solver.apply_with_work(targets)

    assert potential.dtype == np.dtype(dtype)
    np.testing.assert_allclose(potential, reference.apply(targets), rtol=2.0e-6, atol=2.0e-8)
    np.testing.assert_array_equal(solver.apply(targets), potential)
    assert work.leaf_source_transforms == len(densities)
    assert work.target_interpolations == targets.shape[0]
    assert work.local_remainder_evaluations <= 27 * targets.shape[0]


def test_fast_dmk_matches_continuous_on_a_two_to_one_partition() -> None:
    root, densities = _adaptive_constant_densities()
    settings = dict(
        root=root,
        densities=densities,
        tolerance=1.0e-8,
        gaussian_order=14,
        source_quadrature_order=14,
        local_quadrature_order=9,
    )
    solver = FastDmk(**settings, interpolation_order=8)
    targets = np.array(
        [[-0.36, 0.12, 0.04], [0.18, 0.29, 0.31], [1.2, -0.1, 0.2]],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        solver.apply(targets),
        ContinuousDmk(**settings).apply(targets),
        rtol=3.0e-6,
        atol=3.0e-8,
    )


def test_fast_dmk_far_work_depends_on_target_cells_not_target_count() -> None:
    root, densities = _uniform_densities(np.ones((1, 1, 1), dtype=np.float64))
    solver = FastDmk(
        root=root,
        densities=densities,
        tolerance=1.0e-7,
        gaussian_order=10,
        source_quadrature_order=10,
        local_quadrature_order=8,
        interpolation_order=6,
    )
    one = np.array([[-0.31, -0.29, -0.27]], dtype=np.float64)
    many = np.repeat(one, 24, axis=0)

    _, one_work = solver.apply_with_work(one)
    _, many_work = solver.apply_with_work(many)

    assert many_work.same_level_translations == one_work.same_level_translations
    assert many_work.downward_interpolations == one_work.downward_interpolations
    assert many_work.target_interpolations == 24 * one_work.target_interpolations
    assert many_work.local_remainder_evaluations == 24 * one_work.local_remainder_evaluations
    with pytest.raises(FrozenInstanceError):
        many_work.same_level_translations = 0  # type: ignore[misc]


def test_fast_dmk_work_scales_with_hierarchy_not_leaf_target_product() -> None:
    work_by_level = []
    for level in (2, 3):
        root, densities = _uniform_constant_densities(level)
        solver = FastDmk(
            root=root,
            densities=densities,
            tolerance=1.0e-1,
            gaussian_order=1,
            source_quadrature_order=1,
            local_quadrature_order=1,
            interpolation_order=1,
        )
        target_count = len(densities)
        targets = np.column_stack(
            (
                10.0 + 2.0 * np.arange(target_count, dtype=np.float64),
                np.zeros(target_count, dtype=np.float64),
                np.zeros(target_count, dtype=np.float64),
            )
        )
        _, work = solver.apply_with_work(targets)
        assert work.leaf_source_transforms == target_count
        assert work.target_interpolations == target_count
        assert work.local_remainder_evaluations == 0
        work_by_level.append(work)

    small, large = work_by_level
    assert large.total > small.total
    assert large.total <= 12 * small.total
