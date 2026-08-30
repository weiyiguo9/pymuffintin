from dataclasses import FrozenInstanceError
import builtins

import numpy as np
import pytest

from pymuffintin.coulomb import FastPeriodicDmk, LeafBox, LeafDensity, PeriodicDmk
from pymuffintin.tensor import contract


def _linear_density(value: complex | float) -> tuple[LeafBox, tuple[LeafDensity, ...]]:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    dtype = np.complex128 if isinstance(value, complex) else np.float64
    coefficients = np.zeros((2, 1, 1), dtype=dtype)
    coefficients[1, 0, 0] = value
    return root, (LeafDensity(root, coefficients),)


def _checkerboard(level: int) -> tuple[LeafBox, tuple[LeafDensity, ...]]:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    count = 2**level
    width = 1.0 / count
    densities = []
    for index in np.ndindex(count, count, count):
        center = root.lower + width * (np.asarray(index, dtype=np.float64) + 0.5)
        sign = -1.0 if sum(index) % 2 else 1.0
        densities.append(
            LeafDensity(
                LeafBox(center=np.asarray(center, dtype=np.float64), width=float(width)),
                np.array([[[sign]]], dtype=np.float64),
            )
        )
    return root, tuple(densities)


@pytest.mark.parametrize("value", [1.0, 0.7 - 0.35j])
def test_fast_periodic_dmk_matches_dense_oracle_and_wraps(value: complex | float) -> None:
    root, densities = _linear_density(value)
    settings = dict(
        root=root,
        densities=densities,
        tolerance=1.0e-6,
        source_quadrature_order=24,
        local_quadrature_order=14,
    )
    fast = FastPeriodicDmk(**settings, gaussian_order=20, interpolation_order=24)
    dense = PeriodicDmk(**settings)
    targets = np.array(
        [
            [0.13, 0.11, -0.07],
            [-0.37, 0.22, 0.31],
            [0.5 - 1.0e-7, -0.18, 0.24],
            [1.13, -0.89, 0.93],
        ],
        dtype=np.float64,
    )

    actual = fast.apply(targets)

    np.testing.assert_allclose(actual, dense.apply(targets), rtol=2.0e-6, atol=2.0e-8)
    np.testing.assert_allclose(actual[0], actual[3], rtol=0.0, atol=2.0e-8)


def test_fast_periodic_reciprocal_nufft_matches_dense_sum() -> None:
    root, densities = _linear_density(1.0 + 0.2j)
    solver = FastPeriodicDmk(
        root=root,
        densities=densities,
        tolerance=1.0e-7,
        gaussian_order=12,
        source_quadrature_order=12,
        local_quadrature_order=9,
        interpolation_order=8,
    )
    targets = np.array(
        [[0.17, -0.21, 0.33], [-1.14, 0.72, 1.48], [0.49, -0.41, -0.29]],
        dtype=np.float64,
    )
    wrapped = root.lower + np.mod(targets - root.lower, root.width)
    phase_x = np.exp(1j * wrapped[:, 0, None] * solver.wave_numbers[None, :])
    phase_y = np.exp(1j * wrapped[:, 1, None] * solver.wave_numbers[None, :])
    phase_z = np.exp(1j * wrapped[:, 2, None] * solver.wave_numbers[None, :])
    expected = contract(
        "ti,tj,tk,ijk->t",
        phase_x,
        phase_y,
        phase_z,
        solver.reciprocal_coefficients,
    )

    np.testing.assert_allclose(
        solver._reciprocal_space(wrapped), expected, rtol=5.0e-8, atol=5.0e-10
    )


def test_fast_periodic_dependency_error_and_neutrality(monkeypatch: pytest.MonkeyPatch) -> None:
    root, densities = _linear_density(1.0)
    solver = FastPeriodicDmk(
        root=root,
        densities=densities,
        tolerance=1.0e-4,
        gaussian_order=8,
        source_quadrature_order=6,
        local_quadrature_order=6,
        interpolation_order=6,
    )
    original_import = builtins.__import__

    def missing_finufft(name: str, *args: object, **kwargs: object) -> object:
        if name == "finufft":
            error = ModuleNotFoundError("No module named 'finufft'")
            error.name = "finufft"
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_finufft)
    with pytest.raises(
        ModuleNotFoundError,
        match=r"FastPeriodicDmk requires optional dependency 'finufft'; install pymuffintin\[nufft\]\.",
    ):
        solver.apply(np.zeros((1, 3), dtype=np.float64))

    charged = LeafDensity(root, np.ones((1, 1, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="zero total charge"):
        FastPeriodicDmk(root=root, densities=(charged,))


def test_fast_periodic_work_has_fixed_precision_linear_source_scaling() -> None:
    root_one, densities_one = _checkerboard(1)
    root_two, densities_two = _checkerboard(2)
    settings = dict(
        tolerance=1.0e-4,
        gaussian_order=6,
        source_quadrature_order=6,
        local_quadrature_order=5,
        interpolation_order=4,
    )
    targets_one = np.asarray([density.box.center for density in densities_one], dtype=np.float64)
    targets_two = np.asarray([density.box.center for density in densities_two], dtype=np.float64)
    _, work_one = FastPeriodicDmk(root=root_one, densities=densities_one, **settings).apply_with_work(
        targets_one
    )
    _, work_two = FastPeriodicDmk(root=root_two, densities=densities_two, **settings).apply_with_work(
        targets_two
    )

    assert work_one.reciprocal_modes == work_two.reciprocal_modes
    assert work_one.nufft_targets == len(densities_one)
    assert work_two.nufft_targets == len(densities_two)
    assert work_one.reciprocal_source_transforms == len(densities_one)
    assert work_two.reciprocal_source_transforms == len(densities_two)
    assert work_two.total / work_one.total <= 10.0
    with pytest.raises(FrozenInstanceError):
        work_two.real_local_pairs = 0  # type: ignore[misc]
