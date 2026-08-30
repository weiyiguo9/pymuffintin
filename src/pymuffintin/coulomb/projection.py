from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pymuffintin.contracts import require_array
from pymuffintin.tensor import contract

from .density import DensityArray, LeafDensity, _legendre_basis, _quadrature_rule
from .tree import LeafBox


FloatArray = NDArray[np.float64]
ScalarDensity = Callable[[FloatArray], DensityArray]


def project_density(
    root: LeafBox,
    density: ScalarDensity,
    *,
    level: int,
    degree_count: int,
    quadrature_order: int,
    remove_mean: bool = False,
) -> tuple[tuple[LeafDensity, ...], float | complex]:
    """Project a scalar density onto uniform dyadic Legendre leaves.

    ``density`` receives a ``float64`` array with shape ``(n, 3)`` and must
    return a ``float64`` or ``complex128`` array with shape ``(n,)``.  Leaves
    are ordered by their Cartesian dyadic indices, with the last index varying
    fastest.  The returned scalar is the volume-weighted mean removed from the
    constant coefficients, or zero when ``remove_mean`` is false.
    """
    if not isinstance(root, LeafBox):
        raise TypeError("root must be a LeafBox")
    if not callable(density):
        raise TypeError("density must be callable")
    if type(level) is not int or level < 0:
        raise ValueError("level must be a non-negative int")
    if type(degree_count) is not int or degree_count <= 0:
        raise ValueError("degree_count must be a positive int")
    if type(quadrature_order) is not int or quadrature_order <= 0:
        raise ValueError("quadrature_order must be a positive int")
    if type(remove_mean) is not bool:
        raise TypeError("remove_mean must be a bool")

    nodes, weights = _quadrature_rule(quadrature_order)
    basis = _legendre_basis(quadrature_order, degree_count)
    degrees = np.arange(degree_count, dtype=np.float64)
    projection = np.asarray(
        basis * weights[:, np.newaxis] * (degrees + 0.5)[np.newaxis, :],
        dtype=np.float64,
    )

    cell_count = 2**level
    width = root.width / cell_count
    coefficients: list[DensityArray] = []
    boxes: list[LeafBox] = []
    value_dtype: np.dtype[np.float64] | np.dtype[np.complex128] | None = None
    for index in np.ndindex(cell_count, cell_count, cell_count):
        center = root.lower + width * (np.asarray(index, dtype=np.float64) + 0.5)
        box = LeafBox(center=np.asarray(center, dtype=np.float64), width=float(width))
        axes = [box.center[axis] + 0.5 * width * nodes for axis in range(3)]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        points = np.asarray(
            np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1),
            dtype=np.float64,
        )
        values = density(points)
        if not isinstance(values, np.ndarray):
            raise TypeError("density values must be a numpy.ndarray")
        if values.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
            raise TypeError("density values must have dtype float64 or complex128")
        values = require_array(
            "density values", values, values.dtype, (quadrature_order**3,)
        )
        if value_dtype is None:
            value_dtype = values.dtype
        elif values.dtype != value_dtype:
            raise TypeError("density values must have a consistent dtype across leaves")

        value_tensor = values.reshape(
            quadrature_order, quadrature_order, quadrature_order
        )
        leaf_coefficients = np.asarray(
            contract(
                "ia,jb,kc,ijk->abc",
                projection,
                projection,
                projection,
                value_tensor,
            ),
            dtype=values.dtype,
        )
        boxes.append(box)
        coefficients.append(leaf_coefficients)

    assert value_dtype is not None
    removed_mean: float | complex = 0j if value_dtype == np.dtype(np.complex128) else 0.0
    if remove_mean:
        volumes = np.asarray([box.width**3 for box in boxes], dtype=np.float64)
        constant_coefficients = np.asarray(
            [items[0, 0, 0] for items in coefficients], dtype=value_dtype
        )
        removed_mean = (
            np.sum(volumes * constant_coefficients) / np.sum(volumes)
        ).item()
        for items in coefficients:
            items[0, 0, 0] -= removed_mean

    return (
        tuple(
            LeafDensity(box=box, coefficients=items)
            for box, items in zip(boxes, coefficients, strict=True)
        ),
        removed_mean,
    )
