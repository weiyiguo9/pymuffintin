"""Correctness-first continuous-source DMK-lite for free-space Coulomb.

The solver keeps the DMK telescoping kernel split, but evaluates every smooth
Gaussian band directly with separable leaf-polynomial transforms.  It does not
yet implement the short plane-wave translations or the upward/downward passes
needed for an asymptotically fast DMK implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pymuffintin.contracts import require_array

from .density import DensityArray, LeafDensity, _quadrature_rule
from .kernels import CoulombKernelSplit
from .tree import AdaptiveTree, LeafBox


FloatArray = NDArray[np.float64]
RadialKernel = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class ContinuousDmk:
    """Apply ``1/r`` to tensor-product Legendre densities on dyadic leaves.

    Smooth coarse and correction bands are continuous Gaussian convolutions;
    the singular local remainder is integrated with a Duffy transform when a
    target lies in the source leaf.
    """

    root: LeafBox
    densities: tuple[LeafDensity, ...]
    tolerance: float = 1.0e-10
    gaussian_order: int = 16
    source_quadrature_order: int = 16
    local_quadrature_order: int = 10
    tree: AdaptiveTree = field(init=False)
    kernel_splits: tuple[CoulombKernelSplit, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.root, LeafBox):
            raise TypeError("root must be a LeafBox")
        if not isinstance(self.densities, tuple) or not self.densities:
            raise ValueError("densities must be a nonempty tuple of LeafDensity objects")
        if any(not isinstance(density, LeafDensity) for density in self.densities):
            raise TypeError("densities must be a tuple of LeafDensity objects")
        if not np.isfinite(self.tolerance) or not 0.0 < self.tolerance < 1.0:
            raise ValueError("tolerance must lie strictly between zero and one")
        for name, value in (
            ("gaussian_order", self.gaussian_order),
            ("source_quadrature_order", self.source_quadrature_order),
            ("local_quadrature_order", self.local_quadrature_order),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        polynomial_order = max(max(density.coefficients.shape) for density in self.densities)
        if self.source_quadrature_order < polynomial_order:
            raise ValueError("source_quadrature_order must cover every Legendre coefficient")
        if self.local_quadrature_order < polynomial_order:
            raise ValueError("local_quadrature_order must cover every Legendre coefficient")

        tree = AdaptiveTree(self.root, tuple(density.box for density in self.densities))
        splits = tuple(
            CoulombKernelSplit(
                root_width=self.root.width,
                max_level=level,
                tolerance=self.tolerance,
                gaussian_order=self.gaussian_order,
            )
            for level in tree.levels
        )
        object.__setattr__(self, "tree", tree)
        object.__setattr__(self, "kernel_splits", splits)

    def apply(self, targets: FloatArray) -> DensityArray:
        """Return the free-space Coulomb potential at ``targets``."""
        targets = require_array("targets", targets, np.float64, (None, 3))
        dtype = np.result_type(*(density.coefficients.dtype for density in self.densities))
        potential = np.zeros(targets.shape[0], dtype=dtype)

        for density, split in zip(self.densities, self.kernel_splits, strict=True):
            potential += density.radial_potential(
                targets,
                split.coarse_kernel,
                self.source_quadrature_order,
            )
            for band in split.correction_bands:
                selected = _distances_to_box(targets, density.box) <= band.cutoff_radius
                if not np.any(selected):
                    continue
                band_targets = targets[selected]
                for inverse_length, weight in zip(band.nodes, band.weights, strict=True):
                    potential[selected] += weight * density.gaussian_potential(
                        band_targets,
                        float(inverse_length),
                        self.source_quadrature_order,
                    )

        for density, split in zip(self.densities, self.kernel_splits, strict=True):
            for target_index, target in enumerate(targets):
                if density.box.distance_to(target) <= split.local_cutoff_radius:
                    potential[target_index] += _local_remainder_integral(
                        density,
                        target,
                        split.local_kernel,
                        self.local_quadrature_order,
                    )
        return potential


def _distances_to_box(targets: FloatArray, box: LeafBox) -> FloatArray:
    displacement = np.maximum(
        np.maximum(box.lower[None, :] - targets, targets - box.upper[None, :]),
        0.0,
    )
    return np.linalg.norm(displacement, axis=1)


def _local_remainder_integral(
    density: LeafDensity,
    target: FloatArray,
    kernel: RadialKernel,
    order: int,
) -> np.generic:
    box = density.box
    if box.distance_to(target) <= box.width:
        return _duffy_local_integral(density, target, kernel, order)
    return _smooth_local_integral(density, target, kernel, order)


def _smooth_local_integral(
    density: LeafDensity,
    target: FloatArray,
    kernel: RadialKernel,
    order: int,
) -> np.generic:
    nodes, weights = _quadrature_rule(order)
    half_width = 0.5 * density.box.width
    axes = [density.box.center[axis] + half_width * nodes for axis in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    wx, wy, wz = np.meshgrid(weights, weights, weights, indexing="ij")
    volume_weights = (half_width**3) * (wx * wy * wz).ravel()
    distances = np.linalg.norm(points - target[None, :], axis=1)
    return np.sum(volume_weights * kernel(distances) * density.evaluate(points))


def _duffy_local_integral(
    density: LeafDensity,
    target: FloatArray,
    kernel: RadialKernel,
    order: int,
) -> np.generic:
    """Integrate a singular leaf by splitting at the target into Duffy charts."""
    nodes, weights = _quadrature_rule(order)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    u, v, w = np.meshgrid(unit_nodes, unit_nodes, unit_nodes, indexing="ij")
    wu, wv, ww = np.meshgrid(unit_weights, unit_weights, unit_weights, indexing="ij")
    u = u.ravel()
    v = v.ravel()
    w = w.ravel()
    tensor_weights = (wu * wv * ww).ravel()

    anchor = np.clip(target, density.box.lower, density.box.upper)
    negative_lengths = anchor - density.box.lower
    positive_lengths = density.box.upper - anchor
    result = np.zeros((), dtype=density.coefficients.dtype)

    for positive in product((False, True), repeat=3):
        lengths = np.array(
            [positive_lengths[axis] if positive[axis] else negative_lengths[axis] for axis in range(3)]
        )
        if np.any(lengths == 0.0):
            continue
        signs = np.array([1.0 if direction else -1.0 for direction in positive])
        chart_volume = float(np.prod(lengths))

        for dominant in range(3):
            remaining = [axis for axis in range(3) if axis != dominant]
            normalized = np.empty((u.size, 3), dtype=np.float64)
            normalized[:, dominant] = u
            normalized[:, remaining[0]] = u * v
            normalized[:, remaining[1]] = u * w
            points = anchor[None, :] + normalized * (signs * lengths)[None, :]
            distances = np.linalg.norm(points - target[None, :], axis=1)
            jacobian = chart_volume * np.square(u)
            result += np.sum(
                tensor_weights
                * jacobian
                * kernel(distances)
                * density.evaluate(points)
            )
    return result[()]
