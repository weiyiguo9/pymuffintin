from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pymuffintin.contracts import require_array
from pymuffintin.coulomb.tree import LeafBox
from pymuffintin.tensor import contract


FloatArray = NDArray[np.float64]
DensityArray = NDArray[np.float64] | NDArray[np.complex128]
RadialKernel = Callable[[FloatArray], FloatArray]


@lru_cache(maxsize=None)
def _quadrature_rule(order: int) -> tuple[FloatArray, FloatArray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


@lru_cache(maxsize=None)
def _legendre_basis(order: int, degree_count: int) -> FloatArray:
    nodes, _ = _quadrature_rule(order)
    basis = np.asarray(
        np.polynomial.legendre.legvander(nodes, degree_count - 1),
        dtype=np.float64,
    )
    basis.setflags(write=False)
    return basis


@dataclass(frozen=True)
class LeafDensity:
    """Tensor-product Legendre expansion on a cubic leaf.

    Coefficient index ``(a, b, c)`` multiplies
    ``P_a(xi_x) P_b(xi_y) P_c(xi_z)``, with each leaf coordinate ``xi`` in
    ``[-1, 1]``.
    """

    box: LeafBox
    coefficients: DensityArray

    def __post_init__(self) -> None:
        if not isinstance(self.box, LeafBox):
            raise TypeError("box must be a LeafBox")
        if not isinstance(self.coefficients, np.ndarray):
            raise TypeError("coefficients must be a numpy.ndarray")
        if self.coefficients.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
            raise TypeError("coefficients must have dtype float64 or complex128")
        if self.coefficients.ndim != 3 or any(size == 0 for size in self.coefficients.shape):
            raise ValueError("coefficients must be a nonempty three-dimensional array")

    def evaluate(self, points: FloatArray) -> DensityArray:
        """Evaluate the Legendre expansion at an arbitrary ``(n, 3)`` point array."""
        points = require_array("points", points, np.float64, (None, 3))
        coordinates = 2.0 * (points - self.box.center) / self.box.width
        basis_x = np.polynomial.legendre.legvander(coordinates[:, 0], self.coefficients.shape[0] - 1)
        basis_y = np.polynomial.legendre.legvander(coordinates[:, 1], self.coefficients.shape[1] - 1)
        basis_z = np.polynomial.legendre.legvander(coordinates[:, 2], self.coefficients.shape[2] - 1)
        return contract("na,nb,nc,abc->n", basis_x, basis_y, basis_z, self.coefficients)

    def gaussian_potential(
        self,
        targets: FloatArray,
        inverse_length: float,
        quadrature_order: int,
    ) -> DensityArray:
        """Convolve this leaf with ``exp(-inverse_length**2 * |r-r'|**2)``.

        Gauss--Legendre integration is separated by Cartesian direction; the
        final fixed tensor contraction is routed through
        :func:`pymuffintin.tensor.contract`.
        """
        targets = require_array("targets", targets, np.float64, (None, 3))
        if (
            type(inverse_length) is not float
            or not np.isfinite(inverse_length)
            or inverse_length <= 0.0
        ):
            raise ValueError("inverse_length must be a positive finite float")
        if type(quadrature_order) is not int or quadrature_order <= 0:
            raise ValueError("quadrature_order must be a positive int")

        nodes, weights = _quadrature_rule(quadrature_order)
        half_width = 0.5 * self.box.width
        matrices: list[FloatArray] = []
        for axis, degree_count in enumerate(self.coefficients.shape):
            basis = _legendre_basis(quadrature_order, degree_count)
            source_points = self.box.center[axis] + half_width * nodes
            kernel = np.exp(
                -(inverse_length**2)
                * (targets[:, axis, np.newaxis] - source_points[np.newaxis, :]) ** 2
            )
            matrices.append(half_width * ((kernel * weights) @ basis))

        return contract(
            "na,nb,nc,abc->n",
            matrices[0],
            matrices[1],
            matrices[2],
            self.coefficients,
        )

    def fourier_tensor(
        self,
        wave_numbers: FloatArray,
        quadrature_order: int,
    ) -> NDArray[np.complex128]:
        """Return ``int rho(r) exp(-i k.r) dr`` on a Cartesian k tensor."""
        wave_numbers = require_array("wave_numbers", wave_numbers, np.float64, (None,))
        if type(quadrature_order) is not int or quadrature_order <= 0:
            raise ValueError("quadrature_order must be a positive int")

        nodes, weights = _quadrature_rule(quadrature_order)
        half_width = 0.5 * self.box.width
        matrices: list[NDArray[np.complex128]] = []
        for axis, degree_count in enumerate(self.coefficients.shape):
            basis = _legendre_basis(quadrature_order, degree_count)
            source_points = self.box.center[axis] + half_width * nodes
            phase = np.exp(-1j * wave_numbers[:, None] * source_points[None, :])
            matrices.append(
                np.asarray(half_width * ((phase * weights) @ basis), dtype=np.complex128)
            )

        return np.asarray(
            contract(
                "ia,jb,kc,abc->ijk",
                matrices[0],
                matrices[1],
                matrices[2],
                self.coefficients,
            ),
            dtype=np.complex128,
        )

    def radial_potential(
        self,
        targets: FloatArray,
        kernel: RadialKernel,
        quadrature_order: int,
    ) -> DensityArray:
        """Integrate a smooth radial kernel against this leaf density."""
        targets = require_array("targets", targets, np.float64, (None, 3))
        if type(quadrature_order) is not int or quadrature_order <= 0:
            raise ValueError("quadrature_order must be a positive int")

        nodes, weights = _quadrature_rule(quadrature_order)
        half_width = 0.5 * self.box.width
        axes = [self.box.center[axis] + half_width * nodes for axis in range(3)]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        wx, wy, wz = np.meshgrid(weights, weights, weights, indexing="ij")
        weighted_density = (
            (half_width**3)
            * (wx * wy * wz).ravel()
            * self.evaluate(points)
        )
        distances = np.linalg.norm(targets[:, None, :] - points[None, :, :], axis=2)
        return kernel(np.asarray(distances, dtype=np.float64)) @ weighted_density
