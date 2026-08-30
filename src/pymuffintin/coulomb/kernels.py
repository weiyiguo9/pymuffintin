"""Gaussian telescoping split of the three-dimensional Coulomb kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, erfc, inf, log, pi, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GaussianBand:
    """A quadrature approximation to one interval of the Gaussian integral."""

    lower: float
    upper: float
    nodes: NDArray[np.float64]
    weights: NDArray[np.float64]
    cutoff_radius: float

    def evaluate(self, distances: ArrayLike) -> NDArray[np.float64]:
        """Evaluate this (untruncated) Gaussian band at non-negative distances."""
        radii = np.asarray(distances, dtype=np.float64)
        if np.any(radii < 0.0):
            raise ValueError("distances must be non-negative")
        return np.sum(
            self.weights * np.exp(-np.square(radii[..., None] * self.nodes)),
            axis=-1,
        )


@dataclass(frozen=True)
class CoulombKernelSplit:
    """Dyadic Gaussian-band split of ``1/r`` for a 3D multilevel hierarchy.

    The root inverse width is ``t_0 = 1 / root_width``.  The coarse band is
    ``[0, t_0]`` and correction level ``ell`` covers
    ``[t_ell, t_(ell + 1)]`` for ``ell = 0, ..., max_level``.  Consequently
    the complementary local kernel starts at twice the finest leaf inverse
    width.
    """

    root_width: float
    max_level: int
    tolerance: float = 1.0e-12
    gaussian_order: int = 16
    coarse_band: GaussianBand = field(init=False)
    correction_bands: tuple[GaussianBand, ...] = field(init=False)
    terminal_scale: float = field(init=False)
    local_cutoff_radius: float = field(init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.root_width) or self.root_width <= 0.0:
            raise ValueError("root_width must be a finite positive float")
        if isinstance(self.max_level, bool) or not isinstance(self.max_level, int):
            raise TypeError("max_level must be an integer")
        if self.max_level < 0:
            raise ValueError("max_level must be non-negative")
        if not np.isfinite(self.tolerance) or not 0.0 < self.tolerance < 1.0:
            raise ValueError("tolerance must be finite and lie strictly between zero and one")
        if isinstance(self.gaussian_order, bool) or not isinstance(self.gaussian_order, int):
            raise TypeError("gaussian_order must be an integer")
        if self.gaussian_order < 1:
            raise ValueError("gaussian_order must be positive")

        root_scale = 1.0 / self.root_width
        coarse = self._make_band(0.0, root_scale, inf)
        corrections = tuple(
            self._make_band(
                root_scale * 2.0**level,
                root_scale * 2.0 ** (level + 1),
                sqrt(-log(self.tolerance)) / (root_scale * 2.0**level),
            )
            for level in range(self.max_level + 1)
        )
        terminal_scale = root_scale * 2.0 ** (self.max_level + 1)

        object.__setattr__(self, "coarse_band", coarse)
        object.__setattr__(self, "correction_bands", corrections)
        object.__setattr__(self, "terminal_scale", terminal_scale)
        object.__setattr__(
            self,
            "local_cutoff_radius",
            sqrt(-log(self.tolerance)) / terminal_scale,
        )

    @property
    def bands(self) -> tuple[GaussianBand, ...]:
        """All Gaussian bands, ordered from global coarse to finest correction."""
        return (self.coarse_band, *self.correction_bands)

    def local_kernel(self, distances: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``erfc(terminal_scale * r) / r`` without truncation.

        The value at zero remains infinite: treatment of the singular cell
        belongs to the downstream DMK discretization rather than this split.
        """
        radii = np.asarray(distances, dtype=np.float64)
        if np.any(radii < 0.0):
            raise ValueError("distances must be non-negative")
        result = np.empty_like(radii)
        zero = radii == 0.0
        result[zero] = inf
        nonzero = ~zero
        scaled = self.terminal_scale * radii[nonzero]
        result[nonzero] = np.fromiter(
            (erfc(float(value)) for value in scaled.flat),
            dtype=np.float64,
            count=scaled.size,
        ).reshape(scaled.shape) / radii[nonzero]
        return result

    def coarse_kernel(self, distances: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the exact coarse kernel ``erf(r/root_width) / r``."""
        radii = np.asarray(distances, dtype=np.float64)
        if np.any(radii < 0.0):
            raise ValueError("distances must be non-negative")
        result = np.empty_like(radii)
        zero = radii == 0.0
        root_scale = 1.0 / self.root_width
        result[zero] = 2.0 * root_scale / sqrt(pi)
        scaled = root_scale * radii[~zero]
        result[~zero] = np.fromiter(
            (erf(float(value)) for value in scaled.flat),
            dtype=np.float64,
            count=scaled.size,
        ).reshape(scaled.shape) / radii[~zero]
        return result

    def reconstruct(self, r: ArrayLike) -> NDArray[np.float64]:
        """Reconstruct the split kernel at strictly positive radii."""
        radii = np.asarray(r, dtype=np.float64)
        if np.any(radii <= 0.0):
            raise ValueError("reconstruct requires strictly positive radii")
        corrections = sum(
            (band.evaluate(radii) for band in self.correction_bands),
            np.zeros_like(radii),
        )
        return self.coarse_kernel(radii) + corrections + self.local_kernel(radii)

    def _make_band(self, lower: float, upper: float, cutoff_radius: float) -> GaussianBand:
        abscissae, quadrature_weights = np.polynomial.legendre.leggauss(self.gaussian_order)
        half_width = 0.5 * (upper - lower)
        midpoint = 0.5 * (upper + lower)
        nodes = np.asarray(midpoint + half_width * abscissae, dtype=np.float64)
        weights = np.asarray(
            (2.0 / sqrt(pi)) * half_width * quadrature_weights,
            dtype=np.float64,
        )
        return GaussianBand(lower, upper, nodes, weights, cutoff_radius)
