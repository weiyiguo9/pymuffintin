"""Three-dimensional cubic-periodic continuous Coulomb reference."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import erfc, log, pi, sqrt

import numpy as np
from numpy.typing import NDArray

from pymuffintin.contracts import require_array
from pymuffintin.tensor import contract

from .density import LeafDensity
from .dmk import FloatArray, _local_remainder_integral
from .tree import AdaptiveTree, LeafBox


@dataclass(frozen=True)
class PeriodicDmk:
    """Apply the zero-mean cubic-periodic ``1/r`` Green function.

    The top level uses an Ewald split with dense tensor Fourier transforms.
    Input density in the unit cell must be neutral.  FINUFFT can later replace
    only the reciprocal evaluation without changing this leaf-density contract.
    """

    root: LeafBox
    densities: tuple[LeafDensity, ...]
    tolerance: float = 1.0e-10
    source_quadrature_order: int = 16
    local_quadrature_order: int = 10
    tree: AdaptiveTree = field(init=False)
    ewald_alpha: float = field(init=False)
    real_cutoff: float = field(init=False)
    reciprocal_order: int = field(init=False)
    wave_numbers: NDArray[np.float64] = field(init=False)
    reciprocal_coefficients: NDArray[np.complex128] = field(init=False)

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
            ("source_quadrature_order", self.source_quadrature_order),
            ("local_quadrature_order", self.local_quadrature_order),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")

        tree = AdaptiveTree(self.root, tuple(density.box for density in self.densities))
        polynomial_order = max(max(density.coefficients.shape) for density in self.densities)
        if self.local_quadrature_order < polynomial_order:
            raise ValueError("local_quadrature_order must cover every Legendre coefficient")
        charges = np.asarray(
            [density.box.width**3 * density.coefficients[0, 0, 0] for density in self.densities],
            dtype=np.complex128,
        )
        neutrality_tolerance = 64.0 * np.finfo(np.float64).eps * max(
            float(np.sum(np.abs(charges))), 1.0
        )
        if abs(np.sum(charges)) > neutrality_tolerance:
            raise ValueError("periodic Coulomb density must have zero total charge")

        logarithm = -log(self.tolerance)
        alpha = sqrt(logarithm) / self.root.width
        reciprocal_order = int(np.ceil(logarithm / pi))
        required_source_order = max(polynomial_order, 2 * reciprocal_order)
        if self.source_quadrature_order < required_source_order:
            raise ValueError(
                f"source_quadrature_order must be at least {required_source_order}"
            )
        indices = np.arange(-reciprocal_order, reciprocal_order + 1, dtype=np.float64)
        wave_numbers = np.asarray(2.0 * pi * indices / self.root.width, dtype=np.float64)
        rho_hat = np.zeros((indices.size, indices.size, indices.size), dtype=np.complex128)
        for density in self.densities:
            rho_hat += density.fourier_tensor(wave_numbers, self.source_quadrature_order)
        kx, ky, kz = np.meshgrid(wave_numbers, wave_numbers, wave_numbers, indexing="ij")
        k_squared = kx * kx + ky * ky + kz * kz
        green = np.zeros_like(k_squared)
        nonzero = k_squared > 0.0
        green[nonzero] = (
            (4.0 * pi / self.root.width**3)
            * np.exp(-k_squared[nonzero] / (4.0 * alpha**2))
            / k_squared[nonzero]
        )
        reciprocal_coefficients = np.asarray(green * rho_hat, dtype=np.complex128)
        wave_numbers.setflags(write=False)
        reciprocal_coefficients.setflags(write=False)
        object.__setattr__(self, "tree", tree)
        object.__setattr__(self, "ewald_alpha", alpha)
        object.__setattr__(self, "real_cutoff", sqrt(logarithm) / alpha)
        object.__setattr__(self, "reciprocal_order", reciprocal_order)
        object.__setattr__(self, "wave_numbers", wave_numbers)
        object.__setattr__(self, "reciprocal_coefficients", reciprocal_coefficients)

    def apply(self, targets: FloatArray) -> NDArray[np.complex128]:
        """Return the zero-mean periodic potential at Cartesian targets."""
        targets = require_array("targets", targets, np.float64, (None, 3))
        wrapped = self.root.lower + np.mod(targets - self.root.lower, self.root.width)
        potential = self._real_space(wrapped)
        potential += self._reciprocal_space(wrapped)
        return potential

    def _real_space(self, targets: FloatArray) -> NDArray[np.complex128]:
        potential = np.zeros(targets.shape[0], dtype=np.complex128)
        image_layers = int(np.ceil(self.real_cutoff / self.root.width)) + 1
        kernel = self._real_kernel

        for image in product(range(-image_layers, image_layers + 1), repeat=3):
            shift = self.root.width * np.asarray(image, dtype=np.float64)
            for density in self.densities:
                if _shifted_box_distance(self.root, density.box, shift) > self.real_cutoff:
                    continue
                for target_index, target in enumerate(targets):
                    potential[target_index] += _local_remainder_integral(
                        density,
                        np.asarray(target - shift, dtype=np.float64),
                        kernel,
                        self.local_quadrature_order,
                    )
        return potential

    def _reciprocal_space(self, targets: FloatArray) -> NDArray[np.complex128]:
        phase_x = np.exp(1j * targets[:, 0, None] * self.wave_numbers[None, :])
        phase_y = np.exp(1j * targets[:, 1, None] * self.wave_numbers[None, :])
        phase_z = np.exp(1j * targets[:, 2, None] * self.wave_numbers[None, :])
        return np.asarray(
            contract(
                "ti,tj,tk,ijk->t",
                phase_x,
                phase_y,
                phase_z,
                self.reciprocal_coefficients,
            ),
            dtype=np.complex128,
        )

    def _real_kernel(self, distances: FloatArray) -> FloatArray:
        radii = np.asarray(distances, dtype=np.float64)
        result = np.empty_like(radii)
        zero = radii == 0.0
        result[zero] = np.inf
        scaled = self.ewald_alpha * radii[~zero]
        result[~zero] = np.fromiter(
            (erfc(float(value)) for value in scaled.flat),
            dtype=np.float64,
            count=scaled.size,
        ).reshape(scaled.shape) / radii[~zero]
        return result


def _shifted_box_distance(root: LeafBox, box: LeafBox, shift: FloatArray) -> float:
    shifted_lower = box.lower + shift
    shifted_upper = box.upper + shift
    displacement = np.maximum(
        np.maximum(root.lower - shifted_upper, shifted_lower - root.upper),
        0.0,
    )
    return float(np.linalg.norm(displacement))
