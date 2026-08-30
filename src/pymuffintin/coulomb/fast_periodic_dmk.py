"""Hierarchical zero-mean cubic-periodic Coulomb solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import ceil, erfc, log, pi, sqrt
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from pymuffintin.contracts import require_array
from pymuffintin.tensor import contract

from .density import LeafDensity, _quadrature_rule
from .dmk import _local_remainder_integral
from .fast_dmk import (
    CellIndex,
    CellKey,
    EquivalentArray,
    FloatArray,
    _box_distance_cells,
    _build_hierarchy,
    _build_source_hierarchy,
    _cell_center,
    _child_interpolation_matrices,
    _evaluate_target_locals,
    _gaussian_translation_matrices,
    _group_targets,
)
from .kernels import GaussianBand
from .tree import LeafBox


@dataclass(frozen=True)
class FastPeriodicDmkWork:
    """Deterministic logical work for one periodic application."""

    real_leaf_source_transforms: int
    real_upward_translations: int
    real_same_level_translations: int
    real_downward_interpolations: int
    real_target_interpolations: int
    real_local_pairs: int
    reciprocal_source_transforms: int
    reciprocal_modes: int
    nufft_targets: int

    @property
    def total(self) -> int:
        return (
            self.real_leaf_source_transforms
            + self.real_upward_translations
            + self.real_same_level_translations
            + self.real_downward_interpolations
            + self.real_target_interpolations
            + self.real_local_pairs
            + self.reciprocal_source_transforms
            + self.reciprocal_modes
            + self.nufft_targets
        )


@dataclass(frozen=True)
class FastPeriodicDmk:
    """Apply the zero-mean cubic-periodic ``1/r`` Green function.

    The Ewald real part uses periodic same-level Gaussian translations and a
    bounded leaf tail.  The fixed reciprocal grid is accumulated from leaf
    Fourier transforms and evaluated at targets by FINUFFT type 2.
    """

    root: LeafBox
    densities: tuple[LeafDensity, ...]
    tolerance: float = 1.0e-10
    gaussian_order: int = 16
    source_quadrature_order: int = 16
    local_quadrature_order: int = 10
    interpolation_order: int = 24
    levels: tuple[int, ...] = field(init=False)
    max_level: int = field(init=False)
    leaf_keys: tuple[CellKey, ...] = field(init=False)
    cells_by_level: tuple[tuple[CellKey, ...], ...] = field(init=False)
    ewald_alpha: float = field(init=False)
    real_bands: tuple[GaussianBand, ...] = field(init=False)
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
            ("gaussian_order", self.gaussian_order),
            ("source_quadrature_order", self.source_quadrature_order),
            ("local_quadrature_order", self.local_quadrature_order),
            ("interpolation_order", self.interpolation_order),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")

        polynomial_order = max(max(density.coefficients.shape) for density in self.densities)
        if self.local_quadrature_order < polynomial_order:
            raise ValueError("local_quadrature_order must cover every Legendre coefficient")
        if self.interpolation_order < polynomial_order:
            raise ValueError("interpolation_order must cover every Legendre coefficient")
        if 2 * self.source_quadrature_order - 1 < self.interpolation_order + polynomial_order - 2:
            raise ValueError(
                "source_quadrature_order must integrate the equivalent-source transform"
            )
        logarithm = -log(self.tolerance)
        reciprocal_order = int(np.ceil(logarithm / pi))
        required_source_order = max(polynomial_order, 2 * reciprocal_order)
        if self.source_quadrature_order < required_source_order:
            raise ValueError(
                f"source_quadrature_order must be at least {required_source_order}"
            )

        charges = np.asarray(
            [density.box.width**3 * density.coefficients[0, 0, 0] for density in self.densities],
            dtype=np.complex128,
        )
        neutrality_tolerance = 64.0 * np.finfo(np.float64).eps * max(
            float(np.sum(np.abs(charges))), 1.0
        )
        if abs(np.sum(charges)) > neutrality_tolerance:
            raise ValueError("periodic Coulomb density must have zero total charge")

        levels, leaf_keys, cells_by_level = _build_hierarchy(
            self.root,
            tuple(density.box for density in self.densities),
        )
        max_level = max(levels)
        alpha = sqrt(logarithm) / self.root.width
        bands = tuple(
            _make_gaussian_band(
                alpha * 2**level,
                alpha * 2 ** (level + 1),
                self.root.width / 2**level,
                self.gaussian_order,
            )
            for level in range(max_level + 1)
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

        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "max_level", max_level)
        object.__setattr__(self, "leaf_keys", leaf_keys)
        object.__setattr__(self, "cells_by_level", cells_by_level)
        object.__setattr__(self, "ewald_alpha", alpha)
        object.__setattr__(self, "real_bands", bands)
        object.__setattr__(self, "reciprocal_order", reciprocal_order)
        object.__setattr__(self, "wave_numbers", wave_numbers)
        object.__setattr__(self, "reciprocal_coefficients", reciprocal_coefficients)

    def apply(self, targets: FloatArray) -> NDArray[np.complex128]:
        """Return the periodic potential at Cartesian targets."""
        potential, _ = self.apply_with_work(targets)
        return potential

    def apply_with_work(
        self,
        targets: FloatArray,
    ) -> tuple[NDArray[np.complex128], FastPeriodicDmkWork]:
        """Return the potential and immutable logical work counts."""
        targets = require_array("targets", targets, np.float64, (None, 3))
        wrapped = self.root.lower + np.mod(targets - self.root.lower, self.root.width)
        real, upward_count, same_level_count, downward_count, local_pairs = self._real_space(
            wrapped
        )
        real += self._reciprocal_space(wrapped)
        work = FastPeriodicDmkWork(
            real_leaf_source_transforms=len(self.densities),
            real_upward_translations=upward_count,
            real_same_level_translations=same_level_count,
            real_downward_interpolations=downward_count,
            real_target_interpolations=targets.shape[0],
            real_local_pairs=local_pairs,
            reciprocal_source_transforms=len(self.densities),
            reciprocal_modes=self.reciprocal_coefficients.size - 1,
            nufft_targets=targets.shape[0],
        )
        return real, work

    def _real_space(
        self,
        wrapped: FloatArray,
    ) -> tuple[NDArray[np.complex128], int, int, int, int]:
        if wrapped.shape[0] == 0:
            return np.zeros(0, dtype=np.complex128), 0, 0, 0, 0
        interpolation_nodes, _ = _quadrature_rule(self.interpolation_order)
        sources, upward_count = _build_source_hierarchy(
            self.densities,
            self.leaf_keys,
            self.max_level,
            interpolation_nodes,
            self.source_quadrature_order,
            np.dtype(np.complex128),
        )
        target_cells = tuple(
            _group_targets(wrapped, self.root, level) for level in range(self.max_level + 1)
        )
        child_interpolation = _child_interpolation_matrices(interpolation_nodes)
        locals_by_cell: dict[CellIndex, EquivalentArray] = {}
        same_level_count = 0
        downward_count = 0

        for level in range(self.max_level + 1):
            shape = (self.interpolation_order,) * 3
            if level == 0:
                next_locals: dict[CellIndex, EquivalentArray] = {
                    cell: np.zeros(shape, dtype=np.complex128)
                    for cell in sorted(target_cells[level])
                }
            else:
                next_locals = {}
                for cell in sorted(target_cells[level]):
                    parent = tuple(index // 2 for index in cell)
                    bits = tuple(index - 2 * parent_axis for index, parent_axis in zip(cell, parent))
                    matrices = tuple(child_interpolation[bit] for bit in bits)
                    next_locals[cell] = np.asarray(
                        contract(
                            "ia,jb,kc,abc->ijk",
                            matrices[0],
                            matrices[1],
                            matrices[2],
                            locals_by_cell[parent],
                        ),
                        dtype=np.complex128,
                    )
                    downward_count += 1
            translated, count = _translate_periodic_band(
                level,
                self.root,
                self.real_bands[level],
                interpolation_nodes,
                sources[level],
                tuple(sorted(target_cells[level])),
            )
            same_level_count += count
            for cell, values in translated.items():
                next_locals[cell] += values
            locals_by_cell = next_locals

        potential = np.asarray(
            _evaluate_target_locals(
                wrapped,
                self.root,
                self.max_level,
                interpolation_nodes,
                target_cells[self.max_level],
                locals_by_cell,
                np.dtype(np.complex128),
            ),
            dtype=np.complex128,
        )
        local_pairs = self._add_periodic_tails(potential, wrapped, target_cells)
        return potential, upward_count, same_level_count, downward_count, local_pairs

    def _add_periodic_tails(
        self,
        potential: NDArray[np.complex128],
        wrapped: FloatArray,
        target_cells: tuple[dict[CellIndex, tuple[int, ...]], ...],
    ) -> int:
        count = 0
        length = self.root.width
        for density, key in zip(self.densities, self.leaf_keys, strict=True):
            level, x, y, z = key
            n_cells = 2**level
            width = length / n_cells
            cutoff = 0.5 * width
            terminal_scale = self.ewald_alpha * 2 ** (level + 1)
            kernel = lambda distances, scale=terminal_scale: _tail_kernel(scale, distances)
            groups = target_cells[level]
            source = (x, y, z)
            for image in product((-1, 0, 1), repeat=3):
                absolute_source = tuple(
                    index + shift * n_cells for index, shift in zip(source, image)
                )
                shift = length * np.asarray(image, dtype=np.float64)
                candidate_indices: list[int] = []
                for dx in (-2, -1, 0, 1, 2):
                    for dy in (-2, -1, 0, 1, 2):
                        for dz in (-2, -1, 0, 1, 2):
                            cell = (
                                absolute_source[0] + dx,
                                absolute_source[1] + dy,
                                absolute_source[2] + dz,
                            )
                            if all(0 <= axis < n_cells for axis in cell):
                                candidate_indices.extend(groups.get(cell, ()))
                for target_index in sorted(candidate_indices):
                    target = np.asarray(wrapped[target_index] - shift, dtype=np.float64)
                    if density.box.distance_to(target) > cutoff:
                        continue
                    potential[target_index] += _local_remainder_integral(
                        density,
                        target,
                        kernel,
                        self.local_quadrature_order,
                    )
                    count += 1
        return count

    def _reciprocal_space(self, wrapped: FloatArray) -> NDArray[np.complex128]:
        finufft = _load_finufft()
        if wrapped.shape[0] == 0:
            return np.zeros(0, dtype=np.complex128)
        theta = np.mod(
            (2.0 * pi / self.root.width) * wrapped + pi,
            2.0 * pi,
        ) - pi
        result = finufft.nufft3d2(
            np.ascontiguousarray(theta[:, 0], dtype=np.float64),
            np.ascontiguousarray(theta[:, 1], dtype=np.float64),
            np.ascontiguousarray(theta[:, 2], dtype=np.float64),
            np.ascontiguousarray(self.reciprocal_coefficients),
            eps=self.tolerance,
            isign=+1,
        )
        return np.asarray(result, dtype=np.complex128)


def _make_gaussian_band(
    lower: float,
    upper: float,
    cutoff_radius: float,
    order: int,
) -> GaussianBand:
    abscissae, quadrature_weights = np.polynomial.legendre.leggauss(order)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    nodes = np.asarray(midpoint + half_width * abscissae, dtype=np.float64)
    weights = np.asarray(
        (2.0 / sqrt(pi)) * half_width * quadrature_weights,
        dtype=np.float64,
    )
    return GaussianBand(lower, upper, nodes, weights, cutoff_radius)


def _translate_periodic_band(
    level: int,
    root: LeafBox,
    band: GaussianBand,
    interpolation_nodes: FloatArray,
    sources: dict[CellIndex, EquivalentArray],
    target_cells: tuple[CellIndex, ...],
) -> tuple[dict[CellIndex, EquivalentArray], int]:
    width = root.width / 2**level
    n_cells = 2**level
    order = interpolation_nodes.size
    translated = {
        cell: np.zeros((order, order, order), dtype=np.complex128)
        for cell in target_cells
    }
    radius = int(ceil(band.cutoff_radius / width)) + 1
    matrix_cache: dict[
        CellIndex,
        tuple[tuple[FloatArray, FloatArray, FloatArray], ...],
    ] = {}
    count = 0

    for target in target_cells:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    absolute_source = (target[0] + dx, target[1] + dy, target[2] + dz)
                    if _box_distance_cells(target, absolute_source, width) > band.cutoff_radius:
                        continue
                    source = tuple(index % n_cells for index in absolute_source)
                    source_values = sources.get(source)
                    if source_values is None:
                        continue
                    relative = tuple(
                        target_axis - source_axis
                        for target_axis, source_axis in zip(target, absolute_source)
                    )
                    matrices_by_node = matrix_cache.get(relative)
                    if matrices_by_node is None:
                        matrices_by_node = _gaussian_translation_matrices(
                            width,
                            relative,
                            band.nodes,
                            interpolation_nodes,
                        )
                        matrix_cache[relative] = matrices_by_node
                    for weight, matrices in zip(band.weights, matrices_by_node, strict=True):
                        translated[target] += float(weight) * contract(
                            "ia,jb,kc,abc->ijk",
                            matrices[0],
                            matrices[1],
                            matrices[2],
                            source_values,
                        )
                    count += 1
    return translated, count


def _tail_kernel(scale: float, distances: FloatArray) -> FloatArray:
    radii = np.asarray(distances, dtype=np.float64)
    result = np.empty_like(radii)
    zero = radii == 0.0
    result[zero] = np.inf
    scaled = scale * radii[~zero]
    result[~zero] = np.fromiter(
        (erfc(float(value)) for value in scaled.flat),
        dtype=np.float64,
        count=scaled.size,
    ).reshape(scaled.shape) / radii[~zero]
    return result


def _load_finufft() -> ModuleType:
    try:
        import finufft
    except ModuleNotFoundError as error:
        if error.name != "finufft":
            raise
        raise ModuleNotFoundError(
            "FastPeriodicDmk requires optional dependency 'finufft'; "
            "install pymuffintin[nufft]."
        ) from error
    return finufft
