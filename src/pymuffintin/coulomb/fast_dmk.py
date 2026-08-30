"""Hierarchical free-space DMK for continuous dyadic leaf densities.

The smooth Gaussian bands are evaluated with fixed-order tensor-product
equivalent sources and local interpolation.  Source data move upward through a
hash-indexed dyadic hierarchy, while target locals move downward through a
uniform (and possibly out-of-root) target path.  Only the complementary local
kernel is integrated against individual leaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from pymuffintin.contracts import require_array
from pymuffintin.tensor import contract

from .density import DensityArray, LeafDensity, _legendre_basis, _quadrature_rule
from .dmk import _local_remainder_integral
from .kernels import CoulombKernelSplit, GaussianBand
from .tree import LeafBox


FloatArray = NDArray[np.float64]
EquivalentArray: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
CellIndex: TypeAlias = tuple[int, int, int]
CellKey: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True)
class FastDmkWork:
    """Logical work performed by one :meth:`FastDmk.apply_with_work` call."""

    leaf_source_transforms: int
    upward_translations: int
    same_level_translations: int
    downward_interpolations: int
    target_interpolations: int
    local_remainder_evaluations: int

    @property
    def total_translations(self) -> int:
        return (
            self.upward_translations
            + self.same_level_translations
            + self.downward_interpolations
        )

    @property
    def total(self) -> int:
        return (
            self.leaf_source_transforms
            + self.total_translations
            + self.target_interpolations
            + self.local_remainder_evaluations
        )


@dataclass(frozen=True)
class FastDmk:
    """Apply ``1/r`` without a far-field leaf-by-target traversal.

    The density boxes must be a complete, mildly 2:1-balanced dyadic
    partition of ``root``.  ``interpolation_order`` controls the fixed-rank
    equivalent-source and local representations; the two quadrature orders
    retain the contracts of :class:`ContinuousDmk`.
    """

    root: LeafBox
    densities: tuple[LeafDensity, ...]
    tolerance: float = 1.0e-10
    gaussian_order: int = 16
    source_quadrature_order: int = 16
    local_quadrature_order: int = 10
    interpolation_order: int = 8
    levels: tuple[int, ...] = field(init=False)
    max_level: int = field(init=False)
    leaf_keys: tuple[CellKey, ...] = field(init=False)
    cells_by_level: tuple[tuple[CellKey, ...], ...] = field(init=False)
    kernel_split: CoulombKernelSplit = field(init=False)

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
        if self.source_quadrature_order < polynomial_order:
            raise ValueError("source_quadrature_order must cover every Legendre coefficient")
        if self.local_quadrature_order < polynomial_order:
            raise ValueError("local_quadrature_order must cover every Legendre coefficient")
        if self.interpolation_order < polynomial_order:
            raise ValueError("interpolation_order must cover every Legendre coefficient")

        levels, leaf_keys, cells_by_level = _build_hierarchy(
            self.root,
            tuple(density.box for density in self.densities),
        )
        max_level = max(levels)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "max_level", max_level)
        object.__setattr__(self, "leaf_keys", leaf_keys)
        object.__setattr__(self, "cells_by_level", cells_by_level)
        object.__setattr__(
            self,
            "kernel_split",
            CoulombKernelSplit(
                root_width=self.root.width,
                max_level=max_level,
                tolerance=self.tolerance,
                gaussian_order=self.gaussian_order,
            ),
        )

    def apply(self, targets: FloatArray) -> DensityArray:
        """Return the free-space Coulomb potential at batched ``targets``."""
        potential, _ = self.apply_with_work(targets)
        return potential

    def apply_with_work(self, targets: FloatArray) -> tuple[DensityArray, FastDmkWork]:
        """Return the potential and deterministic immutable logical work counts."""
        targets = require_array("targets", targets, np.float64, (None, 3))
        dtype = np.dtype(
            np.result_type(*(density.coefficients.dtype for density in self.densities))
        )
        if targets.shape[0] == 0:
            work = FastDmkWork(len(self.densities), 0, 0, 0, 0, 0)
            return np.zeros(0, dtype=dtype), work

        interpolation_nodes, _ = _quadrature_rule(self.interpolation_order)
        sources, upward_count = self._source_hierarchy(interpolation_nodes, dtype)
        target_cells = tuple(
            _group_targets(targets, self.root, level) for level in range(self.max_level + 1)
        )

        locals_by_cell: dict[CellIndex, EquivalentArray] = {}
        same_level_count = 0
        downward_count = 0
        child_interpolation = _child_interpolation_matrices(interpolation_nodes)

        for level in range(self.max_level + 1):
            next_locals: dict[CellIndex, EquivalentArray] = {}
            if level > 0:
                for cell in sorted(target_cells[level]):
                    parent = tuple(index // 2 for index in cell)
                    bits = tuple(index - 2 * parent_axis for index, parent_axis in zip(cell, parent))
                    parent_local = locals_by_cell[parent]
                    matrices = tuple(child_interpolation[bit] for bit in bits)
                    next_locals[cell] = np.asarray(
                        contract(
                            "ia,jb,kc,abc->ijk",
                            matrices[0],
                            matrices[1],
                            matrices[2],
                            parent_local,
                        ),
                        dtype=dtype,
                    )
                    downward_count += 1
            else:
                shape = (self.interpolation_order,) * 3
                next_locals = {
                    cell: np.zeros(shape, dtype=dtype) for cell in sorted(target_cells[0])
                }

            bands = (
                (self.kernel_split.coarse_band, self.kernel_split.correction_bands[0])
                if level == 0
                else (self.kernel_split.correction_bands[level],)
            )
            for band in bands:
                translated, interaction_count = _translate_band(
                    level=level,
                    root=self.root,
                    band=band,
                    interpolation_nodes=interpolation_nodes,
                    sources=sources[level],
                    target_cells=tuple(sorted(target_cells[level])),
                    dtype=dtype,
                )
                same_level_count += interaction_count
                for cell, values in translated.items():
                    next_locals[cell] += values
            locals_by_cell = next_locals

        potential = _evaluate_target_locals(
            targets,
            self.root,
            self.max_level,
            interpolation_nodes,
            target_cells[self.max_level],
            locals_by_cell,
            dtype,
        )
        local_count = self._add_local_remainders(potential, targets, target_cells)
        work = FastDmkWork(
            leaf_source_transforms=len(self.densities),
            upward_translations=upward_count,
            same_level_translations=same_level_count,
            downward_interpolations=downward_count,
            target_interpolations=targets.shape[0],
            local_remainder_evaluations=local_count,
        )
        return potential, work

    def _source_hierarchy(
        self,
        interpolation_nodes: FloatArray,
        dtype: np.dtype[np.float64] | np.dtype[np.complex128],
    ) -> tuple[tuple[dict[CellIndex, EquivalentArray], ...], int]:
        levels: list[dict[CellIndex, EquivalentArray]] = [
            {} for _ in range(self.max_level + 1)
        ]
        for density, key in zip(self.densities, self.leaf_keys, strict=True):
            level, x, y, z = key
            levels[level][(x, y, z)] = _leaf_source_transform(
                density,
                interpolation_nodes,
                self.source_quadrature_order,
                dtype,
            )

        transfer = _child_source_transfer_matrices(interpolation_nodes)
        upward_count = 0
        for level in range(self.max_level, 0, -1):
            for child in sorted(levels[level]):
                parent = tuple(index // 2 for index in child)
                bits = tuple(index - 2 * parent_axis for index, parent_axis in zip(child, parent))
                matrices = tuple(transfer[bit] for bit in bits)
                contribution = contract(
                    "ia,jb,kc,abc->ijk",
                    matrices[0],
                    matrices[1],
                    matrices[2],
                    levels[level][child],
                )
                if parent in levels[level - 1]:
                    levels[level - 1][parent] += contribution
                else:
                    levels[level - 1][parent] = np.asarray(contribution, dtype=dtype)
                upward_count += 1
        return tuple(levels), upward_count

    def _add_local_remainders(
        self,
        potential: EquivalentArray,
        targets: FloatArray,
        target_cells: tuple[dict[CellIndex, tuple[int, ...]], ...],
    ) -> int:
        count = 0
        split_by_level = tuple(
            CoulombKernelSplit(
                root_width=self.root.width,
                max_level=level,
                tolerance=self.tolerance,
                gaussian_order=self.gaussian_order,
            )
            for level in range(self.max_level + 1)
        )
        for density, key in zip(self.densities, self.leaf_keys, strict=True):
            level, x, y, z = key
            split = split_by_level[level]
            width = self.root.width / 2**level
            radius = int(ceil(split.local_cutoff_radius / width)) + 1
            groups = target_cells[level]
            candidate_indices: list[int] = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        candidate_indices.extend(groups.get((x + dx, y + dy, z + dz), ()))
            for target_index in sorted(candidate_indices):
                target = targets[target_index]
                if density.box.distance_to(target) > split.local_cutoff_radius:
                    continue
                potential[target_index] += _local_remainder_integral(
                    density,
                    target,
                    split.local_kernel,
                    self.local_quadrature_order,
                )
                count += 1
        return count


def _build_hierarchy(
    root: LeafBox,
    leaves: tuple[LeafBox, ...],
) -> tuple[tuple[int, ...], tuple[CellKey, ...], tuple[tuple[CellKey, ...], ...]]:
    tolerance = 32.0 * np.finfo(np.float64).eps * max(
        1.0, root.width, float(np.max(np.abs(root.center)))
    )
    root_lower = root.lower
    levels: list[int] = []
    keys: list[CellKey] = []
    leaf_set: set[CellKey] = set()
    internal_set: set[CellKey] = set()

    for leaf_index, leaf in enumerate(leaves):
        ratio = root.width / leaf.width
        level = int(np.rint(np.log2(ratio)))
        if level < 0 or not np.isclose(
            leaf.width, root.width / 2**level, rtol=0.0, atol=tolerance
        ):
            raise ValueError(f"leaves[{leaf_index}] width is not dyadic relative to root")
        if np.any(leaf.lower < root.lower - tolerance) or np.any(leaf.upper > root.upper + tolerance):
            raise ValueError(f"leaves[{leaf_index}] lies outside root")
        offsets = (leaf.lower - root_lower) / leaf.width
        rounded = np.rint(offsets)
        if not np.allclose(offsets, rounded, rtol=0.0, atol=tolerance / leaf.width):
            raise ValueError(f"leaves[{leaf_index}] is not aligned to the dyadic root grid")
        index = tuple(int(value) for value in rounded)
        if any(value < 0 or value >= 2**level for value in index):
            raise ValueError(f"leaves[{leaf_index}] lies outside root")
        key = (level, *index)
        if key in leaf_set:
            raise ValueError(f"leaves[{leaf_index}] overlaps another leaf in volume")
        for ancestor_level in range(level):
            shift = level - ancestor_level
            ancestor = (ancestor_level, *(value >> shift for value in index))
            if ancestor in leaf_set:
                raise ValueError(f"leaves[{leaf_index}] overlaps another leaf in volume")
            internal_set.add(ancestor)
        if key in internal_set:
            raise ValueError(f"leaves[{leaf_index}] overlaps another leaf in volume")
        leaf_set.add(key)
        levels.append(level)
        keys.append(key)

    max_level = max(levels)
    represented_cells = sum(2 ** (3 * (max_level - level)) for level in levels)
    if represented_cells != 2 ** (3 * max_level):
        raise ValueError("leaves must form a complete partition of root")

    _validate_mild_balance(tuple(leaf_set), leaf_set, internal_set)
    cells: list[set[CellKey]] = [set() for _ in range(max_level + 1)]
    for key in leaf_set | internal_set:
        cells[key[0]].add(key)
    cells_by_level = tuple(tuple(sorted(level_cells)) for level_cells in cells)
    return tuple(levels), tuple(keys), cells_by_level


def _validate_mild_balance(
    leaves: tuple[CellKey, ...],
    leaf_set: set[CellKey],
    internal_set: set[CellKey],
) -> None:
    for level, x, y, z in leaves:
        size = 2**level
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    neighbor = (x + dx, y + dy, z + dz)
                    if any(index < 0 or index >= size for index in neighbor):
                        continue
                    covering_level = level
                    covering: CellKey | None = (level, *neighbor)
                    while covering_level >= 0 and covering not in leaf_set:
                        covering_level -= 1
                        if covering_level >= 0:
                            shift = level - covering_level
                            covering = (
                                covering_level,
                                *(index >> shift for index in neighbor),
                            )
                    if covering_level >= 0:
                        if level - covering_level > 1:
                            raise ValueError("touching leaves violate 2:1 balance")
                        continue

                    same_level_key = (level, *neighbor)
                    if same_level_key not in internal_set:
                        raise ValueError("leaves must form a complete partition of root")
                    child_choices = []
                    for offset in (dx, dy, dz):
                        child_choices.append((1,) if offset < 0 else (0,) if offset > 0 else (0, 1))
                    for bx in child_choices[0]:
                        for by in child_choices[1]:
                            for bz in child_choices[2]:
                                child = (
                                    level + 1,
                                    2 * neighbor[0] + bx,
                                    2 * neighbor[1] + by,
                                    2 * neighbor[2] + bz,
                                )
                                if child in internal_set:
                                    raise ValueError("touching leaves violate 2:1 balance")


def _lagrange_matrix(nodes: FloatArray, points: FloatArray) -> FloatArray:
    weights = np.ones(nodes.size, dtype=np.float64)
    for index in range(nodes.size):
        weights[index] = 1.0 / np.prod(nodes[index] - np.delete(nodes, index))
    differences = points[:, None] - nodes[None, :]
    matrix = np.empty_like(differences)
    exact = differences == 0.0
    for point_index in range(points.size):
        matches = np.flatnonzero(exact[point_index])
        if matches.size:
            matrix[point_index] = 0.0
            matrix[point_index, matches[0]] = 1.0
        else:
            values = weights / differences[point_index]
            matrix[point_index] = values / np.sum(values)
    return matrix


def _leaf_source_transform(
    density: LeafDensity,
    interpolation_nodes: FloatArray,
    quadrature_order: int,
    dtype: np.dtype[np.float64] | np.dtype[np.complex128],
) -> EquivalentArray:
    quadrature_nodes, quadrature_weights = _quadrature_rule(quadrature_order)
    lagrange = _lagrange_matrix(interpolation_nodes, quadrature_nodes)
    half_width = 0.5 * density.box.width
    matrices = []
    for degree_count in density.coefficients.shape:
        basis = _legendre_basis(quadrature_order, degree_count)
        matrices.append(
            half_width * (lagrange.T @ (quadrature_weights[:, None] * basis))
        )
    return np.asarray(
        contract(
            "ia,jb,kc,abc->ijk",
            matrices[0],
            matrices[1],
            matrices[2],
            density.coefficients,
        ),
        dtype=dtype,
    )


def _child_source_transfer_matrices(nodes: FloatArray) -> tuple[FloatArray, FloatArray]:
    child_points = (-0.5 + 0.5 * nodes, 0.5 + 0.5 * nodes)
    return tuple(_lagrange_matrix(nodes, points).T for points in child_points)  # type: ignore[return-value]


def _child_interpolation_matrices(nodes: FloatArray) -> tuple[FloatArray, FloatArray]:
    child_points = (-0.5 + 0.5 * nodes, 0.5 + 0.5 * nodes)
    return tuple(_lagrange_matrix(nodes, points) for points in child_points)  # type: ignore[return-value]


def _group_targets(
    targets: FloatArray,
    root: LeafBox,
    level: int,
) -> dict[CellIndex, tuple[int, ...]]:
    width = root.width / 2**level
    scaled = (targets - root.lower[None, :]) / width
    indices = np.floor(scaled).astype(np.int64)
    groups: dict[CellIndex, list[int]] = {}
    for target_index, raw_index in enumerate(indices):
        cell = tuple(int(value) for value in raw_index)
        groups.setdefault(cell, []).append(target_index)
    return {cell: tuple(items) for cell, items in groups.items()}


def _cell_center(root: LeafBox, level: int, cell: CellIndex) -> FloatArray:
    width = root.width / 2**level
    return root.lower + width * (np.asarray(cell, dtype=np.float64) + 0.5)


def _translate_band(
    *,
    level: int,
    root: LeafBox,
    band: GaussianBand,
    interpolation_nodes: FloatArray,
    sources: dict[CellIndex, EquivalentArray],
    target_cells: tuple[CellIndex, ...],
    dtype: np.dtype[np.float64] | np.dtype[np.complex128],
) -> tuple[dict[CellIndex, EquivalentArray], int]:
    width = root.width / 2**level
    order = interpolation_nodes.size
    shape = (order, order, order)
    translated = {cell: np.zeros(shape, dtype=dtype) for cell in target_cells}
    interaction_count = 0
    if np.isinf(band.cutoff_radius):
        radius: int | None = None
    else:
        radius = int(ceil(band.cutoff_radius / width)) + 1

    relative_matrices: dict[CellIndex, tuple[tuple[FloatArray, FloatArray, FloatArray], ...]] = {}
    for target_cell in target_cells:
        if radius is None:
            candidates = tuple(sorted(sources))
        else:
            candidates = tuple(
                source_cell
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
                for dz in range(-radius, radius + 1)
                if (source_cell := (
                    target_cell[0] + dx,
                    target_cell[1] + dy,
                    target_cell[2] + dz,
                )) in sources
            )
        for source_cell in candidates:
            if not np.isinf(band.cutoff_radius) and _box_distance_cells(
                target_cell, source_cell, width
            ) > band.cutoff_radius:
                continue
            relative = tuple(
                target - source for target, source in zip(target_cell, source_cell)
            )
            matrices_by_node = relative_matrices.get(relative)
            if matrices_by_node is None:
                target_center = _cell_center(root, level, target_cell)
                source_center = _cell_center(root, level, source_cell)
                matrices: list[tuple[FloatArray, FloatArray, FloatArray]] = []
                for inverse_length in band.nodes:
                    axis_matrices = []
                    for axis in range(3):
                        target_points = (
                            target_center[axis] + 0.5 * width * interpolation_nodes
                        )
                        source_points = (
                            source_center[axis] + 0.5 * width * interpolation_nodes
                        )
                        axis_matrices.append(
                            np.exp(
                                -float(inverse_length) ** 2
                                * np.square(target_points[:, None] - source_points[None, :])
                            )
                        )
                    matrices.append((axis_matrices[0], axis_matrices[1], axis_matrices[2]))
                matrices_by_node = tuple(matrices)
                relative_matrices[relative] = matrices_by_node
            source_values = sources[source_cell]
            for weight, matrices in zip(band.weights, matrices_by_node, strict=True):
                translated[target_cell] += float(weight) * contract(
                    "ia,jb,kc,abc->ijk",
                    matrices[0],
                    matrices[1],
                    matrices[2],
                    source_values,
                )
            interaction_count += 1
    return translated, interaction_count


def _box_distance_cells(left: CellIndex, right: CellIndex, width: float) -> float:
    gaps = np.maximum(np.abs(np.subtract(left, right)) - 1, 0)
    return float(width * np.linalg.norm(gaps))


def _evaluate_target_locals(
    targets: FloatArray,
    root: LeafBox,
    level: int,
    interpolation_nodes: FloatArray,
    groups: dict[CellIndex, tuple[int, ...]],
    locals_by_cell: dict[CellIndex, EquivalentArray],
    dtype: np.dtype[np.float64] | np.dtype[np.complex128],
) -> EquivalentArray:
    width = root.width / 2**level
    potential = np.zeros(targets.shape[0], dtype=dtype)
    for cell in sorted(groups):
        indices = np.asarray(groups[cell], dtype=np.int64)
        center = _cell_center(root, level, cell)
        coordinates = 2.0 * (targets[indices] - center[None, :]) / width
        matrices = tuple(
            _lagrange_matrix(interpolation_nodes, coordinates[:, axis]) for axis in range(3)
        )
        potential[indices] = contract(
            "na,nb,nc,abc->n",
            matrices[0],
            matrices[1],
            matrices[2],
            locals_by_cell[cell],
        )
    return potential
