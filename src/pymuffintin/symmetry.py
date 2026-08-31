"""Unified crystal-symmetry dataset over spglib detection and spgrep irreps.

This mirrors the Rust ``muffintin_symmetry`` IR so both language layers share
one contract: rotations are integer matrices in the input-cell fractional
basis acting as ``rotations[i] @ x + translations[i]``, translations are
fractional, and ``equivalent_atoms[i]`` is the representative site of site
``i``'s crystallographic orbit. Backends (spglib here, moyo in the Rust core,
SPEX import) populate the same dataset, so analysis code never depends on a
backend-native structure.

The irrep helpers wrap spgrep and consume the dataset directly; they assume
the dataset describes a primitive cell, which is spgrep's requirement for
``*_from_primitive_symmetry``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import lcm

import numpy as np
import spglib
import spgrep
from numpy.typing import NDArray

from .contracts import ComplexArray, FloatArray, IntArray


@dataclass(frozen=True)
class SymmetryDataset:
    """Method-neutral symmetry of the input cell; see the module docstring."""

    rotations: IntArray
    translations: FloatArray
    time_reversals: NDArray[np.bool_]
    equivalent_atoms: IntArray
    spacegroup_number: int | None
    hermann_mauguin: str | None
    provenance: str


@dataclass(frozen=True)
class IrreducibleKMesh:
    """Regular reciprocal mesh partitioned into symmetry orbits.

    ``parent_indices[i]`` is the full-mesh index of point ``i``'s orbit
    representative, while ``orbit_indices[i]`` is its index in
    ``representative_points``.  The operation recorded for point ``i`` maps
    its representative onto that point; ``operation_time_reversals`` records
    whether the reciprocal action includes the antiunitary sign reversal.
    """

    divisions: tuple[int, int, int]
    shift: tuple[float, float, float]
    full_points: FloatArray
    representative_indices: IntArray
    representative_points: FloatArray
    parent_indices: IntArray
    orbit_indices: IntArray
    operation_indices: IntArray
    operation_time_reversals: NDArray[np.bool_]
    active_operation_indices: IntArray
    active_operation_time_reversals: NDArray[np.bool_]
    multiplicities: IntArray
    weights: FloatArray


def detect(
    lattice: FloatArray,
    positions: FloatArray,
    numbers: IntArray,
    *,
    symprec: float = 1.0e-5,
) -> SymmetryDataset:
    """Detect space-group symmetry with spglib.

    ``lattice`` rows are direct primitive vectors in Cartesian coordinates
    (any consistent length unit), ``positions`` are fractional, and equal
    ``numbers`` mark interchangeable sites.
    """
    cell = (
        np.asarray(lattice, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(numbers, dtype=np.int64),
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        raise ValueError(f"spglib found no symmetry for the cell at symprec={symprec}")
    return SymmetryDataset(
        rotations=dataset.rotations.astype(np.int64),
        translations=dataset.translations.astype(np.float64),
        time_reversals=np.zeros(len(dataset.rotations), dtype=np.bool_),
        equivalent_atoms=dataset.crystallographic_orbits.astype(np.int64),
        spacegroup_number=int(dataset.number),
        hermann_mauguin=dataset.international,
        provenance="spglib",
    )


def reduce_regular_kmesh(
    dataset: SymmetryDataset,
    divisions: tuple[int, int, int],
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    include_time_reversal: bool = True,
) -> IrreducibleKMesh:
    """Reduce a Gamma-centered or half-shifted regular mesh to the IBZ.

    The full mesh is ordered like ``itertools.product`` over its three axes.
    Only operations that preserve both ``divisions`` and ``shift`` are used.
    Reciprocal actions and mesh closure are evaluated with integer arithmetic;
    output points alone are converted to floating-point fractional coordinates
    normalized to ``[0, 1)``.

    A direct-space operation ``x -> R x + t`` sends a reciprocal row vector
    to ``k @ inv(R)``.  An antiunitary operation additionally changes its sign.
    When ``include_time_reversal`` is true, the dataset operations are enlarged
    by composition with pure time reversal.
    """
    mesh = _mesh_divisions(divisions)
    half_shift = _half_shift(shift)
    normalized_shift = tuple(component / 2.0 for component in half_shift)
    integer_points = np.asarray(
        tuple(product(*(range(size) for size in mesh))), dtype=np.int64
    )
    mesh_array = np.asarray(mesh, dtype=np.int64)
    doubled = 2 * integer_points + np.asarray(half_shift, dtype=np.int64)
    full_points = doubled.astype(np.float64) / (2.0 * mesh_array)

    actions: list[tuple[int, bool, IntArray]] = []
    for operation, rotation in enumerate(dataset.rotations):
        inverse = _inverse_unimodular(np.asarray(rotation, dtype=np.int64))
        antiunitary = bool(dataset.time_reversals[operation])
        actions.append((operation, antiunitary, inverse))
        if include_time_reversal:
            actions.append((operation, not antiunitary, inverse))

    identity = np.eye(3, dtype=np.int64)
    actions.sort(
        key=lambda action: (
            not (np.array_equal(dataset.rotations[action[0]], identity) and not action[1]),
            action[1],
            action[0],
        )
    )
    permutations: list[tuple[int, bool, IntArray]] = []
    scale = lcm(*mesh)
    scaled = doubled * (scale // mesh_array)
    for operation, antiunitary, inverse in actions:
        transformed = scaled @ inverse
        if antiunitary:
            transformed = -transformed
        numerators = transformed * mesh_array
        if np.any(numerators % scale):
            continue
        target_doubled = (numerators // scale) % (2 * mesh_array)
        if np.any(target_doubled % 2 != np.asarray(half_shift) % 2):
            continue
        target_integer = (target_doubled - np.asarray(half_shift)) // 2
        target_indices = (
            (target_integer[:, 0] * mesh[1] + target_integer[:, 1]) * mesh[2]
            + target_integer[:, 2]
        ).astype(np.int64)
        permutations.append((operation, antiunitary, target_indices))

    if not permutations:
        raise ValueError("no symmetry operation preserves the regular k mesh")

    images = np.stack([permutation for _, _, permutation in permutations])
    parent_indices = images.min(axis=0).astype(np.int64)
    representative_indices = np.unique(parent_indices).astype(np.int64)
    representative_lookup = {
        int(representative): orbit
        for orbit, representative in enumerate(representative_indices)
    }
    orbit_indices = np.asarray(
        [representative_lookup[int(parent)] for parent in parent_indices],
        dtype=np.int64,
    )

    operation_indices = np.empty(len(full_points), dtype=np.int64)
    operation_time_reversals = np.empty(len(full_points), dtype=np.bool_)
    for point, representative in enumerate(parent_indices):
        for operation, antiunitary, permutation in permutations:
            if permutation[representative] == point:
                operation_indices[point] = operation
                operation_time_reversals[point] = antiunitary
                break
        else:
            raise ValueError("symmetry operations do not form closed k-point orbits")

    multiplicities = np.bincount(
        orbit_indices, minlength=len(representative_indices)
    ).astype(np.int64)
    return IrreducibleKMesh(
        divisions=mesh,
        shift=normalized_shift,
        full_points=full_points,
        representative_indices=representative_indices,
        representative_points=full_points[representative_indices].copy(),
        parent_indices=parent_indices,
        orbit_indices=orbit_indices,
        operation_indices=operation_indices,
        operation_time_reversals=operation_time_reversals,
        active_operation_indices=np.asarray(
            [operation for operation, _, _ in permutations], dtype=np.int64
        ),
        active_operation_time_reversals=np.asarray(
            [antiunitary for _, antiunitary, _ in permutations], dtype=np.bool_
        ),
        multiplicities=multiplicities,
        weights=multiplicities.astype(np.float64) / len(full_points),
    )


def _mesh_divisions(divisions: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(divisions) != 3 or any(type(size) is not int or size <= 0 for size in divisions):
        raise ValueError("divisions must contain three positive ints")
    return divisions


def _half_shift(shift: tuple[float, float, float]) -> tuple[int, int, int]:
    if len(shift) != 3 or any(component not in (0, 0.0, 0.5) for component in shift):
        raise ValueError("shift components must be 0.0 (Gamma) or 0.5 (half shift)")
    return tuple(0 if component == 0 else 1 for component in shift)


def _inverse_unimodular(rotation: IntArray) -> IntArray:
    a, b, c = (int(value) for value in rotation[0])
    d, e, f = (int(value) for value in rotation[1])
    g, h, i = (int(value) for value in rotation[2])
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) != 1:
        raise ValueError("symmetry rotations must be unimodular 3x3 integer matrices")
    adjugate = np.asarray(
        [
            [e * i - f * h, c * h - b * i, b * f - c * e],
            [f * g - d * i, a * i - c * g, c * d - a * f],
            [d * h - e * g, b * g - a * h, a * e - b * d],
        ],
        dtype=np.int64,
    )
    return adjugate // determinant


def little_group_irreps(
    dataset: SymmetryDataset,
    kpoint: FloatArray,
    *,
    real: bool = False,
) -> tuple[list[ComplexArray], IntArray]:
    """Irreps of the little group at ``kpoint`` (fractional reciprocal).

    Returns ``(irreps, mapping_little_group)``: ``irreps[alpha][i]`` represents
    the little-group operation ``dataset.rotations[mapping_little_group[i]]``.
    """
    return spgrep.get_spacegroup_irreps_from_primitive_symmetry(
        dataset.rotations,
        dataset.translations,
        np.asarray(kpoint, dtype=np.float64),
        real=real,
    )


def little_group_spinor_irreps(
    lattice: FloatArray,
    dataset: SymmetryDataset,
    kpoint: FloatArray,
) -> tuple[list[ComplexArray], ComplexArray, ComplexArray, IntArray]:
    """Double-valued (spinor) irreps of the little group at ``kpoint``.

    ``lattice`` supplies the Cartesian frame for the SU(2) rotations. Returns
    ``(irreps, spinor_factor_system, unitary_rotations, mapping_little_group)``
    in spgrep's conventions.
    """
    return spgrep.get_spacegroup_spinor_irreps_from_primitive_symmetry(
        np.asarray(lattice, dtype=np.float64),
        dataset.rotations,
        dataset.translations,
        kpoint=np.asarray(kpoint, dtype=np.float64),
    )
