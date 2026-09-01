"""Unified crystal-symmetry dataset over spglib detection and spgrep irreps.

``SymmetryDataset`` mirrors the Rust ``muffintin_symmetry`` IR: rotations are
integer matrices in the input-cell fractional basis acting as
``rotations[i] @ x + translations[i]``, translations are fractional, and
``equivalent_atoms[i]`` is the representative site of site ``i``'s
crystallographic orbit. Backends (spglib here, moyo in the Rust core, SPEX
import) populate the same dataset, so analysis code never depends on a
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

    The storage is array-oriented, but ``representative_index`` and ``parent``
    follow Rust ``FullKPoint`` semantics: the former is the representative's
    full-mesh linear index for each point, while the latter indexes
    ``irreducible_points`` and can expand irreducible values over the full BZ.
    """

    divisions: tuple[int, int, int]
    shift: tuple[float, float, float]
    full_points: FloatArray
    representative_index: IntArray
    parent: IntArray
    active_operation_indices: IntArray
    multiplicities: IntArray

    @property
    def irreducible_points(self) -> FloatArray:
        """Orbit representatives in irreducible-point order."""
        return self.full_points[np.unique(self.representative_index)]

    @property
    def weights(self) -> FloatArray:
        """Normalized irreducible weights."""
        return self.multiplicities.astype(np.float64) / len(self.full_points)


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
    mesh = _validated_divisions(divisions)
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
    representative_index = images.min(axis=0).astype(np.int64)
    irreducible_indices = np.unique(representative_index).astype(np.int64)
    representative_lookup = {
        int(representative): orbit
        for orbit, representative in enumerate(irreducible_indices)
    }
    parent = np.asarray(
        [representative_lookup[int(representative)] for representative in representative_index],
        dtype=np.int64,
    )

    multiplicities = np.bincount(
        parent, minlength=len(irreducible_indices)
    ).astype(np.int64)
    return IrreducibleKMesh(
        divisions=mesh,
        shift=normalized_shift,
        full_points=full_points,
        representative_index=representative_index,
        parent=parent,
        active_operation_indices=np.unique(
            np.asarray([operation for operation, _, _ in permutations], dtype=np.int64)
        ),
        multiplicities=multiplicities,
    )


def _validated_divisions(divisions: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(divisions) != 3 or any(
        not isinstance(size, (int, np.integer)) or size <= 0 for size in divisions
    ):
        raise ValueError("divisions must contain three positive ints")
    return tuple(int(size) for size in divisions)


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
