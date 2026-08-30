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

import numpy as np
import spglib
import spgrep

from .contracts import ComplexArray, FloatArray, IntArray


@dataclass(frozen=True)
class SymmetryDataset:
    """Method-neutral symmetry of the input cell; see the module docstring."""

    rotations: IntArray
    translations: FloatArray
    equivalent_atoms: IntArray
    spacegroup_number: int | None
    hermann_mauguin: str | None
    provenance: str


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
        equivalent_atoms=dataset.crystallographic_orbits.astype(np.int64),
        spacegroup_number=int(dataset.number),
        hermann_mauguin=dataset.international,
        provenance="spglib",
    )


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
