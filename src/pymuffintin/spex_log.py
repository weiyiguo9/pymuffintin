"""Parse SPEX stdout into the unified symmetry dataset.

SPEX prints its own symmetry analysis (including magnetic constraints and
time-reversal operations) into the standard output; this module recovers the
operation table, the atom basis, and the irreducible-BZ table from that text,
so SPEX's symmetry can be imported without rerunning detection and without
any SPEX-side dump. Space-group classification lines are deliberately
ignored. Wavefunction irreps (``irrep_sub``) never appear in stdout and stay
outside this parser.

The format is pinned against ``print_symmetries`` (SPEX ``src/symmetry.f``):
operations print in blocks of four, transposed across lines — the line of
operation $i$ carries row $k = ((i-1) \\bmod 4) + 1$ of all four rotation
matrices in the current block. Each line is a fixed-width prefix (operation
index, determinant, axis/angle, symmorphic flag, inverse index, and — for
noncollinear runs — a time-reversal flag) followed by matrix-row groups of
width 11, extended to 18 when any operation carries a nonzero translation
(fractions such as ``1/2`` print in a 7-character field; a blank field marks
an operation whose whole translation vanishes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from .contracts import FloatArray, IntArray
from .symmetry import SymmetryDataset

_OPERATION_COUNT = re.compile(r"Number of symmetry operations\s*=\s*(\d+)")
_ATOM_ROW = re.compile(
    r"^\s*\d+\s+\d+\s+\d+\s+[A-Z][a-z]?\s+(\d+)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
_KPOINT_TOTAL = re.compile(r"Number of k points:\s*(\d+)")
_KPOINT_IBZ = re.compile(r"in IBZ:\s*(\d+)")
_IBZ_ROW = re.compile(
    r"^\s*(\d+)\s+\(\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\)\s+eq:\s*(\d+)"
)
_SITE_MATCH_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class SpexLogSymmetry:
    """Symmetry content recovered from one SPEX stdout log."""

    rotations: IntArray
    translations: FloatArray
    time_reversals: NDArray[np.bool_]
    inverse: IntArray
    site_positions: FloatArray
    atomic_numbers: IntArray
    ibz_kpoints: FloatArray
    ibz_weights: IntArray
    kpoint_total: int

    def atom_map(self) -> IntArray:
        """``atom_map[i, a]``: the site operation ``i`` sends site ``a`` to."""
        n_ops = len(self.rotations)
        n_sites = len(self.site_positions)
        mapping = np.empty((n_ops, n_sites), dtype=np.int64)
        for i in range(n_ops):
            images = self.site_positions @ self.rotations[i].T + self.translations[i]
            for a in range(n_sites):
                delta = self.site_positions - images[a]
                delta -= np.rint(delta)
                distances = np.abs(delta).max(axis=1)
                b = int(np.argmin(distances))
                if distances[b] > _SITE_MATCH_TOLERANCE:
                    raise ValueError(
                        f"operation {i} maps site {a} to no site "
                        f"(closest residue {distances[b]:.2e})"
                    )
                mapping[i, a] = b
        return mapping

    def to_dataset(self) -> SymmetryDataset:
        """View as the method-neutral dataset; classification stays absent."""
        return SymmetryDataset(
            rotations=self.rotations.copy(),
            translations=self.translations.copy(),
            time_reversals=self.time_reversals.copy(),
            equivalent_atoms=self.atom_map().min(axis=0),
            spacegroup_number=None,
            hermann_mauguin=None,
            provenance="spex-log",
        )

    def to_spexsym_v1(self, path: str, producer_version: str) -> None:
        """Write a ``libmuffintin.spexsym`` v1 file readable by `muffintin_io`.

        The log carries the irreducible k-points only, so the k list is the
        IBZ (`irreducible_count` equals its length, every point is its own
        parent through the identity operation) and the irreps section is
        empty.
        """
        import h5py

        identity = np.eye(3, dtype=np.int64)
        candidates = [
            i
            for i in range(len(self.rotations))
            if np.array_equal(self.rotations[i], identity)
            and not self.translations[i].any()
            and not self.time_reversals[i]
        ]
        if not candidates:
            raise ValueError("no identity operation in the parsed table")
        identity_index = candidates[0]

        str_dtype = h5py.string_dtype(encoding="utf-8")

        def dataset(group, name, data, dtype, axes):
            written = group.create_dataset(name, data=np.asarray(data, dtype=dtype))
            written.attrs.create("axes", np.array(axes, dtype=str_dtype))

        n_ibz = len(self.ibz_kpoints)
        with h5py.File(path, "w") as file:
            file.attrs.create("schema_name", "libmuffintin.spexsym", dtype=str_dtype)
            file.attrs.create("schema_version", np.uint32(1))
            file.attrs.create("producer_version", producer_version, dtype=str_dtype)
            symmetry = file.create_group("symmetry")
            dataset(
                symmetry, "rotations", self.rotations, np.int32,
                ["operation", "row", "column"],
            )
            dataset(
                symmetry, "translations", self.translations, np.float64,
                ["operation", "fractional"],
            )
            dataset(
                symmetry, "time_reversal", self.time_reversals, np.int32,
                ["operation"],
            )
            dataset(symmetry, "inverse", self.inverse, np.int32, ["operation"])
            dataset(
                symmetry, "atom_map", self.atom_map(), np.int32,
                ["operation", "site"],
            )
            kpoints = file.create_group("kpoints")
            kpoints.attrs.create("irreducible_count", np.int64(n_ibz))
            dataset(
                kpoints, "fractional", self.ibz_kpoints, np.float64,
                ["kpoint", "fractional"],
            )
            dataset(kpoints, "parent", np.arange(n_ibz), np.int32, ["kpoint"])
            dataset(
                kpoints, "parent_operation",
                np.full(n_ibz, identity_index), np.int32, ["kpoint"],
            )
            irreps = file.create_group("irreps")
            irreps.attrs.create("block_count", np.int64(0))


def parse_spex_log(text: str) -> SpexLogSymmetry:
    """Parse one SPEX stdout log; see the module docstring for the format."""
    lines = text.splitlines()
    rotations, translations, time_reversals, inverse = _parse_operations(lines)
    atomic_numbers, site_positions = _parse_atom_basis(lines)
    ibz_kpoints, ibz_weights, kpoint_total = _parse_ibz(lines)
    return SpexLogSymmetry(
        rotations=rotations,
        translations=translations,
        time_reversals=time_reversals,
        inverse=inverse,
        site_positions=site_positions,
        atomic_numbers=atomic_numbers,
        ibz_kpoints=ibz_kpoints,
        ibz_weights=ibz_weights,
        kpoint_total=kpoint_total,
    )


def _parse_operations(
    lines: list[str],
) -> tuple[IntArray, FloatArray, NDArray[np.bool_], IntArray]:
    start = None
    n_ops = 0
    for index, line in enumerate(lines):
        match = _OPERATION_COUNT.search(line)
        if match:
            start = index
            n_ops = int(match.group(1))
            break
    if start is None:
        raise ValueError("no 'Number of symmetry operations' line in the log")
    header = lines[start + 1]
    noncoll = " TRS " in header
    ltransl = header.rstrip().endswith("trsl")
    prefix = 47 + (5 if noncoll else 0)
    stride = 11 + (7 if ltransl else 0)

    rotations = np.zeros((n_ops, 3, 3), dtype=np.int64)
    translations = np.zeros((n_ops, 3), dtype=np.float64)
    filled = np.zeros((n_ops, 3), dtype=np.bool_)
    time_reversals = np.zeros(n_ops, dtype=np.bool_)
    inverse = np.zeros(n_ops, dtype=np.int64)

    def read_groups(line: str, block_start: int, row: int) -> None:
        section = line[prefix:].ljust(4 * stride)
        for offset, j in enumerate(range(block_start, min(block_start + 4, n_ops))):
            group = section[offset * stride : (offset + 1) * stride]
            rotations[j, row] = [
                int(group[2:5]),
                int(group[5:8]),
                int(group[8:11]),
            ]
            if ltransl:
                translations[j, row] = _parse_translation_field(group[11:18])
            filled[j, row] = True

    for i in range(1, n_ops + 1):
        line = lines[start + 1 + i]
        if int(line[2:4]) != i:
            raise ValueError(f"operation line {i} out of order: {line!r}")
        inverse[i - 1] = int(line[41:46]) - 1
        if noncoll:
            time_reversals[i - 1] = line[46:51].strip() == "Yes"
        k = (i - 1) % 4
        if k <= 2:
            read_groups(line, ((i - 1) // 4) * 4, k)
    remainder = n_ops % 4
    extra = start + 2 + n_ops
    last_block = (n_ops - 1) // 4 * 4
    if remainder == 1:
        read_groups(lines[extra], last_block, 1)
        read_groups(lines[extra + 1], last_block, 2)
    elif remainder == 2:
        read_groups(lines[extra], last_block, 2)
    if not filled.all():
        raise ValueError("incomplete rotation-matrix table in the log")
    return rotations, translations, time_reversals, inverse


def _parse_translation_field(field: str) -> float:
    token = field.strip()
    if not token or token == "0":
        return 0.0
    if "/" in token:
        return float(Fraction(token))
    return float(token)


def _parse_atom_basis(lines: list[str]) -> tuple[IntArray, FloatArray]:
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Atom basis:")
    except StopIteration:
        raise ValueError("no 'Atom basis:' block in the log") from None
    numbers: list[int] = []
    positions: list[list[float]] = []
    for line in lines[start + 1 :]:
        if line.lstrip().startswith(("SPEX-INFO", "SPEX-WARNING")):
            continue
        match = _ATOM_ROW.match(line)
        if match:
            numbers.append(int(match.group(1)))
            positions.append([float(match.group(g)) for g in (2, 3, 4)])
            continue
        if positions and "in absolute" in line:
            break
        if positions:
            break
    if not positions:
        raise ValueError("empty 'Atom basis:' table in the log")
    return (
        np.array(numbers, dtype=np.int64),
        np.array(positions, dtype=np.float64),
    )


def _parse_ibz(lines: list[str]) -> tuple[FloatArray, IntArray, int]:
    total = None
    n_ibz = None
    table = None
    for index, line in enumerate(lines):
        if total is None:
            match = _KPOINT_TOTAL.search(line)
            if match:
                total = int(match.group(1))
            continue
        if n_ibz is None:
            match = _KPOINT_IBZ.search(line)
            if match:
                n_ibz = int(match.group(1))
            continue
        if "k points of IBZ:" in line:
            table = index + 1
            break
    if total is None or n_ibz is None or table is None:
        raise ValueError("no irreducible k-point table in the log")
    kpoints = np.empty((n_ibz, 3), dtype=np.float64)
    weights = np.empty(n_ibz, dtype=np.int64)
    for row in range(n_ibz):
        match = _IBZ_ROW.match(lines[table + row])
        if not match:
            raise ValueError(f"malformed IBZ row {row + 1}: {lines[table + row]!r}")
        if int(match.group(1)) != row + 1:
            raise ValueError(f"IBZ row {row + 1} out of order")
        kpoints[row] = [float(match.group(g)) for g in (2, 3, 4)]
        weights[row] = int(match.group(5))
    if int(weights.sum()) != total:
        raise ValueError(
            f"IBZ weights sum to {int(weights.sum())}, expected {total} k points"
        )
    return kpoints, weights, total
