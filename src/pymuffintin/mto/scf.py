"""Python-owned input preparation for self-consistent NMTO calculations.

The TOML and object constructors meet at :class:`NmtoScfInput`.  This module
does not turn the existing frozen-potential NMTO construction into an SCF
solver: the current-potential radial station and the regional NMTO density
assembler remain separate implementation steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import FloatArray, IntArray
from ..symmetry import SymmetryDataset, detect

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def _float_tuple(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} values")
    return result


def _int_tuple(values: Sequence[int], size: int, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} values")
    return result


@dataclass(frozen=True)
class NmtoScfSettings:
    """Algorithm settings shared by direct-Python and TOML inputs.

    Energies and temperatures are in Hartree.  Crystal symmetry is enabled by
    default and is detected once while preparing :class:`NmtoScfInput`.
    """

    electron_count: float
    energy_mesh: tuple[float, ...]
    k_mesh: tuple[int, int, int]
    k_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    l_max: int = 2
    temperature: float = 0.02
    state_degeneracy: float = 2.0
    xc: str = "lda-pw92"
    mixing: float = 0.3
    energy_tolerance: float = 1.0e-5
    density_tolerance: float = 1.0e-5
    max_iterations: int = 40
    minimum_cells: int = 135
    symmetry: bool = True
    symprec: float = 1.0e-5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "energy_mesh",
            tuple(float(energy) for energy in self.energy_mesh),
        )
        object.__setattr__(self, "k_mesh", _int_tuple(self.k_mesh, 3, "k_mesh"))
        object.__setattr__(self, "k_shift", _float_tuple(self.k_shift, 3, "k_shift"))
        if len(self.energy_mesh) < 2 or len(set(self.energy_mesh)) != len(self.energy_mesh):
            raise ValueError("energy_mesh must contain at least two distinct energies")
        if any(size <= 0 for size in self.k_mesh):
            raise ValueError("k_mesh entries must be positive")
        if self.electron_count <= 0.0:
            raise ValueError("electron_count must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.state_degeneracy <= 0.0:
            raise ValueError("state_degeneracy must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.minimum_cells <= 0:
            raise ValueError("minimum_cells must be positive")
        if self.symprec <= 0.0:
            raise ValueError("symprec must be positive")

    @classmethod
    def from_task(cls, task: Mapping[str, Any]) -> NmtoScfSettings:
        """Read the NMTO additions to one Input-V3 ``dft-scf`` task."""

        nmto = task.get("nmto")
        if not isinstance(nmto, Mapping) or "energy-mesh" not in nmto:
            raise ValueError("task.scf.nmto.energy-mesh is required for NMTO SCF")
        symmetry = task.get("symmetry", {})
        return cls(
            electron_count=float(task["electron-count"]),
            energy_mesh=tuple(float(value) for value in nmto["energy-mesh"]),
            k_mesh=_int_tuple(task["k-mesh"]["mesh"], 3, "task.scf.k-mesh.mesh"),
            k_shift=_float_tuple(
                task["k-mesh"].get("shift", (0.0, 0.0, 0.0)),
                3,
                "task.scf.k-mesh.shift",
            ),
            l_max=int(task["basis"]["l-max"]),
            temperature=float(task["occupations"]["temperature"]),
            state_degeneracy=float(nmto.get("state-degeneracy", 2.0)),
            xc=str(task["xc"]["kind"]),
            mixing=float(task["mixing"]["beta"]),
            energy_tolerance=float(task["convergence"]["energy-tolerance"]),
            density_tolerance=float(task["convergence"]["density-tolerance"]),
            max_iterations=int(task["convergence"]["max-iterations"]),
            minimum_cells=int(nmto.get("minimum-cells", 135)),
            symmetry=bool(symmetry.get("enabled", True)),
            symprec=float(symmetry.get("symprec", 1.0e-5)),
        )


@dataclass(frozen=True)
class NmtoScfInput:
    """Prepared native context plus method-neutral crystal metadata."""

    structure: object
    field_layout: object
    initial_density: object
    lattice: FloatArray
    site_ids: tuple[str, ...]
    atomic_numbers: IntArray
    fractional_positions: FloatArray
    muffin_tin_radii: FloatArray
    settings: NmtoScfSettings
    checkpoint: object | None = None
    symmetry_dataset: SymmetryDataset | None = field(init=False)

    def __post_init__(self) -> None:
        lattice = np.asarray(self.lattice, dtype=np.float64)
        positions = np.asarray(self.fractional_positions, dtype=np.float64)
        numbers = np.asarray(self.atomic_numbers, dtype=np.int64)
        radii = np.asarray(self.muffin_tin_radii, dtype=np.float64)
        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape (3, 3)")
        if positions.shape != (len(self.site_ids), 3):
            raise ValueError("fractional_positions must have shape (n_site, 3)")
        if numbers.shape != (len(self.site_ids),) or radii.shape != (len(self.site_ids),):
            raise ValueError("atomic_numbers and muffin_tin_radii must match site_ids")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "fractional_positions", positions)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "muffin_tin_radii", radii)
        dataset = (
            detect(lattice, positions, numbers, symprec=self.settings.symprec)
            if self.settings.symmetry
            else None
        )
        object.__setattr__(self, "symmetry_dataset", dataset)

    @classmethod
    def from_python(
        cls,
        *,
        structure: object,
        field_layout: object,
        initial_density: object,
        lattice: Sequence[Sequence[float]],
        site_ids: Sequence[str],
        atomic_numbers: Sequence[int],
        fractional_positions: Sequence[Sequence[float]],
        muffin_tin_radii: Sequence[float],
        settings: NmtoScfSettings,
        checkpoint: object | None = None,
    ) -> NmtoScfInput:
        """Prepare an input without serializing a checkpoint or SCF TOML file."""

        return cls(
            structure=structure,
            field_layout=field_layout,
            initial_density=initial_density,
            lattice=np.asarray(lattice, dtype=np.float64),
            site_ids=tuple(site_ids),
            atomic_numbers=np.asarray(atomic_numbers, dtype=np.int64),
            fractional_positions=np.asarray(fractional_positions, dtype=np.float64),
            muffin_tin_radii=np.asarray(muffin_tin_radii, dtype=np.float64),
            settings=settings,
            checkpoint=checkpoint,
        )

    @classmethod
    def from_toml(
        cls,
        path: str | Path,
        *,
        native: ModuleType | None = None,
    ) -> NmtoScfInput:
        """Prepare the same input from Input-V3 and its referenced checkpoint."""

        input_path = Path(path)
        with input_path.open("rb") as stream:
            document = tomllib.load(stream)
        task = _single_dft_scf_task(document)
        settings = NmtoScfSettings.from_task(task)

        checkpoint_path = input_path.parent / document["checkpoint"]
        with checkpoint_path.open("rb") as stream:
            checkpoint_document = tomllib.load(stream)
        native = import_module("libmuffintin") if native is None else native
        checkpoint = native.load_checkpoint(checkpoint_path)
        physics = native.CheckpointPhysics(checkpoint)
        initial_density = physics.restart_density()
        if initial_density is None:
            raise ValueError("NMTO SCF requires a checkpoint restart density")

        geometry = _checkpoint_geometry(checkpoint_document)
        structure = native.Structure(**geometry["structure"])
        density = checkpoint_document["initial"]["density"]
        charge = density["n"]
        g_vectors = [entry["g"] for entry in charge["interstitial"]["coefficients"]]
        muffin_tin_l_max = max(
            int(channel["l"])
            for site in charge["muffin_tins"]
            for channel in site["channels"]
        )
        field_layout = native.RegionalFieldLayout(
            structure,
            g_vectors=g_vectors,
            muffin_tin_l_max=muffin_tin_l_max,
        )
        return cls.from_python(
            structure=structure,
            field_layout=field_layout,
            initial_density=initial_density,
            lattice=geometry["lattice"],
            site_ids=geometry["site_ids"],
            atomic_numbers=geometry["atomic_numbers"],
            fractional_positions=geometry["fractional_positions"],
            muffin_tin_radii=geometry["muffin_tin_radii"],
            settings=settings,
            checkpoint=checkpoint,
        )


def _single_dft_scf_task(document: Mapping[str, Any]) -> Mapping[str, Any]:
    tasks = document["task"]
    selected = [
        tasks[name]
        for name in document["workflow"]["tasks"]
        if tasks[name].get("kind") == "dft-scf"
    ]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one dft-scf task, found {len(selected)}")
    return selected[0]


def _checkpoint_geometry(document: Mapping[str, Any]) -> dict[str, Any]:
    geometry = document["geometry"]
    lattice = geometry["lattice"]["vectors"]
    sites = geometry["sites"]
    radial_by_site = {basis["site_id"]: basis for basis in geometry["radial_basis"]}
    radial_meshes = []
    radial_equations = []
    linearization_energies = []
    for site in sites:
        basis = radial_by_site[site["id"]]
        mesh = basis["mesh"]
        radial_meshes.append(
            (float(mesh["first"]), float(mesh["log_increment"]), int(mesh["point_count"]))
        )
        radial_equations.append(str(basis["radial_equation"]))
        linearization_energies.append(
            [
                (int(value["l"]), float(value["energy"]))
                for value in basis["linearization"]["linearization_energies"]
            ]
        )
    site_ids = tuple(str(site["id"]) for site in sites)
    atomic_numbers = tuple(int(site["atomic_number"]) for site in sites)
    fractional_positions = tuple(site["fractional_position"] for site in sites)
    muffin_tin_radii = tuple(float(site["muffin_tin_radius"]) for site in sites)
    return {
        "lattice": lattice,
        "site_ids": site_ids,
        "atomic_numbers": atomic_numbers,
        "fractional_positions": fractional_positions,
        "muffin_tin_radii": muffin_tin_radii,
        "structure": {
            "lattice": lattice,
            "site_ids": list(site_ids),
            "atomic_numbers": list(atomic_numbers),
            "fractional_positions": list(fractional_positions),
            "radial_meshes": radial_meshes,
            "radial_equations": radial_equations,
            "linearization_energies": linearization_energies,
        },
    }
