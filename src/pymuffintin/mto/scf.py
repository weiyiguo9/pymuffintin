"""Python-owned self-consistent scalar NMTO calculation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from importlib import import_module
from itertools import product
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence

import numpy as np
from ..contracts import FloatArray, IntArray
from ..symmetry import IrreducibleKMesh, SymmetryDataset, detect, reduce_regular_kmesh
from .density import (
    ScalarRadialSamples,
    assemble_nmto_regional_density,
)
from .electrons import (
    NmtoBands,
    NmtoOccupations,
    fermi_dirac_occupations,
    solve_nmto_bands,
)
from .kink import BoundaryJets, build_kink_mesh
from .nmto import LowdinResult, NmtoResult, build_nmto
from .parallel import NmtoParallel
from .periodic import PeriodicUswGeometry, PeriodicUswSample
from .periodic_density import PeriodicNmtoBasisEvaluator
from .full_potential import full_potential_corrections
from .omt import nearest_neighbor_distance, overlap_fractions
from .potential import (
    build_nmto_potential,
    sample_nmto_radials,
    sample_omt_radials,
    solve_nmto_core,
)
from .shell import snap_to_exponential_mesh
from .usw import RealHarmonic

if TYPE_CHECKING:
    from mpi4py import MPI

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


REFERENCE_POTENTIALS = frozenset({"spherical-mt", "omt"})


@dataclass(frozen=True)
class NmtoScfSettings:
    """Algorithm settings shared by direct-Python and TOML inputs.

    Energies and temperatures are in Hartree.  Crystal symmetry is enabled by
    default and is detected once while preparing :class:`NmtoScfInput`.
    """

    electron_count: float
    energy_mesh: tuple[float, ...]
    k_mesh: tuple[int, int, int]
    reciprocal_cutoff: float
    lattice_sum_radius: float
    reference_energy: float
    mixing_kind: Literal["linear", "pulay"]
    mixing_history: int | None
    matrix_angular_order: int
    k_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    l_max: int = 2
    temperature: float = 0.02
    state_degeneracy: float = 2.0
    xc: str = "lda-pw92"
    mixing: float = 0.3
    energy_tolerance: float = 1.0e-5
    density_tolerance: float = 1.0e-5
    max_iterations: int = 40
    symmetry: bool = True
    symprec: float = 1.0e-5
    include_time_reversal: bool = True
    reference_potential: str = "spherical-mt"
    potential_radius_scale: float = 1.0
    potential_radii: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "energy_mesh",
            tuple(float(energy) for energy in self.energy_mesh),
        )
        object.__setattr__(self, "k_mesh", _int_tuple(self.k_mesh, 3, "k_mesh"))
        object.__setattr__(self, "k_shift", _float_tuple(self.k_shift, 3, "k_shift"))
        if not self.energy_mesh or len(set(self.energy_mesh)) != len(self.energy_mesh):
            raise ValueError("energy_mesh must contain at least one distinct energy")
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
        if self.reciprocal_cutoff <= 0 or self.lattice_sum_radius <= 0:
            raise ValueError("periodic USW cutoffs must be positive")
        if self.reference_energy >= 0:
            raise ValueError("periodic USW reference energy must be negative")
        if self.matrix_angular_order <= 0:
            raise ValueError("matrix angular quadrature order must be positive")
        if self.mixing_kind not in ("linear", "pulay"):
            raise ValueError("NMTO mixing kind must be linear or pulay")
        if self.mixing_kind == "pulay" and (
            self.mixing_history is None or self.mixing_history <= 0
        ):
            raise ValueError("Pulay mixing requires a positive history length")
        if self.symprec <= 0.0:
            raise ValueError("symprec must be positive")
        if self.reference_potential not in REFERENCE_POTENTIALS:
            raise ValueError(
                f"reference_potential must be one of {sorted(REFERENCE_POTENTIALS)}"
            )
        if self.potential_radius_scale < 1.0:
            raise ValueError("potential_radius_scale must not be below one")
        if self.potential_radii is not None:
            object.__setattr__(
                self,
                "potential_radii",
                {str(key): float(value) for key, value in self.potential_radii.items()},
            )
        if self.reference_potential != "omt" and (
            self.potential_radius_scale != 1.0 or self.potential_radii
        ):
            raise ValueError(
                "potential spheres beyond the hard spheres require reference_potential='omt'"
            )

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
            mixing_kind=task["mixing"]["kind"],
            mixing_history=(
                int(task["mixing"]["history"])
                if task["mixing"]["kind"] == "pulay" else None
            ),
            energy_tolerance=float(task["convergence"]["energy-tolerance"]),
            density_tolerance=float(task["convergence"]["density-tolerance"]),
            max_iterations=int(task["convergence"]["max-iterations"]),
            reciprocal_cutoff=float(nmto["reciprocal-cutoff"]),
            lattice_sum_radius=float(nmto["lattice-sum-radius"]),
            reference_energy=float(nmto["reference-energy"]),
            matrix_angular_order=int(nmto["matrix-angular-order"]),
            reference_potential=str(nmto.get("reference-potential", "spherical-mt")),
            potential_radius_scale=float(nmto.get("potential-radius-scale", 1.0)),
            potential_radii=(
                None
                if nmto.get("potential-radii") is None
                else dict(nmto["potential-radii"])
            ),
            symmetry=bool(symmetry.get("enabled", True)),
            symprec=float(symmetry.get("symprec", 1.0e-5)),
            include_time_reversal=bool(symmetry.get("include-time-reversal", True)),
        )


@dataclass(frozen=True)
class NmtoScfInput:
    """Prepared native context plus method-neutral crystal metadata."""

    native: ModuleType
    structure: object
    field_layout: object
    initial_density: object
    core_station: object
    lattice: FloatArray
    site_ids: tuple[str, ...]
    atomic_numbers: IntArray
    fractional_positions: FloatArray
    muffin_tin_radii: FloatArray
    g_vectors: IntArray
    density_l_max: int
    radial_equations: tuple[str, ...]
    settings: NmtoScfSettings
    checkpoint: object | None = None
    potential_sphere_radii: FloatArray | None = None
    symmetry_dataset: SymmetryDataset | None = field(init=False)
    k_mesh_reduction: IrreducibleKMesh | None = field(init=False)

    def __post_init__(self) -> None:
        lattice = np.asarray(self.lattice, dtype=np.float64)
        positions = np.asarray(self.fractional_positions, dtype=np.float64)
        numbers = np.asarray(self.atomic_numbers, dtype=np.int64)
        radii = np.asarray(self.muffin_tin_radii, dtype=np.float64)
        g_vectors = np.asarray(self.g_vectors, dtype=np.int64)
        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape (3, 3)")
        if positions.shape != (len(self.site_ids), 3):
            raise ValueError("fractional_positions must have shape (n_site, 3)")
        if numbers.shape != (len(self.site_ids),) or radii.shape != (len(self.site_ids),):
            raise ValueError("atomic_numbers and muffin_tin_radii must match site_ids")
        if g_vectors.ndim != 2 or g_vectors.shape[1] != 3:
            raise ValueError("g_vectors must have shape (n_g, 3)")
        if len(self.radial_equations) != len(self.site_ids):
            raise ValueError("radial_equations must contain one value per site")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "fractional_positions", positions)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "muffin_tin_radii", radii)
        object.__setattr__(self, "g_vectors", g_vectors)
        potential_radii = (
            radii.copy()
            if self.potential_sphere_radii is None
            else np.asarray(self.potential_sphere_radii, dtype=np.float64)
        )
        if potential_radii.shape != radii.shape:
            raise ValueError("potential_sphere_radii must contain one value per site")
        if np.any(potential_radii < radii - 1.0e-12):
            raise ValueError(
                "potential sphere radii must not be smaller than the hard-sphere radii"
            )
        if self.settings.reference_potential != "omt" and np.any(
            potential_radii > radii + 1.0e-12
        ):
            raise ValueError(
                "potential spheres beyond the hard spheres require reference_potential='omt'"
            )
        object.__setattr__(self, "potential_sphere_radii", potential_radii)
        dataset = (
            detect(lattice, positions, numbers, symprec=self.settings.symprec)
            if self.settings.symmetry
            else None
        )
        object.__setattr__(self, "symmetry_dataset", dataset)
        object.__setattr__(
            self,
            "k_mesh_reduction",
            None
            if dataset is None
            else reduce_regular_kmesh(
                dataset,
                self.settings.k_mesh,
                self.settings.k_shift,
                include_time_reversal=self.settings.include_time_reversal,
            ),
        )
        if self.k_mesh_reduction is not None:
            _validate_symmetry_layout(self)

    @classmethod
    def from_python(
        cls,
        *,
        native: ModuleType,
        structure: object,
        field_layout: object,
        initial_density: object,
        core_station: object,
        lattice: Sequence[Sequence[float]],
        site_ids: Sequence[str],
        atomic_numbers: Sequence[int],
        fractional_positions: Sequence[Sequence[float]],
        muffin_tin_radii: Sequence[float],
        g_vectors: Sequence[Sequence[int]],
        density_l_max: int,
        radial_equations: Sequence[str],
        settings: NmtoScfSettings,
        checkpoint: object | None = None,
        potential_sphere_radii: Sequence[float] | None = None,
    ) -> NmtoScfInput:
        """Prepare an input without serializing a checkpoint or SCF TOML file.

        ``potential_sphere_radii`` are the overlapping-muffin-tin potential
        radii ``s``; they default to the hard-sphere radii and must be exact
        points of each site's exponential radial mesh when larger.
        """

        return cls(
            native=native,
            structure=structure,
            field_layout=field_layout,
            initial_density=initial_density,
            core_station=core_station,
            lattice=np.asarray(lattice, dtype=np.float64),
            site_ids=tuple(site_ids),
            atomic_numbers=np.asarray(atomic_numbers, dtype=np.int64),
            fractional_positions=np.asarray(fractional_positions, dtype=np.float64),
            muffin_tin_radii=np.asarray(muffin_tin_radii, dtype=np.float64),
            g_vectors=np.asarray(g_vectors, dtype=np.int64),
            density_l_max=int(density_l_max),
            radial_equations=tuple(radial_equations),
            settings=settings,
            checkpoint=checkpoint,
            potential_sphere_radii=(
                None
                if potential_sphere_radii is None
                else np.asarray(potential_sphere_radii, dtype=np.float64)
            ),
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
        structure = physics.structure()
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
        core_station = _core_station(native, task, geometry["site_ids"])
        return cls.from_python(
            native=native,
            structure=structure,
            field_layout=field_layout,
            initial_density=initial_density,
            core_station=core_station,
            lattice=geometry["lattice"],
            site_ids=geometry["site_ids"],
            atomic_numbers=geometry["atomic_numbers"],
            fractional_positions=geometry["fractional_positions"],
            muffin_tin_radii=geometry["muffin_tin_radii"],
            g_vectors=g_vectors,
            density_l_max=muffin_tin_l_max,
            radial_equations=geometry["radial_equations"],
            settings=settings,
            checkpoint=checkpoint,
            potential_sphere_radii=resolve_potential_sphere_radii(
                settings,
                geometry["site_ids"],
                geometry["muffin_tin_radii"],
                geometry["structure"]["radial_meshes"],
            ),
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


def resolve_potential_sphere_radii(
    settings: NmtoScfSettings,
    site_ids: Sequence[str],
    muffin_tin_radii: Sequence[float],
    radial_meshes: Sequence[tuple[float, float, int]],
) -> FloatArray:
    """Return per-site potential radii snapped up to each exponential mesh.

    A species entry in ``settings.potential_radii`` (keyed by the site-id
    prefix before ``-``) takes precedence over ``potential_radius_scale``
    times the hard-sphere radius.  Requests not above the hard sphere leave
    the site without a shell; larger requests are rounded up to the next
    native mesh point so the shell knots are mesh points.
    """

    hard = np.asarray(muffin_tin_radii, dtype=np.float64)
    resolved = hard.copy()
    for site, site_id in enumerate(site_ids):
        species = str(site_id).split("-", 1)[0]
        requested = (
            settings.potential_radii[species]
            if settings.potential_radii is not None and species in settings.potential_radii
            else settings.potential_radius_scale * hard[site]
        )
        if requested > hard[site] + 1.0e-12:
            first, increment, _ = radial_meshes[site]
            resolved[site] = snap_to_exponential_mesh(float(first), float(increment), float(requested))
    return resolved


def _validate_symmetry_layout(scf_input: NmtoScfInput) -> None:
    dataset = scf_input.symmetry_dataset
    reduction = scf_input.k_mesh_reduction
    assert dataset is not None and reduction is not None
    operations = np.unique(reduction.active_operation_indices)
    vectors = {tuple(vector) for vector in scf_input.g_vectors}
    for operation in operations:
        rotation = dataset.rotations[operation]
        translation = dataset.translations[operation]
        for vector in scf_input.g_vectors:
            source = tuple(rotation.T @ vector)
            if source not in vectors:
                raise ValueError(
                    f"g_vectors are not closed under symmetry operation {operation}: "
                    f"missing {source}"
                )
        images = np.mod(
            scf_input.fractional_positions @ rotation.T + translation,
            1.0,
        )
        for source, image in enumerate(images):
            delta = scf_input.fractional_positions - image
            delta -= np.rint(delta)
            distances = np.linalg.norm(delta @ scf_input.lattice, axis=1)
            target = int(np.argmin(distances))
            if distances[target] > scf_input.settings.symprec:
                raise ValueError(
                    f"symmetry operation {operation} does not map site {source}"
                )
            if (
                scf_input.atomic_numbers[source] != scf_input.atomic_numbers[target]
                or abs(
                    scf_input.muffin_tin_radii[source]
                    - scf_input.muffin_tin_radii[target]
                )
                > scf_input.settings.symprec
                or abs(
                    scf_input.potential_sphere_radii[source]
                    - scf_input.potential_sphere_radii[target]
                )
                > scf_input.settings.symprec
                or scf_input.radial_equations[source]
                != scf_input.radial_equations[target]
            ):
                raise ValueError(
                    f"symmetry operation {operation} maps incompatible sites "
                    f"{source} and {target}"
                )


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
        "radial_equations": tuple(radial_equations),
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


def _core_station(
    native: ModuleType,
    task: Mapping[str, Any],
    site_ids: Sequence[str],
) -> object:
    recipes = task["basis"].get("channels", {})
    sites = []
    for site_index, site_id in enumerate(site_ids):
        species = site_id.split("-", 1)[0]
        recipe = recipes.get(species, {})
        states = []
        for token in recipe.get("core", ()):
            n = int(token[:-1])
            l = "spdfg".index(token[-1])
            if l == 0:
                states.append(native.CoreState(n, -1, 2.0))
            else:
                states.append(native.CoreState(n, l, float(2 * l)))
                states.append(native.CoreState(n, -(l + 1), float(2 * (l + 1))))
        sites.append(native.CoreSite(site_index, site_id, states))
    return native.CoreStation(sites)


@dataclass(frozen=True)
class NmtoScfResult:
    """Converged self-consistent NMTO state and its iteration history."""

    iterations: int
    total_energy: float
    chemical_potential: float
    density: object
    potential: object
    bands: NmtoBands
    occupations: NmtoOccupations
    energy_history: FloatArray
    convergence_history: FloatArray
    valence_normalization_history: FloatArray
    reference_constant_history: FloatArray
    reference_rms_history: FloatArray
    k_sampling: IrreducibleKMesh | None
    _restart_checkpoint: object | None

    def restart_checkpoint(self) -> object:
        if self._restart_checkpoint is None:
            raise ValueError("NMTO SCF input has no checkpoint context")
        return self._restart_checkpoint


@dataclass(frozen=True)
class _NmtoIteration:
    bands: NmtoBands
    occupations: NmtoOccupations
    output_density: object | None
    valence_normalization: float
    reference_constant: float = np.nan
    reference_rms: float = np.nan


def _record_scf_timing(
    callback: Callable[[int, str, float], None] | None,
    iteration: int,
    phase: str,
    started: float,
    parallel: NmtoParallel | None,
) -> None:
    if parallel is None:
        if callback is not None:
            callback(iteration, phase, perf_counter() - started)
        return
    with parallel.local_stage():
        if parallel.rank == 0 and callback is not None:
            callback(iteration, phase, perf_counter() - started)


def run_nmto_scf(
    scf_input: NmtoScfInput | str | Path,
    *,
    comm: MPI.Comm | None = None,
    timing_callback: Callable[[int, str, float], None] | None = None,
) -> NmtoScfResult | None:
    """Run the full scalar NMTO density/potential/mixing loop in Python.

    When ``comm`` is supplied, Python schedules fixed potential, core, radial,
    energy, k-point, and density blocks across its ranks. Native block kernels
    write disjoint node-shared outputs; deterministic object assembly, energy,
    mixing, restart construction, and the result remain root-only. All other
    ranks return ``None``.
    """

    if not isinstance(scf_input, NmtoScfInput):
        scf_input = NmtoScfInput.from_toml(scf_input)

    native = scf_input.native
    settings = scf_input.settings
    density = scf_input.initial_density
    mixer = None
    previous_total = None
    energy_history = []
    convergence_history = []
    valence_normalization_history = []
    reference_constant_history = []
    reference_rms_history = []
    for iteration in range(1, settings.max_iterations + 1):
        context = nullcontext(None) if comm is None else NmtoParallel(comm)
        with context as parallel:
            root = parallel is None or parallel.rank == 0
            timing = (
                None
                if timing_callback is None
                else lambda phase, seconds: timing_callback(iteration, phase, seconds)
            )
            stage = nullcontext() if parallel is None else parallel.local_stage()
            with stage:
                if root:
                    if iteration == 1:
                        mixer = (
                            native.DensityMixer.linear(settings.mixing)
                            if settings.mixing_kind == "linear"
                            else native.DensityMixer.pulay_anderson(
                                settings.mixing, settings.mixing_history
                            )
                        )
            potential = build_nmto_potential(
                native,
                density if root else None,
                settings.xc,
                parallel=parallel,
                timing=timing,
            )
            core = solve_nmto_core(
                native,
                scf_input.core_station,
                potential,
                parallel=parallel,
                timing=timing,
            )
            started = perf_counter()
            if parallel is None:
                current = _solve_nmto_iteration(
                    scf_input,
                    potential,
                    core,
                    timing=timing,
                )
            else:
                current = _solve_nmto_iteration(
                    scf_input,
                    potential,
                    core,
                    parallel=parallel,
                    timing=timing,
                )
            _record_scf_timing(
                timing_callback, iteration, "nmto", started, parallel
            )
            valence_normalization_history.append(current.valence_normalization)
            reference_constant_history.append(current.reference_constant)
            reference_rms_history.append(current.reference_rms)
            iteration_state = None
            started = perf_counter()
            stage = nullcontext() if parallel is None else parallel.local_stage()
            with stage:
                if root:
                    energy = native.evaluate_total_energy(
                        potential,
                        current.output_density,
                        current.occupations.band_energy,
                        core.core_eigenvalue_sum,
                        current.occupations.minus_temperature_entropy,
                        previous_total,
                    )
                    total = float(energy.total)
                    density_rms = float(energy.density_rms)
                    energy_change = (
                        None
                        if energy.energy_change is None
                        else float(energy.energy_change)
                    )
                    change = np.inf if energy_change is None else abs(energy_change)
                    converged = (
                        density_rms <= settings.density_tolerance
                        and energy_change is not None
                        and change <= settings.energy_tolerance
                    )
                    iteration_state = (
                        total,
                        density_rms,
                        energy_change,
                        change,
                        converged,
                    )
            if parallel is not None:
                iteration_state = parallel.comm.bcast(iteration_state, root=0)
            _record_scf_timing(
                timing_callback, iteration, "energy", started, parallel
            )
            total, density_rms, energy_change, change, converged = iteration_state
            energy_history.append(total)
            convergence_history.append((density_rms, change))
            if converged:
                restart_checkpoint = None
                if scf_input.checkpoint is not None:
                    stage = nullcontext() if parallel is None else parallel.local_stage()
                    with stage:
                        if root:
                            checkpoint_physics = native.CheckpointPhysics(scf_input.checkpoint)
                            restart_checkpoint = checkpoint_physics.restart_checkpoint(
                                density,
                                potential,
                                _checkpoint_annotations(
                                    scf_input,
                                    iteration,
                                    total,
                                    reference_constant=current.reference_constant,
                                    reference_rms=current.reference_rms,
                                ),
                            )
                if not root:
                    return None
                return NmtoScfResult(
                    iterations=iteration,
                    total_energy=total,
                    chemical_potential=current.occupations.chemical_potential,
                    density=density,
                    potential=potential,
                    bands=current.bands,
                    occupations=current.occupations,
                    energy_history=np.asarray(energy_history),
                    convergence_history=np.asarray(convergence_history),
                    valence_normalization_history=np.asarray(
                        valence_normalization_history
                    ),
                    reference_constant_history=np.asarray(reference_constant_history),
                    reference_rms_history=np.asarray(reference_rms_history),
                    k_sampling=scf_input.k_mesh_reduction,
                    _restart_checkpoint=restart_checkpoint,
                )
            stage = nullcontext() if parallel is None else parallel.local_stage()
            started = perf_counter()
            with stage:
                if root:
                    density = mixer.step(density, current.output_density).density()
            _record_scf_timing(
                timing_callback, iteration, "mix", started, parallel
            )
            previous_total = total
    raise RuntimeError(
        f"NMTO SCF did not converge after {settings.max_iterations} iterations; "
        f"density_rms={convergence_history[-1][0]}, energy_change={convergence_history[-1][1]}"
    )


def recipe_annotations(scf_input: NmtoScfInput) -> dict[str, str]:
    """Describe the method recipe actually executed by :func:`run_nmto_scf`.

    Every entry names one construction choice so that two "FP-NMTO" runs can
    be told apart from their checkpoints alone: the reference potential and
    its constant, the roles played by each sphere radius, the envelope,
    augmentation, field representation, Coulomb solver, and full-potential
    matrix scheme.
    """

    settings = scf_input.settings
    centers = scf_input.fractional_positions @ scf_input.lattice
    hard = scf_input.muffin_tin_radii
    potential = scf_input.potential_sphere_radii
    omt = settings.reference_potential == "omt"
    shells = bool(np.any(potential > hard + 1.0e-12))
    return {
        "nmto.recipe.family": "fp-nmto",
        "nmto.recipe.reference_potential": (
            "omt-shells" if omt else "spherical-mt"
        ),
        "nmto.recipe.reference_constant": (
            "least-squares" if omt else "interstitial-g0"
        ),
        "nmto.recipe.hard_sphere_radii_bohr": ",".join(repr(float(r)) for r in hard),
        "nmto.recipe.potential_sphere_radii_bohr": ",".join(
            repr(float(r)) for r in potential
        ),
        "nmto.recipe.hard_sphere_roles": (
            "usw-hard-sphere,kink-sphere,augmentation-partition,regional-field-boundary"
        ),
        "nmto.recipe.nearest_neighbor_bohr": repr(
            nearest_neighbor_distance(scf_input.lattice, centers)
        ),
        "nmto.recipe.hard_sphere_max_overlap": repr(
            float(np.max(overlap_fractions(scf_input.lattice, centers, hard)))
        ),
        "nmto.recipe.potential_sphere_max_overlap": repr(
            float(np.max(overlap_fractions(scf_input.lattice, centers, potential)))
        ),
        "nmto.recipe.envelope": "periodic-usw-hard-sphere",
        "nmto.recipe.augmentation": (
            "single-hard-sphere+additive-shells" if shells else "single-hard-sphere"
        ),
        "nmto.recipe.energy_nodes_hartree": ",".join(
            repr(float(e)) for e in settings.energy_mesh
        ),
        "nmto.recipe.l_max": str(settings.l_max),
        "nmto.recipe.field_representation": "regional-mt-harmonics+interstitial-fourier",
        "nmto.recipe.coulomb": "libmuffintin-regional",
        "nmto.recipe.full_potential_matrix": "direct-quadrature",
        "nmto.recipe.relativity": ",".join(scf_input.radial_equations),
        "nmto.recipe.xc": settings.xc,
    }


def _checkpoint_annotations(
    scf_input: NmtoScfInput,
    iterations: int,
    total_energy: float,
    *,
    reference_constant: float = np.nan,
    reference_rms: float = np.nan,
) -> dict[str, str]:
    annotations = {
        "nmto.scf.iterations": str(iterations),
        "nmto.scf.total_energy_hartree": repr(total_energy),
        "scf.k_sampling.divisions": ",".join(map(str, scf_input.settings.k_mesh)),
        "scf.k_sampling.shift": ",".join(map(str, scf_input.settings.k_shift)),
    }
    annotations.update(recipe_annotations(scf_input))
    if np.isfinite(reference_constant):
        annotations["nmto.scf.reference_constant_hartree"] = repr(float(reference_constant))
    if np.isfinite(reference_rms):
        annotations["nmto.scf.reference_fit_rms_hartree"] = repr(float(reference_rms))
    reduction = scf_input.k_mesh_reduction
    if reduction is None:
        annotations["scf.k_sampling.kind"] = "full"
        annotations["scf.k_sampling.full_point_count"] = str(
            int(np.prod(scf_input.settings.k_mesh))
        )
    else:
        annotations.update(
            {
                "scf.k_sampling.kind": "symmetry-reduced",
                "scf.k_sampling.symprec_bohr": repr(scf_input.settings.symprec),
                "scf.k_sampling.include_time_reversal": str(
                    scf_input.settings.include_time_reversal
                ).lower(),
                "scf.k_sampling.spacegroup_number": str(
                    scf_input.symmetry_dataset.spacegroup_number
                ),
                "scf.k_sampling.full_point_count": str(len(reduction.full_points)),
                "scf.k_sampling.irreducible_point_count": str(
                    len(reduction.irreducible_points)
                ),
                "scf.k_sampling.multiplicities": ",".join(
                    map(str, reduction.multiplicities)
                ),
                "scf.k_sampling.operation_count": str(
                    len(np.unique(reduction.active_operation_indices))
                ),
                "scf.k_sampling.symmetry_provenance": scf_input.symmetry_dataset.provenance,
            }
        )
    return annotations


def _solve_nmto_iteration(
    scf_input: NmtoScfInput,
    potential: object | None,
    core: object | None,
    *,
    parallel: NmtoParallel | None = None,
    timing: Callable[[str, float], None] | None = None,
) -> _NmtoIteration:
    settings = scf_input.settings
    direct = scf_input.lattice
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct).T
    if scf_input.k_mesh_reduction is None:
        k_fractional, k_weights = _regular_k_mesh(settings.k_mesh, settings.k_shift)
    else:
        k_fractional = scf_input.k_mesh_reduction.irreducible_points
        k_weights = scf_input.k_mesh_reduction.weights
    k_cartesian = k_fractional @ reciprocal
    site_cartesian = scf_input.fractional_positions @ direct
    channels = tuple(
        RealHarmonic(l, m)
        for l in range(settings.l_max + 1)
        for m in range(-l, l + 1)
    )
    reference = None
    potential_samples = None
    reference_rms = np.nan
    if settings.reference_potential == "omt":
        started = perf_counter()
        (
            reference,
            radial_samples,
            jets,
            interstitial_zero,
            potential_samples,
        ) = sample_omt_radials(
            scf_input.native,
            potential,
            scf_input.radial_equations,
            settings.energy_mesh,
            settings.l_max,
            channels,
            direct,
            scf_input.fractional_positions,
            scf_input.muffin_tin_radii,
            scf_input.potential_sphere_radii,
            settings.matrix_angular_order,
            parallel=parallel,
            timing=timing,
        )
        reference_rms = reference.diagnostics.weighted_rms
        exported_potential = None
        if parallel is None and timing is not None:
            timing("radial", perf_counter() - started)
    elif parallel is None:
        started = perf_counter()
        radial_samples, jets = _current_radials(scf_input, potential, channels)
        if timing is not None:
            timing("radial", perf_counter() - started)
        exported_potential = potential.export_interstitial()
        zero = np.flatnonzero(np.all(exported_potential["g_vectors"] == 0, axis=1))
        if len(zero) != 1:
            raise ValueError("current potential must contain exactly one interstitial G=0")
        interstitial_zero = float(np.real(exported_potential["components"][0, zero[0]]))
    else:
        radial_samples, jets, interstitial_zero = sample_nmto_radials(
            scf_input.native,
            potential,
            scf_input.radial_equations,
            settings.energy_mesh,
            settings.l_max,
            scf_input.muffin_tin_radii,
            channels,
            parallel=parallel,
            timing=timing,
        )
        exported_potential = None
        with parallel.local_stage():
            if parallel.rank == 0:
                exported_potential = potential.export_interstitial()
    interstitial_energies = np.asarray(settings.energy_mesh) - interstitial_zero
    started = perf_counter()
    periodic_samples = _periodic_samples(
        direct,
        site_cartesian,
        scf_input.muffin_tin_radii,
        channels,
        k_cartesian,
        interstitial_energies,
        settings,
        parallel,
    )
    if parallel is None:
        if timing is not None:
            timing("nmto.periodic_samples", perf_counter() - started)
    else:
        with parallel.local_stage():
            if parallel.rank == 0 and timing is not None:
                timing("nmto.periodic_samples", perf_counter() - started)

    started = perf_counter()
    results = _periodic_nmto_results(
        periodic_samples,
        settings.energy_mesh,
        jets,
        parallel,
    )
    if parallel is None:
        if timing is not None:
            timing("nmto.matrices", perf_counter() - started)
    else:
        with parallel.local_stage():
            if parallel.rank == 0 and timing is not None:
                timing("nmto.matrices", perf_counter() - started)

    if parallel is None:
        core_electrons = float(np.sum(core.requested_charges()))
    else:
        core_electrons = None
        with parallel.local_stage():
            if parallel.rank == 0:
                core_electrons = float(np.sum(core.requested_charges()))
        core_electrons = parallel.comm.bcast(core_electrons, root=0)
    started = perf_counter()
    bands, occupations = _bands_and_occupations(
        results, k_weights, settings, core_electrons, parallel
    )
    if parallel is None:
        if timing is not None:
            timing("nmto.bands.initial", perf_counter() - started)
    else:
        with parallel.local_stage():
            if parallel.rank == 0 and timing is not None:
                timing("nmto.bands.initial", perf_counter() - started)

    evaluator = PeriodicNmtoBasisEvaluator(
        direct_lattice=direct,
        site_fractional=scf_input.fractional_positions,
        muffin_tin_radii=scf_input.muffin_tin_radii,
        channels=channels,
        energies=np.asarray(settings.energy_mesh),
        interstitial_energies=interstitial_energies,
        k_cartesian=k_cartesian,
        k_weights=k_weights,
        results=results,
        bands=bands,
        occupations=occupations,
        periodic_samples=periodic_samples,
        radial_samples=radial_samples,
        symmetry=scf_input.symmetry_dataset,
        symmetry_operation_indices=(
            None
            if scf_input.k_mesh_reduction is None
            else scf_input.k_mesh_reduction.active_operation_indices
        ),
        potential_sphere_radii=scf_input.potential_sphere_radii,
    )
    started = perf_counter()
    corrections = full_potential_corrections(
        evaluator,
        exported_potential,
        interstitial_zero,
        angular_order=settings.matrix_angular_order,
        parallel=parallel,
        reference=reference,
        samples=potential_samples,
    )
    if parallel is None:
        if timing is not None:
            timing("nmto.full_potential", perf_counter() - started)
    else:
        with parallel.local_stage():
            if parallel.rank == 0 and timing is not None:
                timing("nmto.full_potential", perf_counter() - started)

    results = tuple(
        replace(
            result,
            hamiltonian=result.hamiltonian + correction,
            lowdin=replace(
                result.lowdin,
                hamiltonian=result.lowdin.hamiltonian
                + result.lowdin.transformation.conj().T
                @ correction
                @ result.lowdin.transformation,
            ),
        )
        for result, correction in zip(results, corrections, strict=True)
    )
    started = perf_counter()
    bands, occupations = _bands_and_occupations(
        results, k_weights, settings, core_electrons, parallel
    )
    if parallel is None:
        if timing is not None:
            timing("nmto.bands.corrected", perf_counter() - started)
    else:
        with parallel.local_stage():
            if parallel.rank == 0 and timing is not None:
                timing("nmto.bands.corrected", perf_counter() - started)

    evaluator = replace(evaluator, results=results, bands=bands, occupations=occupations)
    started = perf_counter()
    valence = assemble_nmto_regional_density(
        scf_input.native,
        scf_input.structure,
        scf_input.field_layout,
        scf_input.g_vectors,
        scf_input.density_l_max,
        evaluator,
        parallel=parallel,
    )
    if parallel is None:
        if timing is not None:
            timing("nmto.density", perf_counter() - started)
    else:
        with parallel.local_stage():
            if parallel.rank == 0 and timing is not None:
                timing("nmto.density", perf_counter() - started)

    started = perf_counter()
    if parallel is None:
        represented_electrons = float(valence.electron_count())
        valence_normalization = occupations.electron_count / represented_electrons
        zero = valence.difference(valence)
        valence = zero.add_scaled(valence_normalization, valence)
        output_density = valence.add_scaled(1.0, core.density())
    else:
        valence_normalization = None
        output_density = None
        with parallel.local_stage():
            if parallel.rank == 0:
                represented_electrons = float(valence.electron_count())
                valence_normalization = occupations.electron_count / represented_electrons
                zero = valence.difference(valence)
                valence = zero.add_scaled(valence_normalization, valence)
                output_density = valence.add_scaled(1.0, core.density())
        valence_normalization = parallel.comm.bcast(valence_normalization, root=0)
    if parallel is None:
        if timing is not None:
            timing("nmto.density.normalization", perf_counter() - started)
    else:
        with parallel.local_stage():
            if parallel.rank == 0 and timing is not None:
                timing("nmto.density.normalization", perf_counter() - started)
    return _NmtoIteration(
        bands,
        occupations,
        output_density,
        valence_normalization,
        reference_constant=float(interstitial_zero),
        reference_rms=float(reference_rms),
    )


def _periodic_samples(
    direct_lattice: FloatArray,
    site_cartesian: FloatArray,
    muffin_tin_radii: FloatArray,
    channels: tuple[RealHarmonic, ...],
    k_cartesian: FloatArray,
    interstitial_energies: FloatArray,
    settings: NmtoScfSettings,
    parallel: NmtoParallel | None,
) -> tuple[tuple[PeriodicUswSample, ...], ...]:
    def build(k_point: FloatArray) -> tuple[PeriodicUswSample, ...]:
        geometry = PeriodicUswGeometry(
            lattice=direct_lattice,
            sites=site_cartesian,
            radii=muffin_tin_radii,
            channels=channels,
            k=k_point,
            g_cutoff=settings.reciprocal_cutoff,
            reference_energy=settings.reference_energy,
            lattice_radius=settings.lattice_sum_radius,
        )
        return tuple(
            geometry.sample(float(energy)) for energy in interstitial_energies
        )

    if parallel is None:
        return tuple(build(k_point) for k_point in k_cartesian)

    owned: dict[int, tuple[PeriodicUswSample, ...]] = {}
    with parallel.local_stage():
        for k_index in parallel.indices(len(k_cartesian)):
            owned[k_index] = build(k_cartesian[k_index])
    result = []
    for k_index, k_point in enumerate(k_cartesian):
        owner = k_index % parallel.size
        result.append(
            _share_periodic_sample_set(
                owned.pop(k_index, None),
                owner,
                direct_lattice,
                site_cartesian,
                muffin_tin_radii,
                channels,
                k_point,
                interstitial_energies,
                settings,
                parallel,
            )
        )
    return tuple(result)


def _share_periodic_sample_set(
    source: tuple[PeriodicUswSample, ...] | None,
    owner: int,
    direct_lattice: FloatArray,
    site_cartesian: FloatArray,
    muffin_tin_radii: FloatArray,
    channels: tuple[RealHarmonic, ...],
    k_point: FloatArray,
    interstitial_energies: FloatArray,
    settings: NmtoScfSettings,
    parallel: NmtoParallel,
) -> tuple[PeriodicUswSample, ...]:
    geometry_source = None if source is None else source[0].geometry
    geometry = object.__new__(PeriodicUswGeometry)
    for name, value in (
        ("lattice", direct_lattice),
        ("sites", site_cartesian),
        ("radii", muffin_tin_radii),
        ("channels", channels),
        ("k", k_point),
        ("g_cutoff", settings.reciprocal_cutoff),
        ("reference_energy", settings.reference_energy),
        ("lattice_radius", settings.lattice_sum_radius),
        ("volume", float(abs(np.linalg.det(direct_lattice)))),
    ):
        object.__setattr__(geometry, name, value)
    for name in (
        "translations",
        "wave_vectors",
        "reciprocal_indices",
        "kinetic_energies",
        "form_factors",
        "reference_boundary",
        "reference_regular",
        "reference_hankel",
    ):
        object.__setattr__(
            geometry,
            name,
            parallel.broadcast_array(
                None if geometry_source is None else getattr(geometry_source, name),
                owner=owner,
            ),
        )
    samples = []
    for energy_index, energy in enumerate(interstitial_energies):
        sample_source = None if source is None else source[energy_index]
        arrays = {
            name: parallel.broadcast_array(
                None if sample_source is None else getattr(sample_source, name),
                owner=owner,
            )
            for name in (
                "slope",
                "slope_derivative",
                "boundary_inverse",
                "fourier_coefficients",
            )
        }
        samples.append(
            PeriodicUswSample(
                geometry=geometry,
                energy=float(energy),
                **arrays,
            )
        )
    return tuple(samples)


def _periodic_nmto_results(
    periodic_samples: tuple[tuple[PeriodicUswSample, ...], ...],
    energy_mesh: Sequence[float],
    jets: BoundaryJets,
    parallel: NmtoParallel | None,
) -> tuple[NmtoResult, ...]:
    def build(samples: tuple[PeriodicUswSample, ...]) -> NmtoResult:
        return build_nmto(
            build_kink_mesh(
                np.asarray(energy_mesh),
                np.asarray([sample.slope for sample in samples]),
                np.asarray([sample.slope_derivative for sample in samples]),
                jets,
                jets.potential_radii,
            )
        )

    if parallel is None:
        return tuple(build(samples) for samples in periodic_samples)

    k_count = len(periodic_samples)
    energy_count = len(energy_mesh)
    basis_size = len(jets.potential_radii)
    result_shape = (k_count, energy_count, basis_size, basis_size)
    matrix_shape = (k_count, basis_size, basis_size)
    green = parallel.shared_array(result_shape, np.complex128)
    green_derivatives = parallel.shared_array(result_shape, np.complex128)
    lagrange_matrices = parallel.shared_array(result_shape, np.complex128)
    hamiltonians = parallel.shared_array(matrix_shape, np.complex128)
    overlaps = parallel.shared_array(matrix_shape, np.complex128)
    lowdin_transformations = parallel.shared_array(matrix_shape, np.complex128)
    lowdin_hamiltonians = parallel.shared_array(matrix_shape, np.complex128)
    overlap_eigenvalues = parallel.shared_array((k_count, basis_size), np.float64)
    with parallel.local_stage():
        for k_index in parallel.indices(k_count):
            result = build(periodic_samples[k_index])
            green[k_index] = result.green
            green_derivatives[k_index] = result.green_derivatives
            lagrange_matrices[k_index] = result.lagrange_matrices
            hamiltonians[k_index] = result.hamiltonian
            overlaps[k_index] = result.overlap
            lowdin_transformations[k_index] = result.lowdin.transformation
            lowdin_hamiltonians[k_index] = result.lowdin.hamiltonian
            overlap_eigenvalues[k_index] = result.lowdin.overlap_eigenvalues
    for array in (
        green,
        green_derivatives,
        lagrange_matrices,
        hamiltonians,
        overlaps,
        lowdin_transformations,
        lowdin_hamiltonians,
        overlap_eigenvalues,
    ):
        parallel.publish(array)
    return tuple(
        NmtoResult(
            energies=np.asarray(energy_mesh),
            green=green[k_index],
            green_derivatives=green_derivatives[k_index],
            lagrange_matrices=lagrange_matrices[k_index],
            hamiltonian=hamiltonians[k_index],
            overlap=overlaps[k_index],
            lowdin=LowdinResult(
                transformation=lowdin_transformations[k_index],
                hamiltonian=lowdin_hamiltonians[k_index],
                overlap_eigenvalues=overlap_eigenvalues[k_index],
            ),
        )
        for k_index in range(k_count)
    )


def _bands_and_occupations(
    results: tuple[NmtoResult, ...],
    k_weights: FloatArray,
    settings: NmtoScfSettings,
    core_electrons: float,
    parallel: NmtoParallel | None,
) -> tuple[NmtoBands, NmtoOccupations]:
    state = None
    stage = nullcontext() if parallel is None else parallel.local_stage()
    with stage:
        if parallel is None or parallel.rank == 0:
            bands = solve_nmto_bands(results)
            occupations = fermi_dirac_occupations(
                bands.energies,
                k_weights,
                settings.electron_count - core_electrons,
                settings.temperature,
                state_degeneracy=settings.state_degeneracy,
            )
            state = (
                bands.energies,
                bands.orthonormal_coefficients,
                bands.coefficients,
                occupations.chemical_potential,
                occupations.values,
                occupations.electron_count,
                occupations.band_energy,
                occupations.minus_temperature_entropy,
            )
    if parallel is not None:
        state = parallel.comm.bcast(state, root=0)
        bands = NmtoBands(
            energies=np.asarray(state[0]),
            orthonormal_coefficients=np.asarray(state[1]),
            coefficients=np.asarray(state[2]),
        )
        occupations = NmtoOccupations(
            chemical_potential=state[3],
            values=np.asarray(state[4]),
            electron_count=state[5],
            band_energy=state[6],
            minus_temperature_entropy=state[7],
        )
    return bands, occupations


def _current_radials(
    scf_input: NmtoScfInput,
    potential: object,
    channels: tuple[RealHarmonic, ...],
) -> tuple[dict[tuple[int, int], ScalarRadialSamples], BoundaryJets]:
    energies = list(scf_input.settings.energy_mesh)
    radial_samples = {}
    values = []
    radial_derivatives = []
    energy_derivatives = []
    energy_radial_derivatives = []
    potential_radii = []
    inverse_masses = []
    energy_inverse_masses = []
    for site, equation in enumerate(scf_input.radial_equations):
        by_l = {
            l: potential.sample_scalar_radials(site, equation, l, energies)
            for l in range(scf_input.settings.l_max + 1)
        }
        radial_samples.update(
            ((site, l), ScalarRadialSamples.from_export(exported))
            for l, exported in by_l.items()
        )
        for channel in channels:
            boundary = np.asarray(by_l[channel.l]["boundary_radial"])
            boundary_energy = np.asarray(
                by_l[channel.l]["energy_derivative_boundary_radial"]
            )
            values.append(boundary[:, 0])
            radial_derivatives.append(boundary[:, 1])
            energy_derivatives.append(boundary_energy[:, 0])
            energy_radial_derivatives.append(boundary_energy[:, 1])
            potential_radii.append(scf_input.muffin_tin_radii[site])
            inverse_masses.append(by_l[channel.l]["boundary_inverse_mass"])
            energy_inverse_masses.append(
                by_l[channel.l]["boundary_energy_inverse_mass"]
            )
    jets = BoundaryJets(
        inverse_masses=np.asarray(inverse_masses).T,
        energy_inverse_masses=np.asarray(energy_inverse_masses).T,
        potential_radii=np.asarray(potential_radii),
        values=np.stack(values, axis=1),
        radial_derivatives=np.stack(radial_derivatives, axis=1),
        energy_derivatives=np.stack(energy_derivatives, axis=1),
        energy_radial_derivatives=np.stack(energy_radial_derivatives, axis=1),
    )
    return radial_samples, jets


def _regular_k_mesh(
    mesh: tuple[int, int, int], shift: tuple[float, float, float]
) -> tuple[FloatArray, FloatArray]:
    points = np.asarray(tuple(product(*(range(size) for size in mesh))), dtype=np.float64)
    fractional = (points + np.asarray(shift)) / np.asarray(mesh)
    return fractional, np.full(len(fractional), 1.0 / len(fractional))
