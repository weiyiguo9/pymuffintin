"""Python-owned self-consistent scalar NMTO calculation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib import import_module
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..contracts import ComplexArray, FloatArray, IntArray
from ..symmetry import IrreducibleKMesh, SymmetryDataset, detect, reduce_regular_kmesh
from ..tensor import contract
from .density import (
    NmtoBasisEvaluator,
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
from .nmto import NmtoResult, build_nmto
from .usw import (
    RealHarmonic,
    bloch_fold_usw_coefficients,
    usw_matrices_with_energy_derivative,
)

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
    include_time_reversal: bool = True

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
    ) -> NmtoScfInput:
        """Prepare an input without serializing a checkpoint or SCF TOML file."""

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
    output_density: object
    valence_normalization: float


def run_nmto_scf(scf_input: NmtoScfInput | str | Path) -> NmtoScfResult:
    """Run the full scalar NMTO density/potential/mixing loop in Python."""

    if not isinstance(scf_input, NmtoScfInput):
        scf_input = NmtoScfInput.from_toml(scf_input)

    native = scf_input.native
    settings = scf_input.settings
    density = scf_input.initial_density
    mixer = native.DensityMixer.linear(settings.mixing)
    previous_total = None
    energy_history = []
    convergence_history = []
    valence_normalization_history = []
    for iteration in range(1, settings.max_iterations + 1):
        potential = native.build_regional_potential(density, xc=settings.xc)
        core = scf_input.core_station.solve(potential)
        current = _solve_nmto_iteration(scf_input, potential, core)
        valence_normalization_history.append(current.valence_normalization)
        energy = native.evaluate_total_energy(
            potential,
            current.output_density,
            current.occupations.band_energy,
            core.core_eigenvalue_sum,
            current.occupations.minus_temperature_entropy,
            previous_total,
        )
        energy_history.append(float(energy.total))
        change = np.inf if energy.energy_change is None else abs(float(energy.energy_change))
        convergence_history.append((float(energy.density_rms), change))
        if (
            energy.density_rms <= settings.density_tolerance
            and energy.energy_change is not None
            and change <= settings.energy_tolerance
        ):
            restart_checkpoint = None
            if scf_input.checkpoint is not None:
                checkpoint_physics = native.CheckpointPhysics(scf_input.checkpoint)
                restart_checkpoint = checkpoint_physics.restart_checkpoint(
                    density,
                    potential,
                    _checkpoint_annotations(
                        scf_input,
                        iteration,
                        float(energy.total),
                    ),
                )
            return NmtoScfResult(
                iterations=iteration,
                total_energy=float(energy.total),
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
                k_sampling=scf_input.k_mesh_reduction,
                _restart_checkpoint=restart_checkpoint,
            )
        density = mixer.step(density, current.output_density).density()
        previous_total = float(energy.total)
    raise RuntimeError(
        f"NMTO SCF did not converge after {settings.max_iterations} iterations; "
        f"density_rms={convergence_history[-1][0]}, energy_change={convergence_history[-1][1]}"
    )


def _checkpoint_annotations(
    scf_input: NmtoScfInput,
    iterations: int,
    total_energy: float,
) -> dict[str, str]:
    annotations = {
        "nmto.scf.iterations": str(iterations),
        "nmto.scf.total_energy_hartree": repr(total_energy),
        "scf.k_sampling.divisions": ",".join(map(str, scf_input.settings.k_mesh)),
        "scf.k_sampling.shift": ",".join(map(str, scf_input.settings.k_shift)),
    }
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
    potential: object,
    core: object,
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
    integers, translations = _translation_cluster(direct, settings.minimum_cells)
    center_translation = int(np.flatnonzero(np.all(integers == 0, axis=1))[0])
    site_cartesian = scf_input.fractional_positions @ direct
    centers = (translations[:, None, :] + site_cartesian[None, :, :]).reshape((-1, 3))
    cluster_radii = np.tile(scf_input.muffin_tin_radii, len(translations))
    channels = tuple(
        RealHarmonic(l, m)
        for l in range(settings.l_max + 1)
        for m in range(-l, l + 1)
    )
    radial_samples, jets = _current_radials(scf_input, potential, channels)

    exported_potential = potential.export_interstitial()
    zero = np.flatnonzero(np.all(exported_potential["g_vectors"] == 0, axis=1))
    if len(zero) != 1:
        raise ValueError("current potential must contain exactly one interstitial G=0")
    interstitial_zero = float(np.real(exported_potential["components"][0, zero[0]]))
    interstitial_energies = np.asarray(settings.energy_mesh) - interstitial_zero

    coefficients = []
    slopes = []
    slope_derivatives = []
    for energy in interstitial_energies:
        coefficient, slope, derivative = usw_matrices_with_energy_derivative(
            float(energy), centers, cluster_radii, channels
        )
        coefficients.append(coefficient)
        slopes.append(slope)
        slope_derivatives.append(derivative)

    results = []
    folded_coefficients = []
    for k_point in k_cartesian:
        folded_slopes = np.stack(
            tuple(
                _bloch_fold_matrix(
                    value,
                    translations,
                    center_translation,
                    len(site_cartesian),
                    len(channels),
                    k_point,
                )
                for value in slopes
            )
        )
        folded_derivatives = np.stack(
            tuple(
                _bloch_fold_matrix(
                    value,
                    translations,
                    center_translation,
                    len(site_cartesian),
                    len(channels),
                    k_point,
                )
                for value in slope_derivatives
            )
        )
        results.append(
            build_nmto(
                build_kink_mesh(
                    np.asarray(settings.energy_mesh),
                    folded_slopes,
                    folded_derivatives,
                    jets,
                    jets.potential_radii,
                )
            )
        )
        folded_coefficients.append(
            np.stack(
                tuple(
                    bloch_fold_usw_coefficients(
                        value,
                        translations,
                        len(site_cartesian),
                        len(channels),
                        k_point,
                    )
                    for value in coefficients
                )
            )
        )
    results = tuple(results)
    bands = solve_nmto_bands(results)
    core_electrons = float(np.sum(core.requested_charges()))
    occupations = fermi_dirac_occupations(
        bands.energies,
        k_weights,
        settings.electron_count - core_electrons,
        settings.temperature,
        state_degeneracy=settings.state_degeneracy,
    )
    evaluator = NmtoBasisEvaluator(
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
        translations=translations,
        centers=centers,
        folded_coefficients=np.asarray(folded_coefficients),
        radial_samples=radial_samples,
        symmetry=scf_input.symmetry_dataset,
        symmetry_operation_indices=(
            None
            if scf_input.k_mesh_reduction is None
            else scf_input.k_mesh_reduction.active_operation_indices
        ),
    )
    evaluator = replace(
        evaluator,
        basis_corrections=_represented_basis_corrections(evaluator),
    )
    valence = assemble_nmto_regional_density(
        scf_input.native,
        scf_input.structure,
        scf_input.field_layout,
        scf_input.g_vectors,
        scf_input.density_l_max,
        evaluator,
    )
    represented_electrons = float(valence.electron_count())
    valence_normalization = occupations.electron_count / represented_electrons
    zero = valence.difference(valence)
    valence = zero.add_scaled(valence_normalization, valence)
    output_density = valence.add_scaled(1.0, core.density())
    return _NmtoIteration(
        bands, occupations, output_density, valence_normalization
    )


def _represented_basis_corrections(evaluator: NmtoBasisEvaluator) -> ComplexArray:
    """Align sampled basis overlaps with the analytic NMTO overlap matrices."""

    axis = (np.arange(18) + 0.5) / 18
    fractional = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    points = fractional @ evaluator.direct_lattice
    volume_weight = abs(np.linalg.det(evaluator.direct_lattice)) / len(points)
    corrections = []
    for k_index, result in enumerate(evaluator.results):
        large, small = evaluator._basis_values(points, k_index)
        represented = volume_weight * (
            large.conj().T @ large + small.conj().T @ small
        )
        represented = 0.5 * (represented + represented.conj().T)
        target = 0.5 * (result.overlap + result.overlap.conj().T)
        represented_cholesky = np.linalg.cholesky(represented)
        target_cholesky = np.linalg.cholesky(target)
        corrections.append(
            np.linalg.solve(
                represented_cholesky.conj().T,
                target_cholesky.conj().T,
            )
        )
    return np.asarray(corrections)


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
    jets = BoundaryJets(
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


def _translation_cluster(
    direct_lattice: FloatArray, minimum_cells: int
) -> tuple[IntArray, FloatArray]:
    extent = 1
    while True:
        integers = np.asarray(
            tuple(product(range(-extent, extent + 1), repeat=3)), dtype=np.int64
        )
        cartesian = integers @ direct_lattice
        distances = np.linalg.norm(cartesian, axis=1)
        if len(distances) < minimum_cells:
            extent += 1
            continue
        shell_radius = float(np.partition(distances, minimum_cells - 1)[minimum_cells - 1])
        selected = distances <= shell_radius + 1.0e-10
        if np.all(np.max(np.abs(integers[selected]), axis=0) < extent):
            order = np.lexsort(
                (
                    integers[selected, 2],
                    integers[selected, 1],
                    integers[selected, 0],
                    distances[selected],
                )
            )
            return integers[selected][order], cartesian[selected][order]
        extent += 1


def _bloch_fold_matrix(
    matrix: NDArray,
    translations: FloatArray,
    center_translation: int,
    site_count: int,
    channel_count: int,
    k_cartesian: FloatArray,
) -> ComplexArray:
    cell_count = len(translations)
    blocks = np.asarray(matrix).reshape(
        (cell_count, site_count, channel_count, cell_count, site_count, channel_count)
    )
    central_rows = blocks[center_translation]
    phase = np.exp(1j * (translations @ k_cartesian))
    return contract("t,aitbj->aibj", phase, central_rows).reshape(
        (site_count * channel_count, site_count * channel_count)
    )
