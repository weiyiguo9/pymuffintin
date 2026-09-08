"""MPI-owned potential, core, and radial block scheduling for NMTO SCF."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from ..contracts import FloatArray
from .density import ScalarRadialSamples
from .kink import BoundaryJets
from .parallel import NmtoParallel
from .usw import RealHarmonic


PhaseTiming = Callable[[str, float], None]

_MT_RADIAL_BLOCK_SIZE = 32
_INTERSTITIAL_BLOCK_SIZE = 256


def _record_timing(
    timing: PhaseTiming | None,
    phase: str,
    started: float,
    parallel: NmtoParallel | None,
) -> None:
    if parallel is None:
        if timing is not None:
            timing(phase, perf_counter() - started)
        return
    with parallel.local_stage():
        if parallel.rank == 0 and timing is not None:
            timing(phase, perf_counter() - started)


def _record_duration(
    timing: PhaseTiming | None,
    phase: str,
    seconds: float | None,
    parallel: NmtoParallel,
) -> None:
    with parallel.local_stage():
        if parallel.rank == 0 and timing is not None and seconds is not None:
            timing(phase, seconds)


def _fixed_blocks(count: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def build_nmto_potential(
    native: object,
    density: object | None,
    xc: str,
    *,
    parallel: NmtoParallel | None = None,
    timing: PhaseTiming | None = None,
) -> object | None:
    """Build one potential, distributing scalar XC blocks when MPI is active."""

    if parallel is None:
        started = perf_counter()
        potential = native.build_regional_potential(density, xc=xc)
        _record_timing(timing, "potential", started, parallel)
        return potential

    root = parallel.rank == 0
    prepared = None
    exported = None
    started = perf_counter()
    with parallel.local_stage():
        if root:
            prepared = native.prepare_scf_potential_blocks(density, xc)
            exported = prepared.export_xc_inputs()
    scalars = parallel.comm.bcast(
        None
        if not root
        else {
            "functional": exported["functional"],
            "angular_basis": exported["angular_basis"],
            "angular_point_count": exported["angular_point_count"],
            "output_l_max": exported["output_l_max"],
            "output_lm_count": exported["output_lm_count"],
            "is_point_weight": exported["is_point_weight"],
        },
        root=0,
    )
    arrays = {
        name: parallel.broadcast_array(exported[name] if root else None)
        for name in (
            "mt_components",
            "mt_input_l_max",
            "mt_site_sample_offsets",
            "mt_radial_offsets",
            "mt_mesh_radii",
            "is_density_derivatives",
            "is_theta",
        )
    }
    _record_timing(timing, "potential.prepare", started, parallel)
    native_timings = prepared.timings() if root else None
    for phase, key in (
        ("potential.hartree", "hartree"),
        ("potential.xc_prepare", "xc_prepare"),
        ("potential.fft_plan", "fft_plan"),
    ):
        _record_duration(
            timing,
            phase,
            None if not root else native_timings[key],
            parallel,
        )

    total_radial = len(arrays["mt_mesh_radii"])
    nlm = int(scalars["output_lm_count"])
    mt_potential = parallel.shared_array(
        (total_radial, 4, nlm), np.complex128
    )
    mt_integrands = parallel.shared_array((total_radial, 2), np.float64)
    mt_tasks = []
    radial_offsets = arrays["mt_radial_offsets"]
    for site in range(len(radial_offsets) - 1):
        global_site_start = int(radial_offsets[site])
        radial_count = int(radial_offsets[site + 1]) - global_site_start
        mt_tasks.extend(
            (
                site,
                local_start,
                local_stop,
                global_site_start + local_start,
                global_site_start + local_stop,
            )
            for local_start, local_stop in _fixed_blocks(
                radial_count, _MT_RADIAL_BLOCK_SIZE
            )
        )
    started = perf_counter()
    with parallel.local_stage():
        for task_index in parallel.indices(len(mt_tasks)):
            site, local_start, _, global_start, global_stop = mt_tasks[task_index]
            input_l_max = int(arrays["mt_input_l_max"][site])
            sample_start = int(arrays["mt_site_sample_offsets"][site])
            sample_stop = int(arrays["mt_site_sample_offsets"][site + 1])
            site_radial_start = int(radial_offsets[site])
            site_radial_stop = int(radial_offsets[site + 1])
            radii = arrays["mt_mesh_radii"][site_radial_start:site_radial_stop]
            charge_channels = arrays["mt_components"][0, sample_start:sample_stop].reshape(
                (input_l_max + 1) ** 2, len(radii)
            )
            native.evaluate_muffin_tin_xc_block(
                scalars["functional"],
                scalars["angular_basis"],
                radii,
                charge_channels,
                input_l_max,
                scalars["output_l_max"],
                scalars["angular_point_count"],
                local_start,
                mt_potential[global_start:global_stop],
                mt_integrands[global_start:global_stop],
            )
    mt_spans = [(task[3], task[4]) for task in mt_tasks]
    parallel.publish_blocks(mt_potential, mt_spans)
    parallel.publish_blocks(mt_integrands, mt_spans)
    _record_timing(timing, "potential.mt_xc", started, parallel)

    point_count = len(arrays["is_theta"])
    is_potential = parallel.shared_array((point_count, 4), np.float64)
    is_integrands = parallel.shared_array((point_count, 2), np.float64)
    is_blocks = _fixed_blocks(point_count, _INTERSTITIAL_BLOCK_SIZE)
    started = perf_counter()
    with parallel.local_stage():
        for block_index in parallel.indices(len(is_blocks)):
            start, stop = is_blocks[block_index]
            native.evaluate_interstitial_xc_block(
                scalars["functional"],
                arrays["is_density_derivatives"],
                arrays["is_theta"],
                scalars["is_point_weight"],
                start,
                is_potential[start:stop],
                is_integrands[start:stop],
            )
    parallel.publish_blocks(is_potential, is_blocks)
    parallel.publish_blocks(is_integrands, is_blocks)
    _record_timing(timing, "potential.interstitial_xc", started, parallel)

    started = perf_counter()
    potential = None
    with parallel.local_stage():
        if root:
            potential = native.assemble_scf_potential_blocks(
                prepared,
                mt_potential,
                mt_integrands,
                is_potential,
                is_integrands,
            )
    _record_timing(timing, "potential.assemble", started, parallel)
    native_timings = prepared.timings() if root else None
    for phase, key in (
        ("potential.fft_transforms", "fft_transforms"),
        ("potential.xc_assemble", "xc_assemble"),
    ):
        _record_duration(
            timing,
            phase,
            None if not root else native_timings[key],
            parallel,
        )
    return potential


def solve_nmto_core(
    native: object,
    core_station: object,
    potential: object | None,
    *,
    parallel: NmtoParallel | None = None,
    timing: PhaseTiming | None = None,
) -> object | None:
    """Solve requested core states, returning an assembled result on root."""

    if parallel is None:
        started = perf_counter()
        core = core_station.solve(potential)
        _record_timing(timing, "core", started, parallel)
        return core

    root = parallel.rank == 0
    plan = None
    exported = None
    started = perf_counter()
    with parallel.local_stage():
        if root:
            plan = native.prepare_core_radial_blocks(core_station, potential)
            exported = plan.export()
    speed_of_light = parallel.comm.bcast(
        exported["speed_of_light"] if root else None, root=0
    )
    tasks = parallel.comm.bcast(exported["tasks"] if root else None, root=0)
    arrays = {
        name: parallel.broadcast_array(exported[name] if root else None)
        for name in (
            "site_offsets",
            "mesh_first",
            "mesh_increment",
            "nuclear_charges",
            "muffin_tin_radii",
            "potential_values",
            "task_radial_offsets",
        )
    }
    _record_timing(timing, "core.prepare", started, parallel)

    task_count = len(tasks)
    if task_count == 0:
        started = perf_counter()
        _record_timing(timing, "core.solve", started, parallel)
        core = None
        started = perf_counter()
        with parallel.local_stage():
            if root:
                empty = np.empty(0, dtype=np.float64)
                core = plan.assemble(
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    np.empty((0, 2), dtype=np.float64),
                )
        _record_timing(timing, "core.assemble", started, parallel)
        return core
    radial_count = int(arrays["task_radial_offsets"][-1]) if task_count else 0
    large = parallel.shared_array((radial_count,), np.float64)
    small = parallel.shared_array((radial_count,), np.float64)
    scalars = parallel.shared_array((task_count, 6), np.float64)
    task_rows = [(task, task + 1) for task in range(task_count)]
    radial_spans = [
        (
            int(arrays["task_radial_offsets"][task]),
            int(arrays["task_radial_offsets"][task + 1]),
        )
        for task in range(task_count)
    ]
    started = perf_counter()
    with parallel.local_stage():
        for task in parallel.indices(task_count):
            site, _, n, kappa = map(int, tasks[task])
            site_start = int(arrays["site_offsets"][site])
            site_stop = int(arrays["site_offsets"][site + 1])
            radial_start, radial_stop = radial_spans[task]
            native.solve_core_radial_block(
                arrays["potential_values"][site_start:site_stop],
                arrays["mesh_first"][site],
                arrays["mesh_increment"][site],
                n,
                kappa,
                arrays["nuclear_charges"][site],
                arrays["muffin_tin_radii"][site],
                large[radial_start:radial_stop],
                small[radial_start:radial_stop],
                scalars[task],
                speed_of_light,
            )
    parallel.publish_blocks(scalars, task_rows)
    parallel.publish_blocks(large, radial_spans)
    parallel.publish_blocks(small, radial_spans)
    _record_timing(timing, "core.solve", started, parallel)

    started = perf_counter()
    core = None
    with parallel.local_stage():
        if root:
            core = plan.assemble(
                scalars[:, 0],
                large,
                small,
                scalars[:, 1],
                scalars[:, 2],
                scalars[:, 3],
                scalars[:, 4:6],
            )
    _record_timing(timing, "core.assemble", started, parallel)
    return core


def sample_nmto_radials(
    native: object,
    potential: object | None,
    radial_equations: Sequence[str],
    energy_mesh: Sequence[float],
    l_max: int,
    muffin_tin_radii: FloatArray,
    channels: tuple[RealHarmonic, ...],
    *,
    parallel: NmtoParallel,
    timing: PhaseTiming | None = None,
) -> tuple[dict[tuple[int, int], ScalarRadialSamples], BoundaryJets, float]:
    """Sample scalar radial tasks into node-shared arrays."""

    root = parallel.rank == 0
    exported = None
    interstitial_zero = None
    started = perf_counter()
    with parallel.local_stage():
        if root:
            exported = native.prepare_scalar_radial_blocks(potential)
            interstitial = potential.export_interstitial()
            zero = np.flatnonzero(np.all(interstitial["g_vectors"] == 0, axis=1))
            if len(zero) != 1:
                raise ValueError(
                    "current potential must contain exactly one interstitial G=0"
                )
            interstitial_zero = float(
                np.real(interstitial["components"][0, zero[0]])
            )
    interstitial_zero = parallel.comm.bcast(interstitial_zero, root=0)
    equations = parallel.comm.bcast(tuple(radial_equations) if root else None, root=0)
    arrays = {
        name: parallel.broadcast_array(exported[name] if root else None)
        for name in (
            "site_offsets",
            "mesh_first",
            "mesh_increment",
            "mesh_count",
            "mesh_radii",
            "potential_values",
        )
    }
    _record_timing(timing, "radial.prepare", started, parallel)

    tasks = [
        (site, l, energy_index)
        for site in range(len(equations))
        for l in range(l_max + 1)
        for energy_index in range(len(energy_mesh))
    ]
    task_offsets = [0]
    for site, _, _ in tasks:
        task_offsets.append(task_offsets[-1] + int(arrays["mesh_count"][site]))
    task_rows = [(task, task + 1) for task in range(len(tasks))]
    radial_spans = [
        (task_offsets[task], task_offsets[task + 1]) for task in range(len(tasks))
    ]
    large = parallel.shared_array((task_offsets[-1],), np.float64)
    small = parallel.shared_array((task_offsets[-1],), np.float64)
    inverse_mass = parallel.shared_array((task_offsets[-1],), np.float64)
    boundary_jets = parallel.shared_array((len(tasks), 7), np.float64)
    started = perf_counter()
    with parallel.local_stage():
        for task in parallel.indices(len(tasks)):
            site, l, energy_index = tasks[task]
            site_start = int(arrays["site_offsets"][site])
            site_stop = int(arrays["site_offsets"][site + 1])
            radial_start, radial_stop = radial_spans[task]
            native.solve_scalar_radial_block(
                arrays["potential_values"][site_start:site_stop],
                arrays["mesh_first"][site],
                arrays["mesh_increment"][site],
                site,
                equations[site],
                l,
                energy_mesh[energy_index],
                large[radial_start:radial_stop],
                small[radial_start:radial_stop],
                inverse_mass[radial_start:radial_stop],
                boundary_jets[task],
            )
    parallel.publish_blocks(large, radial_spans)
    parallel.publish_blocks(small, radial_spans)
    parallel.publish_blocks(inverse_mass, radial_spans)
    parallel.publish_blocks(boundary_jets, task_rows)
    _record_timing(timing, "radial.solve", started, parallel)

    radial_samples = {}
    by_site_l = {}
    task = 0
    for site in range(len(equations)):
        count = int(arrays["mesh_count"][site])
        mesh_start = int(arrays["site_offsets"][site])
        mesh_stop = int(arrays["site_offsets"][site + 1])
        mesh = arrays["mesh_radii"][mesh_start:mesh_stop]
        for l in range(l_max + 1):
            first = task
            start = task_offsets[first]
            task += len(energy_mesh)
            stop = task_offsets[task]
            radial_samples[site, l] = ScalarRadialSamples(
                mesh_radii=mesh,
                large=large[start:stop].reshape(len(energy_mesh), count),
                small=small[start:stop].reshape(len(energy_mesh), count),
                boundary_values=boundary_jets[first:task, 0],
                inverse_mass=inverse_mass[start:stop].reshape(
                    len(energy_mesh), count
                ),
                inverse_speed_of_light=float(boundary_jets[first, 6]),
            )
            by_site_l[site, l] = (
                boundary_jets[first:task, :2],
                boundary_jets[first:task, 2:4],
                boundary_jets[first:task, 4],
                boundary_jets[first:task, 5],
            )

    values = []
    radial_derivatives = []
    energy_derivatives = []
    energy_radial_derivatives = []
    inverse_masses = []
    energy_inverse_masses = []
    potential_radii = []
    for site in range(len(equations)):
        for channel in channels:
            (
                boundary_values,
                boundary_energy,
                boundary_inverse_mass,
                boundary_energy_inverse_mass,
            ) = by_site_l[site, channel.l]
            values.append(boundary_values[:, 0])
            radial_derivatives.append(boundary_values[:, 1])
            energy_derivatives.append(boundary_energy[:, 0])
            energy_radial_derivatives.append(boundary_energy[:, 1])
            inverse_masses.append(boundary_inverse_mass)
            energy_inverse_masses.append(boundary_energy_inverse_mass)
            potential_radii.append(muffin_tin_radii[site])
    jets = BoundaryJets(
        potential_radii=np.asarray(potential_radii),
        values=np.stack(values, axis=1),
        radial_derivatives=np.stack(radial_derivatives, axis=1),
        energy_derivatives=np.stack(energy_derivatives, axis=1),
        energy_radial_derivatives=np.stack(energy_radial_derivatives, axis=1),
        inverse_masses=np.stack(inverse_masses, axis=1),
        energy_inverse_masses=np.stack(energy_inverse_masses, axis=1),
    )
    return radial_samples, jets, interstitial_zero
