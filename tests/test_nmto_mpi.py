from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pymuffintin.mto.parallel import NmtoParallel
from pymuffintin.mto.scf import _solve_nmto_iteration


LIBMUFFINTIN = Path(__file__).resolve().parents[2] / "libmuffintin"
FIXTURES = LIBMUFFINTIN / "python" / "tests" / "fixtures"


def _mpi():
    return pytest.importorskip("mpi4py.MPI")


def _hydrogen_iteration_input():
    sys.path.insert(0, str(LIBMUFFINTIN / "python"))
    native = pytest.importorskip("libmuffintin")
    checkpoint_path = FIXTURES / "hydrogen_checkpoint.toml"
    checkpoint = native.load_checkpoint(checkpoint_path)
    physics = native.CheckpointPhysics(checkpoint)
    structure = physics.structure()
    g_vectors = np.asarray([[0, 0, 0]], dtype=np.int64)
    field_layout = native.RegionalFieldLayout(
        structure,
        g_vectors.tolist(),
        muffin_tin_l_max=0,
    )
    # This fixture supplies a frozen potential, not an SCF restart density.
    # Use its real native radial solver and the native regional-density type.
    frozen = physics.export_frozen_potential()
    potential = SimpleNamespace(
        sample_scalar_radials=lambda site, equation, l, energies: (
            physics.sample_frozen_scalar_radials("H-1", l, energies)
        ),
        export_interstitial=lambda: {
            "g_vectors": g_vectors,
            "components": np.zeros((4, 1), dtype=np.complex128),
        },
    )
    zero_density = native.RegionalDensity(
        structure,
        field_layout,
        "complex-condon-shortley",
        np.zeros((4, 1), dtype=np.complex128),
        np.asarray([[0, 0, 0]], dtype=np.int64),
        np.asarray([0, len(frozen["mt_mesh_radii"])], dtype=np.int64),
        np.zeros((4, len(frozen["mt_mesh_radii"])), dtype=np.complex128),
    )
    core = SimpleNamespace(
        requested_charges=lambda: np.zeros(1),
        density=lambda: zero_density,
    )
    projected = {}

    def regional_density(*args):
        projected["interstitial"] = args[3].copy()
        projected["muffin_tin"] = args[6].copy()
        return native.RegionalDensity(*args)
    settings = SimpleNamespace(
        energy_mesh=(-0.18, -0.14, -0.10),
        k_mesh=(2, 1, 1),
        k_shift=(0.0, 0.0, 0.0),
        l_max=0,
        minimum_cells=27,
        electron_count=1.0,
        temperature=0.02,
        state_degeneracy=2.0,
    )
    scf_input = SimpleNamespace(
        native=SimpleNamespace(RegionalDensity=regional_density),
        structure=structure,
        field_layout=field_layout,
        lattice=8.0 * np.eye(3),
        fractional_positions=np.asarray([[1.25, -0.5, 0.5]], dtype=np.float64),
        muffin_tin_radii=np.asarray([1.0], dtype=np.float64),
        g_vectors=g_vectors,
        density_l_max=0,
        radial_equations=("scalar-koelling-harmon",),
        settings=settings,
        symmetry_dataset=None,
        k_mesh_reduction=None,
    )
    return scf_input, potential, core, projected


def _export_density(density):
    exported = density.export_interstitial()
    return {
        key: np.asarray(value).copy()
        for key, value in exported.items()
        if isinstance(value, np.ndarray)
    }


def test_nmto_parallel_sharing_ownership_failures_and_lifetime() -> None:
    MPI = _mpi()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    task_count = 3
    row_count = 7
    with NmtoParallel(comm) as parallel:
        tasks = parallel.shared_array((task_count, 2), np.int64)
        for task in parallel.indices(task_count):
            tasks[task] = (rank, task)
        parallel.publish(tasks)
        expected_tasks = np.asarray(
            [(task % size, task) for task in range(task_count)], dtype=np.int64
        )
        np.testing.assert_array_equal(tasks, expected_tasks)

        rows = parallel.shared_array((row_count,), np.float64)
        owned = parallel.point_slice(row_count)
        rows[owned] = rank + np.arange(owned.stop - owned.start, dtype=np.float64)
        parallel.publish_rows(rows)
        expected_rows = np.empty(row_count, dtype=np.float64)
        for owner in range(size):
            start = row_count * owner // size
            stop = row_count * (owner + 1) // size
            expected_rows[start:stop] = owner + np.arange(stop - start)
        np.testing.assert_allclose(rows, expected_rows, rtol=0.0, atol=0.0)

        tasks_copy = tasks.copy()
        rows_copy = rows.copy()

        with pytest.raises(RuntimeError, match=r"rank \d+: ValueError: MPI stage"):  # noqa: SIM117
            with parallel.local_stage():
                if rank == size - 1:
                    raise ValueError("MPI stage")

    comm.Barrier()
    np.testing.assert_array_equal(tasks_copy, expected_tasks)
    np.testing.assert_allclose(rows_copy, expected_rows, rtol=0.0, atol=0.0)


def test_nmto_hydrogen_iteration_serial_matches_mpi_density_and_bands() -> None:
    MPI = _mpi()
    comm = MPI.COMM_WORLD
    scf_input, potential, core, projected = _hydrogen_iteration_input()

    serial = _solve_nmto_iteration(scf_input, potential, core)
    serial_projected = {key: value.copy() for key, value in projected.items()}
    with NmtoParallel(comm) as parallel:
        distributed = _solve_nmto_iteration(
            scf_input,
            potential,
            core,
            parallel=parallel,
        )
        distributed_density = _export_density(distributed.output_density)

    for key, reference in serial_projected.items():
        assert np.all(np.isfinite(projected[key]))
        np.testing.assert_allclose(projected[key], reference, rtol=1.0e-10, atol=1.0e-12)

    np.testing.assert_allclose(
        distributed.bands.energies,
        serial.bands.energies,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        distributed.bands.coefficients,
        serial.bands.coefficients,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        distributed.occupations.chemical_potential,
        serial.occupations.chemical_potential,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        distributed.occupations.values,
        serial.occupations.values,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        distributed.occupations.electron_count,
        serial.occupations.electron_count,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        distributed.valence_normalization,
        serial.valence_normalization,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    for key, serial_value in _export_density(serial.output_density).items():
        np.testing.assert_allclose(
            distributed_density[key],
            serial_value,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
