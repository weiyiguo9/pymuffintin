from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pymuffintin.mto.parallel import NmtoParallel
from pymuffintin.mto.scf import _current_radials, _solve_nmto_iteration


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

    def frozen_scalar_radials(site, equation, l, energies):
        exported = physics.sample_frozen_scalar_radials("H-1", l, energies)
        inverse_c = 1.0 / 137.0359895
        spherical = frozen["mt_components"][0].real / np.sqrt(4.0 * np.pi)
        inverse_mass = 1.0 / (2.0 + (np.asarray(energies)[:, None] - spherical) * inverse_c**2)
        exported.update(
            inverse_mass=inverse_mass,
            inverse_speed_of_light=inverse_c,
            boundary_inverse_mass=inverse_mass[:, -1],
            boundary_energy_inverse_mass=-inverse_c**2 * inverse_mass[:, -1]**2,
        )
        return exported

    potential = SimpleNamespace(
        sample_scalar_radials=frozen_scalar_radials,
        export_interstitial=lambda: frozen,
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
        energy_mesh=(-0.18, -0.10),
        k_mesh=(2, 1, 1),
        k_shift=(0.0, 0.0, 0.0),
        l_max=0,
        reciprocal_cutoff=2.0,
        lattice_sum_radius=8.1,
        reference_energy=-1.0,
        matrix_angular_order=2,
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


def test_nmto_fixed_blocks_share_readonly_source() -> None:
    comm = _mpi().COMM_WORLD
    expected = np.arange(18, dtype=np.float64).reshape(9, 2)
    blocks = [(0, 4), (4, 8), (8, 9)]
    with NmtoParallel(comm) as parallel:
        source = parallel.broadcast_array(expected if comm.rank == 0 else None)
        assert not source.flags.writeable
        output = parallel.shared_array(source.shape, source.dtype)
        for block in parallel.indices(len(blocks)):
            start, stop = blocks[block]
            output[start:stop] = 2.0 * source[start:stop]
        parallel.publish_blocks(output, blocks)
        np.testing.assert_array_equal(output, 2.0 * expected)


def test_nmto_hydrogen_iteration_serial_matches_mpi_density_and_bands(monkeypatch) -> None:
    MPI = _mpi()
    comm = MPI.COMM_WORLD
    scf_input, potential, core, projected = _hydrogen_iteration_input()

    # The frozen fixture is not a ScfPotentialBuild. Keep this test focused on
    # the downstream NMTO solve; real potential/core blocks have a native case.
    def frozen_radials(native, built_potential, equations, energies, l_max,
                       radii, channels, *, parallel, **kwargs):
        data = None
        if parallel.rank == 0:
            radial, jets = _current_radials(scf_input, potential, channels)
            data = (radial, jets, 0.0)
        return parallel.comm.bcast(data, root=0)

    monkeypatch.setattr("pymuffintin.mto.scf.sample_nmto_radials", frozen_radials)

    serial = _solve_nmto_iteration(scf_input, potential, core)
    serial_projected = {key: value.copy() for key, value in projected.items()}
    with NmtoParallel(comm) as parallel:
        distributed = _solve_nmto_iteration(
            scf_input,
            potential if comm.rank == 0 else None,
            core if comm.rank == 0 else None,
            parallel=parallel,
        )
        if comm.rank == 0:
            distributed_density = _export_density(distributed.output_density)
        else:
            assert distributed.output_density is None

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
    if comm.rank == 0:
        for key, serial_value in _export_density(serial.output_density).items():
            np.testing.assert_allclose(
                distributed_density[key],
                serial_value,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
