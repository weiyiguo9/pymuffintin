"""Native scalar potential/core/radial parity through MPI array blocks."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pymuffintin.mto.parallel import NmtoParallel
from pymuffintin.mto.potential import (
    build_nmto_potential,
    sample_nmto_radials,
    solve_nmto_core,
)
from pymuffintin.mto.scf import _current_radials
from pymuffintin.mto.usw import RealHarmonic


def test_native_carbon_blocks_match_serial() -> None:
    mt = pytest.importorskip("libmuffintin")
    comm = pytest.importorskip("mpi4py.MPI").COMM_WORLD
    root = comm.rank == 0
    timings = {}
    timing = (lambda phase, seconds: timings.__setitem__(phase, seconds)) if root else None
    density = reference = reference_core = None
    station = mt.CoreStation([mt.CoreSite(0, "C-1", [mt.CoreState(1, -1, occupation=2.0)])])
    channels = tuple(RealHarmonic(l, m) for l in range(2) for m in range(-l, l + 1))
    settings = SimpleNamespace(energy_mesh=(-0.3, 0.1), l_max=1)
    inp = SimpleNamespace(settings=settings, radial_equations=("scalar-koelling-harmon",),
                          muffin_tin_radii=np.array([1.5]))
    with NmtoParallel(comm) as parallel:
        with parallel.local_stage():
            if root:
                structure = mt.Structure(
                    lattice=(4.0 * np.eye(3)).tolist(), site_ids=["C-1"],
                    atomic_numbers=[6], fractional_positions=[[0.5, 0.5, 0.5]],
                    radial_meshes=[(1e-4, math.log(1.5 / 1e-4) / 60, 61)],
                    radial_equations=list(inp.radial_equations),
                    linearization_energies=[[(0, -0.3)]],
                )
                layout = mt.RegionalFieldLayout.from_g_cutoff(
                    structure, g_cutoff=4.0, muffin_tin_l_max=2
                )
                controls = mt.FreeAtomControls(
                    mesh_first=1e-6, mesh_log_increment=0.01, mesh_point_count=1683,
                    mixing=0.3, potential_tolerance=2e-5, tail_tolerance=1e-7,
                    max_iterations=120, angular_points=50,
                )
                start = mt.materialize_atomic_start(
                    structure, layout, xc="lda-pw92", free_atom_controls=controls
                )
                density = mt.CheckpointPhysics(start.checkpoint).restart_density()
                reference = mt.build_regional_potential(density, "pbe")
                reference_core = station.solve(reference)
                reference_radials, reference_jets = _current_radials(inp, reference, channels)

        potential = build_nmto_potential(mt, density, "pbe", parallel=parallel, timing=timing)
        core = solve_nmto_core(mt, station, potential, parallel=parallel, timing=timing)
        radials, jets, zero = sample_nmto_radials(
            mt, potential, inp.radial_equations, settings.energy_mesh, settings.l_max,
            inp.muffin_tin_radii, channels, parallel=parallel, timing=timing,
        )
        with parallel.local_stage():
            if not root:
                assert potential is None and core is None
            else:
                assert timings and all(np.isfinite(value) and value >= 0 for value in timings.values())
                for actual, expected in ((potential, reference),
                                         (core.density(), reference_core.density())):
                    actual_arrays = actual.export_interstitial()
                    expected_arrays = expected.export_interstitial()
                    for key in ("components", "mt_components"):
                        assert np.isfinite(actual_arrays[key]).all()
                        np.testing.assert_allclose(actual_arrays[key], expected_arrays[key],
                                                   rtol=1e-10, atol=1e-12)
                for name in ("madelung", "coulomb", "exchange_correlation",
                             "exchange_correlation_potential"):
                    np.testing.assert_allclose(getattr(potential, name), getattr(reference, name),
                                               rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(core.core_eigenvalue_sum,
                                           reference_core.core_eigenvalue_sum,
                                           rtol=1e-10, atol=1e-12)
                for key in reference_radials:
                    for name in ("large", "small", "boundary_values", "inverse_mass",
                                 "inverse_speed_of_light"):
                        np.testing.assert_allclose(getattr(radials[key], name),
                                                   getattr(reference_radials[key], name),
                                                   rtol=1e-10, atol=1e-12)
                for name in ("values", "radial_derivatives", "energy_derivatives",
                             "energy_radial_derivatives", "inverse_masses",
                             "energy_inverse_masses"):
                    np.testing.assert_allclose(getattr(jets, name), getattr(reference_jets, name),
                                               rtol=1e-10, atol=1e-12)
                exported = reference.export_interstitial()
                origin = np.flatnonzero(np.all(exported["g_vectors"] == 0, axis=1))[0]
                np.testing.assert_allclose(zero, exported["components"][0, origin].real,
                                           rtol=1e-10, atol=1e-12)
