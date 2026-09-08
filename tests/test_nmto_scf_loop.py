from types import SimpleNamespace

import numpy as np
import pytest

from pymuffintin.mto.electrons import NmtoBands, NmtoOccupations
from pymuffintin.mto.scf import (
    NmtoScfInput,
    NmtoScfSettings,
    _NmtoIteration,
    run_nmto_scf,
)


@pytest.mark.parametrize("use_mpi", [False, True])
@pytest.mark.parametrize("mixing_kind", ["linear", "pulay"])
def test_nmto_scf_mixes_until_energy_and_density_converge(monkeypatch, use_mpi, mixing_kind) -> None:
    comm = pytest.importorskip("mpi4py.MPI").COMM_WORLD if use_mpi else None
    root = comm is None or comm.rank == 0
    native_calls = {name: 0 for name in ("potential", "core", "mixer", "mix")}
    density = object()
    output_density = object()
    potential = object()
    core = SimpleNamespace(core_eigenvalue_sum=-0.5)
    bands = NmtoBands(
        energies=np.array([[-1.0, 1.0]]),
        orthonormal_coefficients=np.eye(2)[None],
        coefficients=np.eye(2)[None],
    )
    occupations = NmtoOccupations(
        chemical_potential=0.0,
        values=np.array([[2.0, 0.0]]),
        electron_count=2.0,
        band_energy=-2.0,
        minus_temperature_entropy=-0.01,
    )

    monkeypatch.setattr(
        "pymuffintin.mto.scf._solve_nmto_iteration",
        lambda scf_input, built_potential, built_core, **kwargs: _NmtoIteration(
            bands, occupations, output_density if root else None, 1.0
        ),
    )

    class Mixer:
        @staticmethod
        def step(input_density, result_density):
            native_calls["mix"] += 1
            assert input_density is density
            assert result_density is output_density
            return SimpleNamespace(density=lambda: density)

    calls = []

    def evaluate_total_energy(
        built_potential,
        result_density,
        band_energy,
        core_energy,
        occupation_correction,
        previous_total,
    ):
        calls.append(previous_total)
        converged = previous_total is not None
        return SimpleNamespace(
            total=-1.0 if converged else -1.1,
            density_rms=1.0e-6 if converged else 1.0e-2,
            energy_change=1.0e-6 if converged else None,
        )

    def build_potential(input_density, xc):
        native_calls["potential"] += 1
        return potential

    def solve_core(built_potential):
        native_calls["core"] += 1
        return core

    def make_mixer(beta):
        native_calls["mixer"] += 1
        return Mixer()

    def make_pulay(beta, history):
        assert history == 8
        return make_mixer(beta)

    native = SimpleNamespace(
        DensityMixer=SimpleNamespace(linear=make_mixer, pulay_anderson=make_pulay),
        build_regional_potential=build_potential,
        evaluate_total_energy=evaluate_total_energy,
    )

    def scheduled_potential(native, input_density, xc, **kwargs):
        return build_potential(input_density, xc) if root else None

    def scheduled_core(native, station, built_potential, **kwargs):
        return solve_core(built_potential) if root else None

    monkeypatch.setattr("pymuffintin.mto.scf.build_nmto_potential", scheduled_potential)
    monkeypatch.setattr("pymuffintin.mto.scf.solve_nmto_core", scheduled_core)
    scf_input = NmtoScfInput.from_python(
        native=native,
        structure=object(),
        field_layout=object(),
        initial_density=density,
        core_station=SimpleNamespace(solve=solve_core),
        lattice=np.eye(3),
        site_ids=("H-1",),
        atomic_numbers=(1,),
        fractional_positions=((0.0, 0.0, 0.0),),
        muffin_tin_radii=(0.2,),
        g_vectors=((0, 0, 0),),
        density_l_max=0,
        radial_equations=("schroedinger",),
        settings=NmtoScfSettings(
            electron_count=1.0,
            energy_mesh=(-0.2, 0.1),
            k_mesh=(1, 1, 1),
            reciprocal_cutoff=12.6,
            lattice_sum_radius=16.0,
            reference_energy=-1.0,
            mixing_kind=mixing_kind,
            mixing_history=8 if mixing_kind == "pulay" else None,
            matrix_angular_order=8,
            symmetry=False,
        ),
    )

    timings = []
    result = run_nmto_scf(
        scf_input, comm=comm,
        timing_callback=(lambda iteration, phase, seconds: timings.append((iteration, phase, seconds)))
        if root else None,
    )

    if not root:
        assert result is None
        assert calls == []
        assert native_calls == {"potential": 0, "core": 0, "mixer": 0, "mix": 0}
        return

    assert native_calls == {"potential": 2, "core": 2, "mixer": 1, "mix": 1}
    assert {phase for _, phase, _ in timings} >= {"nmto", "energy", "mix"}
    assert all(np.isfinite(seconds) and seconds >= 0 for _, _, seconds in timings)
    assert result.iterations == 2
    assert result.total_energy == -1.0
    assert result.density is density
    np.testing.assert_allclose(result.energy_history, [-1.1, -1.0])
    assert calls == [None, -1.1]
    np.testing.assert_allclose(result.valence_normalization_history, [1.0, 1.0])
    assert result.k_sampling is None
    assert result._restart_checkpoint is None
