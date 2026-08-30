from types import SimpleNamespace

import numpy as np

from pymuffintin.mto.electrons import NmtoBands, NmtoOccupations
from pymuffintin.mto.scf import (
    NmtoScfInput,
    NmtoScfSettings,
    _NmtoIteration,
    run_nmto_scf,
)


def test_nmto_scf_mixes_until_energy_and_density_converge(monkeypatch) -> None:
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
        lambda scf_input, built_potential, built_core: _NmtoIteration(
            bands, occupations, output_density
        ),
    )

    class Mixer:
        @staticmethod
        def step(input_density, result_density):
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

    native = SimpleNamespace(
        DensityMixer=SimpleNamespace(linear=lambda beta: Mixer()),
        build_regional_potential=lambda input_density, xc: potential,
        evaluate_total_energy=evaluate_total_energy,
    )
    scf_input = NmtoScfInput.from_python(
        native=native,
        structure=object(),
        field_layout=object(),
        initial_density=density,
        core_station=SimpleNamespace(solve=lambda built_potential: core),
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
            symmetry=False,
        ),
    )

    result = run_nmto_scf(scf_input)

    assert result.iterations == 2
    assert result.total_energy == -1.0
    assert result.density is output_density
    np.testing.assert_allclose(result.energy_history, [-1.1, -1.0])
    assert calls == [None, -1.1]
