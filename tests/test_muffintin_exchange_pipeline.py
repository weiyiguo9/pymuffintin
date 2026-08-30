import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pymuffintin.auxiliary import build_hybrid_coulomb, interstitial_thc, muffin_tin_lri
from pymuffintin.backends.muffintin import MuffintinAdapter
from pymuffintin.contracts import FixedOccupation
from pymuffintin.mbpt import compare_exchange, fixed_orbital_exchange


LIBMUFFINTIN = Path(__file__).resolve().parents[2] / "libmuffintin"
FIXTURES = LIBMUFFINTIN / "python" / "tests" / "fixtures"


def _hydrogen_parent_grid(
    adapter: MuffintinAdapter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    site_cartesian, mesh_offsets, mesh_radii = adapter.sampling_metadata(0)
    origin = site_cartesian[0]
    site_start = int(mesh_offsets[0])
    site_stop = int(mesh_offsets[1])
    middle = (site_stop - site_start) // 2
    first_radius = float(mesh_radii[site_start])
    middle_radius = float(mesh_radii[site_start + middle])

    def on_shell(radius: float, direction: tuple[float, float, float]) -> np.ndarray:
        vector = np.asarray(direction, dtype=np.float64)
        return origin + radius * vector / np.linalg.norm(vector)

    points = np.array(
        [
            on_shell(first_radius, (0.4, -0.3, 0.2)),
            on_shell(middle_radius, (1.0, 0.0, 0.0)),
            on_shell(middle_radius, (0.0, 1.0, 0.0)),
            (0.2, 0.2, 0.2),
            (5.0, 4.0, 4.0),
            (2.0, 6.5, 4.0),
        ],
        dtype=np.float64,
    )
    weights = np.array([0.35, 0.0, 0.45, 0.8, 0.15, 0.25], dtype=np.float64)
    regions = np.array(
        [
            [0, 0, 0],
            [0, 0, middle],
            [0, 0, middle],
            [1, -1, -1],
            [1, -1, -1],
            [1, -1, -1],
        ],
        dtype=np.int64,
    )
    return points, weights, regions


def _complex_blocks(values: np.ndarray) -> list[list[list[float]]]:
    return [
        [[float(entry.real), float(entry.imag)] for entry in block.ravel()]
        for block in values
    ]


def test_hydrogen_native_mpb_and_hybrid_exchange_pipeline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sys.path.insert(0, str(LIBMUFFINTIN / "python"))
    native = pytest.importorskip("libmuffintin")
    required_module_api = (
        "build_scalar_mpb",
        "build_scalar_mpb_coulomb",
        "sample_scalar_orbitals",
    )
    assert hasattr(native.CheckpointPhysics, "scalar_product_input")
    assert hasattr(native.CheckpointPhysics, "scalar_q_slice")
    assert all(hasattr(native, name) for name in required_module_api)

    adapter = MuffintinAdapter.from_files(
        FIXTURES / "hydrogen_checkpoint.toml",
        FIXTURES / "hydrogen_input.toml",
    )
    points, weights, regions = _hydrogen_parent_grid(adapter)
    reference_representations = []
    reference_coulomb = []
    hybrid_representations = []
    hybrid_coulomb = []

    for q_index in range(adapter.n_q):
        samples = adapter.sample(q_index, points, weights, regions, spin=0)
        mpb = adapter.build_mpb(
            q_index,
            spin=0,
            product_l_max=2,
            product_g_max=1.5,
            overlap_tolerance=1.0e-12,
        )
        local = muffin_tin_lri(samples, cutoff=1.0e-10)
        interstitial, _ = interstitial_thc(samples, rank=1)
        hybrid, hybrid_v = build_hybrid_coulomb(
            local,
            interstitial,
            adapter,
            gamma_policy="spherical_average_subtracted",
            lexp=2,
        )
        mpb_v = adapter.coulomb(
            mpb,
            gamma_policy="spherical_average_subtracted",
            lexp=2,
        )
        np.testing.assert_allclose(
            hybrid_v.matrix,
            hybrid_v.matrix.conj().T,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            hybrid.coefficients.conj() @ hybrid_v.matrix @ hybrid.coefficients.T,
            mpb.coefficients.conj() @ mpb_v.matrix @ mpb.coefficients.T,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        reference_representations.append(mpb)
        reference_coulomb.append(mpb_v)
        hybrid_representations.append(hybrid)
        hybrid_coulomb.append(hybrid_v)

    window = adapter.orbital_window(0, spin=0)
    occupation = FixedOccupation(
        values=np.ones((window.n_k, window.n_orb), dtype=np.float64),
        k_weights=np.full(window.n_k, 1.0 / window.n_k, dtype=np.float64),
        q_weights=np.full(adapter.n_q, 1.0 / adapter.n_q, dtype=np.float64),
        k_minus_q_indices=np.stack(
            [adapter.k_minus_q_indices(q_index) for q_index in range(adapter.n_q)]
        ),
    )
    reference = fixed_orbital_exchange(
        tuple(reference_representations), tuple(reference_coulomb), occupation
    )
    hybrid = fixed_orbital_exchange(
        tuple(hybrid_representations), tuple(hybrid_coulomb), occupation
    )
    ablation = compare_exchange(reference, hybrid)

    assert reference.sigma_x.shape == (window.n_k, 1, 1)
    assert hybrid.sigma_x.shape == (window.n_k, 1, 1)
    assert np.all(np.isfinite(ablation.sigma_difference))
    report = {
        "native_exchange_pipeline_check": True,
        "scope": "hydrogen fixture native exchange pipeline; not a material-accuracy claim",
        "gamma_policy": "spherical_average_subtracted",
        "reference_mpb_Ex": reference.exchange_energy,
        "hybrid_mt_lri_i_thc_Ex": hybrid.exchange_energy,
        "Ex_difference": ablation.exchange_energy_difference,
        "reference_mpb_Sigma_x_1x1": _complex_blocks(reference.sigma_x),
        "hybrid_mt_lri_i_thc_Sigma_x_1x1": _complex_blocks(hybrid.sigma_x),
        "Sigma_x_difference_1x1": _complex_blocks(ablation.sigma_difference),
    }
    print(json.dumps(report, sort_keys=True))
    assert '"native_exchange_pipeline_check": true' in capsys.readouterr().out
