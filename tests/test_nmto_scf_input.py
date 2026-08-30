from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymuffintin.mto.scf import NmtoScfInput, NmtoScfSettings


DIAMOND_LATTICE = 6.740879853675 * np.array(
    [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
)
DIAMOND_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])


def _settings(*, symmetry: bool = True) -> NmtoScfSettings:
    return NmtoScfSettings(
        electron_count=12.0,
        energy_mesh=(-0.1, 0.4),
        k_mesh=(2, 2, 2),
        symmetry=symmetry,
    )


def test_python_input_detects_diamond_symmetry_by_default() -> None:
    prepared = NmtoScfInput.from_python(
        structure=object(),
        field_layout=object(),
        initial_density=object(),
        lattice=DIAMOND_LATTICE,
        site_ids=("C-1", "C-2"),
        atomic_numbers=(6, 6),
        fractional_positions=DIAMOND_POSITIONS,
        muffin_tin_radii=(1.4, 1.4),
        settings=_settings(),
    )

    assert prepared.symmetry_dataset is not None
    assert prepared.symmetry_dataset.spacegroup_number == 227


def test_python_input_can_disable_symmetry() -> None:
    prepared = NmtoScfInput.from_python(
        structure=object(),
        field_layout=object(),
        initial_density=object(),
        lattice=DIAMOND_LATTICE,
        site_ids=("C-1", "C-2"),
        atomic_numbers=(6, 6),
        fractional_positions=DIAMOND_POSITIONS,
        muffin_tin_radii=(1.4, 1.4),
        settings=_settings(symmetry=False),
    )

    assert prepared.symmetry_dataset is None


def test_toml_input_reaches_the_same_prepared_type(tmp_path) -> None:
    checkpoint_path = tmp_path / "diamond_checkpoint.toml"
    checkpoint_path.write_text(
        """
format = "libmuffintin-checkpoint"
version = 2

[geometry.lattice]
unit = "bohr"
vectors = [[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]]

[[geometry.sites]]
id = "C-1"
atomic_number = 6
fractional_position = [0.0, 0.0, 0.0]
muffin_tin_radius = 1.4

[[geometry.sites]]
id = "C-2"
atomic_number = 6
fractional_position = [0.25, 0.25, 0.25]
muffin_tin_radius = 1.4

[[geometry.radial_basis]]
site_id = "C-1"
radial_equation = "scalar-koelling-harmon"
[geometry.radial_basis.mesh]
first = 0.000002
log_increment = 0.03
point_count = 401
[geometry.radial_basis.linearization]
linearization_energies = [{l = 0, energy = -0.5}]

[[geometry.radial_basis]]
site_id = "C-2"
radial_equation = "scalar-koelling-harmon"
[geometry.radial_basis.mesh]
first = 0.000002
log_increment = 0.03
point_count = 401
[geometry.radial_basis.linearization]
linearization_energies = [{l = 0, energy = -0.5}]

[initial.density]
[[initial.density.n.muffin_tins]]
site_id = "C-1"
channels = [{l = 0, m = 0}]
[[initial.density.n.muffin_tins]]
site_id = "C-2"
channels = [{l = 0, m = 0}]
[initial.density.n.interstitial]
coefficients = [{g = [0, 0, 0]}]
""".strip()
        + "\n"
    )
    input_path = tmp_path / "diamond_input.toml"
    input_path.write_text(
        """
format = "libmuffintin-input"
version = 3
checkpoint = "diamond_checkpoint.toml"

[workflow]
tasks = ["scf"]

[task.scf]
kind = "dft-scf"
electron-count = 12

[task.scf.k-mesh]
mesh = [2, 2, 2]

[task.scf.basis]
l-max = 2

[task.scf.occupations]
temperature = 0.02

[task.scf.xc]
kind = "lda-pw92"

[task.scf.mixing]
beta = 0.3

[task.scf.convergence]
energy-tolerance = 1e-5
density-tolerance = 1e-5
max-iterations = 40

[task.scf.nmto]
energy-mesh = [-0.1, 0.4]
""".strip()
        + "\n"
    )

    density = object()

    class Structure:
        def __init__(self, **values):
            self.values = values

    class FieldLayout:
        def __init__(self, structure, *, g_vectors, muffin_tin_l_max):
            self.structure = structure
            self.g_vectors = g_vectors
            self.muffin_tin_l_max = muffin_tin_l_max

    native = SimpleNamespace(
        load_checkpoint=lambda path: path,
        CheckpointPhysics=lambda checkpoint: SimpleNamespace(
            restart_density=lambda: density
        ),
        Structure=Structure,
        RegionalFieldLayout=FieldLayout,
    )

    prepared = NmtoScfInput.from_toml(input_path, native=native)

    assert prepared.initial_density is density
    assert prepared.settings.symmetry is True
    assert prepared.symmetry_dataset is not None
    assert prepared.settings.energy_mesh == (-0.1, 0.4)
    assert prepared.field_layout.g_vectors == [[0, 0, 0]]
