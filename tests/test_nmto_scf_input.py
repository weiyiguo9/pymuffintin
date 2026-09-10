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
        reciprocal_cutoff=12.6,
        lattice_sum_radius=16.0,
        reference_energy=-1.0,
        mixing_kind="linear",
        mixing_history=None,
        matrix_angular_order=8,
        symmetry=symmetry,
    )


def test_settings_accept_zeroth_order_nmto_mesh() -> None:
    from dataclasses import replace

    settings = replace(_settings(), energy_mesh=(-0.1,))
    assert settings.energy_mesh == (-0.1,)


def test_python_input_detects_diamond_symmetry_by_default() -> None:
    prepared = NmtoScfInput.from_python(
        native=SimpleNamespace(),
        structure=object(),
        field_layout=object(),
        initial_density=object(),
        core_station=object(),
        lattice=DIAMOND_LATTICE,
        site_ids=("C-1", "C-2"),
        atomic_numbers=(6, 6),
        fractional_positions=DIAMOND_POSITIONS,
        muffin_tin_radii=(1.4, 1.4),
        g_vectors=((0, 0, 0),),
        density_l_max=0,
        radial_equations=("scalar-koelling-harmon",) * 2,
        settings=_settings(),
    )

    assert prepared.symmetry_dataset is not None
    assert prepared.symmetry_dataset.spacegroup_number == 227
    assert prepared.k_mesh_reduction is not None
    np.testing.assert_array_equal(prepared.k_mesh_reduction.multiplicities, [1, 4, 3])
    np.testing.assert_allclose(prepared.k_mesh_reduction.weights, [0.125, 0.5, 0.375])


def test_python_input_can_disable_symmetry() -> None:
    prepared = NmtoScfInput.from_python(
        native=SimpleNamespace(),
        structure=object(),
        field_layout=object(),
        initial_density=object(),
        core_station=object(),
        lattice=DIAMOND_LATTICE,
        site_ids=("C-1", "C-2"),
        atomic_numbers=(6, 6),
        fractional_positions=DIAMOND_POSITIONS,
        muffin_tin_radii=(1.4, 1.4),
        g_vectors=((0, 0, 0),),
        density_l_max=0,
        radial_equations=("scalar-koelling-harmon",) * 2,
        settings=_settings(symmetry=False),
    )

    assert prepared.symmetry_dataset is None
    assert prepared.k_mesh_reduction is None


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

[task.scf.basis.channels.C]
core = ["1s"]

[task.scf.occupations]
temperature = 0.02

[task.scf.xc]
kind = "lda-pw92"

[task.scf.mixing]
kind = "linear"
beta = 0.3

[task.scf.convergence]
energy-tolerance = 1e-5
density-tolerance = 1e-5
max-iterations = 40

[task.scf.nmto]
energy-mesh = [-0.1, 0.4]
reciprocal-cutoff = 12.6
lattice-sum-radius = 16.0
reference-energy = -1.0
matrix-angular-order = 8
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
            restart_density=lambda: density,
            structure=lambda: Structure(**{
                "lattice": [[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]],
            }),
        ),
        Structure=Structure,
        RegionalFieldLayout=FieldLayout,
        CoreState=lambda *values: values,
        CoreSite=lambda *values: values,
        CoreStation=lambda sites: sites,
    )

    prepared = NmtoScfInput.from_toml(input_path, native=native)

    assert prepared.initial_density is density
    assert prepared.settings.symmetry is True
    assert prepared.symmetry_dataset is not None
    assert prepared.settings.energy_mesh == (-0.1, 0.4)
    assert prepared.field_layout.g_vectors == [[0, 0, 0]]
    assert len(prepared.core_station) == 2


def _diamond_input(settings: NmtoScfSettings, **overrides):
    kwargs = dict(
        native=SimpleNamespace(),
        structure=object(),
        field_layout=object(),
        initial_density=object(),
        core_station=object(),
        lattice=DIAMOND_LATTICE,
        site_ids=("C-1", "C-2"),
        atomic_numbers=(6, 6),
        fractional_positions=DIAMOND_POSITIONS,
        muffin_tin_radii=(1.4, 1.4),
        g_vectors=((0, 0, 0),),
        density_l_max=0,
        radial_equations=("scalar-koelling-harmon",) * 2,
        settings=settings,
    )
    kwargs.update(overrides)
    return NmtoScfInput.from_python(**kwargs)


def test_settings_validate_the_reference_potential_recipe() -> None:
    import pytest
    from dataclasses import replace

    with pytest.raises(ValueError, match="reference_potential"):
        replace(_settings(), reference_potential="optimized")
    with pytest.raises(ValueError, match="potential_radius_scale"):
        replace(_settings(), reference_potential="omt", potential_radius_scale=0.9)
    with pytest.raises(ValueError, match="require reference_potential='omt'"):
        replace(_settings(), potential_radius_scale=1.2)
    with pytest.raises(ValueError, match="require reference_potential='omt'"):
        replace(_settings(), potential_radii={"C": 1.7})
    omt = replace(_settings(), reference_potential="omt", potential_radii={"C": 1.7})
    assert omt.potential_radii == {"C": 1.7}


def test_settings_read_the_omt_keys_from_a_task() -> None:
    task = {
        "electron-count": 12,
        "k-mesh": {"mesh": [2, 2, 2]},
        "basis": {"l-max": 2},
        "occupations": {"temperature": 0.001},
        "xc": {"kind": "pbe"},
        "mixing": {"kind": "linear", "beta": 0.3},
        "convergence": {
            "energy-tolerance": 1e-5,
            "density-tolerance": 1e-5,
            "max-iterations": 10,
        },
        "nmto": {
            "energy-mesh": [-0.1, 0.2],
            "reciprocal-cutoff": 12.6,
            "lattice-sum-radius": 16.0,
            "reference-energy": -1.0,
            "matrix-angular-order": 8,
            "reference-potential": "omt",
            "potential-radius-scale": 1.2,
            "potential-radii": {"C": 1.75},
        },
    }
    settings = NmtoScfSettings.from_task(task)
    assert settings.reference_potential == "omt"
    assert settings.potential_radius_scale == 1.2
    assert settings.potential_radii == {"C": 1.75}
    del task["nmto"]["reference-potential"]
    del task["nmto"]["potential-radius-scale"]
    del task["nmto"]["potential-radii"]
    default = NmtoScfSettings.from_task(task)
    assert default.reference_potential == "spherical-mt"
    assert default.potential_radii is None


def test_potential_sphere_radii_default_to_the_hard_spheres_and_are_validated() -> None:
    import pytest
    from dataclasses import replace

    prepared = _diamond_input(_settings())
    np.testing.assert_array_equal(prepared.potential_sphere_radii, prepared.muffin_tin_radii)
    with pytest.raises(ValueError, match="require reference_potential='omt'"):
        _diamond_input(_settings(), potential_sphere_radii=(1.7, 1.7))
    with pytest.raises(ValueError, match="not be smaller"):
        _diamond_input(
            replace(_settings(), reference_potential="omt"), potential_sphere_radii=(1.3, 1.4)
        )
    with pytest.raises(ValueError, match="incompatible sites"):
        _diamond_input(
            replace(_settings(), reference_potential="omt"), potential_sphere_radii=(1.7, 1.4)
        )
    omt = _diamond_input(
        replace(_settings(), reference_potential="omt"), potential_sphere_radii=(1.7, 1.7)
    )
    np.testing.assert_array_equal(omt.potential_sphere_radii, [1.7, 1.7])


def test_recipe_annotations_distinguish_the_reference_and_sphere_roles() -> None:
    from dataclasses import replace

    from pymuffintin.mto.scf import _checkpoint_annotations, recipe_annotations

    default = recipe_annotations(_diamond_input(_settings()))
    assert default["nmto.recipe.reference_potential"] == "spherical-mt"
    assert default["nmto.recipe.reference_constant"] == "interstitial-g0"
    assert default["nmto.recipe.augmentation"] == "single-hard-sphere"
    assert default["nmto.recipe.hard_sphere_radii_bohr"] == "1.4,1.4"
    nearest = float(default["nmto.recipe.nearest_neighbor_bohr"])
    np.testing.assert_allclose(nearest, 6.740879853675 * np.sqrt(3.0) / 4.0)
    np.testing.assert_allclose(
        float(default["nmto.recipe.hard_sphere_max_overlap"]), 2.8 / nearest - 1.0
    )
    assert float(default["nmto.recipe.hard_sphere_max_overlap"]) < 0.0

    omt = recipe_annotations(
        _diamond_input(
            replace(_settings(), reference_potential="omt"), potential_sphere_radii=(1.7, 1.7)
        )
    )
    assert omt["nmto.recipe.reference_potential"] == "omt-shells"
    assert omt["nmto.recipe.reference_constant"] == "least-squares"
    assert omt["nmto.recipe.augmentation"] == "single-hard-sphere+additive-shells"
    assert omt["nmto.recipe.potential_sphere_radii_bohr"] == "1.7,1.7"
    np.testing.assert_allclose(
        float(omt["nmto.recipe.potential_sphere_max_overlap"]), 3.4 / nearest - 1.0
    )
    assert float(omt["nmto.recipe.potential_sphere_max_overlap"]) > 0.0
    for key in (
        "nmto.recipe.envelope",
        "nmto.recipe.field_representation",
        "nmto.recipe.coulomb",
        "nmto.recipe.full_potential_matrix",
        "nmto.recipe.hard_sphere_roles",
    ):
        assert default[key] == omt[key]

    annotations = _checkpoint_annotations(
        _diamond_input(_settings()), 3, -1.0, reference_constant=-0.02, reference_rms=float("nan")
    )
    assert annotations["nmto.scf.reference_constant_hartree"] == repr(-0.02)
    assert "nmto.scf.reference_fit_rms_hartree" not in annotations
    assert annotations["nmto.recipe.family"] == "fp-nmto"


def test_resolve_potential_sphere_radii_snaps_up_to_the_native_mesh() -> None:
    from dataclasses import replace

    from pymuffintin.mto.scf import resolve_potential_sphere_radii
    from pymuffintin.mto.shell import exponential_mesh

    first, increment, count = 4.58863382e-05, 0.022, 471
    mesh = exponential_mesh(first, increment, count)
    hard = (float(mesh[-1]),) * 2
    meshes = [(first, increment, count)] * 2
    default = resolve_potential_sphere_radii(_settings(), ("C-1", "C-2"), hard, meshes)
    np.testing.assert_array_equal(default, hard)
    scaled = resolve_potential_sphere_radii(
        replace(_settings(), reference_potential="omt", potential_radius_scale=1.2),
        ("C-1", "C-2"),
        hard,
        meshes,
    )
    expected = first * np.exp(increment * 479)
    assert expected >= 1.2 * hard[0] and expected < 1.2 * hard[0] * np.exp(increment)
    np.testing.assert_allclose(scaled, [expected, expected], rtol=1e-12)
    by_species = resolve_potential_sphere_radii(
        replace(
            _settings(),
            reference_potential="omt",
            potential_radius_scale=1.2,
            potential_radii={"C": float(mesh[-1])},
        ),
        ("C-1", "C-2"),
        hard,
        meshes,
    )
    np.testing.assert_array_equal(by_species, hard)
