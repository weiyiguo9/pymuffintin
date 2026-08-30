from pathlib import Path

import numpy as np

from pymuffintin.spex_log import parse_spex_log

FIXTURE = Path(__file__).parent / "data" / "spex_mnte.log"


def _parsed():
    return parse_spex_log(FIXTURE.read_text())


def test_operation_table() -> None:
    log = _parsed()
    assert log.rotations.shape == (8, 3, 3)
    np.testing.assert_array_equal(log.rotations[0], np.eye(3, dtype=np.int64))
    np.testing.assert_array_equal(log.rotations[1], np.diag([-1, -1, 1]))
    np.testing.assert_array_equal(
        log.rotations[4], [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    )
    np.testing.assert_allclose(log.translations[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(log.translations[1], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(log.translations[6], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(log.translations[7], [0.0, 0.0, 0.5])
    np.testing.assert_array_equal(
        log.time_reversals, [False] * 4 + [True] * 4
    )
    np.testing.assert_array_equal(log.inverse, np.arange(8))


def test_operations_close_and_match_inverses() -> None:
    log = _parsed()
    for i in range(len(log.rotations)):
        j = log.inverse[i]
        np.testing.assert_array_equal(
            log.rotations[i] @ log.rotations[j], np.eye(3, dtype=np.int64)
        )
        residue = log.rotations[i] @ log.translations[j] + log.translations[i]
        np.testing.assert_allclose(residue - np.rint(residue), 0.0, atol=1e-12)


def test_atom_basis_and_orbits() -> None:
    log = _parsed()
    np.testing.assert_array_equal(log.atomic_numbers, [25, 25, 52, 52])
    np.testing.assert_allclose(log.site_positions[1], [0.0, 0.0, 0.5])
    mapping = log.atom_map()
    assert mapping.shape == (8, 4)
    np.testing.assert_array_equal(mapping[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(mapping[1], [1, 0, 3, 2])
    dataset = log.to_dataset()
    np.testing.assert_array_equal(dataset.equivalent_atoms, [0, 0, 2, 2])
    assert dataset.spacegroup_number is None
    assert dataset.provenance == "spex-log"


def test_ibz_table() -> None:
    log = _parsed()
    assert log.kpoint_total == 512
    assert log.ibz_kpoints.shape == (105, 3)
    assert int(log.ibz_weights.sum()) == 512
    np.testing.assert_allclose(log.ibz_kpoints[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(log.ibz_kpoints[104], [0.5, 0.5, 0.5])
    assert log.ibz_weights[0] == 1
    assert log.ibz_weights[6] == 8


def test_spexsym_v1_export_structure(tmp_path) -> None:
    h5py = __import__("h5py")
    log = _parsed()
    path = tmp_path / "mnte_spexsym.h5"
    log.to_spexsym_v1(str(path), "spex-fixture")
    with h5py.File(path, "r") as file:
        assert file.attrs["schema_name"] == "libmuffintin.spexsym"
        assert file.attrs["schema_version"] == 1
        assert file.attrs["schema_version"].dtype == np.uint32
        assert file["symmetry/rotations"].shape == (8, 3, 3)
        assert file["symmetry/rotations"].dtype == np.int32
        axes = [
            value.decode() if isinstance(value, bytes) else value
            for value in file["symmetry/rotations"].attrs["axes"]
        ]
        assert axes == ["operation", "row", "column"]
        assert file["symmetry/time_reversal"][()].sum() == 4
        assert file["kpoints"].attrs["irreducible_count"] == 105
        assert file["kpoints/fractional"].shape == (105, 3)
        np.testing.assert_array_equal(file["kpoints/parent"][()], np.arange(105))
        np.testing.assert_array_equal(file["kpoints/parent_operation"][()], 0)
        assert file["irreps"].attrs["block_count"] == 0
