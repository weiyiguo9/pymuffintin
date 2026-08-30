import numpy as np

from pymuffintin.symmetry import (
    detect,
    little_group_irreps,
    little_group_spinor_irreps,
)

FCC_LATTICE = 3.6 * np.array(
    [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
)


def _fcc_dataset():
    return detect(FCC_LATTICE, np.zeros((1, 3)), np.array([62]))


def test_detect_fcc_primitive_cell() -> None:
    dataset = _fcc_dataset()
    assert dataset.spacegroup_number == 225
    assert dataset.provenance == "spglib"
    assert dataset.rotations.shape == (48, 3, 3)
    assert dataset.translations.shape == (48, 3)
    np.testing.assert_array_equal(dataset.equivalent_atoms, [0])
    identity_hits = (dataset.rotations == np.eye(3, dtype=np.int64)).all(axis=(1, 2))
    assert identity_hits.sum() == 1


def test_gamma_point_irreps_span_group_order() -> None:
    dataset = _fcc_dataset()
    irreps, mapping = little_group_irreps(dataset, np.zeros(3))
    assert mapping.shape == (48,)
    assert sum(irrep.shape[1] ** 2 for irrep in irreps) == 48
    for irrep in irreps:
        assert irrep.shape[0] == 48
        assert irrep.shape[1] == irrep.shape[2]


def test_gamma_point_spinor_irreps_span_group_order() -> None:
    dataset = _fcc_dataset()
    irreps, factor_system, su2, mapping = little_group_spinor_irreps(
        FCC_LATTICE, dataset, np.zeros(3)
    )
    assert mapping.shape == (48,)
    assert su2.shape == (48, 2, 2)
    assert factor_system.shape == (48, 48)
    assert sum(irrep.shape[1] ** 2 for irrep in irreps) == 48
