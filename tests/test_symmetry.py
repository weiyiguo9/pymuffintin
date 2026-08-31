import numpy as np

from pymuffintin.symmetry import (
    SymmetryDataset,
    detect,
    little_group_irreps,
    little_group_spinor_irreps,
    reduce_regular_kmesh,
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


def test_diamond_3x3x3_ibz_has_four_exact_orbits() -> None:
    dataset = detect(
        FCC_LATTICE,
        np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
        np.array([6, 6]),
    )
    reduced = reduce_regular_kmesh(dataset, (3, 3, 3))

    assert reduced.divisions == (3, 3, 3)
    assert reduced.shift == (0.0, 0.0, 0.0)
    assert reduced.full_points.shape == (27, 3)
    assert reduced.representative_points.shape == (4, 3)
    np.testing.assert_array_equal(reduced.multiplicities, [1, 8, 6, 12])
    np.testing.assert_allclose(reduced.weights, np.array([1, 8, 6, 12]) / 27)
    np.testing.assert_array_equal(
        reduced.parent_indices,
        reduced.representative_indices[reduced.orbit_indices],
    )

    for point, representative in enumerate(reduced.parent_indices):
        rotation = dataset.rotations[reduced.operation_indices[point]]
        image = reduced.full_points[representative] @ np.linalg.inv(rotation)
        if reduced.operation_time_reversals[point]:
            image = -image
        residue = image - reduced.full_points[point]
        np.testing.assert_allclose(residue - np.rint(residue), 0.0, atol=1.0e-12)


def test_half_shifted_mesh_is_normalized_and_reduced() -> None:
    reduced = reduce_regular_kmesh(_fcc_dataset(), (2, 2, 2), (0.5, 0.5, 0.5))

    assert reduced.shift == (0.5, 0.5, 0.5)
    assert np.all((reduced.full_points >= 0.0) & (reduced.full_points < 1.0))
    np.testing.assert_array_equal(np.unique(reduced.full_points), [0.25, 0.75])
    np.testing.assert_array_equal(reduced.multiplicities, [2, 6])
    np.testing.assert_allclose(reduced.weights, [0.25, 0.75])


def test_optional_time_reversal_joins_opposite_k_points() -> None:
    identity_only = SymmetryDataset(
        rotations=np.eye(3, dtype=np.int64)[None, :, :],
        translations=np.zeros((1, 3), dtype=np.float64),
        time_reversals=np.zeros(1, dtype=np.bool_),
        equivalent_atoms=np.array([0], dtype=np.int64),
        spacegroup_number=1,
        hermann_mauguin="P1",
        provenance="test",
    )

    unitary = reduce_regular_kmesh(
        identity_only, (3, 1, 1), include_time_reversal=False
    )
    with_time_reversal = reduce_regular_kmesh(identity_only, (3, 1, 1))

    np.testing.assert_array_equal(unitary.multiplicities, [1, 1, 1])
    np.testing.assert_array_equal(with_time_reversal.multiplicities, [1, 2])
    assert with_time_reversal.parent_indices[2] == 1
    assert with_time_reversal.operation_indices[2] == 0
    assert with_time_reversal.operation_time_reversals[2]


def test_dataset_antiunitary_operation_is_kept_without_extra_time_reversal() -> None:
    identity = np.eye(3, dtype=np.int64)
    dataset = SymmetryDataset(
        rotations=np.stack([identity, identity]),
        translations=np.zeros((2, 3), dtype=np.float64),
        time_reversals=np.array([False, True]),
        equivalent_atoms=np.array([0], dtype=np.int64),
        spacegroup_number=1,
        hermann_mauguin="P1",
        provenance="test",
    )

    reduced = reduce_regular_kmesh(
        dataset, (3, 1, 1), include_time_reversal=False
    )

    np.testing.assert_array_equal(reduced.multiplicities, [1, 2])
    assert np.any(reduced.active_operation_time_reversals)


def test_incompatible_shifted_operation_is_excluded_exactly() -> None:
    rotations = np.asarray(
        [np.eye(3, dtype=np.int64), [[0, 1, 0], [1, 0, 0], [0, 0, 1]]]
    )
    dataset = SymmetryDataset(
        rotations=rotations,
        translations=np.zeros((2, 3), dtype=np.float64),
        time_reversals=np.zeros(2, dtype=np.bool_),
        equivalent_atoms=np.array([0], dtype=np.int64),
        spacegroup_number=None,
        hermann_mauguin=None,
        provenance="test",
    )

    reduced = reduce_regular_kmesh(
        dataset,
        (2, 2, 1),
        (0.5, 0.0, 0.0),
        include_time_reversal=False,
    )

    np.testing.assert_array_equal(reduced.multiplicities, [1, 1, 1, 1])
    np.testing.assert_array_equal(reduced.operation_indices, np.zeros(4, dtype=np.int64))
