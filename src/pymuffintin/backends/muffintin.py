from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..contracts import (
    AuxiliaryRepresentation,
    CoulombBlock,
    OrbitalWindow,
    PairLayout,
    PairSamples,
    RegionalChargeExpansion,
)


_SCHEMA = "libmuffintin.pyexport"
_VERSION = 1


def _export(value: object, source: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{source} must return a pyexport dict, got {type(value).__name__}")
    if value.get("schema") != _SCHEMA:
        raise ValueError(f"{source} schema must be {_SCHEMA!r}, got {value.get('schema')!r}")
    if value.get("version") != _VERSION:
        raise ValueError(f"{source} version must be {_VERSION}, got {value.get('version')!r}")
    return value


def _layout(value: Mapping[str, Any]) -> PairLayout:
    return PairLayout(
        n_k=value["n_k"],
        n_orb=value["n_orb"],
        n_columns=value["n_columns"],
        core_orbital=value["core_orbital"],
        pair_order=value["pair_order"],
    )


def _region_name(row: NDArray[np.int64]) -> str:
    kind, first, second, third, fourth = (int(value) for value in row)
    if kind == 0:
        return f"muffin_tin:{first}:l={second}:m={third}:n={fourth}"
    if kind == 1:
        return f"interstitial:g={first},{second},{third}"
    if kind == 2:
        interpolation_region = {0: "muffin_tin", 1: "interstitial", 2: "uniform"}[second]
        return f"interpolation_point:{first}:{interpolation_region}:site={third}"
    raise ValueError(f"unknown native auxiliary-region kind {kind}")


def _representation(
    value: Mapping[str, Any],
    q_index: int,
    layout: PairLayout,
    source: str,
) -> AuxiliaryRepresentation:
    exported = _export(value, source)
    coefficients = exported["coefficients"]
    labels = exported["labels"]
    regions = exported["regions"]
    if not isinstance(coefficients, np.ndarray) or coefficients.dtype != np.complex128:
        raise TypeError(f"{source} coefficients must be a complex128 ndarray")
    if not isinstance(labels, np.ndarray) or labels.dtype != np.int64:
        raise TypeError(f"{source} labels must be an int64 ndarray")
    if not isinstance(regions, np.ndarray) or regions.dtype != np.int64:
        raise TypeError(f"{source} regions must be an int64 ndarray")
    if coefficients.shape != (layout.n_columns, regions.shape[0]):
        raise ValueError(
            f"{source} coefficients shape must be {(layout.n_columns, regions.shape[0])}, "
            f"got {coefficients.shape}"
        )
    if labels.shape != (layout.n_columns, 5) or regions.ndim != 2 or regions.shape[1] != 5:
        raise ValueError(f"{source} labels and regions must have shapes (n_pair, 5) and (n_aux, 5)")

    pair_columns = labels[:, 4]
    if not np.array_equal(np.sort(pair_columns), np.arange(layout.n_columns, dtype=np.int64)):
        raise ValueError(f"{source} must contain every pair column exactly once")
    ordered = np.empty_like(coefficients)
    ordered[pair_columns] = coefficients

    names = tuple(_region_name(row) for row in regions)
    blocks: list[RegionalChargeExpansion] = []
    start = 0
    while start < len(names):
        stop = start + 1
        while stop < len(names) and names[stop] == names[start]:
            stop += 1
        blocks.append(
            RegionalChargeExpansion(
                region=names[start],
                coefficients=np.asarray(ordered[:, start:stop], dtype=np.complex128),
            )
        )
        start = stop
    return AuxiliaryRepresentation(
        q_index=q_index,
        layout=layout,
        expansions=tuple(blocks),
        residual_norm=0.0,
    )


@dataclass(frozen=True)
class _MpbCache:
    spin: int
    product_l_max: int
    product_g_max: float
    overlap_tolerance: float
    handle: object
    representation: AuxiliaryRepresentation


class MuffintinAdapter:
    """DTO adapter over the libmuffintin scalar Stage 2 pyexport v1 API."""

    def __init__(
        self,
        native: ModuleType,
        inputs: tuple[object, ...],
    ) -> None:
        self.__native = native
        self.__inputs = inputs
        self.__mpb: dict[int, _MpbCache] = {}
        self.__mpb_coulomb: dict[tuple[int, int], NDArray[np.complex128]] = {}

    @classmethod
    def from_files(
        cls, checkpoint_path: str | Path, input_path: str | Path
    ) -> MuffintinAdapter:
        native = import_module("libmuffintin")
        checkpoint = native.load_checkpoint(Path(checkpoint_path))
        physics = native.CheckpointPhysics(checkpoint)
        path = Path(input_path)
        gamma = physics.scalar_product_input(path, q=[0.0, 0.0, 0.0])
        q_points = _export(gamma.export_orbitals(), "export_orbitals")["k_fractional"]
        inputs = tuple(
            gamma
            if np.all(np.abs(q) <= 1.0e-12)
            else physics.scalar_product_input(path, q=q)
            for q in q_points
        )
        return cls(native, inputs)

    @property
    def n_q(self) -> int:
        return len(self.__inputs)

    def _input(self, q_index: int) -> object:
        if type(q_index) is not int or not 0 <= q_index < len(self.__inputs):
            raise IndexError(f"q_index {q_index} is outside [0, {len(self.__inputs)})")
        return self.__inputs[q_index]

    def _pair_layout(self, q_index: int) -> PairLayout:
        handle = self._input(q_index)
        return _layout(_export(handle.export_pair_layout(), "export_pair_layout"))

    def sampling_metadata(
        self, q_index: int
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        handle = self._input(q_index)
        geometry = _export(handle.export_geometry(), "export_geometry")
        radials = _export(handle.export_radials(), "export_radials")
        return (
            geometry["site_cartesian"].copy(),
            radials["mesh_offsets"].copy(),
            radials["mesh_radii"].copy(),
        )

    def orbital_window(self, q_index: int, *, spin: int = 0) -> OrbitalWindow:
        handle = self._input(q_index)
        exported = _export(handle.export_orbitals(), "export_orbitals")
        channels = [channel for channel in exported["channels"] if channel["spin"] == spin]
        if len(channels) != 1:
            raise ValueError(f"export_orbitals must contain exactly one spin={spin} channel")
        channel = channels[0]
        return OrbitalWindow(
            k_fractional=exported["k_fractional"],
            energies=channel["energies"],
            eigenvectors=tuple(channel["eigenvectors"]),
            available_bands=channel["available_bands"],
            band_start=exported["band_window_start"],
            spin=spin,
        )

    def scalar_q_slice(self, *, spin: int = 0) -> tuple[OrbitalWindow, ...]:
        return tuple(self.orbital_window(q_index, spin=spin) for q_index in range(self.n_q))

    def k_minus_q_indices(self, q_index: int) -> NDArray[np.int64]:
        exported = _export(self._input(q_index).export_kq_map(), "export_kq_map")
        indices = exported["kq_index"]
        if not isinstance(indices, np.ndarray) or indices.dtype != np.int64:
            raise TypeError("export_kq_map kq_index must be an int64 ndarray")
        return indices.copy()

    def sample(
        self,
        q_index: int,
        points: NDArray[np.float64],
        weights: NDArray[np.float64],
        regions: NDArray[np.int64],
        *,
        spin: int = 0,
    ) -> PairSamples:
        handle = self._input(q_index)
        exported = _export(
            self.__native.sample_scalar_orbitals(
                handle, points, weights, regions, spin=spin
            ),
            "sample_scalar_orbitals",
        )
        layout = self._pair_layout(q_index)
        large = exported["large"]
        small = exported["small"]
        expected_shape = (points.shape[0], layout.n_k, layout.n_orb)
        if (
            not isinstance(large, np.ndarray)
            or large.dtype != np.complex128
            or large.shape != expected_shape
            or not isinstance(small, np.ndarray)
            or small.dtype != np.complex128
            or small.shape != expected_shape
        ):
            raise TypeError(
                f"sample_scalar_orbitals large/small must be complex128 arrays with shape {expected_shape}"
            )

        mapping = _export(handle.export_kq_map(), "export_kq_map")
        k_indices = mapping["k_index"]
        kq_indices = mapping["kq_index"]
        g_wrap = mapping["g_wrap_cartesian"]
        values = np.empty((points.shape[0], layout.n_columns), dtype=np.complex128)
        for map_index in range(layout.n_k):
            k_index = int(k_indices[map_index])
            kq_index = int(kq_indices[map_index])
            phase = np.exp(1j * (points @ g_wrap[map_index]))
            block = (
                large[:, kq_index, :].conj()[:, :, None] * large[:, k_index, :][:, None, :]
                + small[:, kq_index, :].conj()[:, :, None] * small[:, k_index, :][:, None, :]
            )
            start = k_index * layout.n_orb * layout.n_orb
            stop = start + layout.n_orb * layout.n_orb
            values[:, start:stop] = phase[:, None] * block.reshape(points.shape[0], -1)

        site_indices = np.asarray(
            np.where(regions[:, 0] == 0, regions[:, 1], -1), dtype=np.int64
        )
        return PairSamples(
            q_index=q_index,
            layout=layout,
            points=points,
            weights=weights,
            site_indices=site_indices,
            values=values,
        )

    def build_mpb(
        self,
        q_index: int,
        *,
        spin: int = 0,
        product_l_max: int,
        product_g_max: float,
        overlap_tolerance: float,
    ) -> AuxiliaryRepresentation:
        cached = self.__mpb.get(q_index)
        if cached is not None and (
            cached.spin,
            cached.product_l_max,
            cached.product_g_max,
            cached.overlap_tolerance,
        ) == (spin, product_l_max, product_g_max, overlap_tolerance):
            return cached.representation

        layout = self._pair_layout(q_index)
        selections = np.asarray(
            [
                (spin, k_index, left, right)
                for k_index in range(layout.n_k)
                for left in range(layout.n_orb)
                for right in range(layout.n_orb)
            ],
            dtype=np.int64,
        )
        handle = self.__native.build_scalar_mpb(
            self._input(q_index),
            selections,
            product_l_max,
            product_g_max,
            overlap_tolerance,
        )
        representation = _representation(
            handle.export_vertices(), q_index, layout, "ScalarMpbResult.export_vertices"
        )
        self.__mpb[q_index] = _MpbCache(
            spin=spin,
            product_l_max=product_l_max,
            product_g_max=product_g_max,
            overlap_tolerance=overlap_tolerance,
            handle=handle,
            representation=representation,
        )
        self.__mpb_coulomb = {
            key: value for key, value in self.__mpb_coulomb.items() if key[0] != q_index
        }
        return representation

    def coulomb(
        self,
        representation: AuxiliaryRepresentation,
        *,
        gamma_policy: str,
        lexp: int,
    ) -> CoulombBlock:
        if gamma_policy != "spherical_average_subtracted":
            raise ValueError(
                "libmuffintin MPB Coulomb requires gamma_policy='spherical_average_subtracted'"
            )
        cached = self.__mpb.get(representation.q_index)
        if cached is None:
            raise ValueError(f"build_mpb must be called before coulomb at q={representation.q_index}")
        if representation.layout != cached.representation.layout:
            raise ValueError("Coulomb representation and cached MPB pair layouts must match")

        key = (representation.q_index, lexp)
        reference_matrix = self.__mpb_coulomb.get(key)
        if reference_matrix is None:
            handle = self.__native.build_scalar_mpb_coulomb(cached.handle, lexp)
            exported = _export(
                handle.export_matrix(), "ScalarMpbCoulombResult.export_matrix"
            )
            reference_matrix = exported["matrix"]
            if (
                not isinstance(reference_matrix, np.ndarray)
                or reference_matrix.dtype != np.complex128
            ):
                raise TypeError("ScalarMpbCoulombResult matrix must be a complex128 ndarray")
            self.__mpb_coulomb[key] = reference_matrix

        if representation is cached.representation:
            matrix = reference_matrix.copy()
        else:
            reference_coefficients = cached.representation.coefficients
            pair_kernel = (
                reference_coefficients.conj()
                @ reference_matrix
                @ reference_coefficients.T
            )
            trial_map = representation.coefficients.conj()
            projector = np.linalg.pinv(trial_map)
            matrix = projector @ pair_kernel @ projector.conj().T
            matrix = np.asarray(0.5 * (matrix + matrix.conj().T), dtype=np.complex128)
        return CoulombBlock(
            q_index=representation.q_index,
            matrix=matrix,
            gamma_policy=gamma_policy,
        )
