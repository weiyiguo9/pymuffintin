from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]

PAIR_ORDER = "k*n_orb^2 + i*n_orb + j"


def require_array(
    name: str,
    value: Any,
    dtype: np.dtype[Any] | type[Any],
    shape: tuple[int | None, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(value).__name__}")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {value.dtype}")
    if value.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got shape {value.shape}")
    for axis, (actual, expected) in enumerate(zip(value.shape, shape, strict=True)):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} axis {axis} must have length {expected}, got shape {value.shape}"
            )
    return value


@dataclass(frozen=True)
class OrbitalWindow:
    k_fractional: FloatArray
    energies: FloatArray
    eigenvectors: tuple[ComplexArray, ...]
    available_bands: IntArray
    band_start: int = 0
    spin: int = 0

    def __post_init__(self) -> None:
        k_fractional = require_array("k_fractional", self.k_fractional, np.float64, (None, 3))
        energies = require_array("energies", self.energies, np.float64, (k_fractional.shape[0], None))
        require_array("available_bands", self.available_bands, np.int64, (k_fractional.shape[0],))
        if not isinstance(self.eigenvectors, tuple):
            raise TypeError("eigenvectors must be a tuple of complex128 arrays")
        if len(self.eigenvectors) != k_fractional.shape[0]:
            raise ValueError(
                f"eigenvectors must contain one matrix per k point ({k_fractional.shape[0]}), "
                f"got {len(self.eigenvectors)}"
            )
        for k_index, matrix in enumerate(self.eigenvectors):
            require_array(
                f"eigenvectors[{k_index}]", matrix, np.complex128, (None, energies.shape[1])
            )
        if type(self.band_start) is not int or self.band_start < 0:
            raise ValueError("band_start must be a non-negative int")
        if type(self.spin) is not int or self.spin < 0:
            raise ValueError("spin must be a non-negative int")

    @property
    def n_k(self) -> int:
        return self.energies.shape[0]

    @property
    def n_orb(self) -> int:
        return self.energies.shape[1]


@dataclass(frozen=True)
class PairLayout:
    n_k: int
    n_orb: int
    n_columns: int
    core_orbital: int | None = None
    pair_order: str = PAIR_ORDER

    def __post_init__(self) -> None:
        if type(self.n_k) is not int or self.n_k <= 0:
            raise ValueError("n_k must be a positive int")
        if type(self.n_orb) is not int or self.n_orb <= 0:
            raise ValueError("n_orb must be a positive int")
        expected = self.n_k * self.n_orb * self.n_orb
        if type(self.n_columns) is not int or self.n_columns != expected:
            raise ValueError(f"n_columns must equal n_k*n_orb^2 ({expected}), got {self.n_columns}")
        if self.core_orbital is not None and not 0 <= self.core_orbital < self.n_orb:
            raise ValueError("core_orbital must be None or an orbital index in the pair layout")
        if self.pair_order != PAIR_ORDER:
            raise ValueError(f"pair_order must be {PAIR_ORDER!r}, got {self.pair_order!r}")

    def column(self, k_index: int, left: int, right: int) -> int:
        if not 0 <= k_index < self.n_k:
            raise IndexError(f"k_index {k_index} is outside [0, {self.n_k})")
        if not 0 <= left < self.n_orb or not 0 <= right < self.n_orb:
            raise IndexError("orbital index is outside the retained window")
        return k_index * self.n_orb * self.n_orb + left * self.n_orb + right


@dataclass(frozen=True)
class PairSamples:
    q_index: int
    layout: PairLayout
    points: FloatArray
    weights: FloatArray
    site_indices: IntArray
    values: ComplexArray

    def __post_init__(self) -> None:
        points = require_array("points", self.points, np.float64, (None, 3))
        require_array("weights", self.weights, np.float64, (points.shape[0],))
        require_array("site_indices", self.site_indices, np.int64, (points.shape[0],))
        require_array(
            "values", self.values, np.complex128, (points.shape[0], self.layout.n_columns)
        )
        if type(self.q_index) is not int or self.q_index < 0:
            raise ValueError("q_index must be a non-negative int")
        if np.any(self.weights < 0.0):
            raise ValueError("weights must be non-negative")
        if np.any(self.site_indices < -1):
            raise ValueError("site_indices uses -1 for interstitial points and non-negative sites")


@dataclass(frozen=True)
class RegionalChargeExpansion:
    region: str
    coefficients: ComplexArray

    def __post_init__(self) -> None:
        if not isinstance(self.region, str) or not self.region:
            raise ValueError("region must be a non-empty string")
        require_array("coefficients", self.coefficients, np.complex128, (None, None))

    @property
    def n_columns(self) -> int:
        return self.coefficients.shape[0]

    @property
    def n_auxiliary(self) -> int:
        return self.coefficients.shape[1]


@dataclass(frozen=True)
class AuxiliaryRepresentation:
    q_index: int
    layout: PairLayout
    expansions: tuple[RegionalChargeExpansion, ...]
    residual_norm: float

    def __post_init__(self) -> None:
        if type(self.q_index) is not int or self.q_index < 0:
            raise ValueError("q_index must be a non-negative int")
        if not isinstance(self.expansions, tuple) or not self.expansions:
            raise ValueError("expansions must be a non-empty tuple")
        for index, expansion in enumerate(self.expansions):
            if not isinstance(expansion, RegionalChargeExpansion):
                raise TypeError(f"expansions[{index}] must be RegionalChargeExpansion")
            if expansion.n_columns != self.layout.n_columns:
                raise ValueError(
                    f"expansions[{index}] has {expansion.n_columns} pair columns, "
                    f"expected {self.layout.n_columns}"
                )
        if not isinstance(self.residual_norm, (float, np.floating)) or not np.isfinite(
            self.residual_norm
        ) or self.residual_norm < 0.0:
            raise ValueError("residual_norm must be a finite non-negative float")

    @property
    def coefficients(self) -> ComplexArray:
        return np.concatenate([block.coefficients for block in self.expansions], axis=1)

    @property
    def n_auxiliary(self) -> int:
        return sum(block.n_auxiliary for block in self.expansions)


@dataclass(frozen=True)
class CoulombBlock:
    q_index: int
    matrix: ComplexArray
    gamma_policy: str

    def __post_init__(self) -> None:
        matrix = require_array("matrix", self.matrix, np.complex128, (None, None))
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"matrix must be square, got shape {matrix.shape}")
        if type(self.q_index) is not int or self.q_index < 0:
            raise ValueError("q_index must be a non-negative int")
        if not isinstance(self.gamma_policy, str) or not self.gamma_policy:
            raise ValueError("gamma_policy must be an explicit non-empty string")


@dataclass(frozen=True)
class FixedOccupation:
    values: FloatArray
    k_weights: FloatArray
    q_weights: FloatArray
    k_minus_q_indices: IntArray

    def __post_init__(self) -> None:
        values = require_array("values", self.values, np.float64, (None, None))
        require_array("k_weights", self.k_weights, np.float64, (values.shape[0],))
        q_weights = require_array("q_weights", self.q_weights, np.float64, (None,))
        kq = require_array(
            "k_minus_q_indices",
            self.k_minus_q_indices,
            np.int64,
            (q_weights.shape[0], values.shape[0]),
        )
        if np.any(self.values < 0.0):
            raise ValueError("occupation values must be non-negative")
        if np.any(self.k_weights < 0.0) or np.any(self.q_weights < 0.0):
            raise ValueError("k_weights and q_weights must be non-negative")
        if np.any(kq < 0) or np.any(kq >= values.shape[0]):
            raise ValueError("k_minus_q_indices contains an index outside the k mesh")


@dataclass(frozen=True)
class ExchangeResult:
    sigma_x: ComplexArray
    exchange_energy: float

    def __post_init__(self) -> None:
        sigma = require_array("sigma_x", self.sigma_x, np.complex128, (None, None, None))
        if sigma.shape[1] != sigma.shape[2]:
            raise ValueError(f"sigma_x orbital blocks must be square, got shape {sigma.shape}")
        if not isinstance(self.exchange_energy, (float, np.floating)) or not np.isfinite(
            self.exchange_energy
        ):
            raise ValueError("exchange_energy must be a finite float")


@dataclass(frozen=True)
class ExchangeAblation:
    reference: ExchangeResult
    trial: ExchangeResult
    sigma_difference: ComplexArray
    exchange_energy_difference: float

    def __post_init__(self) -> None:
        if self.reference.sigma_x.shape != self.trial.sigma_x.shape:
            raise ValueError("reference and trial sigma_x shapes must match")
        require_array(
            "sigma_difference", self.sigma_difference, np.complex128, self.reference.sigma_x.shape
        )
        if not isinstance(self.exchange_energy_difference, (float, np.floating)) or not np.isfinite(
            self.exchange_energy_difference
        ):
            raise ValueError("exchange_energy_difference must be a finite float")
