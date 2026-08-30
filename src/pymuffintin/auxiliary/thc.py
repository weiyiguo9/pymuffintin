from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..contracts import (
    AuxiliaryRepresentation,
    PairSamples,
    RegionalChargeExpansion,
    require_array,
)
from ..tensor import lstsq


@dataclass(frozen=True)
class IsdfSelection:
    point_indices: NDArray[np.int64]
    zeta: NDArray[np.complex128]
    residual_norm: float

    def __post_init__(self) -> None:
        indices = require_array("point_indices", self.point_indices, np.int64, (None,))
        require_array("zeta", self.zeta, np.complex128, (None, indices.shape[0]))
        if not np.isfinite(self.residual_norm) or self.residual_norm < 0.0:
            raise ValueError("residual_norm must be a finite non-negative float")


def _deterministic_qrcp_columns(matrix: NDArray[np.complex128], rank: int) -> NDArray[np.int64]:
    """Sequential, host-only by design: Gate 2's same-engine THC reproduction
    (doc 21 section 7) depends on this exact pivot order, so it is not routed
    through `tensor.contract` or a backend-dispatched primitive."""
    work = np.array(matrix, dtype=np.complex128, copy=True)
    norms = np.sum(np.abs(work) ** 2, axis=0)
    pivots = np.empty(rank, dtype=np.int64)
    available = np.ones(work.shape[1], dtype=np.bool_)
    for step in range(rank):
        candidates = np.flatnonzero(available)
        pivot = int(candidates[np.argmax(norms[candidates])])
        pivot_norm = float(np.sqrt(norms[pivot]))
        if pivot_norm == 0.0:
            raise ValueError(f"requested ISDF rank {rank} exceeds the weighted pair rank {step}")
        pivots[step] = pivot
        available[pivot] = False
        direction = work[:, pivot] / pivot_norm
        remaining = np.flatnonzero(available)
        if remaining.size:
            projections = direction.conj() @ work[:, remaining]
            work[:, remaining] -= direction[:, None] * projections[None, :]
            norms[remaining] = np.sum(np.abs(work[:, remaining]) ** 2, axis=0)
        norms[pivot] = -1.0
    return pivots


def weighted_isdf(
    values: NDArray[np.complex128], weights: NDArray[np.float64], *, rank: int
) -> IsdfSelection:
    """Select deterministic weighted QRCP rows and solve the ISDF zeta fit."""
    values = require_array("values", values, np.complex128, (None, None))
    weights = require_array("weights", weights, np.float64, (values.shape[0],))
    if type(rank) is not int or not 1 <= rank <= values.shape[0]:
        raise ValueError(f"rank must be in [1, {values.shape[0]}], got {rank}")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")

    weighted = np.sqrt(weights)[:, None] * values
    pivots = _deterministic_qrcp_columns(weighted.T, rank)
    selected = weighted[pivots]
    solution = lstsq(selected.T, weighted.T)
    zeta = np.asarray(solution.T, dtype=np.complex128)
    residual = float(np.linalg.norm(weighted - zeta @ selected))
    return IsdfSelection(point_indices=pivots, zeta=zeta, residual_norm=residual)


def interstitial_thc(samples: PairSamples, *, rank: int) -> tuple[AuxiliaryRepresentation, IsdfSelection]:
    selected = samples.site_indices == -1
    if not np.any(selected):
        raise ValueError("pair samples contain no interstitial points")
    point_rows = np.flatnonzero(selected)
    fit = weighted_isdf(samples.values[selected], samples.weights[selected], rank=rank)
    global_indices = np.asarray(point_rows[fit.point_indices], dtype=np.int64)
    selection = IsdfSelection(
        point_indices=global_indices,
        zeta=fit.zeta,
        residual_norm=fit.residual_norm,
    )
    coefficients = np.asarray(samples.values[global_indices].T, dtype=np.complex128)
    representation = AuxiliaryRepresentation(
        q_index=samples.q_index,
        layout=samples.layout,
        expansions=(
            RegionalChargeExpansion(region="interstitial", coefficients=coefficients),
        ),
        residual_norm=fit.residual_norm,
    )
    return representation, selection
