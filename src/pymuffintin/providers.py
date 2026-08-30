from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .contracts import (
    AuxiliaryRepresentation,
    CoulombBlock,
    OrbitalWindow,
    PairSamples,
)


@runtime_checkable
class Orbitals(Protocol):
    def orbital_window(self, q_index: int, *, spin: int = 0) -> OrbitalWindow: ...

    def sample(
        self,
        q_index: int,
        points: NDArray[np.float64],
        weights: NDArray[np.float64],
        regions: NDArray[np.int64],
        *,
        spin: int = 0,
    ) -> PairSamples: ...


@runtime_checkable
class LocalProduct(Protocol):
    def build_mpb(
        self, q_index: int, *, spin: int = 0, **spec: object
    ) -> AuxiliaryRepresentation: ...


@runtime_checkable
class Coulomb(Protocol):
    def coulomb(
        self,
        representation: AuxiliaryRepresentation,
        *,
        gamma_policy: str,
        **request: object,
    ) -> CoulombBlock: ...
