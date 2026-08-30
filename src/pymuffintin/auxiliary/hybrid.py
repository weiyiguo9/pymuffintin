from __future__ import annotations

import numpy as np

from ..contracts import AuxiliaryRepresentation, CoulombBlock
from ..providers import Coulomb


def hybrid_representation(
    muffin_tin: AuxiliaryRepresentation,
    interstitial: AuxiliaryRepresentation,
) -> AuxiliaryRepresentation:
    if muffin_tin.q_index != interstitial.q_index:
        raise ValueError(
            f"q_index mismatch: muffin-tin {muffin_tin.q_index}, interstitial {interstitial.q_index}"
        )
    if muffin_tin.layout != interstitial.layout:
        raise ValueError("muffin-tin and interstitial pair layouts must match exactly")
    if any(not block.region.startswith("muffin_tin:") for block in muffin_tin.expansions):
        raise ValueError("muffin_tin representation contains a non-muffin-tin block")
    if any(block.region != "interstitial" for block in interstitial.expansions):
        raise ValueError("interstitial representation contains a non-interstitial block")
    return AuxiliaryRepresentation(
        q_index=muffin_tin.q_index,
        layout=muffin_tin.layout,
        expansions=muffin_tin.expansions + interstitial.expansions,
        residual_norm=float(np.hypot(muffin_tin.residual_norm, interstitial.residual_norm)),
    )


def build_hybrid_coulomb(
    muffin_tin: AuxiliaryRepresentation,
    interstitial: AuxiliaryRepresentation,
    provider: Coulomb,
    *,
    gamma_policy: str,
    **request: object,
) -> tuple[AuxiliaryRepresentation, CoulombBlock]:
    """Ask the provider for one full MT–interstitial matrix, including cross blocks."""
    hybrid = hybrid_representation(muffin_tin, interstitial)
    block = provider.coulomb(hybrid, gamma_policy=gamma_policy, **request)
    if block.q_index != hybrid.q_index:
        raise ValueError("Coulomb provider returned a block for a different q_index")
    expected = hybrid.n_auxiliary
    if block.matrix.shape != (expected, expected):
        raise ValueError(
            f"Coulomb provider must return the full ({expected}, {expected}) hybrid matrix, "
            f"got {block.matrix.shape}"
        )
    return hybrid, block
