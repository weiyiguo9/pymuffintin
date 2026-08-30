from __future__ import annotations

import numpy as np

from ..contracts import (
    AuxiliaryRepresentation,
    PairSamples,
    RegionalChargeExpansion,
)


def muffin_tin_lri(samples: PairSamples, *, cutoff: float) -> AuxiliaryRepresentation:
    """Build one overlap-EVD local-RI block for every represented muffin-tin site."""
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("cutoff must be a finite non-negative float")

    expansions: list[RegionalChargeExpansion] = []
    residual_squared = 0.0
    for site in np.unique(samples.site_indices[samples.site_indices >= 0]):
        selected = samples.site_indices == site
        weighted_pairs = np.sqrt(samples.weights[selected])[:, None] * samples.values[selected]
        overlap = weighted_pairs.conj().T @ weighted_pairs
        eigenvalues, eigenvectors = np.linalg.eigh(overlap)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        threshold = cutoff * eigenvalues[0] if eigenvalues.size else 0.0
        keep = eigenvalues > threshold
        if not np.any(keep):
            raise ValueError(f"muffin-tin site {int(site)} has no local-RI mode above cutoff")
        coefficients = np.asarray(
            eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :],
            dtype=np.complex128,
        )
        residual_squared += float(np.sum(eigenvalues[~keep]))
        expansions.append(
            RegionalChargeExpansion(region=f"muffin_tin:{int(site)}", coefficients=coefficients)
        )

    if not expansions:
        raise ValueError("pair samples contain no muffin-tin points")
    return AuxiliaryRepresentation(
        q_index=samples.q_index,
        layout=samples.layout,
        expansions=tuple(expansions),
        residual_norm=float(np.sqrt(residual_squared)),
    )
