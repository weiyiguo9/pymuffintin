from __future__ import annotations

import numpy as np

from ..contracts import (
    AuxiliaryRepresentation,
    CoulombBlock,
    ExchangeAblation,
    ExchangeResult,
    FixedOccupation,
)
from ..tensor import contract


def fixed_orbital_exchange(
    representations: tuple[AuxiliaryRepresentation, ...],
    coulomb_blocks: tuple[CoulombBlock, ...],
    occupation: FixedOccupation,
) -> ExchangeResult:
    """Evaluate fixed-orbital Fock exchange with explicit k, q, and occupation weights."""
    if len(representations) != len(coulomb_blocks):
        raise ValueError("representations and coulomb_blocks must have the same length")
    if len(representations) != occupation.q_weights.shape[0]:
        raise ValueError("one auxiliary representation and Coulomb block are required per q weight")
    if not representations:
        raise ValueError("at least one q point is required")

    layout = representations[0].layout
    if occupation.values.shape != (layout.n_k, layout.n_orb):
        raise ValueError(
            "occupation values shape must match the pair layout: "
            f"expected {(layout.n_k, layout.n_orb)}, got {occupation.values.shape}"
        )
    sigma = np.zeros((layout.n_k, layout.n_orb, layout.n_orb), dtype=np.complex128)

    seen_q: set[int] = set()
    for representation, block in zip(representations, coulomb_blocks, strict=True):
        if representation.layout != layout:
            raise ValueError("all q points must use the same pair layout")
        if representation.q_index != block.q_index:
            raise ValueError("auxiliary representation and Coulomb block q_index must match")
        q_index = representation.q_index
        if q_index in seen_q or not 0 <= q_index < occupation.q_weights.shape[0]:
            raise ValueError(f"q_index {q_index} is duplicated or outside the q weights")
        seen_q.add(q_index)
        if block.matrix.shape != (representation.n_auxiliary, representation.n_auxiliary):
            raise ValueError("Coulomb matrix dimension must match the auxiliary representation")

        # vertices[k, i, j] pairs band i at k-q (conjugated) with band j at k
        # (the MuffintinAdapter.sample convention, doc 21 section 4.2). The
        # occupied/internal band contracted against the Coulomb block is i
        # (at k-q, gathered into f_sel); the external sigma indices are j
        # and l (at k) in 'ki,kija,ab,kilb->kjl'.
        vertices = representation.coefficients.reshape(
            layout.n_k, layout.n_orb, layout.n_orb, representation.n_auxiliary
        )
        q_weight = occupation.q_weights[q_index]
        occupied_k = occupation.k_minus_q_indices[q_index]
        f_sel = occupation.values[occupied_k]
        sigma -= q_weight * contract(
            "ki,kija,ab,kilb->kjl", f_sel, vertices.conj(), block.matrix, vertices
        )

    exchange_energy = 0.0
    for k_index in range(layout.n_k):
        exchange_energy += 0.5 * occupation.k_weights[k_index] * float(
            np.sum(occupation.values[k_index] * np.real(np.diag(sigma[k_index])))
        )
    return ExchangeResult(sigma_x=sigma, exchange_energy=float(exchange_energy))


def compare_exchange(reference: ExchangeResult, trial: ExchangeResult) -> ExchangeAblation:
    if reference.sigma_x.shape != trial.sigma_x.shape:
        raise ValueError("reference and trial sigma_x shapes must match")
    sigma_difference = np.asarray(trial.sigma_x - reference.sigma_x, dtype=np.complex128)
    energy_difference = float(trial.exchange_energy - reference.exchange_energy)
    return ExchangeAblation(
        reference=reference,
        trial=trial,
        sigma_difference=sigma_difference,
        exchange_energy_difference=energy_difference,
    )
