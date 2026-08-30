"""Band, occupation, and sampled-density operations for NMTO results.

Energies and temperatures are in Hartree.  Brillouin-zone weights are
normalized to one, while ``state_degeneracy`` accounts explicitly for spin
or any other state multiplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..tensor import contract, eigh
from .nmto import NmtoResult


FloatArray: TypeAlias = NDArray[np.float64]
NumericArray: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]


@dataclass(frozen=True)
class NmtoBands:
    """NMTO eigenvalues and eigenvectors on a k-point mesh."""

    energies: FloatArray
    orthonormal_coefficients: NumericArray
    coefficients: NumericArray

    def __post_init__(self) -> None:
        if self.energies.ndim != 2:
            raise ValueError("energies must have shape (k, band)")
        expected = (self.energies.shape[0], self.energies.shape[1], self.energies.shape[1])
        if self.orthonormal_coefficients.shape != expected:
            raise ValueError(
                "orthonormal_coefficients must have shape (k, basis, band)"
            )
        if self.coefficients.shape != expected:
            raise ValueError("coefficients must have shape (k, basis, band)")


@dataclass(frozen=True)
class NmtoOccupations:
    """Finite-temperature band occupations and their scalar observables."""

    chemical_potential: float
    values: FloatArray
    electron_count: float
    band_energy: float

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("occupation values must have shape (k, band)")
        if np.any(self.values < 0.0):
            raise ValueError("occupation values must be non-negative")
        if not np.isfinite(self.chemical_potential):
            raise ValueError("chemical_potential must be finite")
        if not np.isfinite(self.electron_count) or self.electron_count < 0.0:
            raise ValueError("electron_count must be finite and non-negative")
        if not np.isfinite(self.band_energy):
            raise ValueError("band_energy must be finite")


def _normalized_k_weights(k_weights: FloatArray, n_k: int) -> FloatArray:
    weights = np.asarray(k_weights, dtype=float)
    if weights.shape != (n_k,):
        raise ValueError(f"k_weights must have shape ({n_k},)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("k_weights must be finite and non-negative")
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("k_weights must sum to one")
    return weights


def solve_nmto_bands(results: Sequence[NmtoResult]) -> NmtoBands:
    """Diagonalize the Löwdin Hamiltonian at every k point."""

    if not results:
        raise ValueError("at least one NMTO result is required")
    hamiltonians = np.stack(tuple(result.lowdin.hamiltonian for result in results))
    transformations = np.stack(tuple(result.lowdin.transformation for result in results))
    if hamiltonians.ndim != 3 or hamiltonians.shape[1] != hamiltonians.shape[2]:
        raise ValueError("Löwdin Hamiltonians must be square matrices of equal size")
    if transformations.shape != hamiltonians.shape:
        raise ValueError("Löwdin transformations must match the Hamiltonian shapes")

    energies, eigenvectors = eigh(hamiltonians)
    coefficients = contract("kab,kbc->kac", transformations, eigenvectors)
    return NmtoBands(
        energies=np.asarray(energies, dtype=np.float64),
        orthonormal_coefficients=eigenvectors,
        coefficients=coefficients,
    )


def fermi_dirac_occupations(
    energies: FloatArray,
    k_weights: FloatArray,
    electron_count: float,
    temperature: float,
    state_degeneracy: float = 2.0,
) -> NmtoOccupations:
    """Fill bands at positive temperature for a normalized k-point mesh."""

    band_energies = np.asarray(energies, dtype=float)
    if band_energies.ndim != 2 or band_energies.size == 0:
        raise ValueError("energies must have nonempty shape (k, band)")
    if np.any(~np.isfinite(band_energies)):
        raise ValueError("energies must be finite")
    weights = _normalized_k_weights(k_weights, band_energies.shape[0])
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(state_degeneracy) or state_degeneracy <= 0.0:
        raise ValueError("state_degeneracy must be finite and positive")

    capacity = state_degeneracy * band_energies.shape[1]
    if not np.isfinite(electron_count) or not 0.0 < electron_count < capacity:
        raise ValueError(f"electron_count must lie strictly between zero and {capacity}")

    filling = electron_count / capacity
    logit = float(np.log(filling / (1.0 - filling)))
    lower = float(np.min(band_energies) + temperature * logit)
    upper = float(np.max(band_energies) + temperature * logit)

    def occupations_at(chemical_potential: float) -> FloatArray:
        exponent = np.clip(
            (band_energies - chemical_potential) / temperature,
            -700.0,
            700.0,
        )
        return state_degeneracy / (np.exp(exponent) + 1.0)

    for _ in range(128):
        chemical_potential = 0.5 * (lower + upper)
        trial = occupations_at(chemical_potential)
        trial_count = float(contract("k,kb->", weights, trial))
        if trial_count < electron_count:
            lower = chemical_potential
        else:
            upper = chemical_potential

    chemical_potential = 0.5 * (lower + upper)
    values = occupations_at(chemical_potential)
    actual_count = float(contract("k,kb->", weights, values))
    band_energy = float(contract("k,kb,kb->", weights, values, band_energies))
    return NmtoOccupations(
        chemical_potential=chemical_potential,
        values=np.asarray(values, dtype=np.float64),
        electron_count=actual_count,
        band_energy=band_energy,
    )


def interpolate_nmto_basis(
    node_values: NumericArray,
    lagrange_matrices: NumericArray,
) -> NumericArray:
    """Interpolate energy-node channel values into the NMTO basis."""

    values = np.asarray(node_values)
    lagrange = np.asarray(lagrange_matrices)
    if values.ndim != 3 or lagrange.ndim != 3:
        raise ValueError("node_values and lagrange_matrices must be three-dimensional")
    if values.shape[0] != lagrange.shape[0] or values.shape[2] != lagrange.shape[1]:
        raise ValueError("energy and channel dimensions must match")
    return contract("epc,ecb->pb", values, lagrange)


def nmto_density_matrices(
    bands: NmtoBands,
    occupations: NmtoOccupations,
) -> NumericArray:
    """Return one occupied density matrix in the nonorthogonal basis per k."""

    if occupations.values.shape != bands.energies.shape:
        raise ValueError("occupation values must match the NMTO band energies")
    return contract(
        "kan,kn,kbn->kab",
        bands.coefficients,
        occupations.values,
        bands.coefficients.conj(),
    )


def sample_nmto_density(
    basis_values: NumericArray,
    bands: NmtoBands,
    occupations: NmtoOccupations,
    k_weights: FloatArray,
) -> FloatArray:
    """Sample the k-weighted scalar density from nonorthogonal NMTO values."""

    values = np.asarray(basis_values)
    if values.ndim != 3:
        raise ValueError("basis_values must have shape (k, point, basis)")
    if values.shape[0] != bands.energies.shape[0] or values.shape[2] != bands.coefficients.shape[1]:
        raise ValueError("basis_values k and basis dimensions must match the NMTO bands")
    weights = _normalized_k_weights(k_weights, values.shape[0])
    density_matrices = nmto_density_matrices(bands, occupations)
    density = contract(
        "k,kpa,kab,kpb->p",
        weights,
        values,
        density_matrices,
        values.conj(),
    )
    return np.asarray(np.real_if_close(density), dtype=np.float64)
