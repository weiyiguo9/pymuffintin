#!/usr/bin/env python3
"""Frozen-potential LAPW--NMTO comparison for primitive diamond carbon."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import libmuffintin as mt
from pymuffintin.mto import (
    BoundaryJets,
    RealHarmonic,
    build_kink_mesh,
    build_nmto,
    fermi_dirac_occupations,
    solve_nmto_bands,
    usw_matrices_with_energy_derivative,
)
from pymuffintin.tensor import contract


def _energy_mesh(value: str) -> NDArray[np.float64]:
    energies = np.asarray([float(item) for item in value.split(",")], dtype=float)
    if energies.size == 0 or np.any(~np.isfinite(energies)):
        raise argparse.ArgumentTypeError("energy mesh must contain finite Hartree values")
    if len(np.unique(energies)) != len(energies):
        raise argparse.ArgumentTypeError("energy mesh values must be distinct")
    return energies


def _translation_cluster(
    direct_lattice: NDArray[np.float64], minimum_cells: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    extent = 1
    while True:
        integers = np.asarray(
            list(product(range(-extent, extent + 1), repeat=3)), dtype=np.int64
        )
        cartesian = integers @ direct_lattice
        distances = np.linalg.norm(cartesian, axis=1)
        if len(distances) < minimum_cells:
            extent += 1
            continue
        shell_radius = float(np.partition(distances, minimum_cells - 1)[minimum_cells - 1])
        selected = distances <= shell_radius + 1.0e-10
        if np.all(np.max(np.abs(integers[selected]), axis=0) < extent):
            integers = integers[selected]
            cartesian = cartesian[selected]
            order = np.lexsort(
                (integers[:, 2], integers[:, 1], integers[:, 0], distances[selected])
            )
            return integers[order], cartesian[order]
        extent += 1


def _bloch_fold(
    matrix: NDArray,
    translations: NDArray[np.float64],
    center_translation: int,
    site_count: int,
    channel_count: int,
    k_cartesian: NDArray[np.float64],
) -> NDArray[np.complex128]:
    cell_count = len(translations)
    blocks = np.asarray(matrix).reshape(
        (cell_count, site_count, channel_count, cell_count, site_count, channel_count)
    )
    central_rows = blocks[center_translation]
    phase = np.exp(1j * (translations @ k_cartesian))
    folded = contract("t,aitbj->aibj", phase, central_rows)
    return folded.reshape((site_count * channel_count, site_count * channel_count))


def _boundary_jets(
    physics: object,
    site_ids: list[str],
    radii: NDArray[np.float64],
    energies: NDArray[np.float64],
    channels: tuple[RealHarmonic, ...],
) -> BoundaryJets:
    values = []
    radial_derivatives = []
    energy_derivatives = []
    energy_radial_derivatives = []
    potential_radii = []
    for site_id, radius in zip(site_ids, radii, strict=True):
        radial_by_l = {
            l: physics.sample_frozen_scalar_radials(
                site_id, l, energies, hard_radius=float(radius)
            )
            for l in range(3)
        }
        for channel in channels:
            boundary = radial_by_l[channel.l]["boundary_radial"]
            boundary_energy = radial_by_l[channel.l][
                "energy_derivative_boundary_radial"
            ]
            values.append(boundary[:, 0])
            radial_derivatives.append(boundary[:, 1])
            energy_derivatives.append(boundary_energy[:, 0])
            energy_radial_derivatives.append(boundary_energy[:, 1])
            potential_radii.append(radius)
    return BoundaryJets(
        potential_radii=np.asarray(potential_radii),
        values=np.stack(values, axis=1),
        radial_derivatives=np.stack(radial_derivatives, axis=1),
        energy_derivatives=np.stack(energy_derivatives, axis=1),
        energy_radial_derivatives=np.stack(energy_radial_derivatives, axis=1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--energy-mesh", required=True, type=_energy_mesh)
    parser.add_argument("--minimum-cells", type=int, default=13)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    physics = mt.CheckpointPhysics(mt.load_checkpoint(args.checkpoint))
    frozen_potential = physics.export_frozen_potential()
    zero_rows = np.flatnonzero(np.all(frozen_potential["g_vectors"] == 0, axis=1))
    if len(zero_rows) != 1:
        raise ValueError("frozen potential must contain exactly one interstitial G=0 term")
    interstitial_potential_zero = float(
        np.real(frozen_potential["components"][0, zero_rows[0]])
    )
    interstitial_energy_mesh = args.energy_mesh - interstitial_potential_zero
    product_input = physics.scalar_product_input(args.input, q=[0.0, 0.0, 0.0])
    geometry = product_input.export_geometry()
    orbitals = product_input.export_orbitals()
    atomic_numbers = np.asarray(geometry["atomic_number"])
    if atomic_numbers.tolist() != [6, 6]:
        raise ValueError("diamond comparison requires a two-carbon primitive cell")

    direct = np.asarray(geometry["direct_lattice"], dtype=float)
    reciprocal = np.asarray(geometry["reciprocal_lattice"], dtype=float)
    basis_positions = np.asarray(geometry["site_cartesian"], dtype=float)
    primitive_radii = np.asarray(geometry["muffin_tin_radius"], dtype=float)
    integers, translations = _translation_cluster(direct, args.minimum_cells)
    center_translation = int(np.flatnonzero(np.all(integers == 0, axis=1))[0])
    centers = (translations[:, None, :] + basis_positions[None, :, :]).reshape((-1, 3))
    cluster_radii = np.tile(primitive_radii, len(translations))
    channels = tuple(RealHarmonic(l, m) for l in range(3) for m in range(-l, l + 1))
    jets = _boundary_jets(
        physics,
        list(geometry["site_id"]),
        primitive_radii,
        args.energy_mesh,
        channels,
    )

    slopes = []
    slope_derivatives = []
    for energy in interstitial_energy_mesh:
        _, slope, slope_derivative = usw_matrices_with_energy_derivative(
            float(energy), centers, cluster_radii, channels
        )
        slopes.append(slope)
        slope_derivatives.append(slope_derivative)

    k_fractional = np.asarray(orbitals["k_fractional"], dtype=float)
    k_cartesian = k_fractional @ reciprocal
    nmto_results = []
    for k_point in k_cartesian:
        folded_slopes = np.stack(
            [
                _bloch_fold(
                    value,
                    translations,
                    center_translation,
                    len(basis_positions),
                    len(channels),
                    k_point,
                )
                for value in slopes
            ]
        )
        folded_derivatives = np.stack(
            [
                _bloch_fold(
                    value,
                    translations,
                    center_translation,
                    len(basis_positions),
                    len(channels),
                    k_point,
                )
                for value in slope_derivatives
            ]
        )
        nmto_results.append(
            build_nmto(
                build_kink_mesh(
                    args.energy_mesh,
                    folded_slopes,
                    folded_derivatives,
                    jets,
                    jets.potential_radii,
                )
            )
        )

    nmto = solve_nmto_bands(tuple(nmto_results))
    lapw_channels = sorted(orbitals["channels"], key=lambda item: item["spin"])
    lapw_energies = np.stack([item["energies"] for item in lapw_channels])
    k_weights = np.full(len(k_fractional), 1.0 / len(k_fractional))
    lapw_for_filling = lapw_energies.transpose(1, 0, 2).reshape(
        (len(k_fractional), -1)
    )
    lapw_occupations = fermi_dirac_occupations(
        lapw_for_filling, k_weights, 8.0, args.temperature, state_degeneracy=1.0
    )
    nmto_occupations = fermi_dirac_occupations(
        nmto.energies, k_weights, 8.0, args.temperature, state_degeneracy=2.0
    )
    np.savez(
        args.output,
        scope=np.asarray("diamond-C frozen-potential LAPW-NMTO"),
        k_fractional=k_fractional,
        k_cartesian=k_cartesian,
        energy_mesh=args.energy_mesh,
        interstitial_potential_zero=np.asarray(interstitial_potential_zero),
        interstitial_energy_mesh=interstitial_energy_mesh,
        translation_integers=integers,
        translation_cartesian=translations,
        lapw_spins=np.asarray([item["spin"] for item in lapw_channels]),
        lapw_energies=lapw_energies,
        lapw_chemical_potential=np.asarray(lapw_occupations.chemical_potential),
        lapw_energies_from_mu=lapw_energies - lapw_occupations.chemical_potential,
        nmto_energies=nmto.energies,
        nmto_chemical_potential=np.asarray(nmto_occupations.chemical_potential),
        nmto_energies_from_mu=nmto.energies - nmto_occupations.chemical_potential,
        nmto_coefficients=nmto.coefficients,
        nmto_orthonormal_coefficients=nmto.orthonormal_coefficients,
    )


if __name__ == "__main__":
    main()
