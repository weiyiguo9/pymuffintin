#!/usr/bin/env python3
"""Compare LAPW and NMTO bands from one scalar frozen-potential checkpoint.

This driver supports a one-site primitive cell only.  It is a frozen-potential
band comparison, not a self-consistent NMTO DFT calculation.
"""

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
    cluster_bloch_sum,
    fermi_dirac_occupations,
    solve_nmto_bands,
    usw_matrices_with_energy_derivative,
)


def _energy_mesh(value: str) -> NDArray[np.float64]:
    try:
        energies = np.asarray([float(item) for item in value.split(",")], dtype=float)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "energy mesh must be comma-separated Hartree values"
        ) from error
    if energies.size == 0 or not np.all(np.isfinite(energies)):
        raise argparse.ArgumentTypeError("energy mesh must contain finite Hartree values")
    if len(np.unique(energies)) != len(energies):
        raise argparse.ArgumentTypeError("energy-mesh values must be distinct")
    return energies


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("cluster shell size must be positive")
    return parsed


def _complete_translation_cluster(
    direct_lattice: NDArray[np.float64], minimum_size: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return all primitive translations through the shell reaching minimum_size."""

    direct = np.asarray(direct_lattice, dtype=float)
    smallest_singular_value = float(np.linalg.svd(direct, compute_uv=False)[-1])
    extent = 1
    while True:
        integers = np.asarray(
            list(product(range(-extent, extent + 1), repeat=3)), dtype=np.int64
        )
        cartesian = integers @ direct
        distances = np.linalg.norm(cartesian, axis=1)
        if len(distances) >= minimum_size:
            shell_radius = float(np.partition(distances, minimum_size - 1)[minimum_size - 1])
            if smallest_singular_value * (extent + 1) > shell_radius:
                break
        extent += 1

    tolerance = 1.0e-10 * max(1.0, shell_radius)
    selected = distances <= shell_radius + tolerance
    integers = integers[selected]
    cartesian = cartesian[selected]
    distances = distances[selected]
    order = np.lexsort(
        (integers[:, 2], integers[:, 1], integers[:, 0], distances)
    )
    return integers[order], cartesian[order]


def _boundary_jets(
    physics: object,
    site_id: str,
    muffin_tin_radius: float,
    energies: NDArray[np.float64],
    channels: tuple[RealHarmonic, ...],
) -> BoundaryJets:
    radials = {
        l: physics.sample_frozen_scalar_radials(
            site_id, l, energies, hard_radius=muffin_tin_radius
        )
        for l in range(3)
    }
    boundary = np.stack(
        [radials[channel.l]["boundary_radial"] for channel in channels], axis=1
    )
    boundary_energy = np.stack(
        [
            radials[channel.l]["energy_derivative_boundary_radial"]
            for channel in channels
        ],
        axis=1,
    )
    potential_radii = np.full(len(channels), muffin_tin_radius)
    return BoundaryJets(
        potential_radii=potential_radii,
        values=boundary[:, :, 0],
        radial_derivatives=boundary[:, :, 1],
        energy_derivatives=boundary_energy[:, :, 0],
        energy_radial_derivatives=boundary_energy[:, :, 1],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LAPW and l<=2 NMTO bands on the same one-site scalar "
            "frozen potential; this is not self-consistent NMTO DFT."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--energy-mesh",
        required=True,
        type=_energy_mesh,
        help="distinct absolute Hartree values separated by commas",
    )
    parser.add_argument(
        "--cluster-shell-size",
        type=_positive_integer,
        default=59,
        help="minimum translations before completing the enclosing shell (default: 59)",
    )
    parser.add_argument("--valence-electrons", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("fe_lapw_nmto.npz"))
    return parser


def main() -> None:
    args = _parser().parse_args()

    checkpoint = mt.load_checkpoint(args.checkpoint)
    physics = mt.CheckpointPhysics(checkpoint)
    frozen_potential = physics.export_frozen_potential()
    zero_rows = np.flatnonzero(np.all(frozen_potential["g_vectors"] == 0, axis=1))
    if len(zero_rows) != 1:
        raise ValueError("frozen potential must contain exactly one interstitial G=0 term")
    interstitial_potential_zero = float(
        np.real(frozen_potential["components"][0, zero_rows[0]])
    )
    interstitial_energy_mesh = args.energy_mesh - interstitial_potential_zero
    product_input = physics.scalar_product_input(
        args.input, q=[0.0, 0.0, 0.0]
    )
    geometry = product_input.export_geometry()
    orbitals = product_input.export_orbitals()

    if len(geometry["site_id"]) != 1:
        raise ValueError("Fe LAPW-NMTO comparison supports one-site primitive cells only")
    if np.asarray(geometry["atomic_number"]).tolist() != [26]:
        raise ValueError("Fe LAPW-NMTO comparison requires atomic number 26")

    site_id = geometry["site_id"][0]
    muffin_tin_radius = float(geometry["muffin_tin_radius"][0])
    direct_lattice = np.asarray(geometry["direct_lattice"], dtype=float)
    reciprocal_lattice = np.asarray(geometry["reciprocal_lattice"], dtype=float)
    cluster_integers, cluster_centers = _complete_translation_cluster(
        direct_lattice, args.cluster_shell_size
    )

    channels = tuple(
        RealHarmonic(l, m) for l in range(3) for m in range(-l, l + 1)
    )
    boundary_jets = _boundary_jets(
        physics,
        site_id,
        muffin_tin_radius,
        args.energy_mesh,
        channels,
    )
    cluster_radii = np.full(len(cluster_centers), muffin_tin_radius)
    cluster_slopes = []
    cluster_slope_derivatives = []
    for energy in interstitial_energy_mesh:
        _, slope, slope_derivative = usw_matrices_with_energy_derivative(
            float(energy), cluster_centers, cluster_radii, channels
        )
        cluster_slopes.append(slope)
        cluster_slope_derivatives.append(slope_derivative)

    k_fractional = np.asarray(orbitals["k_fractional"], dtype=float)
    k_cartesian = k_fractional @ reciprocal_lattice
    nmto_results = []
    for k_point in k_cartesian:
        slopes = np.stack(
            [
                cluster_bloch_sum(
                    slope, cluster_centers, len(channels), k_point
                )
                for slope in cluster_slopes
            ]
        )
        slope_derivatives = np.stack(
            [
                cluster_bloch_sum(
                    derivative, cluster_centers, len(channels), k_point
                )
                for derivative in cluster_slope_derivatives
            ]
        )
        kink_mesh = build_kink_mesh(
            args.energy_mesh,
            slopes,
            slope_derivatives,
            boundary_jets,
            boundary_jets.potential_radii,
        )
        nmto_results.append(build_nmto(kink_mesh))

    nmto_bands = solve_nmto_bands(tuple(nmto_results))
    orbital_channels = sorted(orbitals["channels"], key=lambda channel: channel["spin"])
    lapw_spins = np.asarray([channel["spin"] for channel in orbital_channels], dtype=np.int64)
    lapw_energies = np.stack(
        [np.asarray(channel["energies"], dtype=float) for channel in orbital_channels]
    )
    k_weights = np.full(len(k_fractional), 1.0 / len(k_fractional))
    lapw_for_filling = lapw_energies.transpose(1, 0, 2).reshape(
        (len(k_fractional), -1)
    )
    lapw_occupations = fermi_dirac_occupations(
        lapw_for_filling,
        k_weights,
        args.valence_electrons,
        args.temperature,
        state_degeneracy=1.0,
    )
    nmto_occupations = fermi_dirac_occupations(
        nmto_bands.energies,
        k_weights,
        args.valence_electrons,
        args.temperature,
        state_degeneracy=2.0,
    )

    np.savez(
        args.output,
        comparison=np.asarray("one-site scalar frozen-potential LAPW-NMTO"),
        checkpoint=np.asarray(str(args.checkpoint.resolve())),
        input=np.asarray(str(args.input.resolve())),
        q_fractional=np.zeros(3),
        site_id=np.asarray(site_id),
        atomic_number=np.asarray(geometry["atomic_number"], dtype=np.int64),
        muffin_tin_radius=np.asarray(muffin_tin_radius),
        direct_lattice=direct_lattice,
        reciprocal_lattice=reciprocal_lattice,
        k_fractional=k_fractional,
        k_cartesian=k_cartesian,
        energy_mesh=np.asarray(args.energy_mesh, dtype=float),
        interstitial_potential_zero=np.asarray(interstitial_potential_zero),
        interstitial_energy_mesh=interstitial_energy_mesh,
        real_harmonic_l=np.asarray([channel.l for channel in channels], dtype=np.int64),
        real_harmonic_m=np.asarray([channel.m for channel in channels], dtype=np.int64),
        cluster_integer_translations=cluster_integers,
        cluster_cartesian_translations=cluster_centers,
        lapw_band_window_start=np.asarray(orbitals["band_window_start"], dtype=np.int64),
        lapw_spins=lapw_spins,
        lapw_energies=lapw_energies,
        lapw_chemical_potential=np.asarray(lapw_occupations.chemical_potential),
        lapw_energies_from_mu=lapw_energies - lapw_occupations.chemical_potential,
        nmto_energies=nmto_bands.energies,
        nmto_chemical_potential=np.asarray(nmto_occupations.chemical_potential),
        nmto_energies_from_mu=(
            nmto_bands.energies - nmto_occupations.chemical_potential
        ),
        nmto_coefficients=nmto_bands.coefficients,
        nmto_orthonormal_coefficients=nmto_bands.orthonormal_coefficients,
    )


if __name__ == "__main__":
    main()
