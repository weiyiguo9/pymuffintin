#!/usr/bin/env python3
"""Run the decomposed diamond-C SCF, FastDMK, and USW/NMTO comparison.

The two benchmark axes are kept separate.  ``PeriodicDmk`` and
``FastPeriodicDmk`` receive the same projected converged density, while LAPW
and USW/NMTO bands receive the same converged effective potential and k path.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
from numpy.typing import NDArray

import libmuffintin as mt
from pymuffintin import RegionalScalarSampler
from pymuffintin.coulomb import (
    LeafBox,
    project_density,
)
from pymuffintin.mto import (
    RealHarmonic,
    build_kink_mesh,
    build_nmto,
    solve_nmto_bands,
    usw_matrices_with_energy_derivative,
)

from diamond_lapw_nmto import _bloch_fold, _boundary_jets, _translation_cluster
from diamond_periodic_dmk import TARGETS_FRACTIONAL


FloatArray = NDArray[np.float64]


def _energy_mesh(value: str) -> FloatArray:
    energies = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if energies.size < 2 or np.any(~np.isfinite(energies)):
        raise argparse.ArgumentTypeError("energy mesh needs at least two finite Hartree values")
    if len(np.unique(energies)) != len(energies):
        raise argparse.ArgumentTypeError("energy mesh values must be distinct")
    return energies


def _diamond_path(points_per_segment: int) -> tuple[list[str], FloatArray]:
    if points_per_segment < 2:
        raise ValueError("points_per_segment must be at least two")
    vertices = (
        ("G", np.asarray([0.0, 0.0, 0.0])),
        ("X", np.asarray([0.5, 0.0, 0.5])),
        ("L", np.asarray([0.5, 0.5, 0.5])),
        ("G", np.asarray([0.0, 0.0, 0.0])),
    )
    labels: list[str] = []
    points: list[FloatArray] = []
    for segment, ((left_label, left), (right_label, right)) in enumerate(
        zip(vertices[:-1], vertices[1:], strict=True)
    ):
        for local, fraction in enumerate(np.linspace(0.0, 1.0, points_per_segment)):
            if segment and local == 0:
                continue
            point = (1.0 - fraction) * left + fraction * right
            points.append(point)
            labels.append(left_label if local == 0 else right_label if local == points_per_segment - 1 else "")
    return labels, np.asarray(points, dtype=np.float64)


def _restart_input(source: Path, checkpoint: Path, destination: Path) -> None:
    lines = source.read_text().splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith("checkpoint = ")]
    if len(matches) != 1:
        raise ValueError("SCF input must contain exactly one top-level checkpoint assignment")
    relative_checkpoint = checkpoint.resolve().relative_to(destination.parent.resolve())
    lines[matches[0]] = f"checkpoint = {json.dumps(str(relative_checkpoint))}"
    destination.write_text("\n".join(lines) + "\n")


def _geometry(checkpoint: Path) -> tuple[FloatArray, FloatArray, FloatArray]:
    with checkpoint.open("rb") as stream:
        document = tomllib.load(stream)
    direct = np.asarray(document["geometry"]["lattice"]["vectors"], dtype=np.float64)
    sites = document["geometry"]["sites"]
    fractional = np.asarray([site["fractional_position"] for site in sites], dtype=np.float64)
    radii = np.asarray([site["muffin_tin_radius"] for site in sites], dtype=np.float64)
    return direct, fractional, radii


def _nmto_path(
    physics: object,
    input_path: Path,
    energy_mesh: FloatArray,
    k_fractional: FloatArray,
    minimum_cells: int,
) -> tuple[FloatArray, dict[str, object]]:
    potential = physics.export_frozen_potential()
    zero = np.flatnonzero(np.all(potential["g_vectors"] == 0, axis=1))
    if len(zero) != 1:
        raise ValueError("converged potential must contain exactly one interstitial G=0")
    interstitial_zero = float(np.real(potential["components"][0, zero[0]]))
    interstitial_mesh = energy_mesh - interstitial_zero
    product_input = physics.scalar_product_input(input_path, q=[0.0, 0.0, 0.0])
    geometry = product_input.export_geometry()
    direct = np.asarray(geometry["direct_lattice"], dtype=np.float64)
    reciprocal = np.asarray(geometry["reciprocal_lattice"], dtype=np.float64)
    positions = np.asarray(geometry["site_cartesian"], dtype=np.float64)
    radii = np.asarray(geometry["muffin_tin_radius"], dtype=np.float64)
    if np.asarray(geometry["atomic_number"]).tolist() != [6, 6]:
        raise ValueError("diamond comparison requires a two-carbon primitive cell")
    integers, translations = _translation_cluster(direct, minimum_cells)
    center_translation = int(np.flatnonzero(np.all(integers == 0, axis=1))[0])
    centers = (translations[:, None, :] + positions[None, :, :]).reshape((-1, 3))
    cluster_radii = np.tile(radii, len(translations))
    channels = tuple(RealHarmonic(l, m) for l in range(3) for m in range(-l, l + 1))
    jets = _boundary_jets(
        physics,
        list(geometry["site_id"]),
        radii,
        energy_mesh,
        channels,
    )
    slopes = []
    slope_derivatives = []
    for energy in interstitial_mesh:
        _, slope, derivative = usw_matrices_with_energy_derivative(
            float(energy), centers, cluster_radii, channels
        )
        slopes.append(slope)
        slope_derivatives.append(derivative)
    results = []
    for k_cartesian in k_fractional @ reciprocal:
        folded = np.stack(
            tuple(
                _bloch_fold(
                    value,
                    translations,
                    center_translation,
                    len(positions),
                    len(channels),
                    k_cartesian,
                )
                for value in slopes
            )
        )
        folded_derivative = np.stack(
            tuple(
                _bloch_fold(
                    value,
                    translations,
                    center_translation,
                    len(positions),
                    len(channels),
                    k_cartesian,
                )
                for value in slope_derivatives
            )
        )
        results.append(
            build_nmto(
                build_kink_mesh(
                    energy_mesh,
                    folded,
                    folded_derivative,
                    jets,
                    jets.potential_radii,
                )
            )
        )
    bands = solve_nmto_bands(tuple(results))
    metadata = {
        "interstitial_potential_zero": interstitial_zero,
        "interstitial_energy_mesh": interstitial_mesh,
        "translation_integers": integers,
        "translation_cartesian": translations,
    }
    return bands.energies, metadata


def _midgap(energies: FloatArray, occupied_bands: int) -> float:
    valence_maximum = float(np.max(energies[:, occupied_bands - 1]))
    conduction_minimum = float(np.min(energies[:, occupied_bands]))
    return 0.5 * (valence_maximum + conduction_minimum)


def _revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _dirty(path: Path) -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        ).strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--energy-mesh", type=_energy_mesh, default=_energy_mesh("-0.1,0.4"))
    parser.add_argument("--minimum-cells", type=int, default=135)
    parser.add_argument("--points-per-segment", type=int, default=11)
    parser.add_argument("--band-count", type=int, default=6)
    parser.add_argument("--dmk-level", type=int, default=2)
    parser.add_argument("--dmk-degree-count", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    result = mt.run_dft_scf(args.input)
    restart_checkpoint = args.output_dir / "diamond_restart.toml"
    result.restart_checkpoint().write(restart_checkpoint)
    restart_input = args.output_dir / "diamond_restart_input.toml"
    _restart_input(args.input, restart_checkpoint, restart_input)

    labels, k_fractional = _diamond_path(args.points_per_segment)
    lapw_export = result.band_path(labels, k_fractional, 2 * args.band_count)
    lapw_all = np.asarray(lapw_export["energies"], dtype=np.float64)
    if not np.allclose(lapw_all[:, 0::2], lapw_all[:, 1::2], rtol=0.0, atol=1.0e-10):
        raise ValueError("diamond scalar LAPW path is not spin-degenerate")
    lapw_energies = lapw_all[:, 0::2]

    physics = mt.CheckpointPhysics(mt.load_checkpoint(restart_checkpoint))
    nmto_energies, nmto_metadata = _nmto_path(
        physics,
        restart_input,
        args.energy_mesh,
        k_fractional,
        args.minimum_cells,
    )
    compared = min(args.band_count, nmto_energies.shape[1])
    lapw_reference = _midgap(lapw_energies, 4)
    nmto_reference = _midgap(nmto_energies, 4)
    lapw_aligned = lapw_energies[:, :compared] - lapw_reference
    nmto_aligned = nmto_energies[:, :compared] - nmto_reference
    band_difference = nmto_aligned - lapw_aligned

    direct, site_fractional, radii = _geometry(restart_checkpoint)
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct).T
    k_cartesian = k_fractional @ reciprocal
    path_distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(k_cartesian, axis=0), axis=1)))
    )
    restart_density = physics.restart_density()
    if restart_density is None:
        raise ValueError("converged restart checkpoint has no density")
    sampler = RegionalScalarSampler.from_export(
        restart_density.export_interstitial(),
        direct,
        site_fractional,
        radii,
    )
    supercell = np.asarray([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=int)
    conventional = supercell @ direct
    conventional_width = float(np.mean(np.diag(conventional)))
    np.testing.assert_allclose(conventional, conventional_width * np.eye(3), atol=1.0e-12)
    root = LeafBox(
        center=np.full(3, 0.5 * conventional_width, dtype=np.float64),
        width=conventional_width,
    )
    densities, removed_mean = project_density(
        root,
        sampler,
        level=args.dmk_level,
        degree_count=args.dmk_degree_count,
        quadrature_order=max(args.dmk_degree_count + 2, 6),
        remove_mean=True,
    )
    leaf_centers = np.asarray([density.box.center for density in densities], dtype=np.float64)
    targets = conventional_width * np.mod(TARGETS_FRACTIONAL, 1.0)
    dmk_source = args.output_dir / "diamond_dmk_source.npz"
    dmk_result = args.output_dir / "diamond_dmk_result.npz"
    leaf_coefficients = np.stack(tuple(density.coefficients for density in densities))
    np.savez(
        dmk_source,
        root_center=root.center,
        root_width=np.asarray(root.width),
        leaf_centers=leaf_centers,
        leaf_width=np.asarray(densities[0].box.width),
        leaf_coefficients=leaf_coefficients,
        targets=targets,
        tolerance=np.asarray(1.0e-6),
        source_quadrature_order=np.asarray(12),
        local_quadrature_order=np.asarray(8),
        gaussian_order=np.asarray(12),
        interpolation_order=np.asarray(12),
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("periodic_dmk_compare.py")),
            "--input",
            str(dmk_source),
            "--output",
            str(dmk_result),
        ],
        check=True,
    )
    with np.load(dmk_result) as compared_dmk:
        fast_potential = compared_dmk["fast_potential"].copy()
        dense_potential = compared_dmk["dense_potential"].copy()
        fast_work_total = int(compared_dmk["fast_work_total"])
    dmk_difference = fast_potential - dense_potential

    output = args.output_dir / "diamond_scf_comparison.npz"
    np.savez(
        output,
        scope=np.asarray("diamond-C decomposed SCF FastPeriodicDMK and USW-NMTO comparison"),
        input=np.asarray(str(args.input.resolve())),
        input_sha256=np.asarray(sha256(args.input.read_bytes()).hexdigest()),
        restart_checkpoint=np.asarray(str(restart_checkpoint.resolve())),
        restart_checkpoint_sha256=np.asarray(sha256(restart_checkpoint.read_bytes()).hexdigest()),
        pymuffintin_revision=np.asarray(_revision(Path(__file__).resolve().parents[1])),
        pymuffintin_dirty=np.asarray(_dirty(Path(__file__).resolve().parents[1])),
        libmuffintin_revision=np.asarray(
            _revision(Path(mt.__file__).resolve().parents[2])
        ),
        libmuffintin_dirty=np.asarray(_dirty(Path(mt.__file__).resolve().parents[2])),
        scf_iterations=np.asarray(result.iterations),
        scf_total_energy=np.asarray(result.total_energy),
        scf_energy_history=result.energy_history(),
        scf_convergence_history=result.convergence_history(),
        path_labels=np.asarray(labels),
        path_distance=path_distance,
        k_fractional=k_fractional,
        k_cartesian=k_cartesian,
        lapw_raw_energies=lapw_energies,
        nmto_raw_energies=nmto_energies,
        lapw_spin_degeneracy=np.asarray(2),
        compared_band_count=np.asarray(compared),
        lapw_midgap_reference=np.asarray(lapw_reference),
        nmto_midgap_reference=np.asarray(nmto_reference),
        lapw_aligned_energies=lapw_aligned,
        nmto_aligned_energies=nmto_aligned,
        band_difference=band_difference,
        band_rms=np.asarray(np.sqrt(np.mean(np.abs(band_difference) ** 2))),
        band_maximum_absolute_error=np.asarray(np.max(np.abs(band_difference))),
        energy_mesh=args.energy_mesh,
        **nmto_metadata,
        conventional_width=np.asarray(conventional_width),
        dmk_level=np.asarray(args.dmk_level),
        dmk_degree_count=np.asarray(args.dmk_degree_count),
        dmk_removed_mean=np.asarray(removed_mean),
        dmk_leaf_centers=leaf_centers,
        dmk_target_fractional=np.mod(TARGETS_FRACTIONAL, 1.0),
        dmk_target_cartesian=targets,
        dmk_leaf_coefficients=leaf_coefficients,
        fast_dmk_potential=fast_potential,
        dense_dmk_potential=dense_potential,
        dmk_maximum_absolute_error=np.asarray(np.max(np.abs(dmk_difference))),
        dmk_relative_l2_error=np.asarray(
            np.linalg.norm(dmk_difference) / np.linalg.norm(dense_potential)
        ),
        fast_dmk_work_total=np.asarray(fast_work_total),
    )
    print(output)


if __name__ == "__main__":
    main()
