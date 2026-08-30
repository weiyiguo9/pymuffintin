#!/usr/bin/env python3
"""Materialize neutral bcc-Fe and diamond-C atomic starts from Python."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import libmuffintin as mt


RADIAL_FIRST = 2.0e-6
RADIAL_POINT_COUNT = 401
FIELD_G_CUTOFF = 4.0
FIELD_MUFFIN_TIN_L_MAX = 4


def _radial_mesh(radius: float) -> tuple[float, float, int]:
    log_increment = math.log(radius / RADIAL_FIRST) / (RADIAL_POINT_COUNT - 1)
    return RADIAL_FIRST, log_increment, RADIAL_POINT_COUNT


def _free_atom_controls(material: str) -> mt.FreeAtomControls:
    if material == "diamond-c":
        mesh_log_increment = 0.01
        mesh_point_count = 1683
    else:
        mesh_log_increment = 0.004
        mesh_point_count = (
            math.ceil(math.log(30.0 / 1.0e-6) / mesh_log_increment) + 1
        )
    return mt.FreeAtomControls(
        mesh_first=1.0e-6,
        mesh_log_increment=mesh_log_increment,
        mesh_point_count=mesh_point_count,
        mixing=0.3,
        potential_tolerance=2.0e-5,
        tail_tolerance=1.0e-7,
        max_iterations=120,
        angular_points=50,
    )


def _structure(material: str) -> tuple[mt.Structure, str]:
    if material == "fe-bcc":
        lattice_constant = 5.41689993624
        radius = 2.2
        structure = mt.Structure(
            lattice=[
                [
                    0.5 * lattice_constant,
                    0.5 * lattice_constant,
                    -0.5 * lattice_constant,
                ],
                [
                    -0.5 * lattice_constant,
                    0.5 * lattice_constant,
                    0.5 * lattice_constant,
                ],
                [
                    0.5 * lattice_constant,
                    -0.5 * lattice_constant,
                    0.5 * lattice_constant,
                ],
            ],
            site_ids=["Fe-1"],
            atomic_numbers=[26],
            fractional_positions=[[0.0, 0.0, 0.0]],
            radial_meshes=[_radial_mesh(radius)],
            radial_equations=["scalar-koelling-harmon"],
            linearization_energies=[[(0, -0.3), (1, -0.2), (2, -0.1)]],
        )
        return structure, "fe_bcc_checkpoint.toml"

    lattice_constant = 6.740879853675
    radius = 1.4
    radial_mesh = _radial_mesh(radius)
    linearization_energies = [(0, -0.5), (1, -0.25), (2, 0.05)]
    structure = mt.Structure(
        lattice=[
            [0.0, 0.5 * lattice_constant, 0.5 * lattice_constant],
            [0.5 * lattice_constant, 0.0, 0.5 * lattice_constant],
            [0.5 * lattice_constant, 0.5 * lattice_constant, 0.0],
        ],
        site_ids=["C-1", "C-2"],
        atomic_numbers=[6, 6],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        radial_meshes=[radial_mesh, radial_mesh],
        radial_equations=["scalar-koelling-harmon", "scalar-koelling-harmon"],
        linearization_energies=[linearization_energies, linearization_energies],
    )
    return structure, "diamond_c_checkpoint.toml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a neutral atomic-superposition checkpoint from Python."
    )
    parser.add_argument("material", choices=("fe-bcc", "diamond-c"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "new directory for the checkpoint; defaults to "
            "local_experiment/generated/python_atomic_starts/<material>"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    structure, checkpoint_name = _structure(args.material)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent
            / "local_experiment"
            / "generated"
            / "python_atomic_starts"
            / args.material.replace("-", "_")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / checkpoint_name
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint {checkpoint_path}")

    layout = mt.RegionalFieldLayout.from_g_cutoff(
        structure,
        g_cutoff=FIELD_G_CUTOFF,
        muffin_tin_l_max=FIELD_MUFFIN_TIN_L_MAX,
    )
    start = mt.materialize_atomic_start(
        structure,
        layout,
        xc="lda-pw92",
        free_atom_controls=_free_atom_controls(args.material),
    )
    start.checkpoint.write(checkpoint_path)

    print(f"checkpoint={checkpoint_path.resolve()}")
    for key in (
        "target_electron_count",
        "uncorrected_electron_count",
        "represented_electron_count",
        "zero_mode_coefficient_correction",
        "interstitial_fraction",
        "response_volume",
    ):
        print(f"{key}={start.charge_closure[key]:.16e}")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    print(f"checkpoint_sha256={digest}")


if __name__ == "__main__":
    main()
