#!/usr/bin/env python3
"""Diamond-sized cubic 3D PeriodicDmk validation against analytic Poisson."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import numpy as np
from numpy.typing import NDArray

from pymuffintin.coulomb import LeafBox, LeafDensity, PeriodicDmk
from pymuffintin.tensor import contract


TARGETS_FRACTIONAL = np.asarray(
    [
        [0.137, -0.221, 0.319],
        [-0.413, 0.173, 0.287],
        [0.071, 0.389, -0.247],
        [0.333, 0.117, -0.431],
        [-0.293, -0.347, 0.183],
        [0.219, -0.083, -0.371],
        [0.447, -0.263, -0.129],
        [-0.157, 0.421, 0.091],
    ],
    dtype=np.float64,
)


def _conventional_width(checkpoint: Path) -> float:
    with checkpoint.open("rb") as stream:
        document = tomllib.load(stream)
    primitive = np.asarray(document["geometry"]["lattice"]["vectors"], dtype=float)
    supercell = np.asarray([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=int)
    conventional = supercell @ primitive
    width = float(np.mean(np.diag(conventional)))
    np.testing.assert_allclose(conventional, width * np.eye(3), atol=1.0e-12)
    return width


def _reciprocal_reference(
    width: float, targets_fractional: NDArray[np.float64], order: int
) -> NDArray[np.complex128]:
    modes = np.arange(-order, order + 1, dtype=float)
    legendre_transform = np.zeros_like(modes)
    nonzero = modes != 0.0
    legendre_transform[nonzero] = (
        3.0 * np.power(-1.0, modes[nonzero]) / (np.pi * modes[nonzero]) ** 2
    )
    mx, my, mz = np.meshgrid(modes, modes, modes, indexing="ij")
    g_squared = (2.0 * np.pi / width) ** 2 * (mx * mx + my * my + mz * mz)
    rho_bar = (
        width**-3
        * legendre_transform[:, None, None]
        * legendre_transform[None, :, None]
        * legendre_transform[None, None, :]
    )
    coefficients = np.zeros_like(rho_bar, dtype=np.complex128)
    selected = g_squared > 0.0
    coefficients[selected] = 4.0 * np.pi * rho_bar[selected] / g_squared[selected]
    values = []
    for target in targets_fractional:
        phase = tuple(np.exp(2j * np.pi * coordinate * modes) for coordinate in target)
        values.append(contract("i,j,k,ijk->", *phase, coefficients))
    return np.asarray(values, dtype=np.complex128)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-order", type=int, default=96)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    width = _conventional_width(args.checkpoint)
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=width)
    coefficients = np.zeros((3, 3, 3), dtype=np.float64)
    coefficients[2, 2, 2] = width**-3
    solver = PeriodicDmk(
        root=root,
        densities=(LeafDensity(root, coefficients),),
        tolerance=1.0e-12,
        source_quadrature_order=20,
        local_quadrature_order=14,
    )
    targets = width * TARGETS_FRACTIONAL
    dmk = solver.apply(targets)
    reference = _reciprocal_reference(width, TARGETS_FRACTIONAL, args.reference_order)
    coarse_reference = _reciprocal_reference(
        width, TARGETS_FRACTIONAL, args.reference_order // 2
    )
    difference = dmk - reference
    np.savez(
        args.output,
        scope=np.asarray("diamond-sized conventional-cube 3D periodic Hartree kernel"),
        conventional_width=np.asarray(width),
        source_legendre_coefficients=coefficients,
        target_fractional=TARGETS_FRACTIONAL,
        target_cartesian=targets,
        dmk_potential=dmk,
        reciprocal_reference=reference,
        coarse_reciprocal_reference=coarse_reference,
        reference_max_change=np.asarray(np.max(np.abs(reference - coarse_reference))),
        maximum_absolute_error=np.asarray(np.max(np.abs(difference))),
        relative_l2_error=np.asarray(np.linalg.norm(difference) / np.linalg.norm(reference)),
        maximum_imaginary_leakage=np.asarray(np.max(np.abs(dmk.imag))),
    )


if __name__ == "__main__":
    main()
