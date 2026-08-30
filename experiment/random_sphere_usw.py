#!/usr/bin/env python3
"""Fixed-seed USW experiment on a random non-overlapping sphere structure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pymuffintin.mto import RealHarmonic, usw_matrices, usw_matrices_with_energy_derivative


def _random_spheres(
    seed: int, count: int, cell_width: float
) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(seed)
    centers: list[np.ndarray] = []
    radii: list[float] = []
    while len(centers) < count:
        radius = float(random.uniform(0.45, 0.75))
        center = random.uniform(-0.5 * cell_width, 0.5 * cell_width, size=3)
        if centers:
            displacement = center - np.asarray(centers)
            displacement -= cell_width * np.rint(displacement / cell_width)
            if np.any(np.linalg.norm(displacement, axis=1) <= radius + np.asarray(radii)):
                continue
        centers.append(center)
        radii.append(radius)
    return np.asarray(centers), np.asarray(radii)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--sphere-count", type=int, default=8)
    parser.add_argument("--cell-width", type=float, default=10.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    centers, radii = _random_spheres(args.seed, args.sphere_count, args.cell_width)
    channels = tuple(RealHarmonic(l, m) for l in range(2) for m in range(-l, l + 1))
    energies = np.asarray([-0.4, -0.1, 0.2])
    slopes = []
    slope_derivatives = []
    derivative_errors = []
    symmetry_errors = []
    repeated_radii = np.repeat(radii, len(channels))
    step = 1.0e-6
    for energy in energies:
        _, slope, slope_derivative = usw_matrices_with_energy_derivative(
            float(energy), centers, radii, channels
        )
        _, slope_plus = usw_matrices(float(energy + step), centers, radii, channels)
        _, slope_minus = usw_matrices(float(energy - step), centers, radii, channels)
        finite_difference = (slope_plus - slope_minus) / (2.0 * step)
        slopes.append(slope)
        slope_derivatives.append(slope_derivative)
        derivative_errors.append(np.max(np.abs(slope_derivative - finite_difference)))
        weighted = repeated_radii[:, None] * slope
        symmetry_errors.append(np.max(np.abs(weighted - weighted.T)))

    np.savez(
        args.output,
        scope=np.asarray("fixed-seed random non-overlapping sphere USW"),
        seed=np.asarray(args.seed),
        cell_width=np.asarray(args.cell_width),
        centers=centers,
        radii=radii,
        energies=energies,
        channel_l=np.asarray([channel.l for channel in channels]),
        channel_m=np.asarray([channel.m for channel in channels]),
        slopes=np.stack(slopes),
        slope_derivatives=np.stack(slope_derivatives),
        maximum_derivative_error=np.asarray(derivative_errors),
        maximum_weighted_symmetry_error=np.asarray(symmetry_errors),
    )


if __name__ == "__main__":
    main()
