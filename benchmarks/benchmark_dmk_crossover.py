"""Sample the dense-to-hierarchical periodic DMK target crossover.

Run from the repository root, for example::

    PYTHONPATH=src:/tmp/finufft_site /opt/homebrew/bin/python3 \
        benchmarks/benchmark_dmk_crossover.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from time import perf_counter

import numpy as np

from pymuffintin.coulomb import FastPeriodicDmk, LeafBox, LeafDensity, PeriodicDmk


def _checkerboard(level: int) -> tuple[LeafBox, tuple[LeafDensity, ...]]:
    root = LeafBox(center=np.zeros(3, dtype=np.float64), width=1.0)
    count = 2**level
    width = 1.0 / count
    densities = tuple(
        LeafDensity(
            LeafBox(
                center=np.asarray(
                    root.lower + width * (np.asarray(index, dtype=np.float64) + 0.5),
                    dtype=np.float64,
                ),
                width=float(width),
            ),
            np.array([[[-1.0 if sum(index) % 2 else 1.0]]], dtype=np.float64),
        )
        for index in np.ndindex(count, count, count)
    )
    return root, densities


def _timed_apply(solver: object, targets: np.ndarray) -> tuple[float, np.ndarray]:
    start = perf_counter()
    potential = solver.apply(targets)  # type: ignore[attr-defined]
    return perf_counter() - start, potential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--samples", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--gaussian-order", type=int, default=20)
    parser.add_argument("--interpolation-order", type=int, default=24)
    args = parser.parse_args()

    root, densities = _checkerboard(args.level)
    reciprocal_order = int(np.ceil(-np.log(args.tolerance) / np.pi))
    source_order = max(24, 2 * reciprocal_order)
    common = dict(
        root=root,
        densities=densities,
        tolerance=args.tolerance,
        source_quadrature_order=source_order,
        local_quadrature_order=14,
    )
    direct = PeriodicDmk(**common)
    fast = FastPeriodicDmk(
        **common,
        gaussian_order=args.gaussian_order,
        interpolation_order=args.interpolation_order,
    )
    generator = np.random.default_rng(20260831)
    maximum = max(args.samples)
    target_pool = np.asarray(
        root.lower + root.width * generator.random((maximum, 3)),
        dtype=np.float64,
    )

    crossover: int | None = None
    print(
        f"level={args.level} tolerance={args.tolerance:g} "
        f"gaussian_order={args.gaussian_order} "
        f"interpolation_order={args.interpolation_order}"
    )
    print("targets direct_seconds fast_seconds max_abs_difference")
    for target_count in args.samples:
        targets = target_pool[:target_count]
        direct_seconds, direct_values = _timed_apply(direct, targets)
        start = perf_counter()
        fast_values, work = fast.apply_with_work(targets)
        fast_seconds = perf_counter() - start
        difference = float(np.max(np.abs(fast_values - direct_values)))
        print(
            f"{target_count:7d} {direct_seconds:14.6f} "
            f"{fast_seconds:12.6f} {difference:18.8e}"
        )
        print(f"  deterministic_work={asdict(work)}")
        if crossover is None and fast_seconds <= direct_seconds:
            crossover = target_count

    if crossover is None:
        print(f"crossover: not reached in sampled target counts {args.samples}")
    else:
        print(f"crossover: first sampled target count {crossover}")


if __name__ == "__main__":
    main()
