#!/usr/bin/env python3
"""Apply dense and fast periodic DMK in a native-library-isolated process."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pymuffintin.coulomb import FastPeriodicDmk, LeafBox, LeafDensity, PeriodicDmk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.input) as source:
        root = LeafBox(
            center=np.asarray(source["root_center"], dtype=np.float64),
            width=float(source["root_width"]),
        )
        leaf_width = float(source["leaf_width"])
        densities = tuple(
            LeafDensity(
                LeafBox(center=np.asarray(center, dtype=np.float64), width=leaf_width),
                np.asarray(coefficients),
            )
            for center, coefficients in zip(
                source["leaf_centers"], source["leaf_coefficients"], strict=True
            )
        )
        targets = np.asarray(source["targets"], dtype=np.float64)
        settings = dict(
            root=root,
            densities=densities,
            tolerance=float(source["tolerance"]),
            source_quadrature_order=int(source["source_quadrature_order"]),
            local_quadrature_order=int(source["local_quadrature_order"]),
        )
        fast, work = FastPeriodicDmk(
            **settings,
            gaussian_order=int(source["gaussian_order"]),
            interpolation_order=int(source["interpolation_order"]),
        ).apply_with_work(targets)
        dense = PeriodicDmk(**settings).apply(targets)
    np.savez(
        args.output,
        fast_potential=fast,
        dense_potential=dense,
        fast_work_total=np.asarray(work.total),
    )


if __name__ == "__main__":
    main()
