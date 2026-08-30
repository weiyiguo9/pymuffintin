from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pymuffintin.contracts import require_array


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LeafBox:
    """Axis-aligned cubic leaf of an adaptive dyadic tree."""

    center: FloatArray
    width: float

    def __post_init__(self) -> None:
        center = require_array("center", self.center, np.float64, (3,))
        if not np.all(np.isfinite(center)):
            raise ValueError("center must contain finite values")
        if type(self.width) is not float or not np.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("width must be a positive finite float")

    @property
    def lower(self) -> FloatArray:
        return self.center - 0.5 * self.width

    @property
    def upper(self) -> FloatArray:
        return self.center + 0.5 * self.width

    def distance_to(self, point: FloatArray) -> float:
        """Return the shortest Euclidean distance from ``point`` to the box."""
        point = require_array("point", point, np.float64, (3,))
        displacement = np.maximum(np.maximum(self.lower - point, point - self.upper), 0.0)
        return float(np.linalg.norm(displacement))


@dataclass(frozen=True)
class AdaptiveTree:
    """Validated dyadic leaves and their closure-touching neighbor graph.

    ``levels[i]`` and ``neighbor_lists[i]`` describe ``leaves[i]``.  Neighbor
    lists contain leaf indices and include face-, edge-, and corner-touching
    leaves.
    """

    root: LeafBox
    leaves: tuple[LeafBox, ...]
    levels: tuple[int, ...] = field(init=False)
    max_level: int = field(init=False)
    neighbor_lists: tuple[tuple[int, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.root, LeafBox):
            raise TypeError("root must be a LeafBox")
        if not isinstance(self.leaves, tuple) or not self.leaves:
            raise ValueError("leaves must be a nonempty tuple of LeafBox objects")
        if any(not isinstance(leaf, LeafBox) for leaf in self.leaves):
            raise TypeError("leaves must be a tuple of LeafBox objects")

        tolerance = 32.0 * np.finfo(np.float64).eps * max(
            1.0, self.root.width, float(np.max(np.abs(self.root.center)))
        )
        root_lower = self.root.lower
        root_upper = self.root.upper
        levels: list[int] = []
        lowers: list[FloatArray] = []

        for leaf_index, leaf in enumerate(self.leaves):
            ratio = self.root.width / leaf.width
            level = int(np.rint(np.log2(ratio)))
            if level < 0 or not np.isclose(
                leaf.width, self.root.width / (2**level), rtol=0.0, atol=tolerance
            ):
                raise ValueError(f"leaves[{leaf_index}] width is not dyadic relative to root")

            lower = leaf.lower
            upper = leaf.upper
            if np.any(lower < root_lower - tolerance) or np.any(upper > root_upper + tolerance):
                raise ValueError(f"leaves[{leaf_index}] lies outside root")

            offsets = (lower - root_lower) / leaf.width
            if not np.allclose(offsets, np.rint(offsets), rtol=0.0, atol=tolerance / leaf.width):
                raise ValueError(f"leaves[{leaf_index}] is not aligned to the dyadic root grid")

            levels.append(level)
            lowers.append(lower)

        max_level = max(levels)
        grid_scale = 2**max_level / self.root.width
        grid_lowers = [
            np.rint((lower - root_lower) * grid_scale).astype(np.int64)
            for lower in lowers
        ]
        grid_uppers = [
            lower + 2 ** (max_level - level)
            for lower, level in zip(grid_lowers, levels, strict=True)
        ]

        neighbors: list[list[int]] = [[] for _ in self.leaves]
        for left in range(len(self.leaves)):
            for right in range(left + 1, len(self.leaves)):
                overlap_lengths = np.minimum(
                    grid_uppers[left], grid_uppers[right]
                ) - np.maximum(
                    grid_lowers[left], grid_lowers[right]
                )
                if np.all(overlap_lengths > 0):
                    raise ValueError(f"leaves[{left}] and leaves[{right}] overlap in volume")
                if np.all(overlap_lengths >= 0):
                    if abs(levels[left] - levels[right]) > 1:
                        raise ValueError(
                            f"touching leaves[{left}] and leaves[{right}] violate 2:1 balance"
                        )
                    neighbors[left].append(right)
                    neighbors[right].append(left)

        represented_cells = sum(2 ** (3 * (max_level - level)) for level in levels)
        if represented_cells != 2 ** (3 * max_level):
            raise ValueError("leaves must form a complete partition of root")

        object.__setattr__(self, "levels", tuple(levels))
        object.__setattr__(self, "max_level", max_level)
        object.__setattr__(self, "neighbor_lists", tuple(tuple(items) for items in neighbors))

    def candidate_leaves(self, target: FloatArray, cutoff: float) -> tuple[LeafBox, ...]:
        """Select leaves whose minimum box distance to ``target`` is at most ``cutoff``."""
        target = require_array("target", target, np.float64, (3,))
        if type(cutoff) is not float or not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff must be a non-negative finite float")
        return tuple(leaf for leaf in self.leaves if leaf.distance_to(target) <= cutoff)
