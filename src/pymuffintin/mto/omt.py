"""Periodic optimized-muffin-tin (OMT) potential fitting.

The represented scalar field is

``g + sum_R f_R(|r-R|)``,

where ``g`` is a constant and every radial function is expanded in continuous
piecewise-linear hats.  The last radial knot is the potential-sphere radius
and its value is fixed to zero, so the functions join continuously to zero
outside their spheres.  Cartesian lengths are in Bohr and scalar potentials
are in Hartree.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..tensor import contract, inv, lstsq, solve


FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class OmtFitDiagnostics:
    """Residual and geometric diagnostics for one OMT fit."""

    weighted_residual_norm: float
    weighted_rms: float
    maximum_absolute_residual: float
    overlap_fractions: FloatArray

    @property
    def maximum_overlap_fraction(self) -> float:
        return float(np.max(self.overlap_fractions))


@dataclass(frozen=True)
class OmtFit:
    """A periodic constant-plus-radial-hats OMT representation."""

    lattice: FloatArray
    centers: FloatArray
    potential_radii: FloatArray
    radial_knots: tuple[FloatArray, ...]
    constant: float
    radial_coefficients: tuple[FloatArray, ...]
    diagnostics: OmtFitDiagnostics

    def evaluate(self, points: FloatArray) -> FloatArray:
        """Evaluate the fitted potential at Cartesian ``points``."""

        return evaluate_omt(self, points)


def _nearest_displacement(
    displacement: FloatArray,
    lattice: FloatArray,
    inverse_lattice: FloatArray,
    *,
    exclude_zero_translation: bool = False,
) -> FloatArray:
    """Return the exact closest image of one Cartesian displacement.

    Rounding fractional coordinates is not exact in a skew cell.  After an
    initial candidate is found, the inverse lattice supplies a finite box
    containing every lattice translation which could improve it; that box is
    searched exhaustively.
    """

    fractional = solve(lattice.T, displacement)
    rounded = np.rint(fractional).astype(np.int64)
    if exclude_zero_translation and np.all(rounded == 0):
        initial_translations = np.array(
            [shift for shift in product((-1, 0, 1), repeat=3) if shift != (0, 0, 0)],
            dtype=np.int64,
        )
        initial_images = contract(
            "ni,ij->nj", fractional[None, :] - initial_translations, lattice
        )
        initial_norms = np.linalg.norm(initial_images, axis=1)
        best_index = int(np.argmin(initial_norms))
        best = initial_images[best_index]
        best_norm = float(initial_norms[best_index])
    else:
        best = contract("i,ij->j", fractional - rounded, lattice)
        best_norm = float(np.linalg.norm(best))

    coordinate_bounds = best_norm * np.linalg.norm(inverse_lattice, axis=0)
    lower = np.ceil(fractional - coordinate_bounds).astype(np.int64)
    upper = np.floor(fractional + coordinate_bounds).astype(np.int64)
    translations = np.array(
        list(
            product(
                range(lower[0], upper[0] + 1),
                range(lower[1], upper[1] + 1),
                range(lower[2], upper[2] + 1),
            )
        ),
        dtype=np.int64,
    )
    if exclude_zero_translation:
        translations = translations[np.any(translations != 0, axis=1)]
    images = contract("ni,ij->nj", fractional[None, :] - translations, lattice)
    norms = np.linalg.norm(images, axis=1)
    return images[int(np.argmin(norms))]


def periodic_distances(
    points: FloatArray,
    centers: FloatArray,
    lattice: FloatArray,
) -> FloatArray:
    """Return exact nearest-image distances from points to periodic centers."""

    sample_points = np.asarray(points, dtype=float)
    sites = np.asarray(centers, dtype=float)
    cell = np.asarray(lattice, dtype=float)
    inverse_cell = inv(cell)
    distances = np.empty((len(sample_points), len(sites)), dtype=float)
    for point_index, point in enumerate(sample_points):
        for site_index, center in enumerate(sites):
            image = _nearest_displacement(point - center, cell, inverse_cell)
            distances[point_index, site_index] = np.linalg.norm(image)
    return distances


def radial_hat_matrix(distances: FloatArray, knots: FloatArray) -> FloatArray:
    """Evaluate continuous linear radial hats with a fixed zero outer node.

    ``knots`` must start at zero and end at the potential radius.  There are
    ``len(knots)-1`` fitted hats: their nodal values correspond to every knot
    except the final one, whose value is fixed to zero.
    """

    radii = np.asarray(distances, dtype=float)
    nodes = np.asarray(knots, dtype=float)
    if nodes.ndim != 1 or len(nodes) < 2 or nodes[0] != 0.0 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("radial knots must be a strictly increasing 1D array starting at zero")
    hats = np.zeros((*radii.shape, len(nodes) - 1), dtype=float)
    inside = (radii >= 0.0) & (radii < nodes[-1])
    interval = np.searchsorted(nodes, radii[inside], side="right") - 1
    interval = np.minimum(interval, len(nodes) - 2)
    fraction = (radii[inside] - nodes[interval]) / (
        nodes[interval + 1] - nodes[interval]
    )
    inside_hats = np.zeros((int(np.count_nonzero(inside)), len(nodes) - 1))
    rows = np.arange(len(interval))
    inside_hats[rows, interval] = 1.0 - fraction
    has_free_right_node = interval + 1 < len(nodes) - 1
    inside_hats[rows[has_free_right_node], interval[has_free_right_node] + 1] = fraction[
        has_free_right_node
    ]
    hats[inside] = inside_hats
    return hats


def omt_design_matrix(
    points: FloatArray,
    lattice: FloatArray,
    centers: FloatArray,
    potential_radii: FloatArray,
    radial_knots: Sequence[FloatArray],
) -> FloatArray:
    """Build the constant-plus-periodic-radial-hats least-squares matrix."""

    radii = np.asarray(potential_radii, dtype=float)
    knots = tuple(np.asarray(site_knots, dtype=float) for site_knots in radial_knots)
    if len(knots) != len(radii) or len(radii) != len(centers):
        raise ValueError("centers, potential radii, and radial knot sets must have equal length")
    for site, site_knots in enumerate(knots):
        if site_knots[-1] != radii[site]:
            raise ValueError("each final radial knot must equal its proper potential radius")
    distances = periodic_distances(points, centers, lattice)
    blocks = [np.ones((len(points), 1), dtype=float)]
    blocks.extend(radial_hat_matrix(distances[:, site], knots[site]) for site in range(len(radii)))
    return np.concatenate(blocks, axis=1)


def overlap_fractions(
    lattice: FloatArray,
    centers: FloatArray,
    potential_radii: FloatArray,
) -> FloatArray:
    r"""Return exact periodic sphere-overlap fractions.

    Entry ``(i,j)`` is
    :math:`\omega_{ij}=(s_i+s_j)/d_{ij}-1`, with ``d`` the exact nearest-image
    distance.  Diagonal entries compare a site with its nearest nonzero
    periodic image.
    """

    cell = np.asarray(lattice, dtype=float)
    sites = np.asarray(centers, dtype=float)
    radii = np.asarray(potential_radii, dtype=float)
    inverse_cell = inv(cell)
    result = np.empty((len(sites), len(sites)), dtype=float)
    for left in range(len(sites)):
        for right in range(left, len(sites)):
            displacement = sites[left] - sites[right]
            image = _nearest_displacement(
                displacement,
                cell,
                inverse_cell,
                exclude_zero_translation=left == right,
            )
            distance = np.linalg.norm(image)
            fraction = (radii[left] + radii[right]) / distance - 1.0
            result[left, right] = fraction
            result[right, left] = fraction
    return result


def fit_omt(
    points: FloatArray,
    values: FloatArray,
    weights: FloatArray,
    lattice: FloatArray,
    centers: FloatArray,
    potential_radii: FloatArray,
    radial_knots: Sequence[FloatArray],
) -> OmtFit:
    """Fit a constant and radial hats by weighted least squares."""

    sample_values = np.asarray(values, dtype=float)
    sample_weights = np.asarray(weights, dtype=float)
    design = omt_design_matrix(
        points, lattice, centers, potential_radii, radial_knots
    )
    square_root_weights = np.sqrt(sample_weights)
    weighted_design = design * square_root_weights[:, None]
    weighted_values = sample_values * square_root_weights
    coefficients = lstsq(weighted_design, weighted_values)
    fitted = contract("pi,i->p", design, coefficients)
    residual = fitted - sample_values
    weighted_residual = residual * square_root_weights
    knots = tuple(np.asarray(site_knots, dtype=float) for site_knots in radial_knots)
    counts = tuple(len(site_knots) - 1 for site_knots in knots)
    offsets = np.cumsum((1, *counts))
    radial_coefficients = tuple(
        coefficients[offsets[site] : offsets[site + 1]].copy()
        for site in range(len(counts))
    )
    diagnostics = OmtFitDiagnostics(
        weighted_residual_norm=float(np.linalg.norm(weighted_residual)),
        weighted_rms=float(
            np.sqrt(np.sum(weighted_residual * weighted_residual) / np.sum(sample_weights))
        ),
        maximum_absolute_residual=float(np.max(np.abs(residual))),
        overlap_fractions=overlap_fractions(lattice, centers, potential_radii),
    )
    return OmtFit(
        lattice=np.asarray(lattice, dtype=float),
        centers=np.asarray(centers, dtype=float),
        potential_radii=np.asarray(potential_radii, dtype=float),
        radial_knots=knots,
        constant=float(coefficients[0]),
        radial_coefficients=radial_coefficients,
        diagnostics=diagnostics,
    )


def evaluate_omt(fit: OmtFit, points: FloatArray) -> FloatArray:
    """Evaluate ``fit`` at Cartesian ``points`` in its periodic cell."""

    design = omt_design_matrix(
        points,
        fit.lattice,
        fit.centers,
        fit.potential_radii,
        fit.radial_knots,
    )
    coefficients = np.concatenate(
        (np.array([fit.constant]), *fit.radial_coefficients)
    )
    return contract("pi,i->p", design, coefficients)
