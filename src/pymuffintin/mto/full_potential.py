"""Full-potential matrix corrections in the periodic augmented NMTO basis.

The reference kink Hamiltonian contains spherical active-channel potentials
and a constant free-envelope potential.  Inactive free partial waves inside
the spheres therefore need their spherical potential correction as well as
the nonspherical correction.  The overlap and reference KH kinetic operator
are unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from ..regional import _complex_spherical_harmonics
from ..tensor import contract
from .periodic_density import PeriodicNmtoBasisEvaluator, _angular_values_and_gradients
from .usw import RealHarmonic, real_spherical_harmonics

if TYPE_CHECKING:
    from .parallel import NmtoParallel


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class _FullPotentialQuadrature:
    interstitial_points: FloatArray
    interstitial_weights: FloatArray
    directions: FloatArray
    active_angular: FloatArray
    active_gradients: FloatArray
    site_centers: FloatArray
    radial_offsets: NDArray[np.int64]
    radii: FloatArray
    point_offsets: NDArray[np.int64]
    full_weights: FloatArray
    active_weights: FloatArray
    small_weights: FloatArray


def _real_potential(values: ComplexArray, region: str) -> FloatArray:
    """Require a scalar Hermitian potential before constructing its matrix."""

    real = np.real_if_close(values)
    if np.iscomplexobj(real):
        raise ValueError(
            f"{region} scalar potential has imaginary residual "
            f"{np.max(np.abs(values.imag))}"
        )
    return np.asarray(real, dtype=np.float64)


def _active_sphere_values(
    evaluator: PeriodicNmtoBasisEvaluator,
    site: int,
    radial_indices: NDArray[np.int64],
    angular: FloatArray,
    angular_gradients: FloatArray,
    k_index: int,
) -> tuple[ComplexArray, ComplexArray]:
    """Active radial components on a sphere centered at its original site.

    The quadrature points are ``site_cartesian + r * direction``, including
    points outside the chosen primitive parallelepiped.  Their image relative
    to this site is zero, so their Bloch image phase is exactly one.
    """

    channel_count = len(evaluator.channels)
    basis_size = len(evaluator.site_fractional) * channel_count
    point_count = len(radial_indices)
    large = np.zeros((point_count, basis_size), dtype=np.complex128)
    small = np.zeros((point_count, 4, basis_size), dtype=np.complex128)
    site_rows = slice(site * channel_count, (site + 1) * channel_count)
    for node, lagrange in enumerate(evaluator.results[k_index].lagrange_matrices):
        radial_large = np.empty((point_count, channel_count))
        radial_small = np.empty_like(radial_large)
        radial_tangential = np.empty_like(radial_large)
        for column, channel in enumerate(evaluator.channels):
            radial = evaluator.radial_samples[(site, channel.l)]
            boundary = radial.boundary_values[node]
            radial_large[:, column] = radial.large[node, radial_indices] / boundary
            radial_small[:, column] = radial.small[node, radial_indices] / boundary
            radial_tangential[:, column] = (
                radial.inverse_speed_of_light
                * radial.large[node, radial_indices]
                * radial.inverse_mass[node, radial_indices]
                / radial.mesh_radii[radial_indices]
                / boundary
            )
        coefficients = lagrange[site_rows]
        large += contract("pl,pl,lb->pb", radial_large, angular, coefficients)
        small[:, 0] += contract(
            "pl,pl,lb->pb", radial_small, angular, coefficients
        )
        small[:, 1:] += contract(
            "pl,pcl,lb->pcb", radial_tangential, angular_gradients, coefficients
        )
    return large, small


def _prepare_full_potential_quadrature(
    evaluator: PeriodicNmtoBasisEvaluator,
    potential_export: Mapping[str, object],
    interstitial_zero: float,
    angular_order: int,
) -> _FullPotentialQuadrature:
    """Reconstruct the potential and form all fixed spatial quadrature data."""

    vectors = np.asarray(potential_export["g_vectors"], dtype=np.int64)
    components = np.asarray(potential_export["components"], dtype=np.complex128)
    counts = 2 * np.max(np.abs(vectors), axis=0) + 3
    axes = [(np.arange(count) + 0.5) / count for count in counts]
    fractional = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    points = fractional @ evaluator.direct_lattice
    fourier_grid = np.zeros(tuple(counts), dtype=np.complex128)
    indices = tuple((vectors % counts).T)
    half_grid_phase = np.exp(1j * pi * np.sum(vectors / counts, axis=1))
    fourier_grid[indices] = components[0] * half_grid_phase
    full_interstitial = _real_potential(
        (np.fft.ifftn(fourier_grid) * np.prod(counts)).reshape(-1), "interstitial"
    )
    _, sphere_sites = evaluator._nearest_sites(points)
    outside = sphere_sites < 0
    interstitial_points = points[outside]
    volume_weight = abs(np.linalg.det(evaluator.direct_lattice)) / len(points)
    interstitial_weights = volume_weight * (
        full_interstitial[outside] - interstitial_zero
    )

    mesh_offsets = np.asarray(potential_export["mt_mesh_offsets"], dtype=np.int64)
    mesh_radii = np.asarray(potential_export["mt_mesh_radii"], dtype=np.float64)
    mesh_weights = np.asarray(potential_export["mt_mesh_weights"], dtype=np.float64)
    labels = np.asarray(potential_export["mt_channel_labels"], dtype=np.int64)
    sample_offsets = np.asarray(potential_export["mt_sample_offsets"], dtype=np.int64)
    mt_components = np.asarray(potential_export["mt_components"], dtype=np.complex128)
    cos_theta, theta_weights = np.polynomial.legendre.leggauss(angular_order)
    phi_count = 2 * angular_order
    phi = 2.0 * pi * np.arange(phi_count) / phi_count
    directions = np.asarray(
        [
            [
                np.sqrt(1.0 - z * z) * np.cos(angle),
                np.sqrt(1.0 - z * z) * np.sin(angle),
                z,
            ]
            for z in cos_theta
            for angle in phi
        ]
    )
    angular_weights = np.repeat(theta_weights, phi_count) * (2.0 * pi / phi_count)
    active_angular, active_gradients = _angular_values_and_gradients(
        directions, evaluator.channels
    )
    site_centers = evaluator.site_fractional @ evaluator.direct_lattice
    radial_offsets = [0]
    point_offsets = [0]
    radii_by_site = []
    full_weights_by_site = []
    active_weights_by_site = []
    small_weights_by_site = []
    for site in range(len(site_centers)):
        mesh_slice = slice(int(mesh_offsets[site]), int(mesh_offsets[site + 1]))
        radii = mesh_radii[mesh_slice]
        radial_weights = mesh_weights[mesh_slice] * radii**2
        quadrature_weights = (
            radial_weights[:, None] * angular_weights[None, :]
        ).reshape(-1)
        channel_indices = np.flatnonzero(labels[:, 0] == site)
        channels = tuple(
            RealHarmonic(int(labels[index, 1]), int(labels[index, 2]))
            for index in channel_indices
        )
        convention = str(potential_export["angular_basis"])
        if convention == "complex-condon-shortley":
            harmonics = _complex_spherical_harmonics(directions, channels)
        elif convention == "real-tesseral-condon-shortley":
            harmonics = real_spherical_harmonics(directions, channels)
        else:
            raise ValueError(f"unsupported potential angular basis {convention!r}")
        radial_coefficients = np.stack(
            tuple(
                mt_components[
                    0, int(sample_offsets[index]) : int(sample_offsets[index + 1])
                ]
                for index in channel_indices
            )
        )
        full_potential = _real_potential(
            radial_coefficients.T @ harmonics.T, f"site {site}"
        )
        monopole = channels.index(RealHarmonic(0, 0))
        spherical = _real_potential(
            radial_coefficients[monopole] / np.sqrt(4.0 * pi),
            f"site {site} monopole",
        )
        full_weights = quadrature_weights * (
            full_potential - interstitial_zero
        ).reshape(-1)
        active_weights = (
            radial_weights[:, None]
            * angular_weights[None, :]
            * (spherical[:, None] - interstitial_zero)
        ).reshape(-1)
        small_weights = quadrature_weights * (
            full_potential - spherical[:, None]
        ).reshape(-1)
        radii_by_site.append(radii)
        full_weights_by_site.append(full_weights)
        active_weights_by_site.append(active_weights)
        small_weights_by_site.append(small_weights)
        radial_offsets.append(radial_offsets[-1] + len(radii))
        point_offsets.append(point_offsets[-1] + len(full_weights))

    return _FullPotentialQuadrature(
        interstitial_points=interstitial_points,
        interstitial_weights=interstitial_weights,
        directions=directions,
        active_angular=active_angular,
        active_gradients=active_gradients,
        site_centers=site_centers,
        radial_offsets=np.asarray(radial_offsets, dtype=np.int64),
        radii=np.concatenate(radii_by_site),
        point_offsets=np.asarray(point_offsets, dtype=np.int64),
        full_weights=np.concatenate(full_weights_by_site),
        active_weights=np.concatenate(active_weights_by_site),
        small_weights=np.concatenate(small_weights_by_site),
    )


def full_potential_corrections(
    evaluator: PeriodicNmtoBasisEvaluator,
    potential_export: Mapping[str, object] | None,
    interstitial_zero: float,
    *,
    angular_order: int,
    parallel: NmtoParallel | None = None,
) -> tuple[ComplexArray, ...]:
    """Return one physical nonorthogonal ``Delta H`` matrix per k point.

    Interstitial integration uses the same field-compatible half-shifted
    uniform grid as regional density synthesis, excluding all MT spheres.
    Sphere integration uses native ``dr`` weights multiplied by ``r**2`` and
    a rule with ``angular_order`` Gauss--Legendre nodes and twice as many
    azimuthal nodes.  Angular convergence is caller-controlled.  The
    exported radial weights cover the sampled
    interval, without an extrapolated contribution below the first radius.

    Inside each sphere the correction is
    ``(V-V0) large*large - (Vsph-V0) active*active
    + (V-Vsph) small*small``.  The last product sums the four-component KH
    metric.  No sampled-overlap adjustment or change of kinetic operator is
    made, and all band/occupation updates belong to the caller.
    """

    basis_size = len(evaluator.site_fractional) * len(evaluator.channels)
    prepared = None
    preparation_stage = nullcontext() if parallel is None else parallel.local_stage()
    with preparation_stage:
        if parallel is None or parallel.rank == 0:
            if potential_export is None:
                raise ValueError("potential_export is required on rank zero")
            prepared = _prepare_full_potential_quadrature(
                evaluator, potential_export, interstitial_zero, angular_order
            )
    if parallel is not None:
        prepared = _FullPotentialQuadrature(
            interstitial_points=parallel.broadcast_array(
                None if prepared is None else prepared.interstitial_points
            ),
            interstitial_weights=parallel.broadcast_array(
                None if prepared is None else prepared.interstitial_weights
            ),
            directions=parallel.broadcast_array(
                None if prepared is None else prepared.directions
            ),
            active_angular=parallel.broadcast_array(
                None if prepared is None else prepared.active_angular
            ),
            active_gradients=parallel.broadcast_array(
                None if prepared is None else prepared.active_gradients
            ),
            site_centers=parallel.broadcast_array(
                None if prepared is None else prepared.site_centers
            ),
            radial_offsets=parallel.broadcast_array(
                None if prepared is None else prepared.radial_offsets
            ),
            radii=parallel.broadcast_array(
                None if prepared is None else prepared.radii
            ),
            point_offsets=parallel.broadcast_array(
                None if prepared is None else prepared.point_offsets
            ),
            full_weights=parallel.broadcast_array(
                None if prepared is None else prepared.full_weights
            ),
            active_weights=parallel.broadcast_array(
                None if prepared is None else prepared.active_weights
            ),
            small_weights=parallel.broadcast_array(
                None if prepared is None else prepared.small_weights
            ),
        )

    block_size = 8192
    tasks = [
        (-1, start, min(start + block_size, len(prepared.interstitial_points)))
        for start in range(0, len(prepared.interstitial_points), block_size)
    ]
    for site in range(len(prepared.site_centers)):
        point_count = int(prepared.point_offsets[site + 1] - prepared.point_offsets[site])
        tasks.extend(
            (site, start, min(start + block_size, point_count))
            for start in range(0, point_count, block_size)
        )

    shape = (len(evaluator.k_cartesian), basis_size, basis_size)
    if parallel is None:
        corrections = np.zeros(shape, dtype=np.complex128)
        task_indices = range(len(tasks))
        partials = None
    else:
        corrections = None
        task_indices = parallel.indices(len(tasks))
        partials = parallel.shared_array((len(tasks), *shape), np.complex128)

    stage = nullcontext() if parallel is None else parallel.local_stage()
    with stage:
        for task_index in task_indices:
            site, start, stop = tasks[task_index]
            target = corrections if partials is None else partials[task_index]
            if partials is not None:
                target.fill(0.0)
            if site < 0:
                block = slice(start, stop)
                for k_index in range(len(evaluator.k_cartesian)):
                    large, _ = evaluator._basis_values(
                        prepared.interstitial_points[block], k_index
                    )
                    target[k_index] += contract(
                        "pa,p,pb->ab",
                        large.conj(),
                        prepared.interstitial_weights[block],
                        large,
                    )
                continue

            radial_start = int(prepared.radial_offsets[site])
            radial_stop = int(prepared.radial_offsets[site + 1])
            radii = prepared.radii[radial_start:radial_stop]
            point_start = int(prepared.point_offsets[site])
            point_stop = int(prepared.point_offsets[site + 1])
            full_weights = prepared.full_weights[point_start:point_stop]
            active_weights = prepared.active_weights[point_start:point_stop]
            small_weights = prepared.small_weights[point_start:point_stop]
            indices = np.arange(start, stop)
            radial_indices = indices // len(prepared.directions)
            angular_indices = indices % len(prepared.directions)
            sphere_points = (
                prepared.site_centers[site]
                + radii[radial_indices, None] * prepared.directions[angular_indices]
            )
            for k_index in range(len(evaluator.k_cartesian)):
                large, _ = evaluator._basis_values(sphere_points, k_index)
                active, small = _active_sphere_values(
                    evaluator,
                    site,
                    radial_indices,
                    prepared.active_angular[angular_indices],
                    prepared.active_gradients[angular_indices],
                    k_index,
                )
                target[k_index] += (
                    contract(
                        "pa,p,pb->ab", large.conj(), full_weights[start:stop], large
                    )
                    - contract(
                        "pa,p,pb->ab",
                        active.conj(),
                        active_weights[start:stop],
                        active,
                    )
                    + contract(
                        "pca,p,pcb->ab",
                        small.conj(),
                        small_weights[start:stop],
                        small,
                    )
                )

    if parallel is not None:
        parallel.publish(partials)
        source = None
        with parallel.local_stage():
            if parallel.rank == 0:
                source = np.zeros(shape, dtype=np.complex128)
                for task_index in range(len(tasks)):
                    source += partials[task_index]
        corrections = parallel.broadcast_array(source)
    return tuple(0.5 * (matrix + matrix.conj().T) for matrix in corrections)
