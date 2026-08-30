"""Value-and-derivative functions from energy-divided USWs.

The conventions follow Eqs. (q0), (vandd), (odd), (even), and (DD) of
Andersen *et al.*, arXiv:1604.08097.  Energies and energy-divided differences
are in Hartree throughout.  Energy is the leading axis of sampled quantities,
and the coefficient tensor is indexed as
``[energy, derivative, input_channel, output_channel]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..tensor import contract, solve


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
NumericArray: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]


@dataclass(frozen=True)
class VdCoefficients:
    r"""The ``dmax=3`` transformation from divided USWs to v&d functions.

    ``coefficients[n, d]`` is the matrix :math:`D_{n+1,d}` in Eq. (vandd).
    The four energies and per-channel sphere data are retained to make the
    transformation's normalization explicit.
    """

    energies: FloatArray
    radii: FloatArray
    angular_momenta: IntArray
    coefficients: NumericArray


def divided_differences(
    energies: FloatArray,
    values: NumericArray,
) -> tuple[NumericArray, ...]:
    """Return the triangular table of consecutive energy divided differences.

    The result at ``table[order][start]`` is the divided difference over
    ``energies[start:start + order + 1]``.  All axes of ``values`` after its
    leading energy axis are carried elementwise.  Thus ``table[3][0]`` is the
    four-point quantity denoted, for example, :math:`S_{1234}` in the paper.
    """

    table: list[NumericArray] = [values]
    current = values
    trailing_axes = (1,) * (values.ndim - 1)
    for order in range(1, energies.shape[0]):
        denominator = (energies[order:] - energies[:-order]).reshape(
            (-1, *trailing_axes)
        )
        current = (current[1:] - current[:-1]) / denominator
        table.append(current)
    return tuple(table)


def _product(left: NumericArray, right: NumericArray) -> NumericArray:
    return contract("ij,jk->ik", left, right)


def _right_solve(left: NumericArray, right: NumericArray) -> NumericArray:
    """Return ``left @ inv(right)`` using solve semantics."""

    return solve(right.T, left.T).T


def build_vd_coefficients(
    energies: FloatArray,
    slope_matrices: NumericArray,
    radii: FloatArray,
    angular_momenta: IntArray,
) -> VdCoefficients:
    r"""Construct all sixteen ``dmax=3`` v&d coefficient matrices.

    ``slope_matrices[n]`` is the dimensionless screened slope matrix
    :math:`\mathcal S(\epsilon_{n+1})`.  The implementation uses the forms in
    Eqs. (odd), (even), and (DD), which avoid inversion of :math:`S_{123}`.
    Every structured matrix product is dispatched through
    :func:`pymuffintin.tensor.contract`; matrix divisions use
    :func:`pymuffintin.tensor.solve`.
    """

    hartree_differences = divided_differences(energies, slope_matrices)
    differences = tuple(
        difference / (2.0**order)
        for order, difference in enumerate(hartree_differences)
    )
    script_s1 = differences[0][0]
    script_s2 = differences[0][1]
    s12 = differences[1][0]
    s23 = differences[1][1]
    s123 = differences[2][0]
    s234 = differences[2][1]
    s1234 = differences[3][0]

    n_channels = slope_matrices.shape[1]
    dtype = np.result_type(slope_matrices.dtype, energies.dtype, radii.dtype)
    identity = np.eye(n_channels, dtype=dtype)
    inverse_radii = np.diag(1.0 / radii).astype(dtype, copy=False)

    l_factor = angular_momenta * (angular_momenta + 1)
    w = l_factor / (radii * radii)
    epsilon_minus_w = np.diag(2.0 * energies[0] - w).astype(dtype, copy=False)
    a_w_prime = np.diag(-2.0 * w).astype(dtype, copy=False)

    inverse_s1234_s123 = solve(s1234, s123)
    d33_denominator = s23 - _product(s234, inverse_s1234_s123)
    d33 = -solve(d33_denominator, identity)
    d43 = -_product(inverse_s1234_s123, d33)

    inverse_s23_s234 = solve(s23, s234)
    a_matrix = solve(s1234 - _product(s123, inverse_s23_s234), identity)

    s234_inverse_s1234 = _right_solve(s234, s1234)
    d31 = _product(d33, s234_inverse_s1234 + epsilon_minus_w)
    d41 = a_matrix - _product(
        _product(inverse_s1234_s123, d33), epsilon_minus_w
    )

    d32_bracket = script_s2 - _product(s234_inverse_s1234, s12)
    d32 = _product(-_product(d33, d32_bracket), inverse_radii)
    d42_numerator = -_product(d43, script_s2) + _product(a_matrix, s12)
    d42 = _product(d42_numerator, inverse_radii)

    s12_epsilon_minus_w = _product(s12, epsilon_minus_w)
    d30_bracket = (
        a_w_prime
        + _product(script_s2, epsilon_minus_w)
        + _product(
            s234_inverse_s1234,
            script_s1 - s12_epsilon_minus_w,
        )
    )
    d30 = _product(-_product(d33, d30_bracket), inverse_radii)

    d40_numerator = -_product(
        d43,
        a_w_prime + _product(script_s2, epsilon_minus_w),
    ) - _product(a_matrix, script_s1 - s12_epsilon_minus_w)
    d40 = _product(d40_numerator, inverse_radii)

    zero = np.zeros((n_channels, n_channels), dtype=dtype)
    coefficients = np.empty((4, 4, n_channels, n_channels), dtype=dtype)
    coefficients[:, 3] = np.stack((zero, zero, d33, d43))
    coefficients[:, 1] = np.stack((zero, zero, d31, d41))
    coefficients[:, 2] = np.stack((zero, -inverse_radii, d32, d42))
    coefficients[:, 0] = np.stack(
        (
            inverse_radii,
            -_product(epsilon_minus_w, inverse_radii),
            d30,
            d40,
        )
    )
    coefficients /= np.array([1.0, 2.0, 4.0, 8.0])[:, None, None, None]
    return VdCoefficients(
        energies=energies,
        radii=radii,
        angular_momenta=angular_momenta,
        coefficients=coefficients,
    )


def combine_usw(
    usw_values: NumericArray,
    transformation: VdCoefficients,
) -> NumericArray:
    """Combine four energy-sampled USW sets into the four v&d functions.

    ``usw_values`` has shape ``(4, ..., n_channels)``.  The returned array has
    shape ``(4, ..., n_channels)``, with its leading axis ordered by radial
    derivative ``d = 0, 1, 2, 3``.  This is Eq. (vandd), and therefore applies
    the super-unitary basis transformation at every point in the ``...`` axes.
    """

    table = divided_differences(transformation.energies, usw_values)
    prefixes = np.stack(tuple(table[order][0] for order in range(4)))
    sample_shape = prefixes.shape[1:-1]
    flattened = prefixes.reshape((4, -1, prefixes.shape[-1]))
    combined = contract(
        "npi,ndij->dpj",
        flattened,
        transformation.coefficients,
    )
    return combined.reshape((4, *sample_shape, combined.shape[-1]))


def constrained_vd_weights(
    localized_components: NumericArray,
    extended_components: NumericArray,
    target_values: NumericArray,
) -> NumericArray:
    r"""Return the minimum-norm weights for open-structure constraints.

    The component arrays have shape ``(n_constraints, 4, ...)`` and contain
    :math:`q^l_{c,dRL}` and :math:`q^e_{c,dRL}`, including the supplied
    boundary data.  The result has shape ``(4, ...)`` and supplies
    :math:`\alpha_{dRL}` in Eq. (IandII).  Eqs. (constr), (w), and (lambdas)
    are solved as ``alpha = Delta.T solve(Delta Delta.T, q - q_l)`` without a
    pseudoinverse.
    """

    n_constraints = localized_components.shape[0]
    localized = localized_components.reshape((n_constraints, -1))
    extended = extended_components.reshape((n_constraints, -1))
    delta = extended - localized
    right_hand_side = target_values - localized.sum(axis=1)
    gram = contract("ci,di->cd", delta, delta)
    half_lambdas = solve(gram, right_hand_side)
    weights = contract("ci,c->i", delta, half_lambdas)
    return weights.reshape(localized_components.shape[1:])
