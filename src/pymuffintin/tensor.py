"""Backend-neutral tensor IR: an optimized contraction path plus a small set
of host-side linear-algebra primitives.

This is a two-tier contract. `contract()` is backend-dispatched: it compiles
(and caches) a reusable `opt_einsum` expression for a given subscript and
operand-shape combination, then evaluates it with `backend="auto"` so
`opt_einsum` infers the execution backend from the operands' own array type.
A future CTF backend needs to supply only `asarray`/`to_host` through the
`Backend` protocol below; `opt_einsum` then dispatches the same expression to
`ctf.einsum`/`ctf.tensordot` on ctf arrays without any change at call sites.

`eigh`, `lstsq`, and `pinv` are host-side by declaration, not by omission:
they run sequentially on host (numpy) arrays regardless of the active
backend, either because they lack a distributed CTF-native equivalent or
because a caller's determinism guarantee depends on numpy's exact
eigenvector/pivot ordering. For the same reason, the deterministic QRCP
column selection in `auxiliary/thc.py` stays entirely host-side and is not
routed through this module: Gate 2's same-engine determinism depends on its
exact pivot order.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import opt_einsum as oe
from numpy.typing import NDArray


@runtime_checkable
class Backend(Protocol):
    """What a tensor backend must supply: identity, and a host round trip."""

    name: str

    def asarray(self, host: np.ndarray) -> object: ...

    def to_host(self, array: object) -> np.ndarray: ...


class _NumpyBackend:
    name = "numpy"

    def asarray(self, host: np.ndarray) -> np.ndarray:
        return np.asarray(host)

    def to_host(self, array: object) -> np.ndarray:
        return np.asarray(array)


_BACKENDS: dict[str, Backend] = {}
_ACTIVE = "numpy"


def register_backend(backend: Backend) -> None:
    """Register `backend` under its `name`, replacing a prior registration."""
    _BACKENDS[backend.name] = backend


def set_backend(name: str) -> None:
    """Select the active backend. `name` must already be registered."""
    if name not in _BACKENDS:
        raise ValueError(f"backend {name!r} is not registered")
    global _ACTIVE
    _ACTIVE = name


def get_backend() -> Backend:
    """Return the active backend."""
    return _BACKENDS[_ACTIVE]


register_backend(_NumpyBackend())


_ExpressionKey = tuple[str, tuple[tuple[int, ...], ...]]
_EXPRESSION_CACHE: dict[_ExpressionKey, "oe.contract.ContractExpression"] = {}


def contract(expr: str, *operands: NDArray) -> NDArray:
    """The sole sanctioned entry for fixed-structure multilinear contractions.

    Compiles and caches an `opt_einsum` contraction expression keyed on
    `(expr, operand shapes)`, then evaluates it with `backend="auto"` so
    dispatch follows the operands' own array type. See the module docstring
    for how this becomes the CTF hook.
    """
    shapes = tuple(operand.shape for operand in operands)
    key = (expr, shapes)
    compiled = _EXPRESSION_CACHE.get(key)
    if compiled is None:
        compiled = oe.contract_expression(expr, *shapes)
        _EXPRESSION_CACHE[key] = compiled
    return compiled(*operands, backend="auto")


def eigh(a: NDArray) -> tuple[NDArray, NDArray]:
    """Host-side Hermitian eigendecomposition. See the module docstring."""
    host = get_backend().to_host(a)
    return np.linalg.eigh(host)


def lstsq(a: NDArray, b: NDArray) -> NDArray:
    """Host-side least squares (numpy's current default `rcond`). See the
    module docstring."""
    backend = get_backend()
    solution, _, _, _ = np.linalg.lstsq(backend.to_host(a), backend.to_host(b), rcond=None)
    return solution


def pinv(a: NDArray) -> NDArray:
    """Host-side Moore-Penrose pseudoinverse. See the module docstring."""
    return np.linalg.pinv(get_backend().to_host(a))
