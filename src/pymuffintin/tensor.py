"""Backend-neutral tensor IR: an optimized contraction path plus a small set
of host-side linear-algebra primitives.

This is a two-tier contract. `contract()` is operand-driven: it compiles (and
caches) a reusable `opt_einsum` expression for a given subscript and
operand-shape combination, then evaluates it with `backend="auto"`. NumPy
operands therefore remain local NumPy contractions, while CTF operands are
dispatched dynamically to the installed CTF module. The optional `CtfBackend`
below supplies the explicit NumPy-to-CTF and CTF-to-NumPy conversions; it is
not registered or selected automatically.

`eigh`, `solve`, `inv`, `lstsq`, and `pinv` are host-side by declaration, not by omission:
they run sequentially on host (numpy) arrays regardless of the active
backend, either because they lack a distributed CTF-native equivalent or
because a caller's determinism guarantee depends on numpy's exact
eigenvector/pivot ordering. For the same reason, the deterministic QRCP
column selection in `auxiliary/thc.py` stays entirely host-side and is not
routed through this module: reproducible weighted-QRCP ISDF point selection
depends on its exact pivot order.
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


class CtfBackend:
    """Adapt the optional CTF Python binding to the backend protocol.

    Importing :mod:`pymuffintin.tensor` does not import CTF or initialize an
    MPI world. The caller owns the CTF/MPI lifetime and must call
    ``register_backend(CtfBackend())`` explicitly when CTF is available.
    ``to_host`` on a CTF tensor materializes the complete tensor and is a
    collective operation; every rank in the tensor's CTF world must
    participate. Host NumPy arrays remain local so the existing host-side
    linear-algebra path is safe with this backend selected.
    """

    name = "ctf"

    def asarray(self, host: np.ndarray) -> object:
        """Copy a host array into a CTF tensor using CTF's public helper."""

        import ctf

        return ctf.from_nparray(np.asarray(host))

    def to_host(self, array: object) -> np.ndarray:
        """Materialize a CTF tensor, or preserve an already-host array."""

        if isinstance(array, np.ndarray):
            return np.asarray(array)
        import ctf

        return np.asarray(ctf.to_nparray(array))


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


def solve(a: NDArray, b: NDArray) -> NDArray:
    """Host-side exact solve of a square linear system. See the module docstring."""
    backend = get_backend()
    return np.linalg.solve(backend.to_host(a), backend.to_host(b))


def inv(a: NDArray) -> NDArray:
    """Host-side inverse of a nonsingular square matrix. See the module docstring."""
    return np.linalg.inv(get_backend().to_host(a))


def lstsq(a: NDArray, b: NDArray) -> NDArray:
    """Host-side least squares (numpy's current default `rcond`). See the
    module docstring."""
    backend = get_backend()
    solution, _, _, _ = np.linalg.lstsq(backend.to_host(a), backend.to_host(b), rcond=None)
    return solution


def pinv(a: NDArray) -> NDArray:
    """Host-side Moore-Penrose pseudoinverse. See the module docstring."""
    return np.linalg.pinv(get_backend().to_host(a))
