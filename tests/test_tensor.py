import numpy as np

from pymuffintin import tensor


class _TaggedArray(np.ndarray):
    """Minimal ndarray subclass standing in for a future non-numpy backend array."""


class _TaggingBackend:
    """Toy backend exercising only the asarray/to_host round trip a real
    (e.g. CTF) backend would need to supply."""

    name = "tagging-test"

    def asarray(self, host: np.ndarray) -> _TaggedArray:
        return np.asarray(host).view(_TaggedArray)

    def to_host(self, array: object) -> np.ndarray:
        return np.asarray(array)


def test_contract_matches_direct_einsum_on_the_hf_expression() -> None:
    rng = np.random.default_rng(11)
    n_k, n_orb, n_aux = 2, 3, 4
    f_sel = rng.uniform(0.1, 1.0, size=(n_k, n_orb))
    vertices = (
        rng.standard_normal((n_k, n_orb, n_orb, n_aux))
        + 1j * rng.standard_normal((n_k, n_orb, n_orb, n_aux))
    ).astype(np.complex128)
    generator = rng.standard_normal((n_aux, n_aux)) + 1j * rng.standard_normal((n_aux, n_aux))
    matrix = (generator @ generator.conj().T + n_aux * np.eye(n_aux)).astype(np.complex128)

    expr = "ki,kija,ab,kilb->kjl"
    result = tensor.contract(expr, f_sel, vertices.conj(), matrix, vertices)
    expected = np.einsum(expr, f_sel, vertices.conj(), matrix, vertices)
    np.testing.assert_allclose(result, expected, rtol=1.0e-10, atol=1.0e-12)


def test_expression_cache_returns_the_same_compiled_object_for_repeated_shapes() -> None:
    expr = "xy,yz->xz"
    key = (expr, ((7, 9), (9, 11)))
    tensor._EXPRESSION_CACHE.pop(key, None)

    a = np.random.default_rng(1).standard_normal((7, 9))
    b = np.random.default_rng(2).standard_normal((9, 11))
    tensor.contract(expr, a, b)
    first = tensor._EXPRESSION_CACHE[key]

    a2 = np.random.default_rng(3).standard_normal((7, 9))
    b2 = np.random.default_rng(4).standard_normal((9, 11))
    tensor.contract(expr, a2, b2)
    second = tensor._EXPRESSION_CACHE[key]

    assert first is second


def test_registry_round_trip_and_contract_on_a_registered_backend() -> None:
    backend = _TaggingBackend()
    tensor.register_backend(backend)
    tensor.set_backend("tagging-test")
    try:
        assert tensor.get_backend() is backend

        rng = np.random.default_rng(7)
        host_a = (
            rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
        ).astype(np.complex128)
        host_b = (
            rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))
        ).astype(np.complex128)
        a = backend.asarray(host_a)
        b = backend.asarray(host_b)
        assert isinstance(a, _TaggedArray)
        assert isinstance(b, _TaggedArray)
        np.testing.assert_array_equal(backend.to_host(a), host_a)
        np.testing.assert_array_equal(backend.to_host(b), host_b)

        result = tensor.contract("ab,bc->ac", a, b)
        np.testing.assert_allclose(np.asarray(result), host_a @ host_b, rtol=1.0e-12, atol=1.0e-14)
    finally:
        tensor.set_backend("numpy")

    assert tensor.get_backend().name == "numpy"
