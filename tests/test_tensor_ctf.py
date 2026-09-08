import numpy as np
import pytest

from pymuffintin import tensor


def test_ctf_backend_round_trip_and_complex_contraction() -> None:
    ctf = pytest.importorskip("ctf")
    backend = tensor.CtfBackend()
    host_left = np.array(
        [[1.0 + 2.0j, 2.0 - 1.0j], [0.5 + 0.25j, -3.0 + 1.5j]],
        dtype=np.complex128,
    )
    host_right = np.array(
        [[-2.0 + 0.5j, 1.0 - 1.0j], [4.0 + 2.0j, 0.25 - 3.0j]],
        dtype=np.complex128,
    )

    left = backend.asarray(host_left)
    right = backend.asarray(host_right)
    np.testing.assert_array_equal(backend.to_host(left), host_left)
    np.testing.assert_array_equal(backend.to_host(host_right), host_right)

    result = tensor.contract("ab,bc->ac", left, right)

    assert isinstance(result, ctf.tensor)
    np.testing.assert_allclose(backend.to_host(result), host_left @ host_right)
