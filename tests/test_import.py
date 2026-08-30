import sys


def test_import_does_not_load_native_backend() -> None:
    sys.modules.pop("libmuffintin", None)
    import pymuffintin

    assert "libmuffintin" not in sys.modules
    assert "PairLayout" in pymuffintin.__all__
    assert "Orbitals" in pymuffintin.__all__
