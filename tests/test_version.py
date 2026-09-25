import version


def test_canonical_release_version() -> None:
    assert version.__version__ == "0.1"
    assert version.VERSION == version.__version__
