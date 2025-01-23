"""The first test of the repository."""


def test_version_exists() -> None:
    """A test that ensures importing src succeeds and that there is a version."""
    import sys

    print(sys.path)
    from src import _version

    assert _version.__version__ is not None
