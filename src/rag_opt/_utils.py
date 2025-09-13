
from importlib.util import find_spec

def check_pkg(pkg: str) -> None:
    """Check if a package is installed."""
    if not find_spec(pkg):
        msg = (
            f"Could not import {pkg} python package. "
            f"Please install it with `pip install {pkg}`"
        )
        raise ImportError(msg)