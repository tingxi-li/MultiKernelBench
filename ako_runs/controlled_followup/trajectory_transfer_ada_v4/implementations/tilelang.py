"""TileLang destination adapter."""
from . import _build_destination


def build(spec, mechanism_enabled, route=None):
    return _build_destination("tilelang", spec, mechanism_enabled, route)
