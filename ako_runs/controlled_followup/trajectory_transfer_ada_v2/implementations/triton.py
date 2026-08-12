"""Triton destination adapter."""
from . import _build_destination


def build(spec, mechanism_enabled, route=None):
    return _build_destination("triton", spec, mechanism_enabled, route)
