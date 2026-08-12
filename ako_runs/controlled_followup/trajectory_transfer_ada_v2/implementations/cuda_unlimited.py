"""CUDA-unlimited destination adapter."""
from . import _build_destination


def build(spec, mechanism_enabled, route=None):
    return _build_destination("cuda_unlimited", spec, mechanism_enabled, route)
