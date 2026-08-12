"""CUDA-no-PTX destination adapter."""
from . import _build_destination


def build(spec, mechanism_enabled, route=None):
    return _build_destination("cuda_noptx", spec, mechanism_enabled, route)
