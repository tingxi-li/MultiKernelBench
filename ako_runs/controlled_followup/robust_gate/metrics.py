"""Operation-aware correctness metrics with strict input preflight."""

from __future__ import annotations

import math
from typing import Any

import torch


class MetricInputError(ValueError):
    """The candidate/reference pair is unsafe or meaningless to compare."""


def _preflight(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        raise MetricInputError("reference and candidate must both be tensors")
    if reference.shape != candidate.shape:
        raise MetricInputError(
            f"shape mismatch: reference={tuple(reference.shape)} candidate={tuple(candidate.shape)}"
        )
    if reference.numel() == 0:
        raise MetricInputError("empty outputs are not valid")
    if not torch.is_floating_point(reference) or not torch.is_floating_point(candidate):
        raise MetricInputError("reference and candidate outputs must be floating point")
    if not bool(torch.isfinite(reference).all()):
        raise MetricInputError("reference contains NaN or Inf")
    if not bool(torch.isfinite(candidate).all()):
        raise MetricInputError("candidate contains NaN or Inf")


def _quantile(values: torch.Tensor, probability: float) -> float:
    flat = values.reshape(-1)
    if flat.numel() == 1:
        return float(flat.item())
    return float(torch.quantile(flat.double(), probability).item())


def _common_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.double()
    got = candidate.double()
    error = got - ref
    abs_error = error.abs()
    ref_norm = torch.linalg.vector_norm(ref.reshape(-1))
    error_norm = torch.linalg.vector_norm(error.reshape(-1))
    denominator = max(float(ref_norm.item()), torch.finfo(torch.float64).tiny)
    return {
        "nonfinite_count": 0.0,
        "max_abs_err": float(abs_error.max().item()),
        "mean_abs_err": float(abs_error.mean().item()),
        "nrmse": float(error_norm.item()) / denominator,
        "abs_signed_bias": abs(float(error.mean().item())),
        "reference_abs_mean": float(ref.abs().mean().item()),
        "reference_abs_max": float(ref.abs().max().item()),
    }


def _matmul_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    inputs: dict[str, torch.Tensor],
) -> dict[str, float]:
    if "a" not in inputs or "b" not in inputs:
        raise MetricInputError("matmul metrics require input tensors 'a' and 'b'")
    scale = inputs["a"].double().abs() @ inputs["b"].double().abs()
    if scale.shape != reference.shape:
        raise MetricInputError("matmul backward-error scale has wrong shape")
    floor = max(float(scale.max().item()) * torch.finfo(torch.float64).eps,
                torch.finfo(torch.float64).tiny)
    error = (candidate.double() - reference.double()).abs()
    scale_norm = torch.linalg.vector_norm(scale.reshape(-1))
    error_norm = torch.linalg.vector_norm(error.reshape(-1))
    return {
        "componentwise_backward_error": float((error / (scale + floor)).max().item()),
        # Ordinary ||error||/||reference|| is undefined for the registered exact
        # cancellation case.  |A|@|B| is the standard forward-error scale: it is
        # comparable to |C| for same-sign products and remains nonzero when real
        # products cancel in C.
        "scaled_rmse": float(error_norm.item())
        / max(float(scale_norm.item()), torch.finfo(torch.float64).tiny),
    }


def _probability_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    ref, got = reference.double(), candidate.double()
    rows_ref = ref.reshape(-1, ref.shape[-1])
    rows_got = got.reshape(-1, got.shape[-1])
    row_sum_error = (rows_got.sum(dim=-1) - 1.0).abs()
    tv = 0.5 * (rows_got - rows_ref).abs().sum(dim=-1)

    eps = torch.finfo(torch.float64).tiny
    p = rows_ref.clamp_min(0)
    q = rows_got.clamp_min(0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    midpoint = 0.5 * (p + q)
    kl_p = torch.where(p > 0, p * (p.clamp_min(eps).log() - midpoint.log()), 0.0).sum(-1)
    kl_q = torch.where(q > 0, q * (q.clamp_min(eps).log() - midpoint.log()), 0.0).sum(-1)
    sqrt_js = torch.sqrt((0.5 * (kl_p + kl_q)).clamp_min(0))

    scale = torch.maximum(rows_ref.abs(), torch.full_like(rows_ref, 1.0 / ref.shape[-1]))
    scaled = (rows_got - rows_ref).abs() / scale
    top1_mismatch = (rows_got.argmax(-1) != rows_ref.argmax(-1)).double()
    return {
        "negative_count": float((got < 0).sum().item()),
        "row_sum_error_max": float(row_sum_error.max().item()),
        "tv_max": float(tv.max().item()),
        "tv_q99": _quantile(tv, 0.99),
        "sqrt_js_max": float(sqrt_js.max().item()),
        "scaled_max_abs_err": float(scaled.max().item()),
        "top1_mismatch_rate": float(top1_mismatch.mean().item()),
    }


def _sdpa_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.double().reshape(-1, reference.shape[-1])
    got = candidate.double().reshape(-1, candidate.shape[-1])
    error = got - ref
    eps = torch.finfo(torch.float64).tiny
    ref_l2 = torch.linalg.vector_norm(ref, dim=-1)
    err_l2 = torch.linalg.vector_norm(error, dim=-1)
    relative_l2 = err_l2 / ref_l2.clamp_min(eps)

    got_l2 = torch.linalg.vector_norm(got, dim=-1)
    cosine = (ref * got).sum(-1) / (ref_l2 * got_l2).clamp_min(eps)
    valid = (ref_l2 > eps) & (got_l2 > eps)
    cosine_distance = torch.where(valid, (1.0 - cosine).clamp_min(0), torch.zeros_like(cosine))
    row_rms = ref.square().mean(-1).sqrt().clamp_min(eps)
    scaled_max = error.abs().max(-1).values / row_rms
    return {
        "query_relative_l2_max": float(relative_l2.max().item()),
        "query_relative_l2_q99": _quantile(relative_l2, 0.99),
        "cosine_distance_max": float(cosine_distance.max().item()),
        "scaled_max_abs_err": float(scaled_max.max().item()),
    }


def compute_metrics(
    op: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    inputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Return finite scalar metrics; lower is better for every gate metric."""
    _preflight(reference, candidate)
    metrics = _common_metrics(reference, candidate)
    if op == "matmul":
        metrics.update(_matmul_metrics(reference, candidate, inputs or {}))
    elif op == "fused_softmax":
        metrics.update(_probability_metrics(reference, candidate))
    elif op == "sdpa":
        metrics.update(_sdpa_metrics(reference, candidate))
    else:
        raise MetricInputError(f"unknown op {op!r}")
    for name, value in metrics.items():
        if not math.isfinite(value):
            raise MetricInputError(f"metric {name} is non-finite")
    return metrics
