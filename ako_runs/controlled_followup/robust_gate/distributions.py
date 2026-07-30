"""Deterministic input families used by the robust-gate campaign."""

from __future__ import annotations

import math
from typing import Any

import torch


RMS_U01 = 1.0 / math.sqrt(3.0)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _sample(
    shape: tuple[int, ...], kind: str, seed: int, params: dict[str, Any] | None = None
) -> torch.Tensor:
    params = params or {}
    generator = _generator(seed)
    if kind == "uniform_positive":
        out = torch.rand(shape, generator=generator, dtype=torch.float32)
    elif kind == "normal_zero":
        out = torch.randn(shape, generator=generator, dtype=torch.float32) * RMS_U01
    elif kind == "rademacher":
        signs = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
        out = (signs.to(torch.float32) * 2.0 - 1.0) * RMS_U01
    elif kind == "mean_shifted_normal":
        rho = float(params.get("rho", 0.5))
        if abs(rho) > 1:
            raise ValueError("mean_shifted_normal requires |rho| <= 1")
        sign = float(params.get("mean_sign", 1.0))
        z = torch.randn(shape, generator=generator, dtype=torch.float32)
        out = RMS_U01 * (sign * rho + math.sqrt(1.0 - rho * rho) * z)
    elif kind == "signed_log_uniform":
        low = float(params.get("exp_low", -4.0))
        high = float(params.get("exp_high", 4.0))
        exponents = torch.rand(shape, generator=generator, dtype=torch.float32)
        exponents = low + (high - low) * exponents
        signs = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
        out = torch.pow(2.0, exponents) * (signs.to(torch.float32) * 2.0 - 1.0)
        measured = out.square().mean().sqrt()
        if measured > 0:
            out = out * (RMS_U01 / measured)
    elif kind == "zeros":
        out = torch.zeros(shape, dtype=torch.float32)
    elif kind == "ones":
        out = torch.ones(shape, dtype=torch.float32)
    else:
        raise ValueError(f"unknown distribution kind {kind!r}")
    return out


def _to_device(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def make_matmul_inputs(
    shape: dict[str, int], case: dict[str, Any], seeds: dict[str, int], device: str
) -> dict[str, torch.Tensor]:
    m, k, n = int(shape["M"]), int(shape["K"]), int(shape["N"])
    kind = case.get("distribution", "uniform_positive")
    params = case.get("params", {})
    if kind == "paired_cancellation":
        if k % 2:
            raise ValueError("paired_cancellation requires an even K")
        a_half = _sample((m, k // 2), "normal_zero", seeds["a"])
        b_half = _sample((k // 2, n), "normal_zero", seeds["b"])
        a = torch.stack((a_half, -a_half), dim=2).reshape(m, k)
        b = torch.stack((b_half, b_half), dim=1).reshape(k, n)
    else:
        a_params = dict(params)
        b_params = dict(params)
        if case.get("opposing_means"):
            a_params["mean_sign"] = 1.0
            b_params["mean_sign"] = -1.0
        a = _sample((m, k), kind, seeds["a"], a_params)
        b = _sample((k, n), kind, seeds["b"], b_params)
    return _to_device({"a": a, "b": b}, device)


def make_fused_inputs(
    shape: dict[str, int], case: dict[str, Any], seeds: dict[str, int], device: str
) -> dict[str, torch.Tensor]:
    m, k, n = int(shape["M"]), int(shape["K"]), int(shape["N"])
    activation = case.get("activation_distribution", "uniform_positive")
    params = case.get("params", {})
    x = _sample((m, k), activation, seeds["x"], params)
    bound = 1.0 / math.sqrt(k)
    weight_gen = _generator(seeds["weight"])
    bias_gen = _generator(seeds["bias"])
    weight = (torch.rand((n, k), generator=weight_gen) * 2.0 - 1.0) * bound
    bias = (torch.rand((n,), generator=bias_gen) * 2.0 - 1.0) * bound
    gain = float(case.get("weight_gain", 1.0))
    weight.mul_(gain)
    bias.mul_(gain)
    return _to_device({"x": x, "weight": weight, "bias": bias}, device)


def make_sdpa_inputs(
    shape: dict[str, int], case: dict[str, Any], seeds: dict[str, int], device: str
) -> dict[str, torch.Tensor]:
    full_shape = (
        int(shape["B"]),
        int(shape["H"]),
        int(shape["S"]),
        int(shape["D"]),
    )
    qk_kind = case.get("qk_distribution", "uniform_positive")
    v_kind = case.get("v_distribution", qk_kind)
    params = case.get("params", {})
    q = _sample(full_shape, qk_kind, seeds["q"], params)
    if case.get("correlated_qk"):
        noise = _sample(full_shape, "normal_zero", seeds["k"])
        coefficient = float(case.get("correlation_noise", 0.1))
        k = q + coefficient * noise
        k = k * (RMS_U01 / k.square().mean().sqrt().clamp_min(1e-30))
    else:
        k = _sample(full_shape, qk_kind, seeds["k"], params)
    v = _sample(full_shape, v_kind, seeds["v"], params)
    temperature = float(case.get("score_temperature", 1.0))
    if temperature <= 0:
        raise ValueError("score_temperature must be positive")
    factor = math.sqrt(temperature)
    q.mul_(factor)
    k.mul_(factor)
    return _to_device({"q": q, "k": k, "v": v}, device)


def make_inputs(
    op: str,
    shape: dict[str, int],
    case: dict[str, Any],
    seeds: dict[str, int],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    if op == "matmul":
        return make_matmul_inputs(shape, case, seeds, device)
    if op == "fused_softmax":
        return make_fused_inputs(shape, case, seeds, device)
    if op == "sdpa":
        return make_sdpa_inputs(shape, case, seeds, device)
    raise ValueError(f"unknown op {op!r}")

