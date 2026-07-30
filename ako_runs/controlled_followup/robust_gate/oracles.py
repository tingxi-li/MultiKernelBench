"""Explicit semantic and arithmetic-contract oracles.

Unlike a CPU fp16 matmul, contract oracles round at named boundaries and then
perform the intervening operation in fp64.  That separates representation loss
from implementation/accumulation-order error and works on CPU-only hosts.
"""

from __future__ import annotations

import math
from typing import Any

import torch


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def round_boundary(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype not in DTYPES:
        raise ValueError(f"unsupported boundary dtype {dtype!r}")
    return tensor.to(DTYPES[dtype]).to(torch.float64)


def _gelu_exact(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * value * (1.0 + torch.erf(value / math.sqrt(2.0)))


def _matmul_semantic(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return inputs["a"].double() @ inputs["b"].double()


def _fused_semantic(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    y = inputs["x"].double() @ inputs["weight"].double().transpose(0, 1)
    y = y + inputs["bias"].double()
    y = _gelu_exact(y)
    return torch.softmax(y, dim=-1)


def _sdpa_semantic(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    q, k, v = inputs["q"].double(), inputs["k"].double(), inputs["v"].double()
    outputs = []
    for start in range(0, q.shape[0], 4):
        qq, kk, vv = q[start : start + 4], k[start : start + 4], v[start : start + 4]
        scores = (qq @ kk.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        outputs.append(torch.softmax(scores, dim=-1) @ vv)
    return torch.cat(outputs, dim=0)


def semantic_reference(op: str, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if op == "matmul":
        return _matmul_semantic(inputs)
    if op == "fused_softmax":
        return _fused_semantic(inputs)
    if op == "sdpa":
        return _sdpa_semantic(inputs)
    raise ValueError(f"unknown op {op!r}")


def contract_reference(
    op: str, inputs: dict[str, torch.Tensor], contract: dict[str, Any]
) -> torch.Tensor:
    """Evaluate declared rounding boundaries with fp64 between boundaries."""
    operand_dtype = contract.get("operand_dtype", "fp16")
    output_dtype = contract.get("output_dtype", "fp32")
    if op == "matmul":
        a = round_boundary(inputs["a"], contract.get("a_dtype", operand_dtype))
        b = round_boundary(inputs["b"], contract.get("b_dtype", operand_dtype))
        y = a @ b
        if contract.get("gemm_output_dtype"):
            y = round_boundary(y, contract["gemm_output_dtype"])
        return round_boundary(y, output_dtype)
    if op == "fused_softmax":
        x = round_boundary(inputs["x"], contract.get("x_dtype", operand_dtype))
        weight = round_boundary(
            inputs["weight"], contract.get("weight_dtype", operand_dtype)
        )
        bias = round_boundary(inputs["bias"], contract.get("bias_dtype", "fp32"))
        y = x @ weight.transpose(0, 1)
        if contract.get("gemm_output_dtype"):
            y = round_boundary(y, contract["gemm_output_dtype"])
        y = y + bias
        if contract.get("gelu", "exact_erf") != "exact_erf":
            raise ValueError("the primary fused contract requires exact_erf GELU")
        y = _gelu_exact(y)
        if contract.get("pre_softmax_dtype"):
            y = round_boundary(y, contract["pre_softmax_dtype"])
        y = torch.softmax(y, dim=-1)
        if contract.get("probability_dtype"):
            y = round_boundary(y, contract["probability_dtype"])
        return round_boundary(y, output_dtype)
    if op == "sdpa":
        q = round_boundary(inputs["q"], contract.get("q_dtype", operand_dtype))
        k = round_boundary(inputs["k"], contract.get("k_dtype", operand_dtype))
        v = round_boundary(inputs["v"], contract.get("v_dtype", operand_dtype))
        outputs = []
        for start in range(0, q.shape[0], 4):
            qq, kk, vv = q[start : start + 4], k[start : start + 4], v[start : start + 4]
            scores = (qq @ kk.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            scores = round_boundary(scores, contract.get("score_dtype", "fp32"))
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = round_boundary(
                probabilities, contract.get("probability_dtype", "fp16")
            )
            outputs.append(probabilities @ vv)
        return round_boundary(torch.cat(outputs, dim=0), output_dtype)
    raise ValueError(f"unknown op {op!r}")


def _native_matmul(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return inputs["a"].float() @ inputs["b"].float()


def _native_fused(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    x, weight, bias = inputs["x"].float(), inputs["weight"].float(), inputs["bias"].float()
    y = x @ weight.transpose(0, 1) + bias
    y = _gelu_exact(y)
    return torch.softmax(y, dim=-1)


def _native_sdpa(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    q, k, v = inputs["q"].float(), inputs["k"].float(), inputs["v"].float()
    outputs = []
    for start in range(0, q.shape[0], 4):
        qq, kk, vv = q[start : start + 4], k[start : start + 4], v[start : start + 4]
        scores = (qq @ kk.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        outputs.append(torch.softmax(scores, dim=-1) @ vv)
    return torch.cat(outputs, dim=0)


def native_fp32_reference(op: str, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if op == "matmul":
        return _native_matmul(inputs)
    if op == "fused_softmax":
        return _native_fused(inputs)
    if op == "sdpa":
        return _native_sdpa(inputs)
    raise ValueError(f"unknown op {op!r}")


def native_mixed_reference(
    op: str, inputs: dict[str, torch.Tensor], contract: dict[str, Any]
) -> torch.Tensor:
    """A CPU anchor using fp32 operations on explicitly quantized operands."""
    operand_dtype = contract.get("operand_dtype", "fp16")

    def q(name: str, override: str | None = None) -> torch.Tensor:
        dtype = contract.get(override, operand_dtype) if override else operand_dtype
        return inputs[name].to(DTYPES[dtype]).float()

    if op == "matmul":
        y = q("a", "a_dtype") @ q("b", "b_dtype")
    elif op == "fused_softmax":
        x = q("x", "x_dtype")
        weight = q("weight", "weight_dtype")
        bias_dtype = contract.get("bias_dtype", "fp32")
        bias = inputs["bias"].to(DTYPES[bias_dtype]).float()
        y = _gelu_exact(x @ weight.transpose(0, 1) + bias)
        if contract.get("pre_softmax_dtype"):
            y = y.to(DTYPES[contract["pre_softmax_dtype"]]).float()
        y = torch.softmax(y, dim=-1)
        if contract.get("probability_dtype"):
            y = y.to(DTYPES[contract["probability_dtype"]]).float()
    elif op == "sdpa":
        qv, kv, vv = q("q", "q_dtype"), q("k", "k_dtype"), q("v", "v_dtype")
        outputs = []
        for start in range(0, qv.shape[0], 4):
            qq = qv[start : start + 4]
            kk = kv[start : start + 4]
            value = vv[start : start + 4]
            scores = (qq @ kk.transpose(-2, -1)) / math.sqrt(qv.shape[-1])
            scores = scores.to(DTYPES[contract.get("score_dtype", "fp32")]).float()
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = probabilities.to(
                DTYPES[contract.get("probability_dtype", "fp16")]
            ).float()
            outputs.append(probabilities @ value)
        y = torch.cat(outputs, dim=0)
    else:
        raise ValueError(f"unknown op {op!r}")
    return y.to(DTYPES[contract.get("output_dtype", "fp32")])


def resolve_output(
    kind: str,
    op: str,
    inputs: dict[str, torch.Tensor],
    contract: dict[str, Any] | None = None,
) -> torch.Tensor:
    contract = contract or {}
    if kind == "semantic_fp64":
        return semantic_reference(op, inputs)
    if kind == "contract_mixed":
        return contract_reference(op, inputs, contract)
    if kind == "native_fp32":
        return native_fp32_reference(op, inputs)
    if kind == "native_mixed":
        return native_mixed_reference(op, inputs, contract)
    raise ValueError(f"unknown output kind {kind!r}")
