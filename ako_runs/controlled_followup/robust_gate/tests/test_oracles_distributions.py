from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from ako_runs.controlled_followup.robust_gate.distributions import make_matmul_inputs
from ako_runs.controlled_followup.robust_gate.oracles import (
    _gelu_exact,
    contract_reference,
    semantic_reference,
)


class OracleAndDistributionTests(unittest.TestCase):
    def test_hand_matmul(self) -> None:
        inputs = {
            "a": torch.tensor([[1.0, 2.0], [-1.0, 3.0]]),
            "b": torch.tensor([[4.0, -2.0], [0.5, 5.0]]),
        }
        expected = torch.tensor([[5.0, 8.0], [-2.5, 17.0]], dtype=torch.float64)
        torch.testing.assert_close(semantic_reference("matmul", inputs), expected)

    def test_paired_cancellation_is_structural(self) -> None:
        inputs = make_matmul_inputs(
            {"M": 3, "K": 8, "N": 4},
            {"id": "cancel", "distribution": "paired_cancellation"},
            {"a": 11, "b": 12},
            "cpu",
        )
        self.assertTrue(torch.equal(inputs["a"][:, 0::2], -inputs["a"][:, 1::2]))
        self.assertTrue(torch.equal(inputs["b"][0::2], inputs["b"][1::2]))
        self.assertTrue(torch.equal(semantic_reference("matmul", inputs), torch.zeros(3, 4)))

    def test_contract_rounds_fp16_before_fp64_matmul(self) -> None:
        inputs = {
            "a": torch.tensor([[1.0003, -0.3333]], dtype=torch.float32),
            "b": torch.tensor([[2.0007], [0.1251]], dtype=torch.float32),
        }
        contract = {"operand_dtype": "fp16", "output_dtype": "fp64"}
        got = contract_reference("matmul", inputs, contract)
        expected = inputs["a"].half().double() @ inputs["b"].half().double()
        self.assertTrue(torch.equal(got, expected))

    def test_exact_erf_is_not_tanh_gelu(self) -> None:
        x = torch.tensor([-3.0, 3.0], dtype=torch.float64)
        difference = (_gelu_exact(x) - F.gelu(x, approximate="tanh")).abs().max()
        self.assertGreater(float(difference), 4e-4)

    def test_tiny_sdpa_golden_and_metamorphic_invariants(self) -> None:
        eye = torch.eye(2).reshape(1, 1, 2, 2)
        v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        base = {"q": eye, "k": eye, "v": v}
        expected = torch.tensor(
            [[[[1.6604769013, 2.6604769013], [2.3395230987, 3.3395230987]]]],
            dtype=torch.float64,
        )
        output = semantic_reference("sdpa", base)
        torch.testing.assert_close(output, expected, rtol=1e-10, atol=1e-10)

        permutation = torch.tensor([1, 0])
        joint = {"q": eye, "k": eye[..., permutation, :], "v": v[..., permutation, :]}
        torch.testing.assert_close(semantic_reference("sdpa", joint), output)

        shifted = {"q": eye, "k": eye, "v": v + 7.0}
        torch.testing.assert_close(semantic_reference("sdpa", shifted), output + 7.0)

        ones = {"q": eye, "k": eye, "v": torch.ones_like(v)}
        torch.testing.assert_close(semantic_reference("sdpa", ones), torch.ones_like(output))

        zero_scores = {"q": torch.zeros_like(eye), "k": torch.zeros_like(eye), "v": v}
        expected_mean = v.double().mean(dim=-2, keepdim=True).expand_as(output)
        torch.testing.assert_close(semantic_reference("sdpa", zero_scores), expected_mean)


if __name__ == "__main__":
    unittest.main()

