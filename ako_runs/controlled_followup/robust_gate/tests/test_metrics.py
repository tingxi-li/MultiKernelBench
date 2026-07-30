from __future__ import annotations

import unittest

import torch

from ako_runs.controlled_followup.robust_gate.metrics import (
    MetricInputError,
    compute_metrics,
)


class MetricTests(unittest.TestCase):
    def test_preflight_fails_closed(self) -> None:
        cases = [
            (torch.ones(2), torch.ones(1, 2)),
            (torch.empty(0), torch.empty(0)),
            (torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)),
            (torch.tensor([float("nan")]), torch.tensor([0.0])),
            (torch.tensor([0.0]), torch.tensor([float("inf")])),
        ]
        for reference, candidate in cases:
            with self.subTest(reference=reference, candidate=candidate):
                with self.assertRaises(MetricInputError):
                    compute_metrics("sdpa", reference, candidate)

    def test_matmul_backward_error_golden(self) -> None:
        a = torch.tensor([[1.0, -1.0]])
        b = torch.tensor([[1.0], [1.0]])
        reference = torch.tensor([[0.0]], dtype=torch.float64)
        candidate = torch.tensor([[0.1]], dtype=torch.float64)
        metrics = compute_metrics(
            "matmul", reference, candidate, {"a": a, "b": b}
        )
        self.assertAlmostEqual(metrics["componentwise_backward_error"], 0.05)
        self.assertAlmostEqual(metrics["scaled_rmse"], 0.05)

    def test_softmax_pathology_legacy_passes_but_tv_detects(self) -> None:
        n = 8192
        delta = 5e-5
        reference = torch.full((1, n), 1.0 / n, dtype=torch.float64)
        reference[:, 0::2] += delta
        reference[:, 1::2] -= delta
        candidate = torch.full_like(reference, 1.0 / n)
        self.assertTrue(torch.allclose(reference, candidate, atol=1e-4, rtol=1e-4))
        metrics = compute_metrics("fused_softmax", reference, candidate)
        self.assertAlmostEqual(metrics["tv_max"], 0.2048, places=10)
        self.assertGreater(metrics["sqrt_js_max"], 0.0)

    def test_exact_probability_output_has_zero_distances(self) -> None:
        reference = torch.softmax(torch.randn(3, 9, generator=torch.Generator().manual_seed(4)), -1)
        metrics = compute_metrics("fused_softmax", reference, reference.clone())
        for name in ("max_abs_err", "nrmse", "tv_max", "sqrt_js_max"):
            self.assertEqual(metrics[name], 0.0)


if __name__ == "__main__":
    unittest.main()
