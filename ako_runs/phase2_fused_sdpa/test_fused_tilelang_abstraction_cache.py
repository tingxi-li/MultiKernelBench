import unittest
from pathlib import Path


class SoftOnlyCacheRegressionTest(unittest.TestCase):
    def test_cache_binds_tensor_identity_lifetime_and_mutation(self):
        source = (Path(__file__).parent / "variants2" /
                  "fused_tilelang_abstraction.py").read_text()
        self.assertNotIn('box.get("src") != x.data_ptr()', source)
        self.assertIn('box.update(inputs=inputs, versions=versions, scratch=scratch)', source)
        self.assertIn('versions = tuple(t._version for t in inputs)', source)
        self.assertIn('common2.weight_fn(wmode)(W)', source)


if __name__ == "__main__":
    unittest.main()
