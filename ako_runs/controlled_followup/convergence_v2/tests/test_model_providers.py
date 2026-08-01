from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ako_runs.controlled_followup.convergence_v2.model_resolution import (
    ModelResolutionError,
    load_model_resolution_lock,
)
from ako_runs.controlled_followup.convergence_v2.provider_adapters import (
    AnthropicMessagesAdapter,
    OpenAIResponsesAdapter,
    ProviderRequest,
)


def resolved_lock(path: Path) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "state": "resolved",
        "resolutions": [
            {"provider": "openai", "requested_alias": "gpt-5.6-sol", "immutable_revision": "gpt-5.6-sol-2026-07-31.1", "provider_attested_immutable": True, "resolved_at_utc": "2026-07-31T12:00:00Z", "resolution_evidence_sha256": "1" * 64},
            {"provider": "anthropic", "requested_alias": "claude-opus-4.8", "immutable_revision": "claude-opus-4.8-20260731-r1", "provider_attested_immutable": True, "resolved_at_utc": "2026-07-31T12:01:00Z", "resolution_evidence_sha256": "2" * 64}
        ]
    }))


class RecordingCreate:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class ModelProviderTest(unittest.TestCase):
    def test_unresolved_model_lock_fails(self) -> None:
        path = Path(__file__).resolve().parents[1] / "locks/model_resolution_lock.json"
        with self.assertRaises(ModelResolutionError):
            load_model_resolution_lock(path)

    def test_direct_adapters_normalize_usage_without_seed_or_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "lock.json"
            resolved_lock(lock)
            resolutions = load_model_resolution_lock(lock)
        openai_create = RecordingCreate(SimpleNamespace(
            id="resp-1",
            output_text="candidate",
            usage={"input_tokens": 10, "output_tokens": 7, "total_tokens": 17, "input_tokens_details": {"cached_tokens": 3}, "output_tokens_details": {"reasoning_tokens": 2}},
        ))
        openai = OpenAIResponsesAdapter(SimpleNamespace(responses=openai_create), resolutions["openai:gpt-5.6-sol"])
        request = ProviderRequest("system", "user", requested_sampling_seed=123, stochastic_request_id="stochastic-1")
        result = openai.generate(request)
        self.assertNotIn("seed", openai_create.kwargs)
        self.assertEqual((result.usage.reasoning_tokens, result.usage.cache_read_tokens), (2, 3))
        self.assertFalse(result.sampling_seed_supported)
        self.assertNotIn("api_key", json.dumps(openai.audit_metadata()).lower())

        anthropic_create = RecordingCreate(SimpleNamespace(
            id="msg-1",
            content=[{"type": "text", "text": "candidate-2"}],
            usage={"input_tokens": 11, "output_tokens": 5, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 1},
        ))
        anthropic = AnthropicMessagesAdapter(SimpleNamespace(messages=anthropic_create), resolutions["anthropic:claude-opus-4.8"])
        result2 = anthropic.generate(request)
        self.assertNotIn("seed", anthropic_create.kwargs)
        self.assertEqual((result2.text, result2.usage.cache_read_tokens, result2.usage.cache_write_tokens), ("candidate-2", 4, 1))
        self.assertNotIn("api_key", json.dumps(anthropic.audit_metadata()).lower())


if __name__ == "__main__":
    unittest.main()

