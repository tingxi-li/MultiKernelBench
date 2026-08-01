#!/usr/bin/env python3
"""Direct-provider adapters with normalized, secret-free audit records.

Provider SDKs are optional and imported only by ``from_env``. Tests and other
campaigns can inject a compatible client object.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Mapping, Protocol

try:
    from .model_resolution import ModelResolution
except ImportError:  # direct script execution
    from model_resolution import ModelResolution  # type: ignore


SENSITIVE_KEYS = {"api_key", "authorization", "access_token", "secret", "password"}


class ProviderAdapterError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ProviderRequest:
    system_prompt: str
    user_prompt: str
    tool_contract: tuple[dict[str, Any], ...] = ()
    max_output_tokens: int = 8192
    stochastic_request_id: str = ""
    requested_sampling_seed: int | None = None


@dataclasses.dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    raw_categories: Mapping[str, int] = dataclasses.field(default_factory=dict)

    def as_event_payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ProviderResult:
    text: str
    response_id: str
    resolved_model_revision: str
    usage: ProviderUsage
    sampling_seed_supported: bool
    stochastic_request_id: str


class ProviderAdapter(Protocol):
    provider: str
    sampling_seed_supported: bool

    def generate(self, request: ProviderRequest) -> ProviderResult: ...

    def audit_metadata(self) -> dict[str, Any]: ...


def _get(obj: Any, *path: str, default: Any = 0) -> Any:
    current = obj
    for key in path:
        if current is None:
            return default
        current = current.get(key, default) if isinstance(current, dict) else getattr(current, key, default)
    return current


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and value >= 0 else 0


def _numeric_categories(value: Any, prefix: str = "") -> dict[str, int]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif not isinstance(value, dict) and hasattr(value, "__dict__"):
        value = vars(value)
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if str(key).lower() in SENSITIVE_KEYS:
            continue
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)) and item >= 0:
            result[name] = int(item)
        elif isinstance(item, dict) or hasattr(item, "__dict__") or hasattr(item, "model_dump"):
            result.update(_numeric_categories(item, name))
    return result


def _require_resolution(resolution: ModelResolution, provider: str) -> None:
    if resolution.provider != provider or not resolution.provider_attested_immutable:
        raise ProviderAdapterError(f"adapter requires an attested immutable {provider} resolution")


class OpenAIResponsesAdapter:
    provider = "openai"
    # The Responses API does not guarantee sampling-seed control for these models.
    sampling_seed_supported = False

    def __init__(self, client: Any, resolution: ModelResolution):
        _require_resolution(resolution, self.provider)
        self._client = client
        self._resolution = resolution

    @classmethod
    def from_env(cls, resolution: ModelResolution, env_var: str = "OPENAI_API_KEY") -> "OpenAIResponsesAdapter":
        if not os.environ.get(env_var):
            raise ProviderAdapterError(f"required credential environment variable is absent: {env_var}")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderAdapterError("the openai SDK is not installed") from exc
        return cls(OpenAI(api_key=os.environ[env_var]), resolution)

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "immutable_model": self._resolution.audit_metadata(),
            "sampling_seed_supported": self.sampling_seed_supported,
            "credential_source": "environment",
        }

    def generate(self, request: ProviderRequest) -> ProviderResult:
        kwargs: dict[str, Any] = {
            "model": self._resolution.immutable_revision,
            "instructions": request.system_prompt,
            "input": request.user_prompt,
            "max_output_tokens": request.max_output_tokens,
        }
        if request.tool_contract:
            kwargs["tools"] = list(request.tool_contract)
        response = self._client.responses.create(**kwargs)
        usage = _get(response, "usage", default={})
        input_tokens = _integer(_get(usage, "input_tokens"))
        output_tokens = _integer(_get(usage, "output_tokens"))
        normalized = ProviderUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=_integer(_get(usage, "output_tokens_details", "reasoning_tokens")),
            cache_read_tokens=_integer(_get(usage, "input_tokens_details", "cached_tokens")),
            cache_write_tokens=0,
            total_tokens=_integer(_get(usage, "total_tokens", default=input_tokens + output_tokens)),
            raw_categories=_numeric_categories(usage),
        )
        return ProviderResult(
            text=str(_get(response, "output_text", default="")),
            response_id=str(_get(response, "id", default="")),
            resolved_model_revision=self._resolution.immutable_revision,
            usage=normalized,
            sampling_seed_supported=self.sampling_seed_supported,
            stochastic_request_id=request.stochastic_request_id,
        )


class AnthropicMessagesAdapter:
    provider = "anthropic"
    sampling_seed_supported = False

    def __init__(self, client: Any, resolution: ModelResolution):
        _require_resolution(resolution, self.provider)
        self._client = client
        self._resolution = resolution

    @classmethod
    def from_env(cls, resolution: ModelResolution, env_var: str = "ANTHROPIC_API_KEY") -> "AnthropicMessagesAdapter":
        if not os.environ.get(env_var):
            raise ProviderAdapterError(f"required credential environment variable is absent: {env_var}")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ProviderAdapterError("the anthropic SDK is not installed") from exc
        return cls(Anthropic(api_key=os.environ[env_var]), resolution)

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "immutable_model": self._resolution.audit_metadata(),
            "sampling_seed_supported": self.sampling_seed_supported,
            "credential_source": "environment",
        }

    def generate(self, request: ProviderRequest) -> ProviderResult:
        kwargs: dict[str, Any] = {
            "model": self._resolution.immutable_revision,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.user_prompt}],
            "max_tokens": request.max_output_tokens,
        }
        if request.tool_contract:
            kwargs["tools"] = list(request.tool_contract)
        response = self._client.messages.create(**kwargs)
        usage = _get(response, "usage", default={})
        input_tokens = _integer(_get(usage, "input_tokens"))
        output_tokens = _integer(_get(usage, "output_tokens"))
        content = _get(response, "content", default=[])
        pieces = []
        for block in content if isinstance(content, list) else []:
            if _get(block, "type", default="") == "text":
                pieces.append(str(_get(block, "text", default="")))
        normalized = ProviderUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=_integer(_get(usage, "thinking_tokens")),
            cache_read_tokens=_integer(_get(usage, "cache_read_input_tokens")),
            cache_write_tokens=_integer(_get(usage, "cache_creation_input_tokens")),
            total_tokens=input_tokens + output_tokens,
            raw_categories=_numeric_categories(usage),
        )
        return ProviderResult(
            text="".join(pieces),
            response_id=str(_get(response, "id", default="")),
            resolved_model_revision=self._resolution.immutable_revision,
            usage=normalized,
            sampling_seed_supported=self.sampling_seed_supported,
            stochastic_request_id=request.stochastic_request_id,
        )
