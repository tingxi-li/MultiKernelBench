#!/usr/bin/env python3
"""Reusable immutable-model resolution lock validation.

The effort-frontier controller may import ``load_model_resolution_lock`` and
select ``resolutions["openai:gpt-5.6-sol"]`` without depending on convergence
manifests or provider SDKs.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

try:
    from .campaign import MODELS, model_key
except ImportError:  # direct script execution
    from campaign import MODELS, model_key  # type: ignore


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelResolutionError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class ModelResolution:
    provider: str
    requested_alias: str
    immutable_revision: str
    provider_attested_immutable: bool
    resolved_at_utc: str
    resolution_evidence_sha256: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.requested_alias}"

    def audit_metadata(self) -> dict[str, object]:
        """Return safe metadata; provider credentials are never represented."""
        return dataclasses.asdict(self)


def _expected_keys(expected_models: Iterable[Mapping[str, str]]) -> set[str]:
    return {model_key(dict(model)) for model in expected_models}


def load_model_resolution_lock(
    path: Path,
    *,
    expected_models: Iterable[Mapping[str, str]] = MODELS,
    require_resolved: bool = True,
) -> dict[str, ModelResolution]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelResolutionError(f"cannot read model resolution lock: {exc}") from exc
    expected = _expected_keys(expected_models)
    rows = raw.get("resolutions")
    if raw.get("schema_version") != 1 or not isinstance(rows, list):
        raise ModelResolutionError("model lock must have schema_version=1 and resolutions[]")
    if raw.get("state") != "resolved":
        if require_resolved:
            raise ModelResolutionError("model resolution lock state is not resolved")
        return {}
    resolved: dict[str, ModelResolution] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ModelResolutionError("resolution rows must be objects")
        provider = row.get("provider")
        alias = row.get("requested_alias")
        key = f"{provider}:{alias}"
        immutable = row.get("immutable_revision")
        timestamp = row.get("resolved_at_utc")
        evidence = row.get("resolution_evidence_sha256")
        if key in resolved:
            raise ModelResolutionError(f"duplicate model resolution: {key}")
        if not isinstance(immutable, str) or not immutable or immutable == alias:
            raise ModelResolutionError(f"{key}: immutable_revision must be nonempty and differ from alias")
        if row.get("provider_attested_immutable") is not True:
            raise ModelResolutionError(f"{key}: provider_attested_immutable must be true")
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise ModelResolutionError(f"{key}: resolved_at_utc must be an explicit UTC timestamp")
        if not isinstance(evidence, str) or SHA256_RE.fullmatch(evidence) is None:
            raise ModelResolutionError(f"{key}: invalid resolution evidence SHA-256")
        resolved[key] = ModelResolution(
            provider=str(provider),
            requested_alias=str(alias),
            immutable_revision=immutable,
            provider_attested_immutable=True,
            resolved_at_utc=timestamp,
            resolution_evidence_sha256=evidence,
        )
    if set(resolved) != expected:
        raise ModelResolutionError(
            f"resolution keys differ: expected={sorted(expected)}, actual={sorted(resolved)}"
        )
    return resolved

