#!/usr/bin/env python3
"""Bound wrappers around the completed closure-v2 and reachability-v2 candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from . import core


@dataclass
class BuiltCandidate:
    candidate_id: str
    run: Callable[[Any, Any, Any, Any], Any]
    build_metadata: dict[str, Any]


def build_candidate(
    definition: dict[str, Any], *, expected_candidate_id: str | None = None
) -> BuiltCandidate:
    candidate_id = definition["candidate_id"]
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise core.ClosureError("candidate ID differs from requested binding")
    started = time.perf_counter()
    if definition["implementation"] == "closure_v2":
        from ako_runs.controlled_followup.fused_closure_v2 import (
            candidates as closure_candidates,
            core as closure_core,
        )

        closure_campaign = closure_core.load_campaign()
        source_id = definition["source_candidate_id"]
        source_definition = closure_core.candidates_by_id(closure_campaign).get(source_id)
        if source_definition is None:
            raise core.ClosureError(f"missing closure-v2 candidate {source_id}")
        built = closure_candidates.build_candidate(
            source_definition, expected_candidate_id=source_id
        )
        return BuiltCandidate(
            candidate_id=candidate_id,
            run=built.run,
            build_metadata={
                "implementation_source": "fused_closure_v2",
                "source_candidate_id": source_id,
                "source_candidate_sha256": closure_core.candidate_sha256(
                    source_definition
                ),
                "source_build_metadata": built.build_metadata,
                "wrapper_build_wall_s": time.perf_counter() - started,
            },
        )
    if definition["implementation"] == "reachability_v2":
        from ako_runs.controlled_followup.fused_reachability_v2 import (
            candidate as reach_candidate,
            protocol as reach_protocol,
        )

        lock = reach_protocol.verify_lock()
        jobs = {row["job_id"]: row for row in reach_protocol.read_json(reach_protocol.JOBS)}
        source_id = definition["source_job_id"]
        job = jobs.get(source_id)
        if job is None:
            raise core.ClosureError(f"missing reachability-v2 job {source_id}")
        if reach_protocol.canonical_sha256(job) != lock["job_sha256"].get(source_id):
            raise core.ClosureError(f"reachability-v2 job hash differs: {source_id}")
        cfg = reach_candidate.make_config(job)
        built = reach_candidate.build(job["lane"], cfg)

        def run(_x_fp32, x_fp16, weight, bias):
            return built.run(x_fp16, weight, bias)

        return BuiltCandidate(
            candidate_id=candidate_id,
            run=run,
            build_metadata={
                "implementation_source": "fused_reachability_v2",
                "source_job_id": source_id,
                "source_job": job,
                "source_job_sha256": lock["job_sha256"][source_id],
                "source_build_compile_s": built.compile_s,
                "source_build_artifacts": built.artifacts,
                "wrapper_build_wall_s": time.perf_counter() - started,
            },
        )
    raise core.ClosureError(
        f"unknown implementation source {definition['implementation']!r}"
    )
