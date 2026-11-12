"""Canonical five-tier source identity resolution (Phase 5 corrected model).

Explicitly distinguishes:
1. Physical execution identity (``physical_id``, e.g., ``d_branch_a_s0``).
2. Logical source identity (``source_id`` when provided by environment, else unresolved/tool+args).
3. Content/version identity (``content_digest`` SHA-256 of returned data).
4. Trust provenance (``trust_level_rank``, ``is_independent_trusted``).
5. Transformation lineage (``lineage`` tuple of upstream producer tools/sources).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.monitor.arm_baseline import ARMMonitor


@dataclass(frozen=True)
class LogicalSource:
    """Five-tier canonical identity of a data node in an execution graph."""

    physical_id: str
    logical_id: str
    producer_tool: str | None
    producer_args: dict[str, Any]
    trust_level_rank: int
    is_independent_trusted: bool
    source_id: str | None = None
    content_digest: str | None = None
    lineage: tuple[str, ...] = ()
    identity_certainty: str = "VERIFIED"


@dataclass(frozen=True)
class CanonicalComparisonResult:
    """Structured comparison between two LogicalSource records."""

    status: str  # EQUIVALENT | DIFFERENT | CONTENT_CHANGED | UNRESOLVED
    same_logical_source: bool | None
    same_content: bool | None
    same_trust: bool
    same_lineage: bool
    reason: str


def _canonical_args(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))


def _compute_content_digest(value: Any) -> str | None:
    if value is None:
        return None
    serialized = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def resolve_logical_source(
    node_id: str,
    monitor: ARMMonitor,
    strict_content_trust: bool = True,
) -> LogicalSource:
    """Resolve a physical Data/DataField ID to its five-tier LogicalSource."""
    if not monitor.graph.has_node(node_id):
        return LogicalSource(
            physical_id=node_id,
            logical_id=f"unresolved::missing::{node_id}",
            producer_tool=None,
            producer_args={},
            trust_level_rank=0,
            is_independent_trusted=False,
            source_id=None,
            content_digest=None,
            lineage=(),
            identity_certainty="UNRESOLVED",
        )

    node = monitor.graph.get_node(node_id)
    data_node = (
        monitor.graph.get_node(node.source_id)
        if node.node_type.value == "DataField" and monitor.graph.has_node(node.source_id)
        else node
    )
    field_suffix = (
        f".{node.metadata.get('field')}" if node.node_type.value == "DataField" else ""
    )

    explicit_source_id = data_node.metadata.get("source_id")
    raw_value = node.metadata.get("value") if "value" in node.metadata else data_node.metadata.get("value")
    content_digest = _compute_content_digest(raw_value)

    call_id = data_node.source_id
    if monitor.graph.has_node(call_id):
        call_node = monitor.graph.get_node(call_id)
        tool_name = call_node.source_id
        args = dict(call_node.metadata.get("arguments", {}))
        base_ref = (
            f"src::{explicit_source_id}{field_suffix}"
            if explicit_source_id
            else f"logical::{tool_name}({_canonical_args(args)}){field_suffix}"
        )
    else:
        tool_name = None
        args = {}
        base_ref = (
            f"src::{explicit_source_id}{field_suffix}"
            if explicit_source_id
            else f"logical::{data_node.node_id}{field_suffix}"
        )

    min_trust = monitor.graph.min_trust(node_id)
    has_cf = (
        bool(monitor.graph.counterfactual_chains(call_id))
        if monitor.graph.has_node(call_id)
        else False
    )

    # Build transformation lineage from ancestor Call nodes in deterministic order.
    ancestors = monitor.graph.get_ancestors(node_id)
    lineage = tuple(
        monitor.graph.get_node(aid).source_id
        for aid in ancestors
        if monitor.graph.get_node(aid).node_type.value == "Call"
    )

    if strict_content_trust:
        logical_id = f"{base_ref}|digest={content_digest}|trust={int(min_trust)}"
    else:
        logical_id = base_ref

    certainty = "VERIFIED" if explicit_source_id is not None else "ARGUMENT_INFERRED"
    return LogicalSource(
        physical_id=node_id,
        logical_id=logical_id,
        producer_tool=tool_name,
        producer_args=args,
        trust_level_rank=int(min_trust),
        is_independent_trusted=(int(min_trust) >= 2 and not has_cf),
        source_id=explicit_source_id,
        content_digest=content_digest,
        lineage=lineage,
        identity_certainty=certainty,
    )


def compare_logical_sources(
    src_a: LogicalSource,
    src_b: LogicalSource,
) -> CanonicalComparisonResult:
    """Compare two LogicalSource records without collapsing distinct sources."""
    same_trust = src_a.trust_level_rank == src_b.trust_level_rank
    same_lineage = src_a.lineage == src_b.lineage
    same_content: bool | None = (
        None
        if (src_a.content_digest is None or src_b.content_digest is None)
        else (src_a.content_digest == src_b.content_digest)
    )

    if src_a.identity_certainty == "UNRESOLVED" or src_b.identity_certainty == "UNRESOLVED":
        return CanonicalComparisonResult(
            status="UNRESOLVED",
            same_logical_source=None,
            same_content=same_content,
            same_trust=same_trust,
            same_lineage=same_lineage,
            reason="One or both sources lack registered provenance nodes.",
        )

    # If explicit source_ids are available on both, they govern logical source identity.
    if src_a.source_id is not None and src_b.source_id is not None:
        if src_a.source_id != src_b.source_id:
            return CanonicalComparisonResult(
                status="DIFFERENT",
                same_logical_source=False,
                same_content=same_content,
                same_trust=same_trust,
                same_lineage=same_lineage,
                reason="Distinct explicit source_ids.",
            )
        if same_content is False or not same_trust:
            return CanonicalComparisonResult(
                status="CONTENT_CHANGED",
                same_logical_source=True,
                same_content=same_content,
                same_trust=same_trust,
                same_lineage=same_lineage,
                reason="Same source_id, but content version or trust provenance changed.",
            )
        return CanonicalComparisonResult(
            status="EQUIVALENT",
            same_logical_source=True,
            same_content=same_content,
            same_trust=same_trust,
            same_lineage=same_lineage,
            reason="Matching explicit source_id, content digest, and trust.",
        )

    # Without explicit source_id, tool+args alone cannot prove equivalence if content/trust differ.
    if not same_trust or same_content is False:
        return CanonicalComparisonResult(
            status="DIFFERENT",
            same_logical_source=False,
            same_content=same_content,
            same_trust=same_trust,
            same_lineage=same_lineage,
            reason="Content digest or trust provenance differs.",
        )

    if src_a.producer_tool == src_b.producer_tool and src_a.producer_args == src_b.producer_args:
        # Missing explicit source_id: preserve source-identity uncertainty rather than inventing equivalence.
        return CanonicalComparisonResult(
            status="UNRESOLVED",
            same_logical_source=None,
            same_content=same_content,
            same_trust=same_trust,
            same_lineage=same_lineage,
            reason=(
                "Missing explicit source_id; matching tool, arguments, content digest, "
                "and trust cannot prove identical underlying source."
            ),
        )

    return CanonicalComparisonResult(
        status="DIFFERENT",
        same_logical_source=False,
        same_content=same_content,
        same_trust=same_trust,
        same_lineage=same_lineage,
        reason="Distinct tool/arguments without shared explicit source_id.",
    )


def resolve_step_logical_sources(
    input_node_ids: list[str],
    monitor: ARMMonitor,
) -> list[LogicalSource]:
    """Resolve all input node IDs of a step to their canonical LogicalSources."""
    return [resolve_logical_source(nid, monitor) for nid in input_node_ids]
