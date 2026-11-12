"""Canonical logical identity resolution for cross-branch provenance comparison.

Separates physical branch-local execution IDs (e.g., ``d_branch_a_s1``) from
logical data identities (e.g., ``read_document({"path":"/tmp/notice.txt"})``)
without merging genuinely distinct information sources.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.monitor.arm_baseline import ARMMonitor


@dataclass(frozen=True)
class LogicalSource:
    """Canonical logical identity of a data node in an execution graph."""

    physical_id: str
    logical_id: str
    producer_tool: str | None
    producer_args: dict[str, Any]
    trust_level_rank: int
    is_independent_trusted: bool


def _canonical_args(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))


def resolve_logical_source(
    node_id: str,
    monitor: ARMMonitor,
) -> LogicalSource:
    """Resolve a physical Data/DataField ID to its canonical LogicalSource."""
    if not monitor.graph.has_node(node_id):
        return LogicalSource(
            physical_id=node_id,
            logical_id=f"unregistered::{node_id}",
            producer_tool=None,
            producer_args={},
            trust_level_rank=0,
            is_independent_trusted=False,
        )

    node = monitor.graph.get_node(node_id)
    # DataField -> parent Data -> producer Call
    data_node = (
        monitor.graph.get_node(node.source_id)
        if node.node_type.value == "DataField" and monitor.graph.has_node(node.source_id)
        else node
    )
    field_suffix = f".{node.metadata.get('field')}" if node.node_type.value == "DataField" else ""

    call_id = data_node.source_id
    if monitor.graph.has_node(call_id):
        call_node = monitor.graph.get_node(call_id)
        tool_name = call_node.source_id
        args = dict(call_node.metadata.get("arguments", {}))
        logical_id = f"logical::{tool_name}({_canonical_args(args)}){field_suffix}"
    else:
        tool_name = None
        args = {}
        logical_id = f"logical::{data_node.node_id}{field_suffix}"

    min_trust = monitor.graph.min_trust(node_id)
    # Independent trusted if min_trust >= TOOL_TRUSTED (rank 2) and no counterfactual chain
    has_cf = (
        bool(monitor.graph.counterfactual_chains(call_id))
        if monitor.graph.has_node(call_id)
        else False
    )
    return LogicalSource(
        physical_id=node_id,
        logical_id=logical_id,
        producer_tool=tool_name,
        producer_args=args,
        trust_level_rank=int(min_trust),
        is_independent_trusted=(int(min_trust) >= 2 and not has_cf),
    )


def resolve_step_logical_sources(
    input_node_ids: list[str],
    monitor: ARMMonitor,
) -> list[LogicalSource]:
    """Resolve all input node IDs of a step to their canonical LogicalSources."""
    return [resolve_logical_source(nid, monitor) for nid in input_node_ids]
