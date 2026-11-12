"""NetworkX-backed provenance graph for ARM-inspired runtime enforcement.

Edge direction conventions (following Chinaei, 2026, Section 5.2):
- DIRECT_OUTPUT(call, data): Call -> Data (a call produced a data item).
- INPUT_TO(data_or_field, call): Data|DataField -> Call (data used as call input).
- FIELD_OF(field, data): DataField -> Data (field is a component of structured data).
  Directing FIELD_OF from field to parent data ensures that:
    1. A call consuming the whole parent Data node inherits the minimum trust across
       all constituent fields (via ancestors of Data).
    2. A call consuming a single DataField node inherits only that field's trust,
       without being tainted by sibling fields.
- COUNTERFACTUAL(denied_action, call): DeniedAction -> Call (potential causal
  influence of a denial event on a subsequent tool invocation).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import networkx as nx

from src.monitor.models import (
    DenialEvent,
    EdgeType,
    NodeType,
    ProvenanceEdge,
    ProvenanceNode,
    TrustLevel,
    conservative_join,
    utc_now,
)

_ALLOWED_EDGE_ENDPOINTS: dict[EdgeType, tuple[set[NodeType], set[NodeType]]] = {
    EdgeType.DIRECT_OUTPUT: ({NodeType.CALL}, {NodeType.DATA}),
    EdgeType.INPUT_TO: ({NodeType.DATA, NodeType.DATA_FIELD}, {NodeType.CALL}),
    EdgeType.FIELD_OF: ({NodeType.DATA_FIELD}, {NodeType.DATA}),
    EdgeType.COUNTERFACTUAL: ({NodeType.DENIED_ACTION}, {NodeType.CALL}),
}


class ProvenanceGraphError(ValueError):
    """Raised when a provenance graph invariant or reference check fails."""


class ProvenanceGraph:
    """Deterministic directed provenance graph over tool calls, data, fields, and denials."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, ProvenanceNode] = {}
        self._edges: list[ProvenanceEdge] = []
        self._edge_keys: set[tuple[str, str, EdgeType]] = set()
        self._denials: dict[str, DenialEvent] = {}
        self._denial_order: list[str] = []

    @property
    def node_count(self) -> int:
        """Return total number of registered nodes."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Return total number of registered edges."""
        return len(self._edges)

    @property
    def denials(self) -> list[DenialEvent]:
        """Return recorded DenialEvents in insertion order."""
        return [self._denials[denial_id] for denial_id in self._denial_order]

    def has_node(self, node_id: str) -> bool:
        """Return whether a node ID is registered."""
        return node_id in self._nodes

    def get_node(self, node_id: str) -> ProvenanceNode:
        """Return the ProvenanceNode for ``node_id`` or raise ProvenanceGraphError."""
        if node_id not in self._nodes:
            raise ProvenanceGraphError(f"Unknown provenance node: '{node_id}'.")
        return self._nodes[node_id]

    def register_node(self, node: ProvenanceNode) -> ProvenanceNode:
        """Register a new node, rejecting duplicate node IDs."""
        if node.node_id in self._nodes:
            raise ProvenanceGraphError(f"Duplicate provenance node ID: '{node.node_id}'.")
        self._nodes[node.node_id] = node
        self._graph.add_node(
            node.node_id,
            node_type=node.node_type,
            trust_level=node.trust_level,
            source_id=node.source_id,
        )
        return node

    def register_edge(self, edge: ProvenanceEdge) -> ProvenanceEdge:
        """Register a directed edge after validating endpoints and semantics."""
        if edge.source_id not in self._nodes:
            raise ProvenanceGraphError(
                f"Dangling edge source '{edge.source_id}' is not registered."
            )
        if edge.target_id not in self._nodes:
            raise ProvenanceGraphError(
                f"Dangling edge target '{edge.target_id}' is not registered."
            )

        edge_key = (edge.source_id, edge.target_id, edge.relationship_type)
        if edge_key in self._edge_keys:
            raise ProvenanceGraphError(
                f"Duplicate edge ({edge.source_id} -[{edge.relationship_type.value}]-> {edge.target_id})."
            )

        src_type = self._nodes[edge.source_id].node_type
        dst_type = self._nodes[edge.target_id].node_type
        allowed_src, allowed_dst = _ALLOWED_EDGE_ENDPOINTS[edge.relationship_type]
        if src_type not in allowed_src or dst_type not in allowed_dst:
            raise ProvenanceGraphError(
                f"Invalid endpoint types for {edge.relationship_type.value}: "
                f"{src_type.value} -> {dst_type.value}."
            )

        self._edge_keys.add(edge_key)
        self._edges.append(edge)
        # Note: nx.DiGraph stores one edge per (u, v); our node-type rules make
        # (u, v) pairs unique across edge types except if extended in future.
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            relationship_type=edge.relationship_type,
            timestamp=edge.timestamp,
        )
        return edge

    def record_call_node(
        self,
        call_id: str,
        tool_name: str,
        input_node_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> ProvenanceNode:
        """Register a Call node and link any input Data/DataField nodes."""
        node = self.register_node(
            ProvenanceNode(
                node_id=call_id,
                node_type=NodeType.CALL,
                trust_level=TrustLevel.SYS_INSTR,
                source_id=tool_name,
                metadata=metadata or {},
            )
        )
        ts = timestamp or utc_now()
        for input_id in input_node_ids or []:
            self.register_edge(
                ProvenanceEdge(
                    source_id=input_id,
                    target_id=call_id,
                    relationship_type=EdgeType.INPUT_TO,
                    timestamp=ts,
                )
            )
        return node

    def record_tool_output(
        self,
        call_id: str,
        data_id: str,
        value: Any,
        default_trust: TrustLevel,
        field_trust_overrides: dict[str, TrustLevel] | None = None,
        timestamp: datetime | None = None,
    ) -> list[ProvenanceNode]:
        """Record a Data node (and optional DataField nodes for dict outputs).

        Returns the created Data node followed by any DataField nodes in sorted key order.
        """
        if call_id not in self._nodes:
            raise ProvenanceGraphError(f"Cannot attach output to unknown call '{call_id}'.")

        ts = timestamp or utc_now()
        overrides = field_trust_overrides or {}

        # Propagate call's own input taint into the produced data node conservatively.
        call_min_trust = self.min_trust(call_id)
        effective_data_trust = conservative_join([default_trust, call_min_trust])

        data_node = self.register_node(
            ProvenanceNode(
                node_id=data_id,
                node_type=NodeType.DATA,
                trust_level=effective_data_trust,
                source_id=call_id,
                metadata={"value": value},
            )
        )
        self.register_edge(
            ProvenanceEdge(
                source_id=call_id,
                target_id=data_id,
                relationship_type=EdgeType.DIRECT_OUTPUT,
                timestamp=ts,
            )
        )

        created: list[ProvenanceNode] = [data_node]
        if isinstance(value, dict):
            for field_name in sorted(value.keys()):
                field_value = value[field_name]
                field_trust = overrides.get(field_name, effective_data_trust)
                field_node_id = f"{data_id}.{field_name}"
                field_node = self.register_node(
                    ProvenanceNode(
                        node_id=field_node_id,
                        node_type=NodeType.DATA_FIELD,
                        trust_level=field_trust,
                        source_id=data_id,
                        metadata={"field": field_name, "value": field_value},
                    )
                )
                self.register_edge(
                    ProvenanceEdge(
                        source_id=field_node_id,
                        target_id=data_id,
                        relationship_type=EdgeType.FIELD_OF,
                        timestamp=ts,
                    )
                )
                created.append(field_node)

        return created

    def record_denial(self, denial: DenialEvent) -> ProvenanceNode:
        """Record a DenialEvent and create its DeniedAction node."""
        if denial.denial_id in self._denials:
            raise ProvenanceGraphError(f"Duplicate denial ID: '{denial.denial_id}'.")
        self._denials[denial.denial_id] = denial
        self._denial_order.append(denial.denial_id)

        return self.register_node(
            ProvenanceNode(
                node_id=denial.denial_id,
                node_type=NodeType.DENIED_ACTION,
                trust_level=TrustLevel.TOOL_UNTRUSTED,
                source_id=denial.source_event_id,
                metadata={
                    "denied_action": denial.denied_action,
                    "denial_reason": denial.denial_reason,
                    "feedback_observed": denial.feedback_observed,
                },
            )
        )

    def add_counterfactual_edge(
        self,
        denial_id: str,
        target_call_id: str,
        timestamp: datetime | None = None,
    ) -> ProvenanceEdge:
        """Create a COUNTERFACTUAL edge from a DeniedAction node to a Call node."""
        return self.register_edge(
            ProvenanceEdge(
                source_id=denial_id,
                target_id=target_call_id,
                relationship_type=EdgeType.COUNTERFACTUAL,
                timestamp=timestamp or utc_now(),
            )
        )

    def get_ancestors(self, node_id: str) -> list[str]:
        """Return deterministically sorted ancestor node IDs of ``node_id``."""
        if node_id not in self._nodes:
            raise ProvenanceGraphError(f"Unknown provenance node: '{node_id}'.")
        return sorted(nx.ancestors(self._graph, node_id))

    def get_transitive_dependencies(self, node_id: str) -> list[ProvenanceNode]:
        """Return all ancestor ProvenanceNodes in deterministic sorted ID order."""
        return [self._nodes[aid] for aid in self.get_ancestors(node_id)]

    def min_trust(self, node_id: str) -> TrustLevel:
        """Compute MinTrust(node_id) across reachable Data and DataField ancestors."""
        if node_id not in self._nodes:
            raise ProvenanceGraphError(f"Unknown provenance node: '{node_id}'.")

        candidate_ids = [node_id, *self.get_ancestors(node_id)]
        data_levels = [
            self._nodes[nid].trust_level
            for nid in candidate_ids
            if self._nodes[nid].node_type in {NodeType.DATA, NodeType.DATA_FIELD}
        ]
        return conservative_join(data_levels)

    def counterfactual_chains(self, node_id: str) -> list[list[str]]:
        """Return all simple paths from any DeniedAction node to ``node_id``
        that traverse at least one COUNTERFACTUAL edge.
        """
        if node_id not in self._nodes:
            raise ProvenanceGraphError(f"Unknown provenance node: '{node_id}'.")

        denial_ids = sorted(
            nid
            for nid, node in self._nodes.items()
            if node.node_type == NodeType.DENIED_ACTION
        )
        matching_paths: list[list[str]] = []
        for denial_id in denial_ids:
            if not nx.has_path(self._graph, denial_id, node_id):
                continue
            for path in nx.all_simple_paths(self._graph, denial_id, node_id):
                has_cf = any(
                    self._graph.edges[path[i], path[i + 1]]["relationship_type"]
                    == EdgeType.COUNTERFACTUAL
                    for i in range(len(path) - 1)
                )
                if has_cf:
                    matching_paths.append(path)
        return sorted(matching_paths)

    def validate_graph(self) -> list[str]:
        """Validate structural invariants of the graph. Returns a list of error strings."""
        errors: list[str] = []
        for edge in self._edges:
            if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
                errors.append(f"Dangling edge: {edge.source_id} -> {edge.target_id}")
                continue
            src_type = self._nodes[edge.source_id].node_type
            dst_type = self._nodes[edge.target_id].node_type
            allowed_src, allowed_dst = _ALLOWED_EDGE_ENDPOINTS[edge.relationship_type]
            if src_type not in allowed_src or dst_type not in allowed_dst:
                errors.append(
                    f"Invalid edge types {src_type.value} -> {dst_type.value} "
                    f"for {edge.relationship_type.value}"
                )
        if not nx.is_directed_acyclic_graph(self._graph):
            errors.append("Provenance graph contains a cycle.")
        return errors
