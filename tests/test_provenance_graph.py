"""Deterministic unit tests for the NetworkX-backed ProvenanceGraph."""

import pytest

from src.monitor.models import (
    DenialEvent,
    EdgeType,
    NodeType,
    ProvenanceEdge,
    ProvenanceNode,
    TrustLevel,
)
from src.monitor.provenance import ProvenanceGraph, ProvenanceGraphError


def test_graph_creation_and_direct_dependency_traversal() -> None:
    """Verify node/edge registration and direct dependency lookup."""
    graph = ProvenanceGraph()
    graph.record_call_node("c1", "read_notes")
    graph.record_tool_output(
        call_id="c1",
        data_id="d1",
        value="meeting notes",
        default_trust=TrustLevel.TOOL_TRUSTED,
    )
    graph.record_call_node("c2", "summarize", input_node_ids=["d1"])

    assert graph.node_count == 3
    assert graph.edge_count == 2
    assert graph.get_ancestors("c2") == ["c1", "d1"]
    assert graph.min_trust("c2") == TrustLevel.TOOL_TRUSTED
    assert graph.validate_graph() == []


def test_transitive_dependency_traversal_and_monotonic_taint() -> None:
    """Verify untrusted data propagates transitively across multiple calls."""
    graph = ProvenanceGraph()
    graph.record_call_node("c1", "read_emails")
    graph.record_tool_output(
        call_id="c1",
        data_id="d1",
        value={"body": "untrusted"},
        default_trust=TrustLevel.TOOL_UNTRUSTED,
    )
    graph.record_call_node("c2", "format_response", input_node_ids=["d1"])
    graph.record_tool_output(
        call_id="c2",
        data_id="d2",
        value="formatted",
        default_trust=TrustLevel.TOOL_TRUSTED,
    )
    graph.record_call_node("c3", "send_email", input_node_ids=["d2"])

    assert graph.get_ancestors("c3") == ["c1", "c2", "d1", "d1.body", "d2"]
    assert graph.min_trust("c2") == TrustLevel.TOOL_UNTRUSTED
    assert graph.min_trust("d2") == TrustLevel.TOOL_UNTRUSTED
    assert graph.min_trust("c3") == TrustLevel.TOOL_UNTRUSTED


def test_field_level_provenance_isolation() -> None:
    """Verify trusted and untrusted fields on the same object are isolated."""
    graph = ProvenanceGraph()
    graph.record_call_node("c1", "get_contact")
    graph.record_tool_output(
        call_id="c1",
        data_id="d1",
        value={"name": "Alice", "email": "attacker@evil.test"},
        default_trust=TrustLevel.TOOL_TRUSTED,
        field_trust_overrides={"email": TrustLevel.TOOL_UNTRUSTED},
    )

    graph.record_call_node("c_name", "greet", input_node_ids=["d1.name"])
    graph.record_call_node("c_email", "send_email", input_node_ids=["d1.email"])
    graph.record_call_node("c_whole", "export", input_node_ids=["d1"])

    assert graph.min_trust("c_name") == TrustLevel.TOOL_TRUSTED
    assert graph.min_trust("c_email") == TrustLevel.TOOL_UNTRUSTED
    assert graph.min_trust("c_whole") == TrustLevel.TOOL_UNTRUSTED


def test_denial_recording_and_counterfactual_edges() -> None:
    """Verify DenialEvent recording, DeniedAction node, and counterfactual chains."""
    graph = ProvenanceGraph()
    denial = DenialEvent(
        denial_id="den_1",
        source_event_id="c1",
        denied_action="read_file",
        denial_reason="HB-2: sensitive path",
        feedback_observed="DENIED: sensitive path",
    )
    node = graph.record_denial(denial)
    assert node.node_type == NodeType.DENIED_ACTION
    assert graph.denials == [denial]

    graph.record_call_node("c2", "send_email")
    graph.add_counterfactual_edge("den_1", "c2")

    assert graph.counterfactual_chains("c2") == [["den_1", "c2"]]


def test_missing_provenance_and_invalid_graph_references() -> None:
    """Verify unknown nodes, duplicates, dangling edges, and invalid edge types fail."""
    graph = ProvenanceGraph()
    graph.record_call_node("c1", "tool_a")

    # Call with no data ancestors defaults to SysInstr.
    assert graph.min_trust("c1") == TrustLevel.SYS_INSTR

    # Unknown node queries raise ProvenanceGraphError.
    with pytest.raises(ProvenanceGraphError, match="Unknown provenance node"):
        graph.min_trust("missing_node")

    with pytest.raises(ProvenanceGraphError, match="Unknown provenance node"):
        graph.get_ancestors("missing_node")

    # Duplicate node registration rejected.
    with pytest.raises(ProvenanceGraphError, match="Duplicate provenance node ID"):
        graph.register_node(
            ProvenanceNode(
                node_id="c1",
                node_type=NodeType.CALL,
                trust_level=TrustLevel.SYS_INSTR,
                source_id="tool_a",
            )
        )

    # Dangling edge rejected.
    with pytest.raises(ProvenanceGraphError, match="Dangling edge"):
        graph.register_edge(
            ProvenanceEdge(
                source_id="nonexistent",
                target_id="c1",
                relationship_type=EdgeType.INPUT_TO,
            )
        )

    # Invalid endpoint node types for edge relationship rejected.
    graph.record_call_node("c2", "tool_b")
    with pytest.raises(ProvenanceGraphError, match="Invalid endpoint types"):
        graph.register_edge(
            ProvenanceEdge(
                source_id="c1",
                target_id="c2",
                relationship_type=EdgeType.DIRECT_OUTPUT,
            )
        )
