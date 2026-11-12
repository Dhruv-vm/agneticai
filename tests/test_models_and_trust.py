"""Deterministic unit tests for Pydantic v2 models and the integrity lattice."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.monitor.models import (
    AgentEvent,
    DecisionVerdict,
    DenialEvent,
    EdgeType,
    EventStatus,
    EventType,
    NodeType,
    PolicyDecision,
    ProvenanceEdge,
    ProvenanceNode,
    TrustLevel,
    conservative_join,
)


def test_trust_lattice_total_ordering() -> None:
    """Verify ToolDesc < ToolUntrusted < ToolTrusted < UserInput < SysInstr."""
    ordered = [
        TrustLevel.TOOL_DESC,
        TrustLevel.TOOL_UNTRUSTED,
        TrustLevel.TOOL_TRUSTED,
        TrustLevel.USER_INPUT,
        TrustLevel.SYS_INSTR,
    ]
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            assert ordered[i] < ordered[j]
            assert ordered[i] <= ordered[j]
            assert ordered[j] > ordered[i]
            assert ordered[j] >= ordered[i]

    assert [t.label for t in ordered] == [
        "ToolDesc",
        "ToolUntrusted",
        "ToolTrusted",
        "UserInput",
        "SysInstr",
    ]


def test_conservative_join_propagation() -> None:
    """Verify conservative_join returns the minimum trust level (or SysInstr if empty)."""
    assert conservative_join([]) == TrustLevel.SYS_INSTR
    assert conservative_join([TrustLevel.SYS_INSTR]) == TrustLevel.SYS_INSTR
    assert (
        conservative_join([TrustLevel.SYS_INSTR, TrustLevel.USER_INPUT])
        == TrustLevel.USER_INPUT
    )
    assert (
        conservative_join([TrustLevel.TOOL_TRUSTED, TrustLevel.TOOL_UNTRUSTED])
        == TrustLevel.TOOL_UNTRUSTED
    )
    assert (
        conservative_join(
            [TrustLevel.SYS_INSTR, TrustLevel.TOOL_TRUSTED, TrustLevel.TOOL_DESC]
        )
        == TrustLevel.TOOL_DESC
    )


def test_agent_event_validation_and_serialization() -> None:
    """Verify AgentEvent validates required fields and round-trips JSON."""
    ts = datetime(2026, 11, 12, 0, 0, 0, tzinfo=UTC)
    event = AgentEvent(
        event_id="evt_1",
        session_id="sess_1",
        timestamp=ts,
        event_type=EventType.TOOL_CALL,
        tool_name="read_file",
        arguments={"path": "/tmp/notes.txt"},
        status=EventStatus.PENDING,
        parent_event_ids=["evt_0"],
    )
    dumped = event.model_dump_json()
    restored = AgentEvent.model_validate_json(dumped)
    assert restored == event

    with pytest.raises(ValidationError):
        AgentEvent(
            event_id="   ",
            session_id="sess_1",
            event_type=EventType.TOOL_CALL,
            tool_name="read_file",
            status=EventStatus.PENDING,
        )


def test_denial_event_and_policy_decision_models() -> None:
    """Verify DenialEvent and PolicyDecision validation and serialization."""
    denial = DenialEvent(
        denial_id="den_1",
        source_event_id="evt_1",
        denied_action="read_file",
        denial_reason="HB-2: sensitive path",
        feedback_observed="DENIED: sensitive path",
    )
    assert DenialEvent.model_validate_json(denial.model_dump_json()) == denial

    decision = PolicyDecision(
        decision_id="dec_1",
        event_id="evt_1",
        decision=DecisionVerdict.DENY,
        reason="HB-2: sensitive path",
        evidence={"rule": "HB-2"},
    )
    assert PolicyDecision.model_validate_json(decision.model_dump_json()) == decision


def test_provenance_node_and_edge_validation() -> None:
    """Verify ProvenanceNode and ProvenanceEdge reject invalid inputs and self-loops."""
    node = ProvenanceNode(
        node_id="n1",
        node_type=NodeType.DATA,
        trust_level=TrustLevel.TOOL_TRUSTED,
        source_id="c1",
    )
    assert node.trust_level == TrustLevel.TOOL_TRUSTED

    with pytest.raises(ValidationError):
        ProvenanceEdge(
            source_id="n1",
            target_id="n1",
            relationship_type=EdgeType.INPUT_TO,
        )
