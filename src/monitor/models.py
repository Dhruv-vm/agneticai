"""Validated Pydantic v2 data models and integrity-lattice trust types."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum, IntEnum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class TrustLevel(IntEnum):
    """Five-level integrity lattice from ARM (Chinaei, 2026).

    Ordering: ToolDesc < ToolUntrusted < ToolTrusted < UserInput < SysInstr.
    Integer values enforce lattice ordering without string comparisons.
    """

    TOOL_DESC = 0
    TOOL_UNTRUSTED = 1
    TOOL_TRUSTED = 2
    USER_INPUT = 3
    SYS_INSTR = 4

    @property
    def label(self) -> str:
        """Return the canonical human-readable lattice label."""
        labels = {
            TrustLevel.TOOL_DESC: "ToolDesc",
            TrustLevel.TOOL_UNTRUSTED: "ToolUntrusted",
            TrustLevel.TOOL_TRUSTED: "ToolTrusted",
            TrustLevel.USER_INPUT: "UserInput",
            TrustLevel.SYS_INSTR: "SysInstr",
        }
        return labels[self]


def conservative_join(levels: Iterable[TrustLevel]) -> TrustLevel:
    """Return the minimum (least-trusted) level across contributing sources.

    Returns ``TrustLevel.SYS_INSTR`` when ``levels`` is empty, matching
    ARM's Definition 4 default when no data ancestors exist.
    """
    collected = list(levels)
    if not collected:
        return TrustLevel.SYS_INSTR
    return min(collected)


class EventType(str, Enum):
    """Types of agent session events."""

    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    DENIAL = "DENIAL"
    USER_MESSAGE = "USER_MESSAGE"
    SYSTEM_INSTRUCTION = "SYSTEM_INSTRUCTION"


class EventStatus(str, Enum):
    """Execution status of an agent event."""

    PENDING = "PENDING"
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class NodeType(str, Enum):
    """Provenance graph node categories."""

    CALL = "Call"
    DATA = "Data"
    DATA_FIELD = "DataField"
    DENIED_ACTION = "DeniedAction"


class EdgeType(str, Enum):
    """Provenance graph directed edge relationships."""

    DIRECT_OUTPUT = "DIRECT_OUTPUT"
    INPUT_TO = "INPUT_TO"
    FIELD_OF = "FIELD_OF"
    COUNTERFACTUAL = "COUNTERFACTUAL"


class DecisionVerdict(str, Enum):
    """Final enforcement verdict for a tool invocation."""

    ALLOW = "ALLOW"
    DENY = "DENY"


class LayerVerdict(str, Enum):
    """Per-layer policy evaluation verdict."""

    PASS = "PASS"
    DENY = "DENY"


class AgentEvent(BaseModel):
    """Structured record of an agent or tool-execution event."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=utc_now)
    event_type: EventType
    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    status: EventStatus
    parent_event_ids: list[str] = Field(default_factory=list)

    @field_validator("event_id", "session_id", "tool_name")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Identifier/name fields must not be blank.")
        return value


class DenialEvent(BaseModel):
    """Structured record of a denied tool invocation and observed feedback."""

    model_config = ConfigDict(extra="forbid")

    denial_id: str = Field(..., min_length=1)
    source_event_id: str = Field(..., min_length=1)
    denied_action: str = Field(..., min_length=1)
    denial_reason: str = Field(..., min_length=1)
    feedback_observed: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=utc_now)

    @field_validator("denial_id", "source_event_id", "denied_action", "denial_reason")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("DenialEvent fields must not be blank.")
        return value


class ProvenanceNode(BaseModel):
    """Node in the runtime provenance graph."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(..., min_length=1)
    node_type: NodeType
    trust_level: TrustLevel
    source_id: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("node_id", "source_id")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Node identifiers must not be blank.")
        return value


class ProvenanceEdge(BaseModel):
    """Directed edge in the runtime provenance graph."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    relationship_type: EdgeType
    timestamp: datetime = Field(default_factory=utc_now)

    @field_validator("source_id", "target_id")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Edge endpoint IDs must not be blank.")
        return value

    @model_validator(mode="after")
    def _disallow_self_loop(self) -> ProvenanceEdge:
        if self.source_id == self.target_id:
            raise ValueError(f"Self-loop edges are not permitted: {self.source_id}")
        return self


class PolicyDecision(BaseModel):
    """Deterministic enforcement decision emitted by the monitor."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(..., min_length=1)
    event_id: str = Field(..., min_length=1)
    decision: DecisionVerdict
    reason: str = Field(..., min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)
