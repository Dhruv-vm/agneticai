"""Typed Pydantic v2 models for denial-feedback interventions and attribution."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.monitor.models import DenialEvent, utc_now


class FeedbackCondition(str, Enum):
    """Configurable denial-feedback intervention conditions."""

    ACTUAL = "ACTUAL"
    NEUTRALIZED = "NEUTRALIZED"
    PARAPHRASED = "PARAPHRASED"
    MINIMAL_STRUCTURED = "MINIMAL_STRUCTURED"


class AttributionLabel(str, Enum):
    """Experimental attribution classifications (not definitive causal truths)."""

    DENIAL_SENSITIVE = "DENIAL_SENSITIVE"
    DENIAL_INDEPENDENT = "DENIAL_INDEPENDENT"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class UncertaintyLevel(str, Enum):
    """Categorical uncertainty level (avoids uncalibrated numeric probabilities)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class InterventionRecord(BaseModel):
    """Records an applied denial-feedback intervention while preserving the original."""

    model_config = ConfigDict(extra="forbid")

    intervention_id: str = Field(..., min_length=1)
    original_denial: DenialEvent
    condition: FeedbackCondition
    rendered_feedback: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=utc_now)


class TraceStepRecord(BaseModel):
    """Single step recorded during paired-branch deterministic replay."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    input_node_ids: list[str] = Field(default_factory=list)
    decision: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    is_protected_sink: bool = False
    is_unauthorized_sink: bool = False
    executed: bool = True


class PairedTrace(BaseModel):
    """Execution trace for one branch (Branch A or Branch B) of a paired replay."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    branch_name: str = Field(..., min_length=1)
    condition: FeedbackCondition
    initial_state_hash: str = Field(..., min_length=1)
    permissions_hash: str = Field(..., min_length=1)
    non_denial_outputs: dict[str, Any] = Field(default_factory=dict)
    steps: list[TraceStepRecord] = Field(default_factory=list)
    interventions: list[InterventionRecord] = Field(default_factory=list)
    replay_valid: bool = True
    replay_error: str | None = None


class DivergencePoint(BaseModel):
    """Observable difference between Branch A and Branch B at a trace position."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    branch_a_step_id: str | None = None
    branch_b_step_id: str | None = None
    dimensions_changed: list[str] = Field(default_factory=list)
    branch_a_summary: str
    branch_b_summary: str


class TraceComparison(BaseModel):
    """Structured comparison between paired Branch A and Branch B traces."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(..., min_length=1)
    identical_initial_state: bool
    identical_permissions: bool
    identical_non_denial_outputs: bool
    replay_deterministic: bool
    earliest_divergence: DivergencePoint | None = None
    divergences: list[DivergencePoint] = Field(default_factory=list)
    same_final_outcome: bool
    differing_provenance_same_args: bool = False
    comparison_notes: str = ""


class AttributionResult(BaseModel):
    """Structured attribution result for a downstream action."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(..., min_length=1)
    denial_event_id: str | None = None
    target_action_id: str = Field(..., min_length=1)
    classification: AttributionLabel
    supporting_evidence: list[str] = Field(default_factory=list)
    earliest_divergence: DivergencePoint | None = None
    relevant_nodes: list[str] = Field(default_factory=list)
    relevant_edges: list[str] = Field(default_factory=list)
    branch_a_ref: str = Field(..., min_length=1)
    branch_b_ref: str = Field(..., min_length=1)
    uncertainty: UncertaintyLevel
    reason: str = Field(..., min_length=1)
