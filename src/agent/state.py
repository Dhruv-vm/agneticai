"""Typed Pydantic v2 state models for independent agent execution (Phase 4).

All agent-facing models use ``extra='forbid'`` and strictly exclude any
evaluation-only ground-truth fields (``is_attack``, ``denial_dependent``,
``expected_decision``, ``ground_truth_attribution``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentTask(BaseModel):
    """Observable task specification visible to the agent."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1)
    instructions: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=10, ge=1, le=50)
    seed: int = 42


class AgentObservation(BaseModel):
    """Single tool-execution observation returned to the agent."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    allowed: bool
    output: Any = None
    feedback: str
    data_id: str | None = None


class ProposedAction(BaseModel):
    """Next action (or termination) proposed dynamically by the agent."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    input_node_ids: list[str] = Field(default_factory=list)
    terminate: bool = False
    termination_reason: str | None = None


class AgentMemory:
    """Isolated per-branch mutable execution history (never shared across branches)."""

    def __init__(self) -> None:
        self.observations: list[AgentObservation] = []
        self.tool_results: dict[str, Any] = {}
        self.denial_feedbacks: list[str] = []
        self.step_count: int = 0

    def record(self, obs: AgentObservation) -> None:
        """Append an observation to branch-local memory."""
        self.observations.append(obs)
        self.step_count += 1
        if obs.allowed and obs.data_id is not None:
            self.tool_results[obs.data_id] = obs.output
        elif not obs.allowed:
            self.denial_feedbacks.append(obs.feedback)
