"""Deterministic paired-execution replay engine (no LLM or external API)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from src.attribution.intervention import DenialInterventionEngine
from src.attribution.models import (
    FeedbackCondition,
    InterventionRecord,
    PairedTrace,
    TraceStepRecord,
)
from src.evaluation.scenarios import Phase2Step
from src.monitor.arm_baseline import (
    ARMMonitor,
    CapabilityToken,
    CounterfactualStrategy,
    MonitorConfig,
    ProvenanceMode,
)
from src.monitor.models import DecisionVerdict


def _canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PairedBranchSpec:
    """Deterministic specification for one replay branch."""

    steps: tuple[Phase2Step, ...]
    initial_state: dict[str, Any] = field(default_factory=dict)
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    non_deterministic_flag: bool = False


class DeterministicReplayEngine:
    """Replays paired Branch A (actual) and Branch B (neutralized) traces."""

    def __init__(
        self,
        intervention_engine: DenialInterventionEngine | None = None,
    ) -> None:
        self._intervention = intervention_engine or DenialInterventionEngine()

    def _run_branch(
        self,
        scenario_id: str,
        branch_name: str,
        condition: FeedbackCondition,
        spec: PairedBranchSpec,
        monitor_config: MonitorConfig | None = None,
    ) -> tuple[ARMMonitor, PairedTrace]:
        cfg = monitor_config or MonitorConfig(
            provenance_mode=ProvenanceMode.GRAPH_L2G,
            counterfactual_strategy=CounterfactualStrategy.NONE,
        )
        token = CapabilityToken(
            session_id=f"replay_{scenario_id}_{branch_name}",
            allowed_tools=spec.allowed_tools,
        )
        monitor = ARMMonitor(
            session_id=f"replay_{scenario_id}_{branch_name}",
            capability_token=token,
            config=cfg,
        )

        init_hash = _canonical_hash(spec.initial_state)
        perm_hash = _canonical_hash(sorted(spec.allowed_tools))
        non_denial_outputs = {
            f"{s.tool_name}:{json.dumps(s.arguments, sort_keys=True, separators=(',', ':'))}": {
                "result": s.simulated_result,
                "trust": s.result_trust.label,
            }
            for s in spec.steps
            if s.result_data_id is not None
        }

        if spec.non_deterministic_flag:
            return monitor, PairedTrace(
                trace_id=f"trace_{scenario_id}_{branch_name}",
                scenario_id=scenario_id,
                branch_name=branch_name,
                condition=condition,
                initial_state_hash=init_hash,
                permissions_hash=perm_hash,
                non_denial_outputs=non_denial_outputs,
                replay_valid=False,
                replay_error="Non-deterministic branch cannot be replayed deterministically.",
            )

        trace_steps: list[TraceStepRecord] = []
        interventions: list[InterventionRecord] = []
        recorded_denials = 0

        for step in spec.steps:
            if any(not monitor.graph.has_node(nid) for nid in step.input_node_ids):
                trace_steps.append(
                    TraceStepRecord(
                        step_id=step.event_id,
                        tool_name=step.tool_name,
                        arguments=step.arguments,
                        input_node_ids=list(step.input_node_ids),
                        decision="DENY",
                        reason="Skipped due to missing upstream input dependency.",
                        is_protected_sink=step.is_protected_sink,
                        is_unauthorized_sink=step.is_unauthorized_sink,
                        executed=False,
                    )
                )
                continue

            decision = monitor.evaluate_tool_call(
                event_id=step.event_id,
                tool_name=step.tool_name,
                arguments=step.arguments,
                input_node_ids=list(step.input_node_ids),
            )
            if (
                decision.decision == DecisionVerdict.ALLOW
                and step.result_data_id is not None
            ):
                monitor.record_tool_result(
                    call_event_id=step.event_id,
                    data_id=step.result_data_id,
                    value=step.simulated_result,
                    default_trust=step.result_trust,
                    field_trust_overrides=dict(step.field_trust_overrides),
                )
            elif decision.decision == DecisionVerdict.DENY:
                denials = monitor.graph.denials
                while recorded_denials < len(denials):
                    denial = denials[recorded_denials]
                    interventions.append(
                        self._intervention.apply(denial=denial, condition=condition)
                    )
                    recorded_denials += 1

            trace_steps.append(
                TraceStepRecord(
                    step_id=step.event_id,
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    input_node_ids=list(step.input_node_ids),
                    decision=decision.decision.value,
                    reason=decision.reason,
                    is_protected_sink=step.is_protected_sink,
                    is_unauthorized_sink=step.is_unauthorized_sink,
                    executed=True,
                )
            )

        return monitor, PairedTrace(
            trace_id=f"trace_{scenario_id}_{branch_name}",
            scenario_id=scenario_id,
            branch_name=branch_name,
            condition=condition,
            initial_state_hash=init_hash,
            permissions_hash=perm_hash,
            non_denial_outputs=non_denial_outputs,
            steps=trace_steps,
            interventions=interventions,
            replay_valid=True,
        )

    def replay_pair(
        self,
        scenario_id: str,
        branch_a_spec: PairedBranchSpec,
        branch_b_spec: PairedBranchSpec | None,
        branch_b_condition: FeedbackCondition = FeedbackCondition.NEUTRALIZED,
        monitor_config: MonitorConfig | None = None,
    ) -> tuple[ARMMonitor, PairedTrace, ARMMonitor | None, PairedTrace]:
        """Replay Branch A (ACTUAL) and Branch B (NEUTRALIZED or configured)."""
        mon_a, trace_a = self._run_branch(
            scenario_id=scenario_id,
            branch_name="branch_a",
            condition=FeedbackCondition.ACTUAL,
            spec=branch_a_spec,
            monitor_config=monitor_config,
        )
        if branch_b_spec is None:
            trace_b = PairedTrace(
                trace_id=f"trace_{scenario_id}_branch_b",
                scenario_id=scenario_id,
                branch_name="branch_b",
                condition=branch_b_condition,
                initial_state_hash=trace_a.initial_state_hash,
                permissions_hash=trace_a.permissions_hash,
                non_denial_outputs=trace_a.non_denial_outputs,
                replay_valid=False,
                replay_error="Branch B specification is missing; paired replay unavailable.",
            )
            return mon_a, trace_a, None, trace_b

        mon_b, trace_b = self._run_branch(
            scenario_id=scenario_id,
            branch_name="branch_b",
            condition=branch_b_condition,
            spec=branch_b_spec,
            monitor_config=monitor_config,
        )
        return mon_a, trace_a, mon_b, trace_b
