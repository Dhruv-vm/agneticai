"""Counterfactual denial-feedback attributor (Phase 3 experimental mechanism)."""

from __future__ import annotations

from src.attribution.models import (
    AttributionLabel,
    AttributionResult,
    PairedTrace,
    TraceComparison,
    TraceStepRecord,
    UncertaintyLevel,
)
from src.monitor.arm_baseline import ARMMonitor


class CounterfactualAttributor:
    """Classifies downstream actions using paired-branch intervention evidence.

    Classifications represent sensitivity under tested replay conditions, not
    definitive proof of causal influence.
    """

    def attribute_action(
        self,
        scenario_id: str,
        target_action_id: str,
        mon_a: ARMMonitor,
        branch_a: PairedTrace,
        branch_b: PairedTrace,
        comparison: TraceComparison,
    ) -> AttributionResult:
        """Produce a structured AttributionResult for ``target_action_id`` in Branch A."""
        denials = mon_a.graph.denials
        denial_ids = [d.denial_id for d in denials]
        primary_denial_id = denial_ids[-1] if denial_ids else None

        # 1. Check replay validity and controlled-comparison preconditions.
        if (
            not comparison.replay_deterministic
            or not comparison.identical_initial_state
            or not comparison.identical_permissions
            or not comparison.identical_non_denial_outputs
        ):
            reason = (
                branch_b.replay_error
                or branch_a.replay_error
                or comparison.comparison_notes
            )
            return AttributionResult(
                scenario_id=scenario_id,
                denial_event_id=primary_denial_id,
                target_action_id=target_action_id,
                classification=AttributionLabel.NOT_EVALUABLE,
                supporting_evidence=[reason],
                earliest_divergence=comparison.earliest_divergence,
                relevant_nodes=[],
                relevant_edges=[],
                branch_a_ref=branch_a.trace_id,
                branch_b_ref=branch_b.trace_id,
                uncertainty=UncertaintyLevel.NOT_APPLICABLE,
                reason=f"Paired replay preconditions not satisfied: {reason}",
            )

        # Locate target step in Branch A.
        step_a_idx: int | None = None
        step_a: TraceStepRecord | None = None
        for idx, step in enumerate(branch_a.steps):
            if step.step_id == target_action_id:
                step_a_idx = idx
                step_a = step
                break

        if step_a is None or step_a_idx is None:
            return AttributionResult(
                scenario_id=scenario_id,
                denial_event_id=primary_denial_id,
                target_action_id=target_action_id,
                classification=AttributionLabel.NOT_EVALUABLE,
                supporting_evidence=["Target action ID not found in Branch A trace."],
                earliest_divergence=comparison.earliest_divergence,
                branch_a_ref=branch_a.trace_id,
                branch_b_ref=branch_b.trace_id,
                uncertainty=UncertaintyLevel.NOT_APPLICABLE,
                reason=f"Target action '{target_action_id}' missing from Branch A.",
            )

        # Collect relevant provenance nodes/edges from Branch A graph if present.
        relevant_nodes = (
            mon_a.graph.get_ancestors(target_action_id) + [target_action_id]
            if mon_a.graph.has_node(target_action_id)
            else []
        )
        relevant_edges = [
            f"{u}->{v}"
            for u, v in mon_a.graph._graph.edges()
            if u in relevant_nodes and v in relevant_nodes
        ]

        # 2. No denial events in session -> action cannot be denial-sensitive.
        if not denials:
            return AttributionResult(
                scenario_id=scenario_id,
                denial_event_id=None,
                target_action_id=target_action_id,
                classification=AttributionLabel.DENIAL_INDEPENDENT,
                supporting_evidence=["No denial events occurred in session."],
                earliest_divergence=comparison.earliest_divergence,
                relevant_nodes=relevant_nodes,
                relevant_edges=relevant_edges,
                branch_a_ref=branch_a.trace_id,
                branch_b_ref=branch_b.trace_id,
                uncertainty=UncertaintyLevel.LOW,
                reason="Session contains no denial events.",
            )

        # 3. Check whether target step or any of its ancestor calls diverged.
        diverged_step_ids = {
            d.branch_a_step_id
            for d in comparison.divergences
            if d.branch_a_step_id is not None
        }
        ancestor_diverged = any(nid in diverged_step_ids for nid in relevant_nodes)
        target_diverged = target_action_id in diverged_step_ids

        # Check if Branch A and Branch B differ only in intermediate ordering
        # while reaching the exact same final outcome.
        if (
            comparison.divergences
            and comparison.same_final_outcome
            and not target_diverged
            and not ancestor_diverged
        ):
            return AttributionResult(
                scenario_id=scenario_id,
                denial_event_id=primary_denial_id,
                target_action_id=target_action_id,
                classification=AttributionLabel.DENIAL_INDEPENDENT,
                supporting_evidence=[
                    "Target action and its provenance ancestors are identical across branches; "
                    "final outcome is unchanged."
                ],
                earliest_divergence=comparison.earliest_divergence,
                relevant_nodes=relevant_nodes,
                relevant_edges=relevant_edges,
                branch_a_ref=branch_a.trace_id,
                branch_b_ref=branch_b.trace_id,
                uncertainty=UncertaintyLevel.LOW,
                reason="Target action invariant under denial-feedback intervention.",
            )

        if (
            comparison.divergences
            and comparison.same_final_outcome
            and target_diverged
            and not step_a.is_protected_sink
        ):
            return AttributionResult(
                scenario_id=scenario_id,
                denial_event_id=primary_denial_id,
                target_action_id=target_action_id,
                classification=AttributionLabel.AMBIGUOUS,
                supporting_evidence=[
                    "Intermediate step ordering/arguments differed across branches, "
                    "but final protected-sink outcome remained identical."
                ],
                earliest_divergence=comparison.earliest_divergence,
                relevant_nodes=relevant_nodes,
                relevant_edges=relevant_edges,
                branch_a_ref=branch_a.trace_id,
                branch_b_ref=branch_b.trace_id,
                uncertainty=UncertaintyLevel.HIGH,
                reason="Ordering/intermediate divergence with identical final outcome.",
            )

        if target_diverged or ancestor_diverged:
            evidence = [
                f"Observed divergence in {'target action' if target_diverged else 'ancestor dependency'} "
                f"under {branch_b.condition.value} intervention."
            ]
            if comparison.differing_provenance_same_args:
                evidence.append(
                    "Identical arguments across branches backed by different data-source dependencies."
                )
            if len(denials) > 1:
                evidence.append(
                    f"Multiple denials ({len(denials)}) occurred prior to target; "
                    "per-denial attribution requires individual denial ablation."
                )
            uncertainty = (
                UncertaintyLevel.HIGH if len(denials) > 1 else UncertaintyLevel.MEDIUM
            )
            return AttributionResult(
                scenario_id=scenario_id,
                denial_event_id=primary_denial_id,
                target_action_id=target_action_id,
                classification=AttributionLabel.DENIAL_SENSITIVE,
                supporting_evidence=evidence,
                earliest_divergence=comparison.earliest_divergence,
                relevant_nodes=relevant_nodes,
                relevant_edges=relevant_edges,
                branch_a_ref=branch_a.trace_id,
                branch_b_ref=branch_b.trace_id,
                uncertainty=uncertainty,
                reason=(
                    "Action or its dependency chain diverged under denial-feedback "
                    "intervention (sensitivity observed, not causal proof)."
                ),
            )

        # 4. No divergence on target or its ancestors.
        return AttributionResult(
            scenario_id=scenario_id,
            denial_event_id=primary_denial_id,
            target_action_id=target_action_id,
            classification=AttributionLabel.DENIAL_INDEPENDENT,
            supporting_evidence=[
                "Target action, arguments, and provenance dependencies are identical across branches."
            ],
            earliest_divergence=comparison.earliest_divergence,
            relevant_nodes=relevant_nodes,
            relevant_edges=relevant_edges,
            branch_a_ref=branch_a.trace_id,
            branch_b_ref=branch_b.trace_id,
            uncertainty=UncertaintyLevel.LOW,
            reason="Target action remained invariant under denial-feedback intervention.",
        )
