"""Paired-trace comparison module for Branch A (actual) vs. Branch B (neutralized)."""

from __future__ import annotations

from src.attribution.models import (
    DivergencePoint,
    PairedTrace,
    TraceComparison,
    TraceStepRecord,
)


def _summarize_step(step: TraceStepRecord | None) -> str:
    if step is None:
        return "<missing>"
    return (
        f"{step.tool_name}(args={step.arguments}, inputs={step.input_node_ids}, "
        f"decision={step.decision})"
    )


def compare_paired_traces(
    branch_a: PairedTrace,
    branch_b: PairedTrace,
) -> TraceComparison:
    """Compare Branch A and Branch B traces across tool, argument, and provenance dimensions."""
    same_init = branch_a.initial_state_hash == branch_b.initial_state_hash
    same_perms = branch_a.permissions_hash == branch_b.permissions_hash
    shared_tool_calls = set(branch_a.non_denial_outputs) & set(branch_b.non_denial_outputs)
    same_outputs = all(
        branch_a.non_denial_outputs[k] == branch_b.non_denial_outputs[k]
        for k in shared_tool_calls
    )
    replay_det = branch_a.replay_valid and branch_b.replay_valid

    if not replay_det or not same_init or not same_perms:
        return TraceComparison(
            scenario_id=branch_a.scenario_id,
            identical_initial_state=same_init,
            identical_permissions=same_perms,
            identical_non_denial_outputs=same_outputs,
            replay_deterministic=replay_det,
            earliest_divergence=None,
            divergences=[],
            same_final_outcome=False,
            comparison_notes=(
                "Paired comparison invalid: "
                f"replay_det={replay_det}, same_init={same_init}, same_perms={same_perms}."
            ),
        )

    divergences: list[DivergencePoint] = []
    differing_prov_same_args = False
    max_len = max(len(branch_a.steps), len(branch_b.steps))

    for idx in range(max_len):
        step_a = branch_a.steps[idx] if idx < len(branch_a.steps) else None
        step_b = branch_b.steps[idx] if idx < len(branch_b.steps) else None

        dims: list[str] = []
        if step_a is None or step_b is None:
            dims.append("trace_length")
            dims.append("tool_selection")
        else:
            if step_a.tool_name != step_b.tool_name:
                dims.append("tool_selection")
                dims.append("tool_ordering")
            if step_a.arguments != step_b.arguments:
                dims.append("tool_arguments")
            if step_a.input_node_ids != step_b.input_node_ids:
                dims.append("data_source_dependencies")
                if step_a.tool_name == step_b.tool_name and step_a.arguments == step_b.arguments:
                    differing_prov_same_args = True
            if step_a.is_protected_sink != step_b.is_protected_sink:
                dims.append("protected_sink_attempt")
            if step_a.decision != step_b.decision:
                dims.append("policy_decision")

        if dims:
            divergences.append(
                DivergencePoint(
                    step_index=idx,
                    branch_a_step_id=step_a.step_id if step_a else None,
                    branch_b_step_id=step_b.step_id if step_b else None,
                    dimensions_changed=dims,
                    branch_a_summary=_summarize_step(step_a),
                    branch_b_summary=_summarize_step(step_b),
                )
            )

    final_a = branch_a.steps[-1] if branch_a.steps else None
    final_b = branch_b.steps[-1] if branch_b.steps else None
    same_final = (
        final_a is not None
        and final_b is not None
        and final_a.tool_name == final_b.tool_name
        and final_a.arguments == final_b.arguments
        and final_a.decision == final_b.decision
        and final_a.input_node_ids == final_b.input_node_ids
    )

    return TraceComparison(
        scenario_id=branch_a.scenario_id,
        identical_initial_state=same_init,
        identical_permissions=same_perms,
        identical_non_denial_outputs=same_outputs,
        replay_deterministic=replay_det,
        earliest_divergence=divergences[0] if divergences else None,
        divergences=divergences,
        same_final_outcome=same_final,
        differing_provenance_same_args=differing_prov_same_args,
        comparison_notes=(
            f"Found {len(divergences)} divergence(s); same_final_outcome={same_final}."
        ),
    )
