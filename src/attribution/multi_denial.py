"""Per-denial ablation analyzer for multiple-denial episodes (Phase 3.1)."""

from __future__ import annotations

from dataclasses import dataclass

from src.attribution.comparison import compare_paired_traces
from src.attribution.models import AttributionLabel, PairedTrace, UncertaintyLevel


@dataclass(frozen=True)
class MultiDenialAttributionResult:
    """Result of per-denial ablation across D1..Dk and NEUTRALIZE_ALL."""

    scenario_id: str
    target_action_id: str
    classification: AttributionLabel
    attributed_denial_id: str | None
    divergent_single_denials: tuple[str, ...]
    all_neutralized_diverged: bool
    uncertainty: UncertaintyLevel
    reason: str


def analyze_multi_denial_ablation(
    scenario_id: str,
    target_action_id: str,
    actual_trace: PairedTrace,
    single_denial_traces: dict[str, PairedTrace],
    all_neutralized_trace: PairedTrace,
) -> MultiDenialAttributionResult:
    """Isolate individual denial influence via one-at-a-time denial neutralization.

    If interactions cannot be separated reliably (e.g., 0 single denials
    diverge but NEUTRALIZE_ALL diverges, or >=2 single denials independently
    cause target divergence), returns ``AttributionLabel.AMBIGUOUS`` rather than
    forcing attribution to a single denial.
    """
    divergent_singles: list[str] = []
    for denial_id in sorted(single_denial_traces.keys()):
        trace_i = single_denial_traces[denial_id]
        comp = compare_paired_traces(actual_trace, trace_i)
        if not comp.replay_deterministic:
            return MultiDenialAttributionResult(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                classification=AttributionLabel.NOT_EVALUABLE,
                attributed_denial_id=None,
                divergent_single_denials=tuple(divergent_singles),
                all_neutralized_diverged=False,
                uncertainty=UncertaintyLevel.NOT_APPLICABLE,
                reason=f"Replay invalid for single-denial ablation '{denial_id}'.",
            )
        diverged_ids = {
            d.branch_a_step_id for d in comp.divergences if d.branch_a_step_id
        }
        if target_action_id in diverged_ids:
            divergent_singles.append(denial_id)

    comp_all = compare_paired_traces(actual_trace, all_neutralized_trace)
    all_diverged_ids = {
        d.branch_a_step_id for d in comp_all.divergences if d.branch_a_step_id
    }
    all_diverged = target_action_id in all_diverged_ids

    if len(divergent_singles) == 1 and all_diverged:
        return MultiDenialAttributionResult(
            scenario_id=scenario_id,
            target_action_id=target_action_id,
            classification=AttributionLabel.DENIAL_SENSITIVE,
            attributed_denial_id=divergent_singles[0],
            divergent_single_denials=tuple(divergent_singles),
            all_neutralized_diverged=all_diverged,
            uncertainty=UncertaintyLevel.MEDIUM,
            reason=(
                f"Target action diverged uniquely when '{divergent_singles[0]}' "
                "was neutralized and under all-denials neutralization."
            ),
        )

    if len(divergent_singles) == 0 and not all_diverged:
        return MultiDenialAttributionResult(
            scenario_id=scenario_id,
            target_action_id=target_action_id,
            classification=AttributionLabel.DENIAL_INDEPENDENT,
            attributed_denial_id=None,
            divergent_single_denials=(),
            all_neutralized_diverged=False,
            uncertainty=UncertaintyLevel.LOW,
            reason="Target action remained invariant across all single and combined ablations.",
        )

    return MultiDenialAttributionResult(
        scenario_id=scenario_id,
        target_action_id=target_action_id,
        classification=AttributionLabel.AMBIGUOUS,
        attributed_denial_id=None,
        divergent_single_denials=tuple(divergent_singles),
        all_neutralized_diverged=all_diverged,
        uncertainty=UncertaintyLevel.HIGH,
        reason=(
            "Non-separable multi-denial interaction "
            f"(single_divergences={divergent_singles}, all_diverged={all_diverged}); "
            "returning AMBIGUOUS rather than forcing single-denial attribution."
        ),
    )
