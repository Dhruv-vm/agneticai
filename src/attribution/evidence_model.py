"""Evidence-based counterfactual attribution model with 7 ablation configurations.

Zero keyword matching: decisions rely strictly on observable denial-event
relationships, canonical logical provenance, intervention-sensitive trace
differences, independent-source availability, and protected-sink presence.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.attribution.canonical import resolve_step_logical_sources
from src.attribution.models import (
    AttributionLabel,
    PairedTrace,
    TraceComparison,
    TraceStepRecord,
    UncertaintyLevel,
)
from src.monitor.arm_baseline import ARMMonitor


class AblationConfig(str, Enum):
    """Seven evidence-dimension ablation configurations."""

    A_TRACE_DIVERGENCE_ONLY = "A_TRACE_DIVERGENCE_ONLY"
    B_PROVENANCE_ONLY = "B_PROVENANCE_ONLY"
    C_DENIAL_EVENT_ONLY = "C_DENIAL_EVENT_ONLY"
    D_INDEPENDENT_SOURCE_ONLY = "D_INDEPENDENT_SOURCE_ONLY"
    E_COMBINED = "E_COMBINED"
    F_COMBINED_NO_TEMPORAL = "F_COMBINED_NO_TEMPORAL"
    G_COMBINED_NO_PROVENANCE = "G_COMBINED_NO_PROVENANCE"


class EvidenceExplanation(BaseModel):
    """Structured, non-numeric explanation for an attribution classification."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    target_action_id: str
    relevant_denial_ids: list[str] = Field(default_factory=list)
    evidence_supporting: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    independent_sources_considered: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    classification: AttributionLabel
    uncertainty: UncertaintyLevel


def _find_matching_step_b(
    step_a: TraceStepRecord,
    idx_a: int,
    branch_b: PairedTrace,
) -> TraceStepRecord | None:
    if idx_a < len(branch_b.steps) and branch_b.steps[idx_a].tool_name == step_a.tool_name:
        return branch_b.steps[idx_a]
    return next(
        (s for s in branch_b.steps if s.tool_name == step_a.tool_name),
        None,
    )


class EvidenceBasedAttributor:
    """Multi-evidence counterfactual attributor supporting ablation configs A-G."""

    def attribute(
        self,
        scenario_id: str,
        target_action_id: str,
        mon_a: ARMMonitor,
        mon_b: ARMMonitor,
        branch_a: PairedTrace,
        branch_b: PairedTrace,
        comparison: TraceComparison,
        ablation: AblationConfig = AblationConfig.E_COMBINED,
    ) -> EvidenceExplanation:
        denials = mon_a.graph.denials
        denial_ids = [d.denial_id for d in denials]

        # Precondition checks
        if (
            not comparison.replay_deterministic
            or not comparison.identical_initial_state
            or not comparison.identical_permissions
            or not comparison.identical_non_denial_outputs
        ):
            return EvidenceExplanation(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                relevant_denial_ids=denial_ids,
                missing_evidence=["Controlled paired replay preconditions failed."],
                classification=AttributionLabel.NOT_EVALUABLE,
                uncertainty=UncertaintyLevel.NOT_APPLICABLE,
            )

        idx_a = next(
            (i for i, s in enumerate(branch_a.steps) if s.step_id == target_action_id),
            None,
        )
        if idx_a is None:
            return EvidenceExplanation(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                relevant_denial_ids=denial_ids,
                missing_evidence=["Target action ID absent from Branch A."],
                classification=AttributionLabel.NOT_EVALUABLE,
                uncertainty=UncertaintyLevel.NOT_APPLICABLE,
            )

        step_a = branch_a.steps[idx_a]
        step_b = _find_matching_step_b(step_a, idx_a, branch_b)

        # Canonical logical provenance (never merges genuinely different sources)
        sources_a = resolve_step_logical_sources(step_a.input_node_ids, mon_a)
        sources_b = (
            resolve_step_logical_sources(step_b.input_node_ids, mon_b)
            if step_b is not None
            else []
        )
        logical_ids_a = [s.logical_id for s in sources_a]
        logical_ids_b = [s.logical_id for s in sources_b]
        prov_equivalent = logical_ids_a == logical_ids_b
        both_indep_trusted = (
            bool(sources_a)
            and bool(sources_b)
            and all(s.is_independent_trusted for s in sources_a)
            and all(s.is_independent_trusted for s in sources_b)
        )

        # Canonical trace divergence (ignoring branch_a_ / branch_b_ ID prefixes)
        action_present_in_both = step_b is not None
        args_identical = action_present_in_both and (step_a.arguments == step_b.arguments)
        canonical_diverged = (not action_present_in_both) or (not args_identical) or (not prov_equivalent)

        # Check if an upstream transform step (e.g. calculate/transform_text) was omitted in Branch B
        tools_a = [s.tool_name for s in branch_a.steps]
        tools_b = [s.tool_name for s in branch_b.steps]
        omitted_in_b = len(tools_a) > len(tools_b) and not action_present_in_both

        # Build evidence lists
        supporting: list[str] = []
        against: list[str] = []
        alternatives: list[str] = []
        missing: list[str] = []

        if denials:
            supporting.append(f"{len(denials)} denial event(s) observed in Branch A.")
        else:
            against.append("No denial events occurred in session.")

        if not action_present_in_both:
            supporting.append(
                f"Action '{step_a.tool_name}' executed in Branch A but was omitted under NEUTRALIZED feedback."
            )
        elif args_identical and prov_equivalent:
            against.append("Action, arguments, and canonical logical provenance are identical across branches.")
        elif args_identical and not prov_equivalent:
            if both_indep_trusted:
                against.append("Both branches used independent trusted logical sources.")
                alternatives.append("Different trusted templates/sources selected independently.")
            else:
                supporting.append("Identical arguments backed by non-equivalent/untrusted provenance.")
        else:
            alternatives.append(
                "Both branches executed the same tool with different arguments; "
                "divergence may reflect unrelated variation rather than denial influence."
            )

        if len(denials) > 1:
            missing.append("Session-level neutralization does not isolate individual denials in multi-denial episodes.")

        # Dispatch by AblationConfig
        if ablation == AblationConfig.A_TRACE_DIVERGENCE_ONLY:
            label = (
                AttributionLabel.DENIAL_SENSITIVE
                if canonical_diverged
                else AttributionLabel.DENIAL_INDEPENDENT
            )
            return EvidenceExplanation(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                relevant_denial_ids=denial_ids,
                evidence_supporting=supporting,
                evidence_against=against,
                independent_sources_considered=logical_ids_a,
                alternative_explanations=alternatives,
                missing_evidence=missing,
                classification=label,
                uncertainty=UncertaintyLevel.MEDIUM,
            )

        if ablation == AblationConfig.B_PROVENANCE_ONLY:
            # Provenance-only: flags if ungrounded/untrusted or canonical provenance differs
            label = (
                AttributionLabel.DENIAL_INDEPENDENT
                if (prov_equivalent and bool(sources_a) and all(s.is_independent_trusted for s in sources_a))
                else AttributionLabel.DENIAL_SENSITIVE
            )
            return EvidenceExplanation(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                relevant_denial_ids=denial_ids,
                evidence_supporting=supporting,
                evidence_against=against,
                independent_sources_considered=logical_ids_a,
                alternative_explanations=alternatives,
                missing_evidence=missing,
                classification=label,
                uncertainty=UncertaintyLevel.MEDIUM,
            )

        if ablation == AblationConfig.C_DENIAL_EVENT_ONLY:
            label = (
                AttributionLabel.DENIAL_SENSITIVE
                if denials
                else AttributionLabel.DENIAL_INDEPENDENT
            )
            return EvidenceExplanation(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                relevant_denial_ids=denial_ids,
                evidence_supporting=supporting,
                evidence_against=against,
                independent_sources_considered=logical_ids_a,
                alternative_explanations=alternatives,
                missing_evidence=missing,
                classification=label,
                uncertainty=UncertaintyLevel.HIGH if denials else UncertaintyLevel.LOW,
            )

        if ablation == AblationConfig.D_INDEPENDENT_SOURCE_ONLY:
            label = (
                AttributionLabel.DENIAL_INDEPENDENT
                if both_indep_trusted
                else AttributionLabel.DENIAL_SENSITIVE
            )
            return EvidenceExplanation(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                relevant_denial_ids=denial_ids,
                evidence_supporting=supporting,
                evidence_against=against,
                independent_sources_considered=logical_ids_a,
                alternative_explanations=alternatives,
                missing_evidence=missing,
                classification=label,
                uncertainty=UncertaintyLevel.MEDIUM,
            )

        # Combined models: E_COMBINED, F_COMBINED_NO_TEMPORAL, G_COMBINED_NO_PROVENANCE
        use_provenance = ablation != AblationConfig.G_COMBINED_NO_PROVENANCE

        if not denials:
            return EvidenceExplanation(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                relevant_denial_ids=[],
                evidence_supporting=supporting,
                evidence_against=against,
                independent_sources_considered=logical_ids_a,
                alternative_explanations=alternatives,
                missing_evidence=missing,
                classification=AttributionLabel.DENIAL_INDEPENDENT,
                uncertainty=UncertaintyLevel.LOW,
            )

        if action_present_in_both and args_identical and (not use_provenance or prov_equivalent):
            return EvidenceExplanation(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                relevant_denial_ids=denial_ids,
                evidence_supporting=supporting,
                evidence_against=against,
                independent_sources_considered=logical_ids_a,
                alternative_explanations=alternatives,
                missing_evidence=missing,
                classification=AttributionLabel.DENIAL_INDEPENDENT,
                uncertainty=UncertaintyLevel.LOW,
            )

        if use_provenance and action_present_in_both and args_identical and both_indep_trusted:
            return EvidenceExplanation(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                relevant_denial_ids=denial_ids,
                evidence_supporting=supporting,
                evidence_against=against,
                independent_sources_considered=logical_ids_a + logical_ids_b,
                alternative_explanations=alternatives,
                missing_evidence=missing,
                classification=AttributionLabel.DENIAL_INDEPENDENT,
                uncertainty=UncertaintyLevel.MEDIUM,
            )

        if action_present_in_both and not args_identical and not omitted_in_b:
            # Both branches executed the same tool with different arguments -> AMBIGUOUS
            return EvidenceExplanation(
                scenario_id=scenario_id,
                target_action_id=target_action_id,
                relevant_denial_ids=denial_ids,
                evidence_supporting=supporting,
                evidence_against=against,
                independent_sources_considered=logical_ids_a,
                alternative_explanations=alternatives,
                missing_evidence=missing,
                classification=AttributionLabel.AMBIGUOUS,
                uncertainty=UncertaintyLevel.HIGH,
            )

        # Action omitted under NEUTRALIZED feedback (or non-equivalent untrusted provenance)
        return EvidenceExplanation(
            scenario_id=scenario_id,
            target_action_id=target_action_id,
            relevant_denial_ids=denial_ids,
            evidence_supporting=supporting,
            evidence_against=against,
            independent_sources_considered=logical_ids_a,
            alternative_explanations=alternatives,
            missing_evidence=missing,
            classification=AttributionLabel.DENIAL_SENSITIVE,
            uncertainty=UncertaintyLevel.HIGH if len(denials) > 1 else UncertaintyLevel.MEDIUM,
        )
