"""Experimental policy adapter (disabled by default) for attribution results."""

from __future__ import annotations

from dataclasses import dataclass

from src.attribution.models import AttributionLabel, AttributionResult
from src.monitor.models import DecisionVerdict, PolicyDecision


@dataclass(frozen=True)
class AttributionPolicyConfig:
    """Controls whether attribution results can influence enforcement decisions.

    Kept disabled (``enforce_attribution=False``) by default so attribution
    quality is evaluated independently from enforcement quality.
    """

    enforce_attribution: bool = False
    block_on_sensitive_sink_only: bool = True


class AttributionPolicyAdapter:
    """Optional adapter combining baseline PolicyDecision with AttributionResult."""

    def __init__(self, config: AttributionPolicyConfig | None = None) -> None:
        self.config = config or AttributionPolicyConfig()

    def apply(
        self,
        baseline_decision: PolicyDecision,
        attribution: AttributionResult,
        is_protected_sink: bool,
    ) -> PolicyDecision:
        """Return ``baseline_decision`` unless attribution enforcement is enabled."""
        if not self.config.enforce_attribution:
            return baseline_decision

        if baseline_decision.decision == DecisionVerdict.DENY:
            return baseline_decision

        if attribution.classification == AttributionLabel.DENIAL_SENSITIVE:
            if not self.config.block_on_sensitive_sink_only or is_protected_sink:
                return PolicyDecision(
                    decision_id=f"attr_{baseline_decision.decision_id}",
                    event_id=baseline_decision.event_id,
                    decision=DecisionVerdict.DENY,
                    reason=(
                        "Experimental attribution adapter blocked DENIAL_SENSITIVE "
                        f"protected action ({attribution.reason})."
                    ),
                    evidence={
                        "baseline_decision": baseline_decision.decision.value,
                        "attribution_classification": attribution.classification.value,
                        "uncertainty": attribution.uncertainty.value,
                    },
                )

        return baseline_decision
