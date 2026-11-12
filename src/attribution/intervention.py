"""Denial-feedback intervention engine for controlled counterfactual replay."""

from __future__ import annotations

from src.attribution.models import FeedbackCondition, InterventionRecord
from src.monitor.models import DenialEvent


class DenialInterventionEngine:
    """Applies configurable denial-feedback conditions while preserving original denials."""

    NEUTRALIZED_MESSAGE = "Action was not executed."
    MINIMAL_STRUCTURED_MESSAGE = '{"status": "not_executed"}'

    def apply(
        self,
        denial: DenialEvent,
        condition: FeedbackCondition,
        intervention_id: str | None = None,
    ) -> InterventionRecord:
        """Render feedback under ``condition`` and record the intervention."""
        if condition == FeedbackCondition.ACTUAL:
            rendered = denial.feedback_observed
        elif condition == FeedbackCondition.NEUTRALIZED:
            rendered = self.NEUTRALIZED_MESSAGE
        elif condition == FeedbackCondition.PARAPHRASED:
            rendered = (
                f"The requested operation '{denial.denied_action}' could not be completed "
                "under active runtime constraints."
            )
        elif condition == FeedbackCondition.MINIMAL_STRUCTURED:
            rendered = self.MINIMAL_STRUCTURED_MESSAGE
        else:
            raise ValueError(f"Unsupported feedback condition: {condition}")

        return InterventionRecord(
            intervention_id=intervention_id or f"int_{condition.value}_{denial.denial_id}",
            original_denial=denial,
            condition=condition,
            rendered_feedback=rendered,
        )
