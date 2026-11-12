"""Phase 4.3 independent evaluation, leakage, canonicalization, and control tests."""

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent.environment import EnvironmentState
from src.agent.state import AgentTask
from src.evaluation.phase4_3_metrics import compute_four_class_metrics, wilson_ci
from src.evaluation.phase4_3_runner import (
    run_canonicalization_audit,
    run_intervention_controls,
    run_phase4_3_audit,
)
from src.evaluation.phase4_3_scenarios import generate_phase4_3_dataset


def test_phase4_3_dataset_opaque_ids_and_leakage_isolation() -> None:
    """Verify 104 scenarios have opaque IDs and zero ground-truth fields in tasks."""
    tasks_doc, labels_doc = generate_phase4_3_dataset(seed=43127)
    assert len(tasks_doc["scenarios"]) == 104
    assert len(labels_doc["labels"]) == 104

    forbidden = {
        "ground_truth_class",
        "ground_truth_attribution",
        "is_attack",
        "denial_dependent",
        "expected_decision",
        "category",
        "sub_pattern",
        "failure_mode",
    }
    for sc in tasks_doc["scenarios"]:
        tid = sc["task"]["task_id"]
        assert tid.startswith("IND_SC_")
        assert not (set(sc.keys()) & forbidden)
        assert not (set(sc["task"].keys()) & forbidden)

    # Injecting hidden labels into AgentTask / EnvironmentState must fail.
    with pytest.raises(ValidationError):
        AgentTask(task_id="x", goal="g", ground_truth_class="DENIAL_SENSITIVE")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        EnvironmentState(is_attack=True)  # type: ignore[call-arg]


def test_canonicalization_audit_detects_same_args_different_source_defect() -> None:
    """Verify 11/12 canonicalization cases pass and defect #5 is explicitly caught."""
    cases = {c["case"]: c for c in run_canonicalization_audit()}
    assert len(cases) == 12
    assert cases["1_identical_logical_across_branches"]["passed"] is True
    assert cases["3_different_docs_identical_content"]["passed"] is True
    assert cases["6_nested_reordered_args"]["passed"] is True
    assert cases["12_field_level_distinct"]["passed"] is True
    # Known defect: canonical.py keys on (tool, args) without content/trust digest
    assert cases["5_same_args_different_underlying_source"]["passed"] is False


def test_intervention_controls_all_pass() -> None:
    """Verify sham, positive, negative, independent-source, and repeated controls pass."""
    controls = {c["control"]: c for c in run_intervention_controls()}
    assert len(controls) == 5
    assert all(c["passed"] for c in controls.values())


def test_phase4_3_metrics_reconcile_with_predictions_csv(tmp_path: Path) -> None:
    """Verify 4-class confusion matrix and Wilson CI reconcile with raw predictions."""
    res = run_phase4_3_audit(output_dir=tmp_path)
    for fname in (
        "phase4_3_results.json",
        "phase4_3_predictions.csv",
        "phase4_3_ablation.csv",
        "phase4_3_policy_comparison.csv",
        "phase4_3_canonicalization.csv",
        "phase4_3_intervention_controls.csv",
    ):
        assert (tmp_path / fname).exists()

    with (tmp_path / "phase4_3_predictions.csv").open(encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    assert len(rows) == 104
    recomputed = compute_four_class_metrics(rows)
    assert recomputed["confusion_matrix_4x4"] == res["primary_e_combined_metrics"]["confusion_matrix_4x4"]
    assert recomputed["overall_4class_accuracy"] == res["primary_e_combined_metrics"]["overall_4class_accuracy"]
    assert wilson_ci(0, 0) == (None, None)
