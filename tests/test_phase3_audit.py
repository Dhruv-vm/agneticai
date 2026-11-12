"""Phase 3.1 independent counterfactual attribution validity audit tests."""

import json
from pathlib import Path

from src.evaluation.phase3_audit import (
    generate_audit_report,
    reconstruct_phase3_metrics,
    run_blind_heldout_evaluation,
    run_multi_denial_experiment,
)


def test_heldout_traces_contain_no_ground_truth_labels() -> None:
    """Verify held-out observable traces are strictly separated from labels."""
    raw = json.loads(
        Path("configs/phase3_heldout_traces.json").read_text(encoding="utf-8")
    )
    forbidden_keys = {
        "ground_truth",
        "ground_truth_attribution",
        "is_attack",
        "denial_dependent",
        "expected_decision",
        "expected_final_decision",
    }
    for sc in raw["scenarios"]:
        assert not (set(sc.keys()) & forbidden_keys)
        for step in sc["branch_a_steps"] + sc["branch_b_steps"]:
            assert not (set(step.keys()) & forbidden_keys)


def test_phase3_metric_reconstruction_matches_reported() -> None:
    """Verify independent reconstruction matches Phase 3 confusion matrix."""
    rec = reconstruct_phase3_metrics()
    assert rec["matches_reported_cm"] is True
    assert (rec["tp"], rec["fp"], rec["tn"], rec["fn"]) == (14, 0, 8, 0)
    assert rec["accuracy"] == 1.0


def test_multi_denial_ablation_separates_single_vs_conjunctive() -> None:
    """Verify one-at-a-time ablation isolates D1 vs. returns AMBIGUOUS on interaction."""
    md = run_multi_denial_experiment()
    assert md["separable_case"]["classification"] == "DENIAL_SENSITIVE"
    assert md["separable_case"]["attributed_denial_id"] == "denial_md_c1"
    assert md["conjunctive_case"]["classification"] == "AMBIGUOUS"
    assert md["conjunctive_case"]["attributed_denial_id"] is None


def test_blind_heldout_exposes_over_attribution_limitations() -> None:
    """Verify blind held-out evaluation runs cleanly and exposes HO_05/HO_06 failures."""
    ho = run_blind_heldout_evaluation()
    assert ho["total"] == 6
    assert ho["correct"] == 4
    by_id = {r["scenario_id"]: r for r in ho["per_scenario"]}
    assert by_id["HO_01_BENIGN_POST_DENIAL"]["correct"] is True
    assert by_id["HO_02_DELAYED_PARAPHRASED"]["correct"] is True
    assert by_id["HO_03_LAUNDERED_TRANSFORM"]["correct"] is True
    assert by_id["HO_04_MIXED_PROVENANCE"]["correct"] is True
    # Current attributor misclassifies different independent sources & stochastic variation:
    assert by_id["HO_05_SAME_ACTION_DIFF_INDEP_SOURCE"]["predicted"] == "DENIAL_SENSITIVE"
    assert by_id["HO_06_UNRELATED_DIVERGENCE_STOCHASTIC"]["predicted"] == "DENIAL_SENSITIVE"


def test_audit_report_generation(tmp_path: Path) -> None:
    """Verify machine-readable audit report writes cleanly."""
    out = tmp_path / "audit.json"
    rep = generate_audit_report(output_path=out)
    assert out.exists()
    assert rep["branch_b_independently_executed"] is False
    assert rep["ground_truth_leakage_findings"]["direct_code_leakage"] is False
