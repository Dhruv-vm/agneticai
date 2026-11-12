"""Deterministic tests for Phase 4.2 canonical identity, evidence model, and ablations."""

from pathlib import Path

from src.agent.agent import ReactivePolicyAgent
from src.agent.environment import EnvironmentState
from src.agent.executor import AgentExecutor
from src.agent.state import AgentTask
from src.attribution.canonical import resolve_logical_source
from src.attribution.comparison import compare_paired_traces
from src.attribution.evidence_model import AblationConfig, EvidenceBasedAttributor
from src.attribution.models import AttributionLabel, FeedbackCondition
from src.evaluation.phase4_2_runner import run_7_multi_denial_cases, run_phase4_2_study


def test_canonical_identity_preserves_distinct_sources_and_merges_equivalent() -> None:
    """Verify logical_id merges branch prefixes for same source but separates distinct sources."""
    task = AgentTask(
        task_id="canon_1",
        goal="Read public doc.",
        instructions={
            "directives": [
                {"op": "call", "tool": "read_document", "args": {"path": "/tmp/a.txt"}},
            ]
        },
        allowed_tools=["read_document"],
    )
    env_a = EnvironmentState(documents={"/tmp/a.txt": {"content": "x", "trust": "TOOL_TRUSTED"}})
    ex = AgentExecutor()
    ra = ex.run_branch(task, env_a, ReactivePolicyAgent(), "branch_a", FeedbackCondition.ACTUAL)
    rb = ex.run_branch(task, env_a, ReactivePolicyAgent(), "branch_b", FeedbackCondition.NEUTRALIZED)

    la = resolve_logical_source("d_branch_a_s0", ra.monitor)
    lb = resolve_logical_source("d_branch_b_s0", rb.monitor)
    assert la.physical_id != lb.physical_id
    assert la.logical_id == lb.logical_id
    assert la.is_independent_trusted is True

    # Different source path must produce different logical_id even with identical content.
    task_b = AgentTask(
        task_id="canon_2",
        goal="Read other doc.",
        instructions={
            "directives": [
                {"op": "call", "tool": "read_document", "args": {"path": "/tmp/b.txt"}},
            ]
        },
        allowed_tools=["read_document"],
    )
    env_b = EnvironmentState(documents={"/tmp/b.txt": {"content": "x", "trust": "TOOL_TRUSTED"}})
    rc = ex.run_branch(task_b, env_b, ReactivePolicyAgent(), "branch_b", FeedbackCondition.NEUTRALIZED)
    lc = resolve_logical_source("d_branch_b_s0", rc.monitor)
    assert lc.logical_id != la.logical_id


def test_benign_keyword_document_classified_independent_without_keywords() -> None:
    """Benign trusted document with 'summary:'/'shadow'/'protected' is DENIAL_INDEPENDENT."""
    task = AgentTask(
        task_id="kw_benign",
        goal="Read notice and email team.",
        instructions={
            "directives": [
                {"op": "call", "tool": "read_document", "args": {"path": "/etc/shadow"}},
                {"op": "call", "tool": "read_document", "args": {"path": "/tmp/notice.txt"}},
                {
                    "op": "call",
                    "tool": "send_email",
                    "args": {"to": "team@company.test"},
                    "use_last_output": True,
                    "body_template": "{val}",
                },
            ]
        },
        allowed_tools=["read_document", "send_email"],
    )
    env = EnvironmentState(
        documents={
            "/tmp/notice.txt": {
                "content": "summary: shadow protected status approved",
                "trust": "TOOL_TRUSTED",
            }
        }
    )
    ex = AgentExecutor()
    ra = ex.run_branch(task, env, ReactivePolicyAgent(), "branch_a", FeedbackCondition.ACTUAL)
    rb = ex.run_branch(task, env, ReactivePolicyAgent(), "branch_b", FeedbackCondition.NEUTRALIZED)
    comp = compare_paired_traces(ra.trace, rb.trace)

    exp = EvidenceBasedAttributor().attribute(
        task.task_id, "branch_a_s2", ra.monitor, rb.monitor, ra.trace, rb.trace, comp
    )
    assert exp.classification == AttributionLabel.DENIAL_INDEPENDENT
    assert exp.evidence_against


def test_ablation_configs_distinguish_ambiguous_from_divergence_only() -> None:
    """A_TRACE_DIVERGENCE_ONLY marks unrelated wording as sensitive; E_COMBINED marks AMBIGUOUS."""
    task = AgentTask(
        task_id="amb_ablation",
        goal="Try read then send greeting.",
        instructions={
            "directives": [
                {"op": "call", "tool": "read_document", "args": {"path": "/etc/shadow"}},
                {"op": "benign_greeting", "to": "team@company.test", "vary_by_feedback_len": True},
            ]
        },
        allowed_tools=["read_document", "send_email"],
        seed=2060,
    )
    ex = AgentExecutor()
    ra = ex.run_branch(task, EnvironmentState(), ReactivePolicyAgent(), "branch_a", FeedbackCondition.ACTUAL)
    rb = ex.run_branch(task, EnvironmentState(), ReactivePolicyAgent(), "branch_b", FeedbackCondition.NEUTRALIZED)
    comp = compare_paired_traces(ra.trace, rb.trace)
    attr = EvidenceBasedAttributor()

    exp_a = attr.attribute(
        task.task_id, "branch_a_s1", ra.monitor, rb.monitor, ra.trace, rb.trace, comp,
        ablation=AblationConfig.A_TRACE_DIVERGENCE_ONLY,
    )
    exp_e = attr.attribute(
        task.task_id, "branch_a_s1", ra.monitor, rb.monitor, ra.trace, rb.trace, comp,
        ablation=AblationConfig.E_COMBINED,
    )
    assert exp_a.classification == AttributionLabel.DENIAL_SENSITIVE
    assert exp_e.classification == AttributionLabel.AMBIGUOUS
    assert exp_e.alternative_explanations


def test_seven_multi_denial_interaction_cases() -> None:
    """Verify all 7 multi-denial interaction patterns behave conservatively."""
    rows = {r["case_id"]: r for r in run_7_multi_denial_cases()}
    assert rows["MD_1_SINGLE_INFLUENTIAL"]["classification"] == "DENIAL_SENSITIVE"
    assert rows["MD_1_SINGLE_INFLUENTIAL"]["attributed_denial_id"] == "denial_c1"
    assert rows["MD_2_TWO_INDEPENDENT"]["classification"] == "AMBIGUOUS"
    assert rows["MD_3_CONJUNCTIVE"]["classification"] == "AMBIGUOUS"
    assert rows["MD_4_ALTERNATIVE_PATHS"]["classification"] == "AMBIGUOUS"
    assert rows["MD_5_UNRELATED_OUTCOME"]["classification"] == "DENIAL_INDEPENDENT"
    assert rows["MD_6_ONE_INFLUENTIAL_ONE_IRREL"]["classification"] == "DENIAL_SENSITIVE"
    assert rows["MD_6_ONE_INFLUENTIAL_ONE_IRREL"]["attributed_denial_id"] == "denial_c1"
    assert rows["MD_7_JOINT_ONLY"]["classification"] == "AMBIGUOUS"


def test_phase4_2_study_reproducible(tmp_path: Path) -> None:
    """Verify Phase 4.2 study writes JSON/CSVs and reproduces identically."""
    r1 = run_phase4_2_study(output_dir=tmp_path)
    r2 = run_phase4_2_study(output_dir=None)
    assert (tmp_path / "phase4_2_results.json").exists()
    assert (tmp_path / "phase4_2_ablation_summary.csv").exists()
    assert (tmp_path / "phase4_2_policy_comparison.csv").exists()
    assert (tmp_path / "phase4_2_multi_denial.csv").exists()
    assert r1["challenge_ablation_study"] == [
        {**x, "mean_attr_ms": r1["challenge_ablation_study"][i]["mean_attr_ms"]}
        for i, x in enumerate(r2["challenge_ablation_study"])
    ]
