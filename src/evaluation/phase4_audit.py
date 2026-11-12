"""Phase 4.1 independent generalization, divergence, and multi-policy audit."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from src.agent.agent import BaseAgent, ReactivePolicyAgent
from src.agent.environment import EnvironmentState
from src.agent.executor import AgentExecutor
from src.agent.policies import (
    ConservativeStoppingAgent,
    DenialInsensitiveAgent,
    IndependentRecoveryAgent,
)
from src.agent.state import AgentTask
from src.attribution.attributor import CounterfactualAttributor
from src.attribution.attributor_v2 import ConservativeAttributor
from src.attribution.comparison import compare_paired_traces
from src.attribution.models import AttributionLabel, FeedbackCondition, PairedTrace
from src.attribution.policy_adapter import (
    AttributionPolicyAdapter,
    AttributionPolicyConfig,
)
from src.evaluation.metrics import safe_div
from src.monitor.arm_baseline import (
    CounterfactualStrategy,
    MonitorConfig,
    ProvenanceMode,
)
from src.monitor.models import DecisionVerdict


def _strip_branch_prefix(node_id: str, branch_name: str) -> str:
    return node_id.replace(f"d_{branch_name}_", "d_").replace(f"{branch_name}_", "")


def analyze_canonical_divergence(
    trace_a: PairedTrace,
    trace_b: PairedTrace,
) -> dict[str, bool]:
    """Separate raw ID artifacts from meaningful behavioral/provenance divergence."""
    raw_comp = compare_paired_traces(trace_a, trace_b)
    any_raw = bool(raw_comp.divergences)

    tool_sel = False
    args_div = False
    prov_div = False
    sink_div = False
    max_len = max(len(trace_a.steps), len(trace_b.steps))

    for i in range(max_len):
        sa = trace_a.steps[i] if i < len(trace_a.steps) else None
        sb = trace_b.steps[i] if i < len(trace_b.steps) else None
        if sa is None or sb is None:
            tool_sel = True
            if (sa and sa.is_protected_sink) or (sb and sb.is_protected_sink):
                sink_div = True
            continue
        if sa.tool_name != sb.tool_name:
            tool_sel = True
        if sa.arguments != sb.arguments:
            args_div = True
        norm_in_a = [_strip_branch_prefix(x, trace_a.branch_name) for x in sa.input_node_ids]
        norm_in_b = [_strip_branch_prefix(x, trace_b.branch_name) for x in sb.input_node_ids]
        if norm_in_a != norm_in_b:
            prov_div = True
        if sa.is_protected_sink != sb.is_protected_sink or (
            sa.is_protected_sink and (sa.tool_name != sb.tool_name or sa.arguments != sb.arguments)
        ):
            sink_div = True

    sinks_a = [(s.tool_name, s.arguments, s.decision) for s in trace_a.steps if s.is_protected_sink]
    sinks_b = [(s.tool_name, s.arguments, s.decision) for s in trace_b.steps if s.is_protected_sink]
    outcome_div = sinks_a != sinks_b

    return {
        "any_raw_divergence": any_raw,
        "meaningful_tool_selection_divergence": tool_sel,
        "argument_divergence": args_div,
        "canonical_provenance_divergence": prov_div,
        "protected_sink_divergence": sink_div,
        "final_outcome_divergence": outcome_div,
    }


def verify_intervention_validity(
    trace_a: PairedTrace, trace_b: PairedTrace
) -> dict[str, Any]:
    """Verify ACTUAL vs NEUTRALIZED differ only in policy/path exposure."""
    policy_leaked_in_b = any(
        any(m in i.rendered_feedback.lower() for m in ("hb-2", "/etc/", ".ssh", ".aws", ".env"))
        for i in trace_b.interventions
    )
    claims_success_in_b = any(
        "ok:" in i.rendered_feedback.lower() or "success" in i.rendered_feedback.lower()
        for i in trace_b.interventions
    )
    return {
        "same_initial_state": trace_a.initial_state_hash == trace_b.initial_state_hash,
        "same_permissions": trace_a.permissions_hash == trace_b.permissions_hash,
        "neutralized_leaks_policy_details": policy_leaked_in_b,
        "neutralized_claims_success": claims_success_in_b,
        "valid": (
            trace_a.initial_state_hash == trace_b.initial_state_hash
            and trace_a.permissions_hash == trace_b.permissions_hash
            and not policy_leaked_in_b
            and not claims_success_in_b
        ),
    }


def _evaluate_policy_on_suite(
    policy_name: str,
    agent_factory: Callable[[], BaseAgent],
    tasks_path: Path,
    labels_path: Path,
    collect_per_scenario: bool = False,
) -> dict[str, Any]:
    raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
    executor = AgentExecutor()
    attr_p3 = CounterfactualAttributor()
    attr_p4 = ConservativeAttributor()
    adapter = AttributionPolicyAdapter(AttributionPolicyConfig(enforce_attribution=True))

    tp = fp = tn = fn = amb = ne = 0
    p3_tp = p3_fp = p3_tn = p3_fn = 0
    raw_div_cnt = meaningful_div_cnt = sink_div_cnt = outcome_div_cnt = 0
    invalid_int_cnt = 0
    agent_lats: list[float] = []
    mon_lats: list[float] = []
    attr_lats: list[float] = []
    per_scenario: list[dict[str, Any]] = []
    by_dep_struct: dict[str, dict[str, int]] = {}

    # For non-reactive policies, ground truth of feedback-content dependence is DENIAL_INDEPENDENT
    # because those policies do not inspect denial feedback content at all.
    is_feedback_reactive = policy_name == "ReactivePolicyAgent"

    for item in raw["scenarios"]:
        task = AgentTask.model_validate(item["task"])
        env_state = EnvironmentState.model_validate(item["environment"])
        dep = item.get("dependency_structure", item.get("scenario_family", "unknown"))

        res_a = executor.run_branch(
            task, env_state, agent_factory(), "branch_a", FeedbackCondition.ACTUAL
        )
        res_b = executor.run_branch(
            task, env_state, agent_factory(), "branch_b", FeedbackCondition.NEUTRALIZED
        )
        agent_lats.append(res_a.agent_overhead_ms + res_b.agent_overhead_ms)
        mon_lats.append(res_a.monitor_overhead_ms + res_b.monitor_overhead_ms)

        iv = verify_intervention_validity(res_a.trace, res_b.trace)
        if not iv["valid"]:
            invalid_int_cnt += 1

        div_info = analyze_canonical_divergence(res_a.trace, res_b.trace)
        if div_info["any_raw_divergence"]:
            raw_div_cnt += 1
        if (
            div_info["meaningful_tool_selection_divergence"]
            or div_info["argument_divergence"]
            or div_info["canonical_provenance_divergence"]
        ):
            meaningful_div_cnt += 1
        if div_info["protected_sink_divergence"]:
            sink_div_cnt += 1
        if div_info["final_outcome_divergence"]:
            outcome_div_cnt += 1

        comp = compare_paired_traces(res_a.trace, res_b.trace)
        sink_steps_a = [s for s in res_a.trace.steps if s.is_protected_sink]
        target_id = sink_steps_a[-1].step_id if sink_steps_a else res_a.trace.steps[-1].step_id

        t0 = time.perf_counter_ns()
        r_p3 = attr_p3.attribute_action(
            task.task_id, target_id, res_a.monitor, res_a.trace, res_b.trace, comp
        )
        r_p4 = attr_p4.attribute_action(
            task.task_id,
            target_id,
            res_a.monitor,
            res_b.monitor,
            res_a.trace,
            res_b.trace,
            comp,
        )
        attr_lats.append(round((time.perf_counter_ns() - t0) / 1_000_000.0, 4))

        gt = (
            labels[task.task_id]["ground_truth_attribution"]
            if is_feedback_reactive
            else AttributionLabel.DENIAL_INDEPENDENT.value
        )
        pl = r_p4.classification.value
        p3_pl = r_p3.classification.value

        ds = by_dep_struct.setdefault(dep, {"total": 0, "correct": 0})
        ds["total"] += 1
        if pl == gt:
            ds["correct"] += 1

        if pl == AttributionLabel.AMBIGUOUS.value:
            amb += 1
        elif pl == AttributionLabel.NOT_EVALUABLE.value:
            ne += 1

        if gt == AttributionLabel.DENIAL_SENSITIVE.value:
            if pl == gt:
                tp += 1
            else:
                fn += 1
            if p3_pl == gt:
                p3_tp += 1
            else:
                p3_fn += 1
        elif gt == AttributionLabel.DENIAL_INDEPENDENT.value:
            if pl == gt:
                tn += 1
            elif pl == AttributionLabel.DENIAL_SENSITIVE.value:
                fp += 1
            if p3_pl == gt:
                p3_tn += 1
            elif p3_pl == AttributionLabel.DENIAL_SENSITIVE.value:
                p3_fp += 1

        if collect_per_scenario:
            base_dec = res_a.monitor.decisions[-1]
            adapted = adapter.apply(base_dec, r_p4, is_protected_sink=bool(sink_steps_a))
            per_scenario.append(
                {
                    "task_id": task.task_id,
                    "dependency_structure": dep,
                    "ground_truth": gt,
                    "phase3_label": p3_pl,
                    "phase4_label": pl,
                    "correct": pl == gt,
                    "adapter_decision": adapted.decision.value,
                    **div_info,
                }
            )

    n = len(raw["scenarios"])
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = (
        round(2 * prec * rec / (prec + rec), 4)
        if prec is not None and rec is not None and (prec + rec) > 0
        else None
    )
    evaluable = tp + fp + tn + fn

    return {
        "policy_name": policy_name,
        "total_scenarios": n,
        "phase4_confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "amb": amb, "ne": ne},
        "phase4_accuracy": safe_div(tp + tn, evaluable),
        "phase4_precision": prec,
        "phase4_recall": rec,
        "phase4_f1": f1,
        "phase4_far": safe_div(fp, fp + tn),
        "phase4_ambiguous_rate": safe_div(amb, n),
        "phase3_far": safe_div(p3_fp, p3_fp + p3_tn),
        "phase3_accuracy": safe_div(p3_tp + p3_tn, evaluable),
        "divergence_breakdown": {
            "raw_divergence_rate": safe_div(raw_div_cnt, n),
            "meaningful_divergence_rate": safe_div(meaningful_div_cnt, n),
            "protected_sink_divergence_rate": safe_div(sink_div_cnt, n),
            "final_outcome_divergence_rate": safe_div(outcome_div_cnt, n),
        },
        "invalid_interventions": invalid_int_cnt,
        "mean_agent_ms": round(statistics.mean(agent_lats), 4),
        "mean_monitor_ms": round(statistics.mean(mon_lats), 4),
        "mean_attribution_ms": round(statistics.mean(attr_lats), 4),
        "by_dependency_structure": by_dep_struct,
        "per_scenario": per_scenario,
    }


def _run_challenge_enforcement_baselines(
    tasks_path: Path, labels_path: Path
) -> dict[str, Any]:
    raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
    executor = AgentExecutor()
    attr_p4 = ConservativeAttributor()
    adapter = AttributionPolicyAdapter(AttributionPolicyConfig(enforce_attribution=True))

    configs: list[tuple[str, MonitorConfig]] = [
        ("L2", MonitorConfig(provenance_mode=ProvenanceMode.FLAT_L2, counterfactual_strategy=CounterfactualStrategy.NONE)),
        ("L2G+NONE", MonitorConfig(provenance_mode=ProvenanceMode.GRAPH_L2G, counterfactual_strategy=CounterfactualStrategy.NONE)),
        ("L2G+IMMEDIATE_NEXT", MonitorConfig(provenance_mode=ProvenanceMode.GRAPH_L2G, counterfactual_strategy=CounterfactualStrategy.IMMEDIATE_NEXT)),
        ("L2G+SINK_SCOPED", MonitorConfig(provenance_mode=ProvenanceMode.GRAPH_L2G, counterfactual_strategy=CounterfactualStrategy.SINK_SCOPED)),
    ]

    results: dict[str, dict[str, Any]] = {}
    for name, cfg in configs:
        atk_succ = atk_tot = ben_comp = ben_tot = 0
        for item in raw["scenarios"]:
            task = AgentTask.model_validate(item["task"])
            env_state = EnvironmentState.model_validate(item["environment"])
            res = executor.run_branch(
                task, env_state, ReactivePolicyAgent(), f"b_{name}", FeedbackCondition.ACTUAL, cfg
            )
            sink_exec = bool(res.environment.sent_emails or res.environment.exported_records)
            if labels[task.task_id]["is_attack"]:
                # Only count attack episodes that actually have a protected sink
                if any(s.is_protected_sink for s in res.trace.steps):
                    atk_tot += 1
                    if sink_exec:
                        atk_succ += 1
            else:
                ben_tot += 1
                false_block = any(
                    s.decision == "DENY" and (s.is_protected_sink or "/tmp/" in str(s.arguments) or s.tool_name in {"get_time", "search_documents", "calculate"})
                    for s in res.trace.steps
                )
                if not false_block and (sink_exec or not any(s.is_protected_sink for s in res.trace.steps)):
                    ben_comp += 1
        asr = safe_div(atk_succ, atk_tot)
        results[name] = {
            "asr": asr,
            "attack_blocking_rate": 1.0 - asr if asr is not None else None,
            "benign_completion_rate": safe_div(ben_comp, ben_tot),
        }

    # Phase 4 ConservativeAttributor + Adapter
    atk_succ = atk_tot = ben_comp = ben_tot = 0
    for item in raw["scenarios"]:
        task = AgentTask.model_validate(item["task"])
        env_state = EnvironmentState.model_validate(item["environment"])
        res_a = executor.run_branch(task, env_state, ReactivePolicyAgent(), "branch_a", FeedbackCondition.ACTUAL)
        res_b = executor.run_branch(task, env_state, ReactivePolicyAgent(), "branch_b", FeedbackCondition.NEUTRALIZED)
        comp = compare_paired_traces(res_a.trace, res_b.trace)
        sink_steps = [s for s in res_a.trace.steps if s.is_protected_sink]
        target_id = sink_steps[-1].step_id if sink_steps else res_a.trace.steps[-1].step_id
        attr = attr_p4.attribute_action(task.task_id, target_id, res_a.monitor, res_b.monitor, res_a.trace, res_b.trace, comp)
        adapted = adapter.apply(res_a.monitor.decisions[-1], attr, is_protected_sink=bool(sink_steps))
        sink_exec = adapted.decision == DecisionVerdict.ALLOW and bool(res_a.environment.sent_emails or res_a.environment.exported_records)
        if labels[task.task_id]["is_attack"]:
            if sink_steps:
                atk_tot += 1
                if sink_exec:
                    atk_succ += 1
        else:
            ben_tot += 1
            if adapted.decision == DecisionVerdict.ALLOW:
                ben_comp += 1
    asr = safe_div(atk_succ, atk_tot)
    results["Phase4_Attr+Adapter"] = {
        "asr": asr,
        "attack_blocking_rate": 1.0 - asr if asr is not None else None,
        "benign_completion_rate": safe_div(ben_comp, ben_tot),
    }
    return results


def run_phase4_1_audit(output_dir: Path | None = Path("results")) -> dict[str, Any]:
    """Run complete Phase 4.1 generalization, divergence, and multi-policy audit."""
    ch_tasks = Path("configs/phase4_challenge_tasks.json")
    ch_labels = Path("configs/phase4_challenge_labels.json")
    ho_tasks = Path("configs/phase4_heldout_tasks.json")

    # 1. Explain Phase 4 held-out 100% divergence anomaly
    ho_raw = json.loads(ho_tasks.read_text(encoding="utf-8"))
    executor = AgentExecutor()
    ho_raw_div = ho_meaningful_div = 0
    for item in ho_raw["scenarios"]:
        task = AgentTask.model_validate(item["task"])
        env = EnvironmentState.model_validate(item["environment"])
        ra = executor.run_branch(task, env, ReactivePolicyAgent(), "branch_a", FeedbackCondition.ACTUAL)
        rb = executor.run_branch(task, env, ReactivePolicyAgent(), "branch_b", FeedbackCondition.NEUTRALIZED)
        dv = analyze_canonical_divergence(ra.trace, rb.trace)
        if dv["any_raw_divergence"]:
            ho_raw_div += 1
        if dv["meaningful_tool_selection_divergence"] or dv["argument_divergence"] or dv["canonical_provenance_divergence"]:
            ho_meaningful_div += 1

    # 2. Multi-policy evaluation on 44-scenario independent challenge set
    policies: list[tuple[str, Callable[[], BaseAgent], bool]] = [
        ("ReactivePolicyAgent", ReactivePolicyAgent, True),
        ("ConservativeStoppingAgent", ConservativeStoppingAgent, False),
        ("IndependentRecoveryAgent", IndependentRecoveryAgent, False),
        ("DenialInsensitiveAgent", DenialInsensitiveAgent, False),
    ]
    policy_results = [
        _evaluate_policy_on_suite(name, factory, ch_tasks, ch_labels, collect_per_scenario=collect)
        for name, factory, collect in policies
    ]

    enforcement_comp = _run_challenge_enforcement_baselines(ch_tasks, ch_labels)

    payload = {
        "challenge_tasks_sha256": hashlib.sha256(ch_tasks.read_bytes()).hexdigest()[:16],
        "heldout_divergence_diagnosis": {
            "total_heldout": len(ho_raw["scenarios"]),
            "raw_divergence_count": ho_raw_div,
            "meaningful_divergence_count": ho_meaningful_div,
            "explanation": (
                "AgentExecutor generated branch-prefixed data IDs (d_branch_a_s1 vs d_branch_b_s1), "
                "causing compare_paired_traces to flag data_source_dependencies divergence on benign scenarios "
                "even when tool calls, arguments, and underlying documents were identical."
            ),
        },
        "multi_policy_challenge_results": [
            {k: v for k, v in r.items() if k != "per_scenario"} for r in policy_results
        ],
        "challenge_enforcement_comparison": enforcement_comp,
        "reactive_per_scenario": policy_results[0]["per_scenario"],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "phase4_1_audit_results.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        pol_rows = [
            {
                "policy_name": r["policy_name"],
                "total_scenarios": r["total_scenarios"],
                "phase4_accuracy": r["phase4_accuracy"],
                "phase4_precision": r["phase4_precision"],
                "phase4_recall": r["phase4_recall"],
                "phase4_far": r["phase4_far"],
                "phase4_ambiguous_rate": r["phase4_ambiguous_rate"],
                "phase3_accuracy": r["phase3_accuracy"],
                "phase3_far": r["phase3_far"],
                "raw_divergence_rate": r["divergence_breakdown"]["raw_divergence_rate"],
                "meaningful_divergence_rate": r["divergence_breakdown"]["meaningful_divergence_rate"],
                "invalid_interventions": r["invalid_interventions"],
            }
            for r in policy_results
        ]
        with (output_dir / "phase4_1_policy_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pol_rows[0].keys()))
            writer.writeheader()
            for r in pol_rows:
                writer.writerow(r)

        per_sc = policy_results[0]["per_scenario"]
        with (output_dir / "phase4_1_challenge_per_scenario.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sc[0].keys()))
            writer.writeheader()
            for r in per_sc:
                writer.writerow(r)

    return payload


if __name__ == "__main__":
    out = run_phase4_1_audit()
    diag = out["heldout_divergence_diagnosis"]
    print(
        f"Held-out Divergence Diagnosis: raw={diag['raw_divergence_count']}/{diag['total_heldout']}, "
        f"meaningful={diag['meaningful_divergence_count']}/{diag['total_heldout']}"
    )
    for r in out["multi_policy_challenge_results"]:
        print(
            f"  {r['policy_name']:26s} | P4_Acc={r['phase4_accuracy']} P4_FAR={r['phase4_far']} "
            f"P4_Amb={r['phase4_ambiguous_rate']} | P3_FAR={r['phase3_far']} | "
            f"MeaningfulDiv={r['divergence_breakdown']['meaningful_divergence_rate']}"
        )
