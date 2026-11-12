"""Phase 3 paired-execution counterfactual attribution evaluation runner."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.attribution.attributor import CounterfactualAttributor
from src.attribution.comparison import compare_paired_traces
from src.attribution.models import AttributionLabel, AttributionResult, FeedbackCondition
from src.attribution.policy_adapter import (
    AttributionPolicyAdapter,
    AttributionPolicyConfig,
)
from src.attribution.replay import DeterministicReplayEngine, PairedBranchSpec
from src.evaluation.metrics import safe_div
from src.evaluation.scenarios import Phase2Step, StepRole
from src.monitor.arm_baseline import (
    ARMMonitor,
    CapabilityToken,
    CounterfactualStrategy,
    MonitorConfig,
    ProvenanceMode,
)
from src.monitor.models import DecisionVerdict, TrustLevel

_TRUST_BY_NAME: dict[str, TrustLevel] = {
    "TOOL_DESC": TrustLevel.TOOL_DESC,
    "TOOL_UNTRUSTED": TrustLevel.TOOL_UNTRUSTED,
    "TOOL_TRUSTED": TrustLevel.TOOL_TRUSTED,
    "USER_INPUT": TrustLevel.USER_INPUT,
    "SYS_INSTR": TrustLevel.SYS_INSTR,
}


def _parse_steps(raw_steps: list[dict[str, Any]]) -> tuple[Phase2Step, ...]:
    steps: list[Phase2Step] = []
    for s in raw_steps:
        overrides = {
            k: _TRUST_BY_NAME[v] for k, v in s.get("field_trust_overrides", {}).items()
        }
        steps.append(
            Phase2Step(
                event_id=s["event_id"],
                tool_name=s["tool_name"],
                arguments=s["arguments"],
                role=StepRole(s["role"]),
                expected_decision=DecisionVerdict(s["expected_decision"]),
                is_protected_sink=bool(s["is_protected_sink"]),
                is_unauthorized_sink=bool(s["is_unauthorized_sink"]),
                required_for_benign_completion=bool(s["required_for_benign_completion"]),
                denial_dependent=bool(s["denial_dependent"]),
                input_node_ids=tuple(s.get("input_node_ids", [])),
                simulated_result=s.get("simulated_result"),
                result_data_id=s.get("result_data_id"),
                result_trust=_TRUST_BY_NAME[s.get("result_trust", "TOOL_TRUSTED")],
                field_trust_overrides=overrides,
            )
        )
    return tuple(steps)


@dataclass(frozen=True)
class PairedScenarioEvalRow:
    """Per-scenario evaluation record for Phase 3 attribution."""

    scenario_id: str
    scenario_family: str
    is_attack: bool
    target_action_id: str
    ground_truth: str
    predicted_label: str
    correct: bool
    uncertainty: str
    replay_deterministic: bool
    diverged: bool
    adapter_sink_decision: str
    latency_ms: float
    reason: str


def run_phase3_evaluation(
    config_path: Path = Path("configs/phase3_paired_scenarios.json"),
    output_dir: Path | None = Path("results"),
) -> dict[str, Any]:
    """Run paired-execution attribution evaluation across all Phase 3 scenarios."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(raw["seed"])
    replay_engine = DeterministicReplayEngine()
    attributor = CounterfactualAttributor()
    adapter = AttributionPolicyAdapter(
        AttributionPolicyConfig(enforce_attribution=True)
    )

    rows: list[PairedScenarioEvalRow] = []
    attribution_results: list[dict[str, Any]] = []

    tp = fp = tn = fn = 0
    ambiguous_count = 0
    not_evaluable_count = 0
    replay_det_count = 0
    diverged_count = 0
    evaluable_correct = 0
    evaluable_total = 0
    sink_tp = sink_fp = sink_tn = sink_fn = 0

    for item in raw["scenarios"]:
        sid = item["scenario_id"]
        family = item["scenario_family"]
        is_attack = bool(item["is_attack"])
        allowed = frozenset(item["allowed_tools"])
        target_id = item["target_action_id"]
        gt = item["ground_truth_attribution"]

        steps_a = _parse_steps(item["branch_a_steps"])
        steps_b = _parse_steps(item["branch_b_steps"])
        spec_a = PairedBranchSpec(steps=steps_a, allowed_tools=allowed)
        spec_b = PairedBranchSpec(
            steps=steps_b,
            allowed_tools=allowed,
            non_deterministic_flag=bool(item.get("branch_b_non_deterministic", False)),
        )

        t0 = time.perf_counter_ns()
        mon_a, trace_a, _, trace_b = replay_engine.replay_pair(
            scenario_id=sid,
            branch_a_spec=spec_a,
            branch_b_spec=spec_b,
            branch_b_condition=FeedbackCondition.NEUTRALIZED,
        )
        comparison = compare_paired_traces(trace_a, trace_b)
        attr_res: AttributionResult = attributor.attribute_action(
            scenario_id=sid,
            target_action_id=target_id,
            mon_a=mon_a,
            branch_a=trace_a,
            branch_b=trace_b,
            comparison=comparison,
        )
        elapsed_ms = round((time.perf_counter_ns() - t0) / 1_000_000.0, 4)

        # Evaluate experimental adapter decision on target action (for comparison only).
        base_dec = next(
            (d for d in mon_a.decisions if d.event_id == target_id),
            mon_a.decisions[-1],
        )
        target_step = next(s for s in steps_a if s.event_id == target_id)
        adapted_dec = adapter.apply(
            baseline_decision=base_dec,
            attribution=attr_res,
            is_protected_sink=target_step.is_protected_sink,
        )

        pred = attr_res.classification.value
        if comparison.replay_deterministic:
            replay_det_count += 1
        if comparison.divergences:
            diverged_count += 1

        if pred == AttributionLabel.AMBIGUOUS.value:
            ambiguous_count += 1
        elif pred == AttributionLabel.NOT_EVALUABLE.value:
            not_evaluable_count += 1

        if gt != AttributionLabel.NOT_EVALUABLE.value:
            evaluable_total += 1
            if pred == gt:
                evaluable_correct += 1

            if gt == AttributionLabel.DENIAL_SENSITIVE.value:
                if pred == AttributionLabel.DENIAL_SENSITIVE.value:
                    tp += 1
                else:
                    fn += 1
            elif gt == AttributionLabel.DENIAL_INDEPENDENT.value:
                if pred == AttributionLabel.DENIAL_INDEPENDENT.value:
                    tn += 1
                elif pred == AttributionLabel.DENIAL_SENSITIVE.value:
                    fp += 1

            if target_step.is_unauthorized_sink:
                if adapted_dec.decision == DecisionVerdict.DENY:
                    sink_tp += 1
                else:
                    sink_fn += 1
            elif target_step.expected_decision == DecisionVerdict.ALLOW:
                if adapted_dec.decision == DecisionVerdict.ALLOW:
                    sink_tn += 1
                else:
                    sink_fp += 1

        rows.append(
            PairedScenarioEvalRow(
                scenario_id=sid,
                scenario_family=family,
                is_attack=is_attack,
                target_action_id=target_id,
                ground_truth=gt,
                predicted_label=pred,
                correct=(pred == gt),
                uncertainty=attr_res.uncertainty.value,
                replay_deterministic=comparison.replay_deterministic,
                diverged=bool(comparison.divergences),
                adapter_sink_decision=adapted_dec.decision.value,
                latency_ms=elapsed_ms,
                reason=attr_res.reason,
            )
        )
        attribution_results.append(attr_res.model_dump(mode="json"))

    total = len(rows)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    false_attr_rate = safe_div(fp, fp + tn)
    attr_accuracy = safe_div(evaluable_correct, evaluable_total)
    ambiguous_rate = safe_div(ambiguous_count, total)
    replay_det_rate = safe_div(replay_det_count, total)
    divergence_rate = safe_div(diverged_count, total)
    latencies = [r.latency_ms for r in rows]

    family_summary: dict[str, dict[str, Any]] = {}
    for fam in sorted({r.scenario_family for r in rows}):
        fam_rows = [r for r in rows if r.scenario_family == fam]
        family_summary[fam] = {
            "total": len(fam_rows),
            "correct": sum(1 for r in fam_rows if r.correct),
            "accuracy": safe_div(sum(1 for r in fam_rows if r.correct), len(fam_rows)),
        }

    summary = {
        "suite_config": str(config_path),
        "seed": seed,
        "total_scenarios": total,
        "evaluable_scenarios": evaluable_total,
        "attribution_confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "attribution_accuracy": attr_accuracy,
        "denial_sensitive_precision": precision,
        "denial_sensitive_recall": recall,
        "false_attribution_rate": false_attr_rate,
        "ambiguous_rate": ambiguous_rate,
        "not_evaluable_count": not_evaluable_count,
        "replay_determinism_rate": replay_det_rate,
        "execution_divergence_rate": divergence_rate,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
        "adapter_sink_confusion_matrix": {
            "tp": sink_tp,
            "fp": sink_fp,
            "tn": sink_tn,
            "fn": sink_fn,
        },
        "adapter_attack_success_rate": safe_div(sink_fn, sink_tp + sink_fn),
        "adapter_benign_completion_rate": safe_div(sink_tn, sink_tn + sink_fp),
        "family_breakdown": family_summary,
        "per_scenario_results": [asdict(r) for r in rows],
        "attribution_records": attribution_results,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "phase3_attribution_results.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        with (output_dir / "phase3_per_action.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(asdict(rows[0]).keys())
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))

        summary_row = {
            "total_scenarios": total,
            "evaluable_scenarios": evaluable_total,
            "attribution_accuracy": attr_accuracy,
            "denial_sensitive_precision": precision,
            "denial_sensitive_recall": recall,
            "false_attribution_rate": false_attr_rate,
            "ambiguous_rate": ambiguous_rate,
            "replay_determinism_rate": replay_det_rate,
            "execution_divergence_rate": divergence_rate,
            "adapter_asr": summary["adapter_attack_success_rate"],
            "adapter_benign_completion": summary["adapter_benign_completion_rate"],
            "mean_latency_ms": summary["mean_latency_ms"],
        }
        with (output_dir / "phase3_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(summary_row)

    return summary


if __name__ == "__main__":
    res = run_phase3_evaluation()
    print(
        f"Phase 3 Attribution ({res['total_scenarios']} scenarios, seed={res['seed']}): "
        f"Acc={res['attribution_accuracy']}, Prec={res['denial_sensitive_precision']}, "
        f"Rec={res['denial_sensitive_recall']}, FAR={res['false_attribution_rate']}, "
        f"Amb={res['ambiguous_rate']}, ReplayDet={res['replay_determinism_rate']}, "
        f"MeanLat={res['mean_latency_ms']} ms"
    )
