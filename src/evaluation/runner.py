"""Baseline scenario evaluation runner comparing L2 and L2G."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    AggregateMetrics,
    EpisodeOutcome,
    compute_aggregate_metrics,
)
from src.evaluation.scenarios import (
    Phase2Scenario,
    ScenarioFixture,
    build_default_token,
    evaluate_episode,
    execute_scenario,
    get_phase1_scenarios,
    load_phase2_scenarios,
)
from src.monitor.arm_baseline import (
    ARMMonitor,
    CapabilityToken,
    CounterfactualStrategy,
    MonitorConfig,
    ProvenanceMode,
)


@dataclass(frozen=True)
class ScenarioEvalRow:
    """Single scenario evaluation record for L2 vs L2G comparison."""

    scenario_id: str
    engine_config: str
    expected_outcome: str
    actual_decision: str
    correct: bool
    decision_reason: str
    execution_latency_ms: float


def run_comparison(
    scenarios: list[ScenarioFixture] | None = None,
    output_dir: Path | None = None,
) -> list[ScenarioEvalRow]:
    """Run scenarios against both L2 (flat) and L2G (graph-aware) configurations."""
    selected = scenarios if scenarios is not None else get_phase1_scenarios()
    rows: list[ScenarioEvalRow] = []

    for mode in (ProvenanceMode.FLAT_L2, ProvenanceMode.GRAPH_L2G):
        for scenario in selected:
            config = MonitorConfig(
                provenance_mode=mode,
                counterfactual_strategy=CounterfactualStrategy.IMMEDIATE_NEXT,
            )
            monitor = ARMMonitor(
                session_id=f"eval_{mode.value}_{scenario.scenario_id}",
                capability_token=build_default_token(scenario),
                config=config,
            )
            start_ns = time.perf_counter_ns()
            decisions = execute_scenario(monitor, scenario)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

            final_decision = decisions[-1]
            rows.append(
                ScenarioEvalRow(
                    scenario_id=scenario.scenario_id,
                    engine_config=mode.value,
                    expected_outcome=scenario.expected_final_decision.value,
                    actual_decision=final_decision.decision.value,
                    correct=(final_decision.decision == scenario.expected_final_decision),
                    decision_reason=final_decision.reason,
                    execution_latency_ms=round(elapsed_ms, 4),
                )
            )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "phase1_baseline_comparison.json"
        csv_path = output_dir / "phase1_baseline_comparison.csv"

        json_path.write_text(
            json.dumps([asdict(r) for r in rows], indent=2) + "\n",
            encoding="utf-8",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "scenario_id",
                    "engine_config",
                    "expected_outcome",
                    "actual_decision",
                    "correct",
                    "decision_reason",
                    "execution_latency_ms",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    return rows


PHASE2_CONFIGURATIONS: tuple[tuple[str, ProvenanceMode, CounterfactualStrategy], ...] = (
    ("L2", ProvenanceMode.FLAT_L2, CounterfactualStrategy.NONE),
    ("L2G+IMMEDIATE_NEXT", ProvenanceMode.GRAPH_L2G, CounterfactualStrategy.IMMEDIATE_NEXT),
    ("L2G+NONE", ProvenanceMode.GRAPH_L2G, CounterfactualStrategy.NONE),
    ("L2G+SINK_SCOPED", ProvenanceMode.GRAPH_L2G, CounterfactualStrategy.SINK_SCOPED),
)


def _group_by_family(
    episodes: list[EpisodeOutcome],
) -> dict[str, dict[str, Any]]:
    """Compute per-family breakdown using compute_aggregate_metrics."""
    families = sorted({ep.scenario_family for ep in episodes})
    breakdown: dict[str, dict[str, Any]] = {}
    for family in families:
        fam_eps = [ep for ep in episodes if ep.scenario_family == family]
        breakdown[family] = compute_aggregate_metrics(family, fam_eps).to_dict()
    return breakdown


def run_phase2_comparison(
    config_path: Path = Path("configs/phase2_scenarios.json"),
    output_dir: Path | None = Path("results"),
) -> dict[str, Any]:
    """Run all Phase 2 scenarios across L2 and L2G counterfactual strategies."""
    seed, scenarios = load_phase2_scenarios(config_path)
    all_episodes: list[EpisodeOutcome] = []
    aggregates: list[AggregateMetrics] = []
    family_breakdowns: dict[str, dict[str, dict[str, Any]]] = {}

    for strategy_name, mode, cf_strategy in PHASE2_CONFIGURATIONS:
        strategy_episodes: list[EpisodeOutcome] = []
        for scenario in scenarios:
            monitor_cfg = MonitorConfig(
                provenance_mode=mode,
                counterfactual_strategy=cf_strategy,
            )
            token = CapabilityToken(
                session_id=f"p2_{strategy_name}_{scenario.scenario_id}",
                allowed_tools=scenario.allowed_tools,
            )
            monitor = ARMMonitor(
                session_id=f"p2_{strategy_name}_{scenario.scenario_id}",
                capability_token=token,
                config=monitor_cfg,
            )
            ep_outcome = evaluate_episode(monitor, scenario, strategy_name)
            strategy_episodes.append(ep_outcome)
            all_episodes.append(ep_outcome)

        aggregates.append(compute_aggregate_metrics(strategy_name, strategy_episodes))
        family_breakdowns[strategy_name] = _group_by_family(strategy_episodes)

    payload: dict[str, Any] = {
        "suite_config": str(config_path),
        "seed": seed,
        "total_scenarios": len(scenarios),
        "configurations": [
            {
                "strategy_name": name,
                "provenance_mode": mode.value,
                "counterfactual_strategy": cf.value,
            }
            for name, mode, cf in PHASE2_CONFIGURATIONS
        ],
        "aggregate_results": [agg.to_dict() for agg in aggregates],
        "family_breakdowns": family_breakdowns,
        "per_scenario_results": [asdict(ep) for ep in all_episodes],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "phase2_strategy_comparison.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

        with (output_dir / "phase2_per_scenario.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "strategy_name",
                    "scenario_id",
                    "scenario_family",
                    "is_attack",
                    "attack_succeeded",
                    "attack_blocked",
                    "benign_completed",
                    "benign_false_intervention",
                    "final_sink_decision",
                    "intermediate_denials",
                    "false_positive_actions",
                    "execution_latency_ms",
                    "final_decision_reason",
                ],
            )
            writer.writeheader()
            for ep in all_episodes:
                writer.writerow(
                    {
                        "strategy_name": ep.strategy_name,
                        "scenario_id": ep.scenario_id,
                        "scenario_family": ep.scenario_family,
                        "is_attack": ep.is_attack,
                        "attack_succeeded": ep.attack_succeeded,
                        "attack_blocked": ep.attack_blocked,
                        "benign_completed": ep.benign_completed,
                        "benign_false_intervention": ep.benign_false_intervention,
                        "final_sink_decision": ep.final_sink_decision,
                        "intermediate_denials": ep.intermediate_denials,
                        "false_positive_actions": ep.false_positive_actions,
                        "execution_latency_ms": ep.execution_latency_ms,
                        "final_decision_reason": ep.final_decision_reason,
                    }
                )

        agg_fields = list(aggregates[0].to_dict().keys())
        with (output_dir / "phase2_aggregate_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=agg_fields)
            writer.writeheader()
            for agg in aggregates:
                writer.writerow(agg.to_dict())

    return payload


if __name__ == "__main__":
    summary = run_phase2_comparison()
    print(f"Executed {summary['total_scenarios']} scenarios (seed={summary['seed']}):")
    for agg in summary["aggregate_results"]:
        asr = f"{agg['attack_success_rate']:.3f}" if agg["attack_success_rate"] is not None else "N/A"
        bcr = (
            f"{agg['benign_task_completion_rate']:.3f}"
            if agg["benign_task_completion_rate"] is not None
            else "N/A"
        )
        fpr = f"{agg['action_fpr']:.3f}" if agg["action_fpr"] is not None else "N/A"
        fnr = f"{agg['action_fnr']:.3f}" if agg["action_fnr"] is not None else "N/A"
        print(
            f"  {agg['strategy_name']:18s} | ASR={asr} | BenignComp={bcr} | "
            f"ActionFPR={fpr} | ActionFNR={fnr} | MeanLat={agg['mean_latency_ms']} ms"
        )
