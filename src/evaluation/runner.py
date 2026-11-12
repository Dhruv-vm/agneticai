"""Baseline scenario evaluation runner comparing L2 and L2G."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.evaluation.scenarios import (
    ScenarioFixture,
    build_default_token,
    execute_scenario,
    get_phase1_scenarios,
)
from src.monitor.arm_baseline import (
    ARMMonitor,
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


if __name__ == "__main__":
    results_dir = Path("results")
    eval_rows = run_comparison(output_dir=results_dir)
    for r in eval_rows:
        status = "OK" if r.correct else "MISS"
        print(
            f"[{status}] {r.engine_config:3s} | {r.scenario_id:25s} | "
            f"expected={r.expected_outcome:5s} actual={r.actual_decision:5s} | "
            f"{r.execution_latency_ms:.3f} ms | {r.decision_reason}"
        )
