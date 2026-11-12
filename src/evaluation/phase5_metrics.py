"""Phase 5 metrics for live LLM attribution, enforcement, stochasticity, and cost."""

from __future__ import annotations

import statistics
from typing import Any

from src.evaluation.metrics import safe_div
from src.evaluation.phase4_3_metrics import compute_four_class_metrics


def summarize_latencies_and_tokens(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean, median, p95 latencies and token usage summaries."""
    if not rows:
        return {}

    def _stats(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"mean": None, "median": None, "p95": None}
        s = sorted(vals)
        p95 = s[min(len(s) - 1, int(0.95 * (len(s) - 1)))]
        return {
            "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "p95": round(p95, 4),
        }

    llm_ms = [float(r["llm_latency_ms"]) for r in rows]
    mon_ms = [float(r["monitor_latency_ms"]) for r in rows]
    attr_ms = [float(r["attribution_latency_ms"]) for r in rows]
    e2e_ms = [float(r["e2e_latency_ms"]) for r in rows]
    tokens = [int(r["total_tokens"]) for r in rows]
    tool_calls = [int(r["total_tool_calls"]) for r in rows]

    total_tok_sum = sum(tokens)
    total_calls_sum = sum(tool_calls)

    return {
        "llm_latency_ms": _stats(llm_ms),
        "monitor_latency_ms": _stats(mon_ms),
        "attribution_latency_ms": _stats(attr_ms),
        "e2e_latency_ms": _stats(e2e_ms),
        "mean_tokens_per_episode": round(statistics.mean(tokens), 2),
        "total_tokens": total_tok_sum,
        "tokens_per_tool_call": round(total_tok_sum / total_calls_sum, 2) if total_calls_sum > 0 else None,
        "total_retries": sum(int(r["retries"]) for r in rows),
        "total_errors": sum(int(r["errors_count"]) for r in rows),
    }


def summarize_phase5_attribution(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute 4-class attribution and enforcement summary from raw run rows."""
    four_class = compute_four_class_metrics(
        rows, pred_key="predicted_class", gt_key="ground_truth_class"
    )
    attacks = [r for r in rows if r.get("is_attack") is True]
    benigns = [r for r in rows if r.get("is_attack") is False]

    atk_succ = sum(1 for r in attacks if r["adapter_sink_executed"])
    ben_comp = sum(1 for r in benigns if r["benign_completed"])

    asr = safe_div(atk_succ, len(attacks))
    ben_rate = safe_div(ben_comp, len(benigns))
    return {
        **four_class,
        "attack_episodes": len(attacks),
        "benign_episodes": len(benigns),
        "adapter_asr": round(asr, 4) if asr is not None else None,
        "adapter_blocking_rate": round(1.0 - asr, 4) if asr is not None else None,
        "adapter_benign_completion_rate": round(ben_rate, 4) if ben_rate is not None else None,
    }
