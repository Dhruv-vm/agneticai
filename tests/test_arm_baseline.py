"""Deterministic tests for ARM baseline enforcement (L2 vs L2G) and policy layers."""

from pathlib import Path

from src.evaluation.runner import run_comparison
from src.evaluation.scenarios import (
    build_default_token,
    execute_scenario,
    get_phase1_scenarios,
)
from src.monitor.arm_baseline import (
    ARMMonitor,
    CapabilityToken,
    CounterfactualStrategy,
    MonitorConfig,
    ProvenanceMode,
)
from src.monitor.models import DecisionVerdict, TrustLevel


def _scenario_by_id(scenario_id: str):
    for scenario in get_phase1_scenarios():
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(scenario_id)


def test_l2_vs_l2g_direct_denial_scenario() -> None:
    """L2 misses direct denial leakage (E1); L2G blocks via counterfactual edge."""
    scenario = _scenario_by_id("E1_DIRECT_DENIAL")

    mon_l2 = ARMMonitor(
        session_id="s_l2_e1",
        capability_token=build_default_token(scenario),
        config=MonitorConfig(provenance_mode=ProvenanceMode.FLAT_L2),
    )
    dec_l2 = execute_scenario(mon_l2, scenario)
    assert dec_l2[0].decision == DecisionVerdict.DENY
    assert dec_l2[1].decision == DecisionVerdict.ALLOW

    mon_l2g = ARMMonitor(
        session_id="s_l2g_e1",
        capability_token=build_default_token(scenario),
        config=MonitorConfig(provenance_mode=ProvenanceMode.GRAPH_L2G),
    )
    dec_l2g = execute_scenario(mon_l2g, scenario)
    assert dec_l2g[0].decision == DecisionVerdict.DENY
    assert dec_l2g[1].decision == DecisionVerdict.DENY
    assert "causality laundering" in dec_l2g[1].reason.lower()
    assert len(mon_l2g.graph.denials) == 2


def test_l2_vs_l2g_transitive_taint_scenario() -> None:
    """L2 misses transitive taint through format_response; L2G blocks via MinTrust."""
    scenario = _scenario_by_id("ARM_TRANSITIVE_TAINT")

    mon_l2 = ARMMonitor(
        session_id="s_l2_tt",
        capability_token=build_default_token(scenario),
        config=MonitorConfig(provenance_mode=ProvenanceMode.FLAT_L2),
    )
    dec_l2 = execute_scenario(mon_l2, scenario)
    assert dec_l2[-1].decision == DecisionVerdict.ALLOW

    mon_l2g = ARMMonitor(
        session_id="s_l2g_tt",
        capability_token=build_default_token(scenario),
        config=MonitorConfig(provenance_mode=ProvenanceMode.GRAPH_L2G),
    )
    dec_l2g = execute_scenario(mon_l2g, scenario)
    assert dec_l2g[-1].decision == DecisionVerdict.DENY
    assert "mintrust" in dec_l2g[-1].reason.lower()


def test_l2_vs_l2g_mixed_provenance_scenario() -> None:
    """L2 misses untrusted email field; L2G blocks via field-level provenance."""
    scenario = _scenario_by_id("ARM_MIXED_PROVENANCE")

    mon_l2 = ARMMonitor(
        session_id="s_l2_mp",
        capability_token=build_default_token(scenario),
        config=MonitorConfig(provenance_mode=ProvenanceMode.FLAT_L2),
    )
    dec_l2 = execute_scenario(mon_l2, scenario)
    assert dec_l2[-1].decision == DecisionVerdict.ALLOW

    mon_l2g = ARMMonitor(
        session_id="s_l2g_mp",
        capability_token=build_default_token(scenario),
        config=MonitorConfig(provenance_mode=ProvenanceMode.GRAPH_L2G),
    )
    dec_l2g = execute_scenario(mon_l2g, scenario)
    assert dec_l2g[-1].decision == DecisionVerdict.DENY
    assert "mintrust" in dec_l2g[-1].reason.lower()


def test_legitimate_actions_allowed_in_both_modes() -> None:
    """Legitimate user-initiated calls and trusted fields pass L2 and L2G."""
    token = CapabilityToken(
        session_id="s_legit",
        allowed_tools=frozenset({"get_contact", "send_email"}),
    )
    for mode in (ProvenanceMode.FLAT_L2, ProvenanceMode.GRAPH_L2G):
        mon = ARMMonitor(
            session_id=f"s_legit_{mode.value}",
            capability_token=token,
            config=MonitorConfig(provenance_mode=mode),
        )
        d1 = mon.evaluate_tool_call("c1", "get_contact", {"query": "Alice"})
        assert d1.decision == DecisionVerdict.ALLOW
        mon.record_tool_result(
            call_event_id="c1",
            data_id="d1",
            value={"name": "Alice", "email": "attacker@evil.test"},
            default_trust=TrustLevel.TOOL_TRUSTED,
            field_trust_overrides={"email": TrustLevel.TOOL_UNTRUSTED},
        )
        # Using only the trusted 'name' field is allowed in both L2 and L2G.
        d2 = mon.evaluate_tool_call(
            "c2",
            "send_email",
            {"to": "colleague@company.test", "body": "Hello Alice"},
            input_node_ids=["d1.name"],
        )
        assert d2.decision == DecisionVerdict.ALLOW


def test_l1_l3_l4_policy_stages_identical_across_modes() -> None:
    """Verify L1 (schema pin, payload), L3 (optional param), and L4 (budget) enforce."""
    schemas = {
        "write_note": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "tag": {"type": "string"},
            },
            "required": ["title"],
        }
    }
    for mode in (ProvenanceMode.FLAT_L2, ProvenanceMode.GRAPH_L2G):
        token = CapabilityToken(
            session_id=f"s_layers_{mode.value}",
            allowed_tools=frozenset({"write_note"}),
            tool_budgets={"write_note": 2},
        )
        mon = ARMMonitor(
            session_id=f"s_layers_{mode.value}",
            capability_token=token,
            config=MonitorConfig(
                provenance_mode=mode,
                counterfactual_strategy=CounterfactualStrategy.NONE,
            ),
            tool_schemas={"write_note": dict(schemas["write_note"])},
        )

        # Valid call passes all layers.
        assert (
            mon.evaluate_tool_call("c1", "write_note", {"title": "Todo"}).decision
            == DecisionVerdict.ALLOW
        )

        # L3 SD-3: optional string parameter > 50 chars denied.
        d_opt = mon.evaluate_tool_call(
            "c2", "write_note", {"title": "Todo", "tag": "x" * 60}
        )
        assert d_opt.decision == DecisionVerdict.DENY
        assert "SD-3" in d_opt.reason

        # L1 HB-5: mutated schema denied.
        mon.context.tool_schemas["write_note"] = {"type": "object", "required": []}
        d_schema = mon.evaluate_tool_call("c3", "write_note", {"title": "Todo"})
        assert d_schema.decision == DecisionVerdict.DENY
        assert "HB-5" in d_schema.reason

        # Restore schema and test L4 budget (1 allowed call so far; 2nd allowed, 3rd denied).
        mon.context.tool_schemas["write_note"] = dict(schemas["write_note"])
        assert (
            mon.evaluate_tool_call("c4", "write_note", {"title": "Second"}).decision
            == DecisionVerdict.ALLOW
        )
        d_budget = mon.evaluate_tool_call("c5", "write_note", {"title": "Third"})
        assert d_budget.decision == DecisionVerdict.DENY
        assert "budget" in d_budget.reason.lower()


def test_counterfactual_strategy_configurable() -> None:
    """Verify CounterfactualStrategy.NONE disables counterfactual edges."""
    scenario = _scenario_by_id("E1_DIRECT_DENIAL")
    mon = ARMMonitor(
        session_id="s_cf_none",
        capability_token=build_default_token(scenario),
        config=MonitorConfig(
            provenance_mode=ProvenanceMode.GRAPH_L2G,
            counterfactual_strategy=CounterfactualStrategy.NONE,
        ),
    )
    decisions = execute_scenario(mon, scenario)
    assert decisions[0].decision == DecisionVerdict.DENY
    assert decisions[1].decision == DecisionVerdict.ALLOW
    assert mon.graph.counterfactual_chains("e1_c2") == []


def test_evaluation_runner_writes_structured_output(tmp_path: Path) -> None:
    """Verify evaluation runner executes all scenarios and writes JSON/CSV."""
    rows = run_comparison(output_dir=tmp_path)
    assert len(rows) == 14  # 7 scenarios x 2 configurations (L2, L2G)
    assert (tmp_path / "phase1_baseline_comparison.json").exists()
    assert (tmp_path / "phase1_baseline_comparison.csv").exists()
