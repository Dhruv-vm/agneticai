"""Phase 5 canonical five-tier source identity and agent foundation tests."""

import pytest

from src.agent.agent import OllamaAgentAdapter
from src.agent.environment import EnvironmentState
from src.agent.executor import AgentExecutor
from src.agent.state import AgentMemory, AgentTask, ProposedAction
from src.attribution.canonical import compare_logical_sources, resolve_logical_source
from src.attribution.models import FeedbackCondition
from src.monitor.arm_baseline import ARMMonitor, CapabilityToken
from src.monitor.models import TrustLevel


def _mon_with_node(
    sid: str,
    call_id: str,
    tool: str,
    args: dict,
    data_id: str,
    val,
    trust: TrustLevel,
    source_id: str | None = None,
    input_ids: list[str] | None = None,
) -> ARMMonitor:
    m = ARMMonitor(sid, CapabilityToken(sid, frozenset({tool, "read_document", "transform_text"})))
    if input_ids:
        for iid in input_ids:
            if not m.graph.has_node(iid):
                m.graph.record_call_node(f"pre_{iid}", "read_document", metadata={"arguments": {"path": "/tmp/src.txt"}})
                m.graph.record_tool_output(f"pre_{iid}", iid, "raw", TrustLevel.TOOL_TRUSTED)
    m.graph.record_call_node(call_id, tool, input_node_ids=input_ids, metadata={"arguments": args})
    m.graph.record_tool_output(call_id, data_id, val, trust)
    if source_id is not None:
        m.graph.get_node(data_id).metadata["source_id"] = source_id
    return m


def test_canonicalization_all_required_cases() -> None:
    """Verify all 11 canonicalization & identity-comparison cases."""
    # 1. Same source, different branch physical IDs
    ma = _mon_with_node("1a", "branch_a_s0", "read_document", {"path": "/tmp/a.txt"}, "d_branch_a_s0", "hi", TrustLevel.TOOL_TRUSTED, source_id="doc::a")
    mb = _mon_with_node("1b", "branch_b_s0", "read_document", {"path": "/tmp/a.txt"}, "d_branch_b_s0", "hi", TrustLevel.TOOL_TRUSTED, source_id="doc::a")
    la, lb = resolve_logical_source("d_branch_a_s0", ma), resolve_logical_source("d_branch_b_s0", mb)
    assert la.logical_id == lb.logical_id
    assert compare_logical_sources(la, lb).status == "EQUIVALENT"

    # 2. Different sources, same content
    mc = _mon_with_node("3c", "branch_b_s0", "read_document", {"path": "/tmp/c.txt"}, "d_branch_b_s0", "hi", TrustLevel.TOOL_TRUSTED, source_id="doc::c")
    lc = resolve_logical_source("d_branch_b_s0", mc)
    cmp_diff = compare_logical_sources(la, lc)
    assert cmp_diff.status == "DIFFERENT"
    assert cmp_diff.same_logical_source is False
    assert cmp_diff.same_content is True

    # 3. Same tool and arguments, different source IDs
    md = _mon_with_node("3d", "branch_b_s0", "read_document", {"path": "/tmp/a.txt"}, "d_branch_b_s0", "hi", TrustLevel.TOOL_TRUSTED, source_id="doc::other_mount")
    ld = resolve_logical_source("d_branch_b_s0", md)
    assert compare_logical_sources(la, ld).status == "DIFFERENT"

    # 4. Same source, changed content version
    mb_v2 = _mon_with_node("2b", "branch_b_s0", "read_document", {"path": "/tmp/a.txt"}, "d_branch_b_s0", "hi_v2", TrustLevel.TOOL_TRUSTED, source_id="doc::a")
    cmp_v2 = compare_logical_sources(la, resolve_logical_source("d_branch_b_s0", mb_v2))
    assert cmp_v2.status == "CONTENT_CHANGED"
    assert cmp_v2.same_logical_source is True
    assert cmp_v2.same_content is False

    # 5. Case 5 FIX: Same tool & args, different underlying content & trust (no source_id)
    m5a = _mon_with_node("5a", "s0", "read_document", {"path": "/tmp/dyn.txt"}, "d_a", "v1", TrustLevel.TOOL_TRUSTED)
    m5b = _mon_with_node("5b", "s0", "read_document", {"path": "/tmp/dyn.txt"}, "d_b", "SECRET", TrustLevel.TOOL_UNTRUSTED)
    l5a, l5b = resolve_logical_source("d_a", m5a), resolve_logical_source("d_b", m5b)
    assert l5a.logical_id != l5b.logical_id
    assert compare_logical_sources(l5a, l5b).status == "DIFFERENT"
    assert compare_logical_sources(l5a, l5b).same_trust is False

    # 6. Missing source_id with matching tool/args/content -> UNRESOLVED (no invented certainty)
    m5a_copy = _mon_with_node("5a2", "s0", "read_document", {"path": "/tmp/dyn.txt"}, "d_a2", "v1", TrustLevel.TOOL_TRUSTED)
    cmp_missing_sid = compare_logical_sources(l5a, resolve_logical_source("d_a2", m5a_copy))
    assert cmp_missing_sid.status == "UNRESOLVED"
    assert cmp_missing_sid.same_logical_source is None
    assert cmp_missing_sid.same_content is True

    # 7. Missing registered node -> UNRESOLVED
    l_missing = resolve_logical_source("missing_node", ma)
    assert l_missing.identity_certainty == "UNRESOLVED"
    assert compare_logical_sources(la, l_missing).status == "UNRESOLVED"

    # 8. Missing content digest -> same_content is None
    m_none = _mon_with_node("n", "s0", "read_document", {"path": "/tmp/n.txt"}, "d_n", None, TrustLevel.TOOL_TRUSTED, source_id="doc::n")
    ln = resolve_logical_source("d_n", m_none)
    assert ln.content_digest is None
    assert compare_logical_sources(la, ln).same_content is None

    # 9. Nested and reordered arguments
    m6a = _mon_with_node("6a", "c0", "search_documents", {"a": 1, "b": {"x": 2, "y": 3}}, "d0", [], TrustLevel.TOOL_TRUSTED)
    m6b = _mon_with_node("6b", "c0", "search_documents", {"b": {"y": 3, "x": 2}, "a": 1}, "d0", [], TrustLevel.TOOL_TRUSTED)
    assert resolve_logical_source("d0", m6a).logical_id == resolve_logical_source("d0", m6b).logical_id

    # 10. Transformation lineage
    m_tr = _mon_with_node("tr", "s1", "transform_text", {"text": "raw", "mode": "base64"}, "d_tr", "cmF3", TrustLevel.TOOL_TRUSTED, input_ids=["d_raw"])
    ltr = resolve_logical_source("d_tr", m_tr)
    assert "read_document" in ltr.lineage
    assert compare_logical_sources(la, ltr).same_lineage is False

    # 11. Repeated calls to same source_id
    m_rep = _mon_with_node("rep", "c0", "read_document", {"path": "/tmp/a.txt"}, "d0", "hi", TrustLevel.TOOL_TRUSTED, source_id="doc::a")
    m_rep.graph.record_call_node("c1", "read_document", metadata={"arguments": {"path": "/tmp/a.txt"}})
    m_rep.graph.record_tool_output("c1", "d1", "hi", TrustLevel.TOOL_TRUSTED)
    m_rep.graph.get_node("d1").metadata["source_id"] = "doc::a"
    cmp_rep = compare_logical_sources(resolve_logical_source("d0", m_rep), resolve_logical_source("d1", m_rep))
    assert cmp_rep.status == "EQUIVALENT"


def test_executor_filters_hallucinated_input_ids_and_unknown_tools() -> None:
    """Verify AgentExecutor handles hallucinated input_node_ids & unknown tools safely."""
    class _HallucinatingAgent:
        def __init__(self) -> None:
            self.step = 0

        def propose_next_action(self, task: AgentTask, memory: AgentMemory, available_tools: dict[str, str]) -> ProposedAction:
            self.step += 1
            if self.step == 1:
                return ProposedAction(tool_name="read_document", arguments={"path": "/tmp/a.txt"}, input_node_ids=["nonexistent_data_0"])
            if self.step == 2:
                return ProposedAction(tool_name="hallucinated_tool", arguments={})
            return ProposedAction(terminate=True, termination_reason="done")

    task = AgentTask(task_id="t_hal", goal="test", allowed_tools=["read_document", "hallucinated_tool"], max_steps=5)
    env = EnvironmentState(documents={"/tmp/a.txt": {"content": "ok", "trust": "TOOL_TRUSTED"}})
    res = AgentExecutor().run_branch(task, env, _HallucinatingAgent(), "branch_a", FeedbackCondition.ACTUAL)  # type: ignore[arg-type]

    assert res.trace.steps[0].decision == "ALLOW"
    assert res.monitor.graph.get_node("d_branch_a_s0").metadata["source_id"] == "doc::/tmp/a.txt"
    assert res.trace.steps[1].executed is False
    assert "Unknown simulated tool" in res.memory.observations[1].feedback


def test_ollama_adapter_disabled_by_default_and_reproducible_config() -> None:
    """Verify OllamaAgentAdapter is disabled by default and never falls back to scripts."""
    adapter = OllamaAgentAdapter(model="qwen3.5:4b", enabled=False)
    assert adapter.config_dict()["protocol"] == "structured_json_action"
    with pytest.raises(RuntimeError, match="disabled by default"):
        adapter.propose_next_action(AgentTask(task_id="t", goal="g"), AgentMemory(), {"get_time": "time"})
