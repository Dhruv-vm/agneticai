"""ARM-inspired deterministic baseline enforcement engine.

Implements the four-layer policy pipeline from Chinaei (2026, arXiv:2604.04035)
with a configurable provenance mode selecting either:
- L2 (FlatProvenanceLayer): minimal flat-history grounding baseline.
- L2G (GraphProvenanceLayer): graph-aware provenance with transitive trust
  propagation, field-level provenance, and counterfactual denial edges.

Known simplifications relative to the original ARM prototype are documented in
``docs/RESEARCH_PLAN.md`` (e.g., NetworkX backend instead of rustworkx,
in-process Python interposition instead of MCP proxy, and deterministic
synthetic evaluation).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.monitor.models import (
    AgentEvent,
    DecisionVerdict,
    DenialEvent,
    EventStatus,
    EventType,
    LayerVerdict,
    PolicyDecision,
    TrustLevel,
    utc_now,
)
from src.monitor.provenance import ProvenanceGraph


class ProvenanceMode(str, Enum):
    """Selectable provenance configuration (alternative modes, not sequential)."""

    FLAT_L2 = "L2"
    GRAPH_L2G = "L2G"


class CounterfactualStrategy(str, Enum):
    """Strategy for linking DeniedAction nodes to subsequent tool calls."""

    IMMEDIATE_NEXT = "IMMEDIATE_NEXT"
    ALL_SUBSEQUENT = "ALL_SUBSEQUENT"
    SINK_SCOPED = "SINK_SCOPED"
    NONE = "NONE"


@dataclass(frozen=True)
class CapabilityToken:
    """Immutable session capability token for Layer 4 manual policy."""

    session_id: str
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    tool_budgets: dict[str, int] = field(default_factory=dict)
    blocked_arguments: dict[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Copy mutable dict reference into a read-only proxy-like copy if needed;
        # callers should treat CapabilityToken as immutable.
        object.__setattr__(self, "tool_budgets", dict(self.tool_budgets))
        object.__setattr__(self, "blocked_arguments", dict(self.blocked_arguments))


class MonitorConfig(BaseModel):
    """Configuration for the ARM-inspired baseline monitor."""

    model_config = ConfigDict(extra="forbid")

    provenance_mode: ProvenanceMode = ProvenanceMode.GRAPH_L2G
    counterfactual_strategy: CounterfactualStrategy = CounterfactualStrategy.IMMEDIATE_NEXT
    trust_threshold: TrustLevel = TrustLevel.TOOL_TRUSTED
    max_payload_chars: int = 10_000
    max_calls_per_tool: int = 100
    sensitive_path_patterns: tuple[str, ...] = (
        "/etc/shadow",
        "/etc/passwd",
        "/.ssh/",
        "~/.ssh/",
        "/.aws/",
        "~/.aws/",
        ".env",
    )
    protected_sinks: tuple[str, ...] = (
        "send_email",
        "post_message",
        "write_file",
        "delete_file",
        "transfer_funds",
        "update_record",
        "run_code",
    )


class LayerResult(BaseModel):
    """Result returned by a single policy layer."""

    model_config = ConfigDict(extra="forbid")

    layer_name: str
    verdict: LayerVerdict
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)


@dataclass
class EvaluationContext:
    """Session context passed through the layered policy pipeline."""

    session_id: str
    capability_token: CapabilityToken
    tool_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    pinned_schema_hashes: dict[str, str] = field(default_factory=dict)
    flat_history: list[dict[str, Any]] = field(default_factory=list)


_CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
)


def _iter_string_values(obj: Any) -> list[tuple[str, str]]:
    """Flatten argument dict/list into ``(key_path, string_value)`` pairs."""
    pairs: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            v = obj[k]
            if isinstance(v, str):
                pairs.append((str(k), v))
            elif isinstance(v, (dict, list)):
                pairs.extend(_iter_string_values(v))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            if isinstance(item, str):
                pairs.append((f"[{idx}]", item))
            elif isinstance(item, (dict, list)):
                pairs.extend(_iter_string_values(item))
    return pairs


def _matches_sensitive_path(value: str, patterns: tuple[str, ...]) -> str | None:
    lowered = value.lower()
    for pattern in patterns:
        if pattern.lower() in lowered:
            return pattern
    return None


def _matches_credential(value: str) -> bool:
    return any(regex.search(value) is not None for regex in _CREDENTIAL_PATTERNS)


class HardBoundariesLayer:
    """Layer 1: Unconditional boundary invariants (HB-1 through HB-5)."""

    def evaluate(
        self,
        event: AgentEvent,
        context: EvaluationContext,
        config: MonitorConfig,
    ) -> LayerResult:
        # HB-3: Call-rate limit (checked against count including this call).
        prior_count = context.call_counts.get(event.tool_name, 0)
        if prior_count + 1 > config.max_calls_per_tool:
            return LayerResult(
                layer_name="L1:HardBoundaries",
                verdict=LayerVerdict.DENY,
                reason=f"HB-3: call-rate limit exceeded for tool '{event.tool_name}'.",
                evidence={"rule": "HB-3", "count": prior_count + 1},
            )

        # HB-5: Schema pinning (if schema is provided in context).
        schema = context.tool_schemas.get(event.tool_name)
        if schema is not None:
            canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
            schema_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            pinned = context.pinned_schema_hashes.get(event.tool_name)
            if pinned is None:
                context.pinned_schema_hashes[event.tool_name] = schema_hash
            elif pinned != schema_hash:
                return LayerResult(
                    layer_name="L1:HardBoundaries",
                    verdict=LayerVerdict.DENY,
                    reason=f"HB-5: schema mismatch detected for tool '{event.tool_name}'.",
                    evidence={"rule": "HB-5", "expected": pinned, "actual": schema_hash},
                )

        # HB-1, HB-2, HB-4: Argument string inspections.
        for key, value in _iter_string_values(event.arguments):
            if len(value) > config.max_payload_chars:
                return LayerResult(
                    layer_name="L1:HardBoundaries",
                    verdict=LayerVerdict.DENY,
                    reason=f"HB-1: argument '{key}' exceeds {config.max_payload_chars} chars.",
                    evidence={"rule": "HB-1", "argument": key, "length": len(value)},
                )
            matched_path = _matches_sensitive_path(value, config.sensitive_path_patterns)
            if matched_path is not None:
                return LayerResult(
                    layer_name="L1:HardBoundaries",
                    verdict=LayerVerdict.DENY,
                    reason=f"HB-2: sensitive path pattern '{matched_path}' blocked.",
                    evidence={"rule": "HB-2", "argument": key, "pattern": matched_path},
                )
            if _matches_credential(value):
                return LayerResult(
                    layer_name="L1:HardBoundaries",
                    verdict=LayerVerdict.DENY,
                    reason=f"HB-4: credential pattern detected in argument '{key}'.",
                    evidence={"rule": "HB-4", "argument": key},
                )

        return LayerResult(
            layer_name="L1:HardBoundaries",
            verdict=LayerVerdict.PASS,
            reason="Passed all hard boundary checks.",
        )


class FlatProvenanceLayer:
    """Layer 2 (Baseline): Minimal flat-history provenance check.

    Checks only whether argument values directly match an immediate untrusted
    tool output in flat history. Does NOT perform transitive graph traversal,
    field-level trust resolution, or counterfactual denial tracking.
    """

    def evaluate(
        self,
        event: AgentEvent,
        context: EvaluationContext,
        config: MonitorConfig,
    ) -> LayerResult:
        arg_strings = [val for _, val in _iter_string_values(event.arguments) if val]
        if not arg_strings:
            return LayerResult(
                layer_name="L2:FlatProvenance",
                verdict=LayerVerdict.PASS,
                reason="No string arguments to ground.",
            )

        # Inspect only immediate tool outputs recorded in flat_history.
        # If an argument value is grounded in an immediate prior result whose
        # object-level trust is below threshold, deny; otherwise pass.
        for val in arg_strings:
            for entry in reversed(context.flat_history):
                serialized = entry["serialized_result"]
                if val in serialized:
                    if entry["object_trust"] < config.trust_threshold:
                        return LayerResult(
                            layer_name="L2:FlatProvenance",
                            verdict=LayerVerdict.DENY,
                            reason=(
                                f"Argument value grounded in untrusted tool result "
                                f"'{entry['tool_name']}'."
                            ),
                            evidence={
                                "matched_tool": entry["tool_name"],
                                "trust_level": entry["object_trust"].label,
                            },
                        )
                    # Grounded in an immediately trusted result; flat baseline stops here.
                    break

        return LayerResult(
            layer_name="L2:FlatProvenance",
            verdict=LayerVerdict.PASS,
            reason="Flat provenance check passed.",
        )


class GraphProvenanceLayer:
    """Layer 2G: Graph-aware provenance enforcement using ProvenanceGraph."""

    def evaluate(
        self,
        event: AgentEvent,
        graph: ProvenanceGraph,
        config: MonitorConfig,
    ) -> LayerResult:
        call_trust = graph.min_trust(event.event_id)
        if call_trust < config.trust_threshold:
            return LayerResult(
                layer_name="L2G:GraphProvenance",
                verdict=LayerVerdict.DENY,
                reason=(
                    f"MinTrust({call_trust.label}) below threshold "
                    f"({config.trust_threshold.label})."
                ),
                evidence={
                    "min_trust": call_trust.label,
                    "threshold": config.trust_threshold.label,
                    "ancestors": graph.get_ancestors(event.event_id),
                },
            )

        cf_chains = graph.counterfactual_chains(event.event_id)
        if cf_chains:
            return LayerResult(
                layer_name="L2G:GraphProvenance",
                verdict=LayerVerdict.DENY,
                reason="Causality laundering detected via counterfactual edge.",
                evidence={"counterfactual_chains": cf_chains},
            )

        return LayerResult(
            layer_name="L2G:GraphProvenance",
            verdict=LayerVerdict.PASS,
            reason="Graph-aware provenance check passed.",
            evidence={"min_trust": call_trust.label},
        )


class SchemaDerivedLayer:
    """Layer 3: Auto-generated schema-derived constraints (SD-1 through SD-4)."""

    _SUSPICIOUS_NAMES = frozenset({"note", "comment", "metadata", "description"})
    _FILESYSTEM_NAMES = frozenset({"path", "file", "directory"})
    _NETWORK_NAMES = frozenset({"url", "endpoint", "host"})

    def evaluate(
        self,
        event: AgentEvent,
        context: EvaluationContext,
        config: MonitorConfig,
    ) -> LayerResult:
        schema = context.tool_schemas.get(event.tool_name, {})
        required_params = set(schema.get("required", []))
        flagged_network: list[str] = []

        for key, value in event.arguments.items():
            key_lower = key.lower()
            if isinstance(value, str):
                # SD-1: Suspicious free-text parameter names.
                if key_lower in self._SUSPICIOUS_NAMES:
                    if len(value) > 100 or _matches_credential(value):
                        return LayerResult(
                            layer_name="L3:SchemaDerived",
                            verdict=LayerVerdict.DENY,
                            reason=f"SD-1: suspicious parameter '{key}' violated constraint.",
                            evidence={"rule": "SD-1", "argument": key},
                        )

                # SD-2: Filesystem parameters.
                if key_lower in self._FILESYSTEM_NAMES:
                    matched_path = _matches_sensitive_path(
                        value, config.sensitive_path_patterns
                    )
                    if matched_path is not None:
                        return LayerResult(
                            layer_name="L3:SchemaDerived",
                            verdict=LayerVerdict.DENY,
                            reason=f"SD-2: filesystem parameter '{key}' matched sensitive path.",
                            evidence={"rule": "SD-2", "argument": key, "pattern": matched_path},
                        )

                # SD-3: Optional string parameters (when schema defines properties/required).
                if schema and key not in required_params and len(value) > 50:
                    return LayerResult(
                        layer_name="L3:SchemaDerived",
                        verdict=LayerVerdict.DENY,
                        reason=f"SD-3: optional parameter '{key}' exceeds 50-char limit.",
                        evidence={"rule": "SD-3", "argument": key, "length": len(value)},
                    )

            # SD-4: Network parameters logged for audit.
            if key_lower in self._NETWORK_NAMES:
                flagged_network.append(key)

        return LayerResult(
            layer_name="L3:SchemaDerived",
            verdict=LayerVerdict.PASS,
            reason="Passed schema-derived checks.",
            evidence={"sd4_network_params": flagged_network} if flagged_network else {},
        )


class ManualPolicyLayer:
    """Layer 4: Operator-defined capability-token rules."""

    def evaluate(
        self,
        event: AgentEvent,
        context: EvaluationContext,
    ) -> LayerResult:
        token = context.capability_token
        if token.allowed_tools and event.tool_name not in token.allowed_tools:
            return LayerResult(
                layer_name="L4:ManualPolicy",
                verdict=LayerVerdict.DENY,
                reason=f"Tool '{event.tool_name}' is not permitted by capability token.",
                evidence={"tool": event.tool_name},
            )

        budget = token.tool_budgets.get(event.tool_name)
        if budget is not None:
            prior_count = context.call_counts.get(event.tool_name, 0)
            if prior_count + 1 > budget:
                return LayerResult(
                    layer_name="L4:ManualPolicy",
                    verdict=LayerVerdict.DENY,
                    reason=f"Tool '{event.tool_name}' exceeded capability budget ({budget}).",
                    evidence={"tool": event.tool_name, "budget": budget},
                )

        blocked_vals = token.blocked_arguments.get(event.tool_name, frozenset())
        for key, value in _iter_string_values(event.arguments):
            for blocked in blocked_vals:
                if blocked in value:
                    return LayerResult(
                        layer_name="L4:ManualPolicy",
                        verdict=LayerVerdict.DENY,
                        reason=f"Argument '{key}' matched manual blocklist entry.",
                        evidence={"argument": key, "blocked": blocked},
                    )

        return LayerResult(
            layer_name="L4:ManualPolicy",
            verdict=LayerVerdict.PASS,
            reason="Passed manual policy checks.",
        )


class ARMMonitor:
    """ARM-inspired runtime reference monitor supporting L2 and L2G configurations."""

    def __init__(
        self,
        session_id: str,
        capability_token: CapabilityToken,
        config: MonitorConfig | None = None,
        tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or MonitorConfig()
        self.graph = ProvenanceGraph()
        self.context = EvaluationContext(
            session_id=session_id,
            capability_token=capability_token,
            tool_schemas=tool_schemas or {},
        )
        self.events: list[AgentEvent] = []
        self.decisions: list[PolicyDecision] = []
        self._pending_denial_ids: list[str] = []
        self._all_denial_ids: list[str] = []

        self._l1 = HardBoundariesLayer()
        self._l2 = FlatProvenanceLayer()
        self._l2g = GraphProvenanceLayer()
        self._l3 = SchemaDerivedLayer()
        self._l4 = ManualPolicyLayer()

    def evaluate_tool_call(
        self,
        event_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        input_node_ids: list[str] | None = None,
        timestamp: datetime | None = None,
    ) -> PolicyDecision:
        """Evaluate a proposed tool call through L1 -> (L2|L2G) -> L3 -> L4."""
        ts = timestamp or utc_now()
        event = AgentEvent(
            event_id=event_id,
            session_id=self.context.session_id,
            timestamp=ts,
            event_type=EventType.TOOL_CALL,
            tool_name=tool_name,
            arguments=arguments,
            status=EventStatus.PENDING,
            parent_event_ids=list(input_node_ids or []),
        )
        self.events.append(event)

        # Register Call node and counterfactual edges before policy evaluation.
        self.graph.record_call_node(
            call_id=event_id,
            tool_name=tool_name,
            input_node_ids=input_node_ids,
            metadata={"arguments": arguments},
            timestamp=ts,
        )
        self._attach_counterfactual_edges(
            target_call_id=event_id,
            tool_name=tool_name,
            arguments=arguments,
            input_node_ids=input_node_ids or [],
            timestamp=ts,
        )

        layer_results: list[LayerResult] = []
        l1_res = self._l1.evaluate(event, self.context, self.config)
        layer_results.append(l1_res)
        if l1_res.verdict == LayerVerdict.DENY:
            return self._finalize_denial(event, l1_res, layer_results, ts)

        if self.config.provenance_mode == ProvenanceMode.FLAT_L2:
            l2_res = self._l2.evaluate(event, self.context, self.config)
        else:
            l2_res = self._l2g.evaluate(event, self.graph, self.config)
        layer_results.append(l2_res)
        if l2_res.verdict == LayerVerdict.DENY:
            return self._finalize_denial(event, l2_res, layer_results, ts)

        l3_res = self._l3.evaluate(event, self.context, self.config)
        layer_results.append(l3_res)
        if l3_res.verdict == LayerVerdict.DENY:
            return self._finalize_denial(event, l3_res, layer_results, ts)

        l4_res = self._l4.evaluate(event, self.context)
        layer_results.append(l4_res)
        if l4_res.verdict == LayerVerdict.DENY:
            return self._finalize_denial(event, l4_res, layer_results, ts)

        # Allowed: increment call count and emit ALLOW decision.
        self.context.call_counts[tool_name] = self.context.call_counts.get(tool_name, 0) + 1
        event.status = EventStatus.ALLOWED
        decision = PolicyDecision(
            decision_id=f"dec_{event_id}",
            event_id=event_id,
            decision=DecisionVerdict.ALLOW,
            reason="All policy layers passed.",
            evidence={
                "provenance_mode": self.config.provenance_mode.value,
                "layers": [r.model_dump() for r in layer_results],
            },
            timestamp=ts,
        )
        self.decisions.append(decision)
        return decision

    def record_tool_result(
        self,
        call_event_id: str,
        data_id: str,
        value: Any,
        default_trust: TrustLevel,
        field_trust_overrides: dict[str, TrustLevel] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Record an allowed tool call's returned data in both graph and flat history."""
        ts = timestamp or utc_now()
        self.graph.record_tool_output(
            call_id=call_event_id,
            data_id=data_id,
            value=value,
            default_trust=default_trust,
            field_trust_overrides=field_trust_overrides,
            timestamp=ts,
        )
        call_node = self.graph.get_node(call_event_id)
        serialized = (
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else str(value)
        )
        self.context.flat_history.append(
            {
                "call_id": call_event_id,
                "data_id": data_id,
                "tool_name": call_node.source_id,
                "object_trust": default_trust,
                "serialized_result": serialized,
            }
        )

    def _extract_target_tokens(self, arguments: dict[str, Any]) -> set[str]:
        """Extract non-trivial alphanumeric tokens (>=4 chars) from tool arguments."""
        tokens: set[str] = set()
        for _, value in _iter_string_values(arguments):
            for part in re.split(r"[^a-zA-Z0-9_]+", value.lower()):
                if len(part) >= 4:
                    tokens.add(part)
        return tokens

    def _attach_counterfactual_edges(
        self,
        target_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        input_node_ids: list[str],
        timestamp: datetime,
    ) -> None:
        strategy = self.config.counterfactual_strategy
        if strategy == CounterfactualStrategy.NONE:
            return
        if strategy == CounterfactualStrategy.IMMEDIATE_NEXT:
            while self._pending_denial_ids:
                denial_id = self._pending_denial_ids.pop(0)
                self.graph.add_counterfactual_edge(
                    denial_id=denial_id,
                    target_call_id=target_call_id,
                    timestamp=timestamp,
                )
        elif strategy == CounterfactualStrategy.ALL_SUBSEQUENT:
            for denial_id in self._all_denial_ids:
                self.graph.add_counterfactual_edge(
                    denial_id=denial_id,
                    target_call_id=target_call_id,
                    timestamp=timestamp,
                )
        elif strategy == CounterfactualStrategy.SINK_SCOPED:
            # Only protected sinks are candidates for counterfactual linking.
            if tool_name not in self.config.protected_sinks:
                return
            # If the sink call has explicit independent provenance inputs, do not
            # heuristically attribute it to an earlier unrelated denial.
            if input_node_ids:
                return
            sink_tokens = self._extract_target_tokens(arguments)
            if not sink_tokens:
                return
            for denial_id in self._all_denial_ids:
                denial_node = self.graph.get_node(denial_id)
                source_call = self.graph.get_node(denial_node.source_id)
                denied_args = source_call.metadata.get("arguments", {})
                denied_tokens = self._extract_target_tokens(denied_args)
                if sink_tokens & denied_tokens:
                    self.graph.add_counterfactual_edge(
                        denial_id=denial_id,
                        target_call_id=target_call_id,
                        timestamp=timestamp,
                    )

    def _finalize_denial(
        self,
        event: AgentEvent,
        failing_layer: LayerResult,
        layer_results: list[LayerResult],
        timestamp: datetime,
    ) -> PolicyDecision:
        event.status = EventStatus.DENIED
        denial_id = f"denial_{event.event_id}"
        feedback = f"DENIED ({failing_layer.layer_name}): {failing_layer.reason}"
        denial = DenialEvent(
            denial_id=denial_id,
            source_event_id=event.event_id,
            denied_action=event.tool_name,
            denial_reason=failing_layer.reason,
            feedback_observed=feedback,
            timestamp=timestamp,
        )
        self.graph.record_denial(denial)
        self._pending_denial_ids.append(denial_id)
        self._all_denial_ids.append(denial_id)

        decision = PolicyDecision(
            decision_id=f"dec_{event.event_id}",
            event_id=event.event_id,
            decision=DecisionVerdict.DENY,
            reason=failing_layer.reason,
            evidence={
                "provenance_mode": self.config.provenance_mode.value,
                "failing_layer": failing_layer.layer_name,
                "denial_id": denial_id,
                "layers": [r.model_dump() for r in layer_results],
            },
            timestamp=timestamp,
        )
        self.decisions.append(decision)
        return decision
