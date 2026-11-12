"""Sandboxed simulated tool environment with zero real-world side effects."""

from __future__ import annotations

import ast
import base64
import operator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.monitor.models import TrustLevel

_SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval_expr(expr: str) -> int | float:
    """Safely evaluate basic integer/float arithmetic expressions."""
    node = ast.parse(expr, mode="eval").body

    def _eval(n: ast.AST) -> int | float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_BIN_OPS:
            return _SAFE_BIN_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -_eval(n.operand)
        raise ValueError(f"Unsupported arithmetic expression: {expr}")

    return _eval(node)


class EnvironmentState(BaseModel):
    """Initial deterministic state for a sandboxed environment."""

    model_config = ConfigDict(extra="forbid")

    documents: dict[str, dict[str, Any]] = Field(default_factory=dict)
    current_time: str = "2026-11-12T12:00:00Z"
    seed: int = 42


class SandboxedEnvironment:
    """Isolated simulated tool environment (no real external side effects)."""

    TOOL_DESCRIPTIONS: dict[str, str] = {
        "get_time": "Return the current simulated UTC timestamp.",
        "search_documents": "Search synthetic document titles/content by keyword.",
        "read_document": "Read a synthetic document by path.",
        "calculate": "Evaluate a basic arithmetic expression.",
        "transform_text": "Transform text (summary, base64, hex, or tag).",
        "send_email": "Simulated outbound email sink (in-memory only).",
        "export_data": "Simulated external data export sink (in-memory only).",
    }

    def __init__(self, initial_state: EnvironmentState) -> None:
        self._state = initial_state.model_copy(deep=True)
        self.sent_emails: list[dict[str, Any]] = []
        self.exported_records: list[dict[str, Any]] = []
        self.execution_log: list[dict[str, Any]] = []

    def execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[Any, TrustLevel, dict[str, TrustLevel]]:
        """Execute a simulated tool and return (output, default_trust, field_overrides)."""
        if tool_name == "get_time":
            result: Any = self._state.current_time
            trust = TrustLevel.TOOL_TRUSTED
            overrides: dict[str, TrustLevel] = {}

        elif tool_name == "search_documents":
            query = str(arguments.get("query", "")).lower()
            matches = sorted(
                path
                for path, doc in self._state.documents.items()
                if query in path.lower() or query in str(doc.get("content", "")).lower()
            )
            result = matches
            trust = TrustLevel.TOOL_TRUSTED
            overrides = {}

        elif tool_name == "read_document":
            path = str(arguments.get("path", ""))
            doc = self._state.documents.get(path)
            if doc is None:
                result = {"error": "document_not_found", "path": path}
                trust = TrustLevel.TOOL_TRUSTED
                overrides = {}
            else:
                result = doc.get("content", "")
                raw_trust = doc.get("trust", "TOOL_TRUSTED")
                trust = getattr(TrustLevel, raw_trust, TrustLevel.TOOL_TRUSTED)
                overrides = {}

        elif tool_name == "calculate":
            expr = str(arguments.get("expression", "0"))
            result = str(_safe_eval_expr(expr))
            trust = TrustLevel.TOOL_TRUSTED
            overrides = {}

        elif tool_name == "transform_text":
            text = str(arguments.get("text", ""))
            mode = str(arguments.get("mode", "summary"))
            if mode == "base64":
                result = base64.b64encode(text.encode("utf-8")).decode("ascii")
            elif mode == "hex":
                result = text.encode("utf-8").hex()
            elif mode == "tag":
                result = f"status:{text}"
            else:
                result = f"summary:{text}"
            trust = TrustLevel.TOOL_TRUSTED
            overrides = {}

        elif tool_name == "send_email":
            record = {
                "to": str(arguments.get("to", "")),
                "body": str(arguments.get("body", "")),
            }
            self.sent_emails.append(record)
            result = {"status": "simulated_sent", **record}
            trust = TrustLevel.TOOL_TRUSTED
            overrides = {}

        elif tool_name == "export_data":
            record = {
                "destination": str(arguments.get("destination", "")),
                "payload": str(arguments.get("payload", "")),
            }
            self.exported_records.append(record)
            result = {"status": "simulated_exported", **record}
            trust = TrustLevel.TOOL_TRUSTED
            overrides = {}

        else:
            raise ValueError(f"Unknown simulated tool: '{tool_name}'.")

        self.execution_log.append(
            {"tool_name": tool_name, "arguments": arguments, "result": result}
        )
        return result, trust, overrides
