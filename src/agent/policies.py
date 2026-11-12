"""Additional deterministic agent policies for Phase 4.1 generalization testing.

None of these policies access evaluation ground-truth labels or scripted Branch B traces.
"""

from __future__ import annotations

from typing import Any

from src.agent.agent import BaseAgent
from src.agent.state import AgentMemory, AgentTask, ProposedAction


class ConservativeStoppingAgent(BaseAgent):
    """Terminates immediately whenever any tool invocation is denied."""

    def __init__(self) -> None:
        self._cursor_by_memory: dict[int, int] = {}

    def propose_next_action(
        self,
        task: AgentTask,
        memory: AgentMemory,
        available_tools: dict[str, str],
    ) -> ProposedAction:
        if memory.observations and not memory.observations[-1].allowed:
            return ProposedAction(
                terminate=True,
                termination_reason="ConservativeStoppingAgent halted after denial.",
            )

        mem_key = id(memory)
        directives: list[dict[str, Any]] = list(task.instructions.get("directives", []))
        cursor = self._cursor_by_memory.get(mem_key, 0)
        while cursor < len(directives):
            d = directives[cursor]
            self._cursor_by_memory[mem_key] = cursor + 1
            op = d.get("op")
            if op == "call":
                tool = str(d["tool"])
                if tool not in available_tools:
                    return ProposedAction(terminate=True, termination_reason="Tool unavailable.")
                return ProposedAction(tool_name=tool, arguments=dict(d.get("args", {})))
            if op == "read_optional":
                return ProposedAction(
                    tool_name="read_document",
                    arguments={"path": str(d["path"])},
                )
            cursor += 1
        return ProposedAction(terminate=True, termination_reason="Completed.")


class IndependentRecoveryAgent(BaseAgent):
    """On denial, searches for authorized public documents and completes benignly."""

    def __init__(self) -> None:
        self._cursor_by_memory: dict[int, int] = {}
        self._recovered_by_memory: dict[int, bool] = {}

    def propose_next_action(
        self,
        task: AgentTask,
        memory: AgentMemory,
        available_tools: dict[str, str],
    ) -> ProposedAction:
        mem_key = id(memory)
        if (
            memory.observations
            and not memory.observations[-1].allowed
            and not self._recovered_by_memory.get(mem_key, False)
        ):
            self._recovered_by_memory[mem_key] = True
            if "search_documents" in available_tools:
                return ProposedAction(
                    tool_name="search_documents",
                    arguments={"query": "defaults"},
                )

        directives: list[dict[str, Any]] = list(task.instructions.get("directives", []))
        cursor = self._cursor_by_memory.get(mem_key, 0)
        while cursor < len(directives):
            d = directives[cursor]
            self._cursor_by_memory[mem_key] = cursor + 1
            op = d.get("op")
            if op == "call":
                tool = str(d["tool"])
                if tool not in available_tools:
                    return ProposedAction(terminate=True, termination_reason="Tool unavailable.")
                return ProposedAction(tool_name=tool, arguments=dict(d.get("args", {})))
            if op == "read_optional":
                return ProposedAction(
                    tool_name="read_document",
                    arguments={"path": str(d["path"])},
                )
            cursor += 1
        return ProposedAction(terminate=True, termination_reason="Completed.")


class DenialInsensitiveAgent(BaseAgent):
    """Ignores all denial feedback and executes unconditional directives."""

    def __init__(self) -> None:
        self._cursor_by_memory: dict[int, int] = {}

    def propose_next_action(
        self,
        task: AgentTask,
        memory: AgentMemory,
        available_tools: dict[str, str],
    ) -> ProposedAction:
        mem_key = id(memory)
        directives: list[dict[str, Any]] = list(task.instructions.get("directives", []))
        cursor = self._cursor_by_memory.get(mem_key, 0)
        while cursor < len(directives):
            d = directives[cursor]
            self._cursor_by_memory[mem_key] = cursor + 1
            op = d.get("op")
            if op == "call":
                tool = str(d["tool"])
                if tool not in available_tools:
                    return ProposedAction(terminate=True, termination_reason="Tool unavailable.")
                return ProposedAction(tool_name=tool, arguments=dict(d.get("args", {})))
            if op == "read_optional":
                return ProposedAction(
                    tool_name="read_document",
                    arguments={"path": str(d["path"])},
                )
            cursor += 1
        return ProposedAction(terminate=True, termination_reason="Completed.")
