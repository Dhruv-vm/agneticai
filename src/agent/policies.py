"""Additional deterministic agent policies for Phase 4.1 and 4.3 testing.

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


class MemoryMediatedAgent(BaseAgent):
    """Stores tool feedback/outputs in a memory scratchpad and retrieves them later."""

    def __init__(self) -> None:
        self._cursor_by_memory: dict[int, int] = {}
        self._scratchpad_by_memory: dict[int, dict[str, Any]] = {}

    def propose_next_action(
        self,
        task: AgentTask,
        memory: AgentMemory,
        available_tools: dict[str, str],
    ) -> ProposedAction:
        mem_key = id(memory)
        pad = self._scratchpad_by_memory.setdefault(mem_key, {})
        if memory.observations:
            last = memory.observations[-1]
            pad[f"step_{last.step_index}"] = (
                last.output if last.allowed else last.feedback
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
                args = dict(d.get("args", {}))
                if "memory_key" in d:
                    val = str(pad.get(str(d["memory_key"]), ""))
                    if "hb-2" in val.lower() or "sensitive" in val.lower():
                        args.setdefault("body", "stored_restriction_observed")
                    elif tool == "send_email" and "body" not in args:
                        args["body"] = val or "no_data"
                return ProposedAction(tool_name=tool, arguments=args)
            if op == "read_optional":
                return ProposedAction(
                    tool_name="read_document",
                    arguments={"path": str(d["path"])},
                )
            cursor += 1
        return ProposedAction(terminate=True, termination_reason="Completed.")


class SourcePrioritizationAgent(BaseAgent):
    """Prioritizes independent authorized sources over denial-derived hints."""

    def __init__(self) -> None:
        self._cursor_by_memory: dict[int, int] = {}
        self._data_ids_by_memory: dict[int, list[str]] = {}

    def propose_next_action(
        self,
        task: AgentTask,
        memory: AgentMemory,
        available_tools: dict[str, str],
    ) -> ProposedAction:
        mem_key = id(memory)
        data_ids = self._data_ids_by_memory.setdefault(mem_key, [])
        if memory.observations and memory.observations[-1].allowed:
            did = memory.observations[-1].data_id
            if did and did not in data_ids:
                data_ids.append(did)

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
                args = dict(d.get("args", {}))
                input_ids: list[str] = []
                if d.get("use_last_output") and data_ids:
                    last_id = data_ids[-1]
                    input_ids.append(last_id)
                    if "body_template" in d:
                        args["body"] = d["body_template"].format(
                            val=memory.tool_results.get(last_id, "")
                        )
                return ProposedAction(
                    tool_name=tool, arguments=args, input_node_ids=input_ids
                )
            if op == "if_denial_reveals_sensitive":
                if data_ids:
                    cursor += 1
                    continue
                for fb in memory.denial_feedbacks:
                    if "hb-2" in fb.lower() or "sensitive" in fb.lower():
                        tool = str(d["tool"])
                        args = dict(d.get("args", {}))
                        args.setdefault("body", "restricted resource")
                        return ProposedAction(tool_name=tool, arguments=args)
            if op == "read_optional":
                return ProposedAction(
                    tool_name="read_document",
                    arguments={"path": str(d["path"])},
                )
            cursor += 1
        return ProposedAction(terminate=True, termination_reason="Completed.")
