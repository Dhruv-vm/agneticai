# Denial Attribution Research

A reproducible experimental research project investigating denial-specific causal attribution in tool-calling LLM agents.

> **Research & Novelty Disclaimer:** This repository is an active university conference research project. No experimental results have been collected yet, and no novelty claims are established. Counterfactual attribution, denial-aware provenance, provenance graphs, and multi-step attack detection already exist in prior literature (including ARM, AgentSentry, CausalArmor, FIDES, and PCAS). Our central hypothesis—whether a denial-specific causal attribution mechanism can distinguish genuine denial-induced information leakage from benign subsequent actions while detecting delayed and multi-step laundering attacks—remains **unverified** and subject to empirical falsification.

---

## Research Overview

Tool-calling LLM agents interact with external environments by reading private data, invoking APIs, and triggering side effects. Runtime reference monitors enforce security policies at the tool-execution boundary by allowing or denying tool invocations. However, denying a tool call does not necessarily terminate an information channel: an agent observes the denial outcome (and any associated error message) and may condition subsequent tool calls on what it inferred—a pattern formalized as **causality laundering** (Chinaei, 2026).

This project studies the trade-offs of denial-aware runtime enforcement, starting from a reproducible baseline inspired by the **Agentic Reference Monitor (ARM)**, and systematically investigates whether denial-specific causal attribution can reduce false positives on benign post-denial actions while detecting delayed and multi-step laundering chains.

## Problem Statement

ARM introduces denied actions (`DeniedAction` nodes) into a runtime provenance graph and links them to the immediately following tool call via a `COUNTERFACTUAL` edge. Because an LLM's internal reasoning is opaque to the enforcement layer, ARM's temporal-adjacency heuristic is intentionally conservative and exhibits structural trade-offs acknowledged in the source paper:

1. **Over-approximation (False Positives):** Legitimate subsequent actions that happen to follow a denied tool call may be flagged or blocked due to temporal adjacency alone.
2. **Under-approximation (Delayed & Multi-Step Laundering):** Delayed or multi-step laundering chains—where denial-derived information is transformed across intermediate actions or delayed beyond the immediate next call—require deeper investigation.
3. **Limited Evaluation Scope:** The initial ARM evaluation relies on three manually constructed attack scenarios, without large-scale false-positive evaluation or benchmark-scale experiments with frontier LLMs.
4. **Security–Utility–Overhead Trade-off:** The balance between security enforcement, benign task completion, and runtime/replay overhead is not yet systematically quantified.

## Motivation

If runtime security monitors block too aggressively after a denial, benign agent workflows break; if they only track immediate temporal adjacency, multi-step or delayed denial-feedback leakage may evade detection. Establishing a controlled, reproducible testbed allows us to measure both failure modes quantitatively before proposing or claiming any new attribution mechanism.

## Research Objectives

1. **Reproduce a Transparent Baseline (Phase 1):** Implement a deterministic, well-documented baseline inspired by ARM (tool-call interception, denied-action logging, provenance graph construction, counterfactual edges, trust propagation, and layered policy decisions), explicitly documenting all simplifications relative to the reference paper.
2. **Construct Controlled Scenarios (Phase 2):** Implement deterministic attack and benign scenarios (`E1`–`E5`) covering direct denial leakage, benign post-denial actions, delayed leakage, multi-step laundering, and repeated denial probes.
3. **Investigate Denial-Specific Attribution (Phase 3):** Explore whether targeted denial-signal intervention mechanisms can measurably improve precision/recall over conservative temporal heuristics without prohibitive overhead.
4. **Evaluate with Tool-Calling LLMs (Phase 4–5):** Run repeated experiments with actual tool-calling LLM agents, perform ablation and error analyses, and reassess novelty against prior work.

## Research Questions

- **RQ1 (Precision vs. Recall of Temporal Heuristics):** To what extent does conservative temporal-adjacency counterfactual linking (as in ARM) cause false positives on benign post-denial actions while missing delayed or multi-step denial laundering?
- **RQ2 (Denial-Specific Causal Attribution):** Can a denial-specific causal attribution / intervention mechanism distinguish genuine denial-induced information leakage from unrelated benign actions after a denial?
- **RQ3 (Delayed & Multi-Step Detection):** How effectively does denial-specific attribution detect delayed leakage (`E3`), multi-step laundering (`E4`), and repeated denial probes (`E5`) compared to ARM-style graph provenance and flat provenance baselines?
- **RQ4 (Utility & Overhead Trade-off):** What is the impact on benign task completion rate, runtime enforcement latency, and counterfactual replay overhead across deterministic and LLM-in-the-loop evaluations?

## Existing Research Foundation

- **Primary Foundation:** Mohammad Hossein Chinaei, *"Causality Laundering: Denial-Feedback Leakage in Tool-Calling LLM Agents"*, arXiv:2604.04035 (April 2026). Introduces the causality laundering threat model and the **Agentic Reference Monitor (ARM)**, which models denied tool calls as first-class `DeniedAction` nodes with `COUNTERFACTUAL` edges in a provenance graph and enforces a 4-layer deterministic policy pipeline (`L1` Hard Boundaries, `L2G` Graph-Aware Provenance, `L3` Schema-Derived Constraints, `L4` Manual Policy) over a 5-level integrity lattice.
- **Related Areas:** Information-flow control for agents (e.g., FIDES), capability/secure-by-design architectures (e.g., CaMeL), graph-based policy enforcement (e.g., PCAS), and replay/ablation-based causal attribution for indirect prompt injection (e.g., AgentSentry, CausalArmor). See [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) for detailed comparisons and overlap tracking.

## Proposed Approach

*(Planned — Phase 3, not yet implemented)*

After establishing and benchmarking the ARM-inspired baseline in Phases 1–2, we will investigate whether selective denial-signal intervention (e.g., controlled counterfactual replay or targeted ablation of denial feedback at the agent boundary) can test whether a downstream tool call's occurrence or arguments causally depend on the observed denial outcome rather than mere temporal proximity. This remains a research hypothesis to be tested against ARM and replay-based baselines.

## System Architecture

### Planned Pipeline (Phases 1–3)

1. **Tool-Call Interception (`src/monitor/`):** Mediates every tool invocation before execution and records returned data or denial events.
2. **Provenance Graph (`src/monitor/`):** NetworkX-backed directed graph tracking `Call`, `Data`, `DataField`, and `DeniedAction` nodes with `DIRECT_OUTPUT`, `INPUT_TO`, `FIELD_OF`, and `COUNTERFACTUAL` edges, plus integrity-lattice trust propagation.
3. **Policy Engine (`src/monitor/`):** Deterministic allow/deny enforcement pipeline with structured `PolicyDecision` records.
4. **Counterfactual Replay / Intervention (`src/replay/`):** *(Phase 3)* Mechanism to evaluate causal dependence on denial feedback.
5. **Simulated Tools & Agent Harness (`src/tools/`, `src/agent/`):** Deterministic synthetic tools (no real external side effects) and, in Phase 4, LLM agent integration.
6. **Evaluation Framework (`src/evaluation/`):** Action-level and episode-level metrics computation.

### Implementation Status

| Component | Status | Notes |
|---|---|---|
| Project structure, docs, env, smoke tests | **Implemented (Phase 0)** | Complete |
| Structured event/graph/policy models (`src/monitor/models.py`) | **Implemented (Phase 1)** | Pydantic v2 models + 5-level `TrustLevel` `IntEnum` |
| NetworkX provenance graph (`src/monitor/provenance.py`) | **Implemented (Phase 1)** | Nodes, edges, `MinTrust`, counterfactual chains |
| ARM-inspired baseline monitor (`src/monitor/arm_baseline.py`) | **Implemented (Phase 1)** | `L1`, `L2` vs `L2G`, `L3`, `L4`; simplifications documented |
| Scenario fixtures & comparison runner (`src/evaluation/`) | **Implemented (Phase 2)** | `metrics.py`, `scenarios.py`, `runner.py`; 19-scenario `E1`–`E5` suite in `configs/phase2_scenarios.json` |
| Attack & benign scenario config (`configs/phase2_scenarios.json`) | **Implemented (Phase 2)** | 12 attack scenarios (`E1`, `E3`, `E4`, `E5`) + 7 benign controls (`E2`) |
| Counterfactual attribution engine (`src/attribution/`) | **Implemented (Phase 3)** | `models.py`, `intervention.py`, `replay.py`, `comparison.py`, `attributor.py`, `policy_adapter.py` |
| Paired attribution evaluation (`src/evaluation/phase3_runner.py`) | **Implemented (Phase 3)** | 23 paired scenarios (`configs/phase3_paired_scenarios.json`); outputs to `results/` |
| Independent agent & sandbox (`src/agent/`) | **Implemented (Phase 4)** | `state.py`, `environment.py`, `agent.py` (`ReactivePolicyAgent`, optional `OllamaAgentAdapter`), `executor.py` |
| Conservative attributor (`src/attribution/attributor_v2.py`) | **Implemented (Phase 4)** | Separates unrelated divergence (`AMBIGUOUS`) and independent trusted sources |
| 60-scenario dev/held-out evaluation (`src/evaluation/phase4_runner.py`) | **Implemented (Phase 4)** | 20 dev + 40 held-out scenarios with strict ground-truth isolation |

## Technology Stack

- **Python:** 3.12
- **Dependency Management:** `uv`
- **Provenance Graphs:** `networkx`
- **Data Models & Validation:** `pydantic` (v2)
- **Testing:** `pytest`
- **Later Phases (not yet installed):** `ollama` or API-based LLM clients (Phase 4), `fastapi` only if a service interface becomes necessary.

## Installation Instructions

Prerequisites: Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
cd denial-attribution-research
uv sync --python 3.12
```

## Environment Setup

Copy the example environment file if configuring optional settings or (in Phase 4+) LLM endpoints:

```bash
cp .env.example .env
```

Do **not** commit `.env` or place real credentials in configuration files or experiment logs.

## How to Run Tests

```bash
uv run pytest -v
```

Currently, the test suite (`tests/test_smoke.py`, `tests/test_models_and_trust.py`, `tests/test_provenance_graph.py`, `tests/test_arm_baseline.py`, `tests/test_phase2_evaluation.py`, `tests/test_phase2_audit.py`, `tests/test_phase3_attribution.py`, `tests/test_phase3_audit.py`, `tests/test_phase4_agent_execution.py`) runs 43 deterministic unit, scenario, audit, attribution, and independent-agent tests.

## How to Execute Experiments

Run the Phase 2 deterministic strategy comparison:

```bash
uv run python -m src.evaluation.runner
```

Run the Phase 3 paired-execution denial attribution evaluation:

```bash
uv run python -m src.evaluation.phase3_runner
```

Run the Phase 4 independent agent execution & attribution evaluation (`20` dev + `40` held-out scenarios):

```bash
uv run python -m src.evaluation.phase4_runner
```

This writes structured outputs to `results/phase4_evaluation_results.json`, `results/phase4_heldout_per_scenario.csv`, and `results/phase4_baseline_comparison.csv`. Every executed experiment must be logged in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) with its configuration, inputs, outputs, and saved results in `results/`.

## Project Structure

```text
denial-attribution-research/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── docs/
│   ├── RESEARCH_PLAN.md
│   └── EXPERIMENT_LOG.md
├── src/
│   ├── __init__.py
│   ├── agent/
│   ├── tools/
│   ├── monitor/
│   ├── replay/
│   └── evaluation/
├── scenarios/
│   ├── direct_denial/
│   ├── benign_after_denial/
│   ├── delayed_leakage/
│   └── multi_step_laundering/
├── tests/
│   └── test_smoke.py
├── results/
├── configs/
└── scripts/
```

## Current Development Status

- **Current Phase:** Phase 4 (Independent Agent Execution & Denial-Feedback Attribution) — complete.
- **Next Milestone:** Phase 5 — Statistical analysis, ablations, live-LLM trials via `OllamaAgentAdapter`, and novelty reassessment for conference paper drafting.

## Team Handover Instructions

This repository is maintained by two researchers:

1. **Before starting work:** Run `git status` and `git pull` (when using a remote) to inspect any existing changes. Never overwrite another researcher's work without reviewing diffs.
2. **Before implementing new features:** Read [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) for architecture, data models, and strict research rules.
3. **After completing experiments:** Record every run in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) using the template provided, and store raw/summary outputs under `results/` with reproducible configs in `configs/`.
4. **After meaningful milestones:** Update this `README.md` (status table) and commit changes with clear, descriptive Git commit messages.
