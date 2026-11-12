# Collaborator Handoff Guide — Denial Attribution Research

> **Read this document first** before modifying code, running experiments, or updating documentation. For the complete chronological evidence record and all quantitative tables, read [`output.md`](output.md).

---

## 1. Project Overview

**Project Name:** Denial Attribution Research  
**Research Foundation:** Mohammad Hossein Chinaei, *"Causality Laundering: Denial-Feedback Leakage in Tool-Calling LLM Agents"*, arXiv:2604.04035 (April 2026) — introducing the **Agentic Reference Monitor (ARM)**.

**Problem Statement:** Runtime reference monitors enforce security at the tool-execution boundary by allowing or denying tool invocations. However, denying a tool call does not necessarily close the information channel: an agent observes the denial outcome (and any denial reason) and may infer protected facts (e.g., file existence or sensitivity class) and exfiltrate them through a later seemingly benign tool call (*causality laundering*). ARM's conservative `IMMEDIATE_NEXT` counterfactual heuristic causes high false positives on benign post-denial actions, while target-scoped lexical provenance (`SINK_SCOPED`) misses paraphrased, transformed, or aggregated leakage.

**Finalized Research Question:**
> *"Can an AI-agent security monitor identify whether downstream protected actions are influenced by denial feedback, including paraphrased, transformed, delayed, aggregated, and mixed-provenance feedback, while avoiding false attribution of unrelated legitimate actions?"*

### Core Attribution Classifications
- `DENIAL_SENSITIVE`: The action (or its upstream dependency) diverges under denial-feedback neutralization with observable sink/transform sensitivity. *(Indicates sensitivity under tested conditions, not definitive causal proof.)*
- `DENIAL_INDEPENDENT`: The action and its canonical provenance remain invariant under denial-feedback neutralization, or both branches rely on independent trusted sources.
- `AMBIGUOUS`: Observable evidence cannot distinguish denial sensitivity from unrelated explanations (e.g., stochastic wording variation or non-separable multi-denial interactions).
- `NOT_EVALUABLE`: Controlled paired-execution preconditions failed (e.g., missing branch, mismatched initial state/permissions, or non-deterministic replay).

### Non-Negotiable Scientific Principles
- Temporal correlation is **not** causation.
- Trace divergence alone does **not** prove denial influence.
- No observed divergence does **not** prove independence.
- Identical arguments do **not** prove identical information provenance.
- Different arguments do **not** prove different information.
- Attribution and enforcement are separate research problems.
- Synthetic scripted results do **not** establish real-world effectiveness.
- Ambiguous and negative findings must always be preserved.

---

## 2. Current Status

- **Completed Phases:** Phase 0 through Phase 4.3.
- **Latest Verified Committed Milestone:** `7335414` (`test: independently audit denial attribution generalization`) on branch `main`.
- **Phase 5 Status:** **Planned (NOT COMPLETED).** *(Note: Local working tree may contain uncommitted draft/WIP Phase 5 files; do not treat any Phase 5 experiment as completed or validated until executed and logged end-to-end.)*

---

## 3. Repository Structure

```text
denial-attribution-research/
├── HANDOFF.md                  # Primary collaborator onboarding & handoff guide (this file)
├── README.md                   # Project entry point, setup, and quickstart
├── output.md                   # Central chronological research evidence record (Phases 0–4.3)
├── pyproject.toml              # Python 3.12 + uv project & pytest configuration
├── uv.lock                     # Locked dependency graph
├── .env.example                # Optional environment variables template (never commit .env)
├── docs/
│   ├── RESEARCH_PLAN.md        # Methodology, threat model, baselines, and Phase 5 plan
│   └── EXPERIMENT_LOG.md       # Chronological experiment log (EXP-001 through EXP-008)
├── src/
│   ├── monitor/                # ARM-inspired baseline: models.py, provenance.py, arm_baseline.py
│   ├── attribution/            # Counterfactual intervention, replay, comparison, canonical, evidence_model
│   ├── agent/                  # BaseAgent, ReactivePolicyAgent, policies.py, SandboxedEnvironment, AgentExecutor
│   └── evaluation/             # Scenario suites, metrics, runners, and audit scripts (Phases 1–4.3)
├── configs/                    # Versioned deterministic scenario & held-out label JSONs
├── results/                    # Machine-readable JSON/CSV outputs from EXP-001..EXP-008
├── scripts/                    # Dataset generation scripts
└── tests/                      # Deterministic unit, scenario, audit, and regression tests
```

---

## 4. Architecture & Data Flow

```mermaid
flowchart LR
    T["AgentTask + EnvironmentState<br/>(no ground-truth labels)"] --> A["Agent Policy<br/>(src/agent/)"]
    A -->|ProposedAction| M["ARMMonitor<br/>(L1, L2/L2G, L3, L4)<br/>src/monitor/arm_baseline.py"]
    M -->|ALLOW| E["SandboxedEnvironment<br/>(simulated tools)<br/>src/agent/environment.py"]
    M -->|DENY| I["DenialInterventionEngine<br/>(ACTUAL vs NEUTRALIZED)<br/>src/attribution/intervention.py"]
    E -->|Tool Output + Provenance| G["ProvenanceGraph (NetworkX)<br/>src/monitor/provenance.py"]
    E -->|AgentObservation| A
    I -->|Rendered Feedback| A
    M -->|Branch A & Branch B Traces| C["compare_paired_traces +<br/>resolve_logical_source"]
    C --> ATT["EvidenceBasedAttributor<br/>src/attribution/evidence_model.py"]
    ATT -->|AttributionResult / EvidenceExplanation| ADP["AttributionPolicyAdapter<br/>(disabled by default)"]
```

### Key Modules & Responsibilities
- **`src/monitor/`:**
  - `models.py`: `TrustLevel` `IntEnum` (`TOOL_DESC < TOOL_UNTRUSTED < TOOL_TRUSTED < USER_INPUT < SYS_INSTR`), `AgentEvent`, `DenialEvent`, `ProvenanceNode`, `ProvenanceEdge`, `PolicyDecision`.
  - `provenance.py`: NetworkX `ProvenanceGraph` (`Call`, `Data`, `DataField`, `DeniedAction` nodes; `DIRECT_OUTPUT`, `INPUT_TO`, `FIELD_OF`, `COUNTERFACTUAL` edges; `min_trust`, `counterfactual_chains`).
  - `arm_baseline.py`: `ARMMonitor` with `L1` (`HardBoundariesLayer`), `L2` (`FlatProvenanceLayer`), `L2G` (`GraphProvenanceLayer`), `L3` (`SchemaDerivedLayer`), `L4` (`ManualPolicyLayer`), and `CounterfactualStrategy` (`NONE`, `IMMEDIATE_NEXT`, `ALL_SUBSEQUENT`, `SINK_SCOPED`).
- **`src/attribution/`:**
  - `intervention.py`: `DenialInterventionEngine` (`ACTUAL`, `NEUTRALIZED`, `PARAPHRASED`, `MINIMAL_STRUCTURED`).
  - `replay.py` & `comparison.py`: `DeterministicReplayEngine`, `compare_paired_traces`.
  - `canonical.py`: `resolve_logical_source` (canonical logical identity resolution).
  - `evidence_model.py`: `EvidenceBasedAttributor` (7 ablation configs `A`–`G`, `EvidenceExplanation`).
  - `multi_denial.py`: `analyze_multi_denial_ablation` (one-at-a-time denial neutralization).
  - `policy_adapter.py`: `AttributionPolicyAdapter` (`enforce_attribution=False` by default).
- **`src/agent/`:**
  - `state.py`: `AgentTask`, `AgentObservation`, `ProposedAction` (`extra="forbid"`), isolated `AgentMemory`.
  - `environment.py`: `SandboxedEnvironment` (all tools simulated in-memory; zero real side effects).
  - `agent.py` & `policies.py`: `ReactivePolicyAgent`, `ConservativeStoppingAgent`, `IndependentRecoveryAgent`, `DenialInsensitiveAgent`, `MemoryMediatedAgent`, `SourcePrioritizationAgent`, and `OllamaAgentAdapter` (disabled by default).
  - `executor.py`: `AgentExecutor` mediating every tool call through `ARMMonitor`.

---

## 5. Technology Stack & Setup Instructions

- **Python:** `3.12` (`.venv` managed via `uv`)
- **Core Dependencies (`pyproject.toml`):** `networkx>=3.4`, `pydantic>=2.10`, `pytest>=8.3` (dev)
- **Optional Local LLM (Phase 5+):** Ollama (`ollama --version`, local `qwen3.5:4b` if installed)

```bash
cd "/Users/dhruv/dhruv files/agentic ai/denial-attribution-research"
uv sync --python 3.12
uv run pytest -v
```

---

## 6. How to Run Existing Evaluations (Phases 1–4.3)

```bash
# Phase 2: 19-scenario L2 vs L2G counterfactual strategy comparison
uv run python -m src.evaluation.runner

# Phase 3: 23-scenario scripted paired-execution attribution
uv run python -m src.evaluation.phase3_runner

# Phase 3.1: Blind held-out & multi-denial validity audit
uv run python -m src.evaluation.phase3_audit

# Phase 4: 60-scenario (20 dev + 40 held-out) independent agent execution
uv run python -m src.evaluation.phase4_runner

# Phase 4.1: 44-scenario challenge & multi-policy generalization audit
uv run python -m src.evaluation.phase4_audit

# Phase 4.2: Evidence-based attribution, 7-way ablation, and 7 multi-denial cases
uv run python -m src.evaluation.phase4_2_runner

# Phase 4.3: 104-scenario independent 4-class, 10-way ablation, 6-policy & control audit
uv run python -m src.evaluation.phase4_3_runner
```

---

## 7. Summary of Research Results (Phases 0–4.3)

- **Phase 2 (`EXP-002`, `n=19`):** `L2` (`ASR=1.000, Benign=1.000`), `L2G+NONE` (`ASR=1.000, Benign=1.000`), `L2G+IMMEDIATE_NEXT` (`ASR=0.000, Benign=0.143`), `L2G+SINK_SCOPED` (`ASR=0.750, Benign=1.000`).
- **Phase 3 / 3.1 (`EXP-003`, `EXP-004`):** `EXP-003` (`1.000`) was invalidated as evidence of generalization because Branch B was pre-scripted; blind held-out scored `4/6` (`0.667`).
- **Phase 4 / 4.1 (`EXP-005`, `EXP-006`):** Dynamic paired execution exposed `branch_a_`/`branch_b_` ID artifacts (`raw=40/40`, `meaningful=33/40`) and keyword-heuristic false positives (`FAR=0.167` on `CH_BENIGN_SENSITIVE_WORDS_01..04`).
- **Phase 4.2 (`EXP-007`):** Removed keyword matching and introduced `EvidenceBasedAttributor` + `canonical.py`. *(Caveat: Phase 4.2's `1.000` accuracy excluded `gt == "AMBIGUOUS"` from `tp+fp+tn+fn`.)*
- **Phase 4.3 (`EXP-008`, `n=104`, `seed=43127`):** Full 4-class evaluation (`32` `DENIAL_SENSITIVE`, `32` `DENIAL_INDEPENDENT`, `20` `AMBIGUOUS`, `20` `NOT_EVALUABLE`) with `ReactivePolicyAgent`:
  - **4-Class Accuracy:** `0.9615` (`100/104`, 95% Wilson CI `[0.9053, 0.9849]`)
  - **Macro-F1:** `0.9575`, **Balanced Accuracy:** `0.9500`, **FAR on `DENIAL_INDEPENDENT`:** `0.000`
  - **Coverage:** `0.6538` (`68/104`), **Selective Accuracy:** `0.9412` (`64/68`, 95% Wilson CI `[0.8583, 0.9769]`)
  - **Ablations:** Removing alternative explanations (`H`) drops accuracy to `0.8077`; removing canonical identity (`J`) drops accuracy to `0.7115` (`FAR = 0.8125`). Removing temporal/provenance (`F`, `G`) had no impact on this synthetic set.

---

## 8. Known Issues & Limitations

1. **Canonicalization Case 5 Defect:** In Phase 4.3 (`results/phase4_3_canonicalization.csv`, `11/12`), `resolve_logical_source()` keyed on `(tool, args)` without a content/trust digest, treating identical `(tool, args)` calls with different underlying content/trust as equivalent.
2. **4 Misclassified `AMBIGUOUS` Cases in Phase 4.3:** Coincidental modulo-3 greeting collisions produced identical traces across branches, causing `E_COMBINED` to predict `DENIAL_INDEPENDENT`.
3. **Lack of Live LLM Evaluation:** All committed results (`EXP-001`–`EXP-008`) use deterministic Python policies; LLM non-determinism, prompt-injection sensitivity, and replay token/latency costs are not yet evaluated.
4. **Unproven Temporal & Provenance Ablation Impact:** In Phase 4.3, `F_COMBINED_NO_TEMPORAL` and `G_COMBINED_NO_PROVENANCE` tied with `E_COMBINED` (`0.9615`).
5. **`SourcePrioritizationAgent` Partial Reactivity:** Scored `0.7404` (`FAR = 0.3214`) in Phase 4.3 because it falls back to denial feedback when no independent `data_id` is available yet.

---

## 9. Next Steps — Phase 5 (Planned, NOT COMPLETED)

1. Fix canonical source identity (`Case 5` defect) using explicit environment `source_id` + content digest + trust provenance, and validate with unit tests.
2. Verify local Ollama setup (`ollama serve`, `ollama list`) and run a live LLM smoke test.
3. Build a genuine LLM-driven tool-using agent (structured action generation or native tool calling) with zero label leakage.
4. Conduct paired `ACTUAL` vs. `NEUTRALIZED` trials across multiple stochastic seeds, including sham/negative controls.
5. Evaluate prompt-injection robustness, enforcement/attribution baselines, and latency/token cost.

*Do not claim any Phase 5 results until experiments are executed, verified, and logged.*

---

## 10. Instructions for the Next Collaborator

1. Read `HANDOFF.md` (this file) first, then `output.md` for full evidence.
2. Inspect `src/` and `tests/` before modifying any code.
3. Run `uv run pytest -v` before and after every change.
4. Never modify historical `results/phase1_*` through `results/phase4_3_*` files.
5. Keep runtime agent/environment inputs strictly separated from ground-truth labels.
6. Document every new experiment in `docs/EXPERIMENT_LOG.md` and `output.md`.
7. Commit in meaningful milestones and push only verified work.

---

## 11. Immediate First Task

**Inspect and finalize the canonicalization defect fix (Phase 4.3 Case 5)** in `src/attribution/canonical.py` (ensuring physical ID, logical `source_id`, content digest, trust provenance, and transformation lineage are separated) and verify all canonicalization tests pass **before** running full live-LLM experiments.
