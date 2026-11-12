# Experiment Log

This file records all controlled experiments executed in the **Denial Attribution Research** project.

**Rules:**
- Do **not** populate entries with fabricated, estimated, or placeholder results.
- Every executed experiment must reference a saved configuration in `configs/` and saved outputs in `results/`.
- Distinguish clearly between observed measurements and interpretations.

---

## Experiment Entry Template

Copy this template for each new experiment (`EXP-001`, `EXP-002`, ...):

```markdown
## EXP-XXX: <Short Title>

- **Experiment ID:** EXP-XXX
- **Date:** YYYY-MM-DD
- **Research Objective:** <What this experiment is designed to measure or validate>
- **Hypothesis:** <Falsifiable prediction before running the experiment>
- **Configuration:** `<path/to/config.yaml or json>` (include key parameters and random seed)
- **Dataset / Scenario:** <e.g., E1, E2, E3, E4, E5, or benchmark subset>
- **Model Used:** <e.g., Deterministic trace driver (no LLM), or exact model ID/version>
- **Baseline:** <e.g., Flat L2, ARM-inspired L2G>

### Procedure
1. <Step-by-step execution commands / protocol>

### Results
- <Link to saved result file in `results/` and raw observed summary>

### Metrics
| Metric | Value | Notes |
|---|---|---|
| Attack Success Rate (ASR) | | Episode-level |
| Benign Task Completion Rate | | Episode-level |
| Detection Precision | | Action-level |
| Detection Recall | | Action-level |
| False Positive Rate (FPR) | | Action-level |
| False Negative Rate (FNR) | | Action-level |
| Attribution Accuracy | | Action-level |
| Runtime Enforcement Latency | | Median / p95 (ms) |
| Counterfactual Replay Overhead | | Latency / tokens |
| Provenance Coverage | | Graph coverage |

### Observations
- <Objective observations from the recorded outputs>

### Conclusion
- <Whether the hypothesis was supported or falsified, and next action>

### Reproducibility Information
- **Git Commit:** `<commit-hash>`
- **Python Version:** `3.12.x`
- **Command:** `<exact command executed>`
- **Result Artifact:** `results/<filename>`
```

---

## Recorded Experiments

*No experiments have been executed yet (Phase 0 setup complete).*
