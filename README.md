# Research Experiment: Accuracy and Computational Cost

This repository studies how task performance changes as computational budgets increase and whether the additional gain diminishes. The experimental phase is closed. This final audit used existing records only; it conducted no training, inference, benchmark runs, data collection, or hardware testing.

## Final status

The strict, unchanged completion framework records **0 of 50 registered objectives fully complete** (20 conventional methods, 15 LLM configurations, and 15 literature configurations). That count describes objective completion, not the amount of evidence collected. The repository contains extensive partial and completed runs, but no registered objective met all criteria. The current conclusion is descriptive: gains vary by method, task, and budget; the evidence does not establish a universal diminishing-returns law or a cross-paradigm compute-efficiency ranking.

## Verified evidence at a glance

- **Canonical local index:** 2,612 rows, consisting of 2,548 conventional raw-result rows and 64 selected LLM evaluation rows. It is a heterogeneous index, not 2,612 independent experiments.
- **Analysis:** 3,614 grouped metric summaries, 2,417 paired within-seed budget transitions, and 169 held-out metric groups. Fifteen repeated K-Means records were excluded from its curve analysis, while raw records remain unchanged.
- **Supervised learning:** MLP, LeNet-style CNN, ResNet-18, and the genuine ViT-Tiny-dimension implementation have three-seed curves through 30 epochs on the fixed 10K Fashion-MNIST split. The repository also retains a separate three-layer, width-96 compact transformer proxy. MLP and LeNet have matched five-epoch comparisons at 30K and 58K actual training examples.
- **Unsupervised and semi-supervised learning:** K-Means/GMM, DBSCAN, Autoencoder/VAE, LabelPropagation, pseudo-labeling, Mean Teacher, FixMatch, and MixMatch are documented with their native budgets and metrics. The transformer is a compact proxy; SSL methods are task adaptations.
- **Reinforcement learning:** DQN, Double DQN, A2C, PPO, and SAC are the original scope. PPO-Pendulum is a separately identified recorded extension. Double DQN rows have `completed` status and `local_measurement` provenance after the derived builder repairs a missing raw CSV field; raw records remain unchanged.
- **Language models:** six public checkpoints have full WikiText-2 raw-v1 test evaluations (FP32, Apple M3 MPS, context 1,024, stride 512). Pythia checkpoint-prefix evaluations and a Pythia-70M full validation-split zero-shot run are reported separately. No model was pretrained locally; smoke tests are not reported as full benchmarks.
- **Literature:** 15 configurations have 300 field-level evidence records: 172 reported, 8 derived, 16 estimated, and 104 uncertain. Published configurations were not locally reproduced.

## Key result and interpretation

At five epochs, three-seed test accuracy was 0.8537 (SD 0.0081) for MLP at 30K training examples and 0.8689 (SD 0.0008) at 58K. LeNet was 0.8784 (SD 0.0035) and 0.8989 (SD 0.0035). These fixed-split observations are consistent with higher accuracy at the larger sizes, but do not establish a general data-scaling law. Full WikiText-2 test perplexities ranged from 44.9334 (Pythia-70M) to 15.4583 (Pythia-410M) in the Pythia family, but model size here does not measure original pretraining compute.

The separate ViT-Tiny-dimension implementation has 30-epoch test accuracy 0.7621 across three seeds (SD 0.0616; 95% t CI [0.6090, 0.9152]); this wide interval reflects high seed variability in the observed runs.

## Repository guide

- `results/raw/`: source experimental CSVs. Do not overwrite these records.
- `results/llm/`: full-corpus checkpoint evaluations, checkpoint-prefix trajectories, benchmark outputs, and smoke-test records.
- `results/literature/`: published model ledger and field-level evidence with source locations.
- `results/processed/`: canonical index, audit inventory, statistical tables, marginal transitions, exclusions, and figure catalogue data.
- `results/figures/final/`: 25 publication-style figures in PNG and vector SVG formats.
- `scripts/`: separate read-only analysis, plotting, inventory, and validation workflows. Experimental entry points are retained but are not needed for the final audit.
- `FINAL_RESEARCH_REPORT.md` and `reports/FINAL_RESEARCH_REPORT.pdf`: final research report.
- `FINAL_COMPLETION_MATRIX.md`: 50-objective status and separately listed PPO-Pendulum extension.
- `DATA_INVENTORY.md`, `FIGURE_CATALOGUE.md`, and `PROJECT_AUDIT.md`: file/data inventory, figure provenance, and consistency findings.
- `RESULT_PROVENANCE.md` and `DATA_QUALITY_REGISTER.md`: source hierarchy, canonical count reconciliation, analytical eligibility, and documented anomalies.
- `IDENTIFIER_AUDIT.md` and `IDENTIFIER_RECONCILIATION.csv`: ID scope checks, generated evaluation IDs, and corrected derived mappings.

## Rebuild the derived audit artifacts

Using the recorded `.venv-mac` environment from the repository root:

```sh
.venv-mac/bin/python scripts/build_canonical_results.py
.venv-mac/bin/python scripts/build_verified_dataset.py
.venv-mac/bin/python scripts/analyze_strengthened_runs.py
.venv-mac/bin/python scripts/analyze_llm_full_eval.py
.venv-mac/bin/python scripts/build_literature_evidence.py
.venv-mac/bin/python scripts/plot_final_figures.py
.venv-mac/bin/python scripts/audit_repository_inventory.py
.venv-mac/bin/python scripts/validate_project_outputs.py
```

These workflows read saved data and regenerate analysis artifacts only. They do not complete missing experiments. See [reproducibility notes](REPRODUCIBILITY.md), [final report](FINAL_RESEARCH_REPORT.md), [statistical analysis](STATISTICAL_ANALYSIS_REPORT.md), and [limitations and gaps](REMAINING_WORK.md).
