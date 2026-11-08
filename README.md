# Research Experiment: Accuracy and Computational Cost

This repository studies how task performance changes as computational budgets increase and whether marginal improvements diminish. The research is finalized from verified local experiments, selected checkpoint evaluations, and published literature. One incomplete semi-supervised sweep was continued from its saved epoch-5 checkpoints to the already registered 30-epoch budget, using its existing seeds and data splits; no new seeds, datasets, benchmark protocols, or hardware studies were added. The continuation is recorded in `results/run_logs/` and its original checkpoints are preserved.

## Final status

The 50-objective matrix records **46/50 (92%)**: 20 conventional objectives, 11 LLM configuration objectives with completed full WikiText-2 raw-v1 test endpoints, and 15 literature-only objectives. The LLM count is per model endpoint; multi-model full downstream-benchmark coverage and selected checkpoint-trajectory subtasks remain incomplete. GPT-2 XL has no scored evaluation, and Pythia-2.8B/6.9B/12B were not evaluated because their FP32 weights exceed this Mac’s safe memory budget. Literature completion means the published-study review is complete to the extent sources report data; it does not mean local reproduction or that every source reports every field. Results across paradigms are not directly comparable, and the evidence does not establish a universal diminishing-returns law or cross-paradigm efficiency ranking.

## Verified evidence at a glance

- **Canonical local index:** 3,368 rows. This heterogeneous source index is not a count of independent experiments; it includes checkpoints, repeated exports, historical/superseded evidence, and current measurements.
- **Analysis:** 4,864 grouped metric summaries, 3,167 paired within-seed budget transitions, and 219 held-out metric groups. Fifteen repeated K-Means records were excluded from its curve analysis, while raw records remain unchanged.
- **Supervised learning:** MLP, LeNet-style CNN, ResNet-18, and the genuine ViT-Tiny-dimension implementation have three-seed curves through 30 epochs on the fixed 10K Fashion-MNIST split. The repository also retains a separate three-layer, width-96 compact transformer proxy. MLP and LeNet have matched five-epoch comparisons at 30K and 58K actual training examples.
- **Unsupervised and semi-supervised learning:** K-Means/GMM, DBSCAN, Autoencoder/VAE, LabelPropagation, pseudo-labeling, Mean Teacher, FixMatch, and MixMatch are documented with their native budgets and metrics. Pseudo-labeling and its matched labeled-only baseline now cover five label fractions, three seeds, and 30 epochs. The transformer is a compact proxy; SSL methods are task adaptations.
- **Reinforcement learning:** DQN, Double DQN, A2C, PPO, and SAC are the original scope. PPO-Pendulum is a separately identified recorded extension. Double DQN rows have `completed` status and `local_measurement` provenance after the derived builder repairs a missing raw CSV field; raw records remain unchanged.
- **Language models:** eleven public checkpoints have full WikiText-2 raw-v1 test evaluations (FP32, Apple M3 MPS, context 1,024, stride 512). Five additional full-test endpoints cover GPT-2 Medium/Large, Pythia-1B/1.4B, and OPT-1.3B. Pythia checkpoint-prefix evaluations and a Pythia-70M full validation-split zero-shot run are reported separately. No model was pretrained locally; smoke tests are not reported as full benchmarks.
- **Literature:** 15 configurations have 300 field-level evidence records and pass the literature-only completion audit. Reported, derived, estimated, uncertain, and unavailable values are distinguished; gaps are explicit and are not treated as zero. Published configurations were not locally reproduced.

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
- `results/processed/literature_completion_audit.csv`: per-entry source and field validation for the completed literature-only objectives.
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
.venv-mac/bin/python scripts/audit_conventional_completion.py
.venv-mac/bin/python scripts/audit_literature_completion.py
.venv-mac/bin/python scripts/generate_completion_matrix.py
.venv-mac/bin/python scripts/build_experiment_run_registry.py
.venv-mac/bin/python scripts/plot_final_figures.py
.venv-mac/bin/python scripts/audit_repository_inventory.py
.venv-mac/bin/python scripts/audit_identifiers.py
.venv-mac/bin/python scripts/build_final_report_pdf.py
.venv-mac/bin/python scripts/validate_project_outputs.py
```

These workflows read saved data and regenerate analysis artifacts only. They do not train models or run inference. See [reproducibility notes](REPRODUCIBILITY.md), [final report](FINAL_RESEARCH_REPORT.md), [statistical analysis](STATISTICAL_ANALYSIS_REPORT.md), and [limitations](REMAINING_WORK.md).
