# Research Experiment: Accuracy and Computational Cost

This repository examines how model performance changes as computational budgets increase, including whether marginal gains diminish. It brings together verified local studies in supervised, unsupervised, semi-supervised, and reinforcement learning; checkpoint evaluations for public language models; and a source-checked literature review.

## Project status

The per-configuration completion score is **46/50 (92%)**: 20 conventional objectives, 11 LLM configurations with completed full WikiText-2 raw-v1 test endpoints, and 15 literature-only objectives. The four LLM configurations without a completed endpoint are GPT-2 XL (interrupted before scoring) and Pythia-2.8B, 6.9B, and 12B (not evaluated because their FP32 weights exceed this Mac’s safe memory budget). The score counts one full-test endpoint as completion of that model’s configuration objective. Multi-model downstream benchmark and additional checkpoint-trajectory coverage remain incomplete and are described separately in the reports.

No language model was pretrained locally. Full-test checkpoint evaluations measure inference on released checkpoints, not their original training compute. Metrics and budgets remain separate by paradigm; the current evidence does not establish a universal diminishing-returns law or a cross-paradigm efficiency ranking.

## Verified results at a glance

- The canonical index contains **3,368 source-located rows**. It catalogs heterogeneous records and is not a count of independent experiments.
- Analysis includes **4,864 grouped metric summaries**, **3,167 within-seed paired budget transitions**, and **219 held-out metric groups**.
- Supervised, unsupervised, semi-supervised, and RL results preserve their native metrics and budgets. Fifteen duplicate K-Means checkpoint-reuse rows remain in raw data and are excluded from the analytical curve.
- Eleven public checkpoints have full WikiText-2 raw-v1 test evaluations using FP32 on Apple M3 MPS, context 1,024, and stride 512. Pythia prefix trajectories, the one full zero-shot harness evaluation, and restricted smoke tests are kept as separate protocols.
- Fifteen literature configurations have completed source-based reviews. Reported, derived, estimated, uncertain, and unavailable fields are classified separately.

## Repository layout

- `configs/` — experiment and model registries.
- `docs/` — methodology, setup, reproducibility, provenance, audits, and the [original project proposal](docs/original_research_proposal.pdf).
- `results/raw/` — preserved source experimental records.
- `results/llm/` — saved language-model checkpoint evaluations, benchmark outputs, and smoke-test records.
- `results/literature/` — published-model ledger and field-level source evidence.
- `results/processed/` — canonical and verified datasets, statistical tables, audits, identifier reconciliation, and run registry.
- `results/figures/final/` — 25 publication-style figures in PNG and SVG formats.
- `reports/` — final research report, matrices, statistical analyses, figure catalogue, and validation reports.
- `scripts/` — analysis, plotting, audit, report, and experiment scripts. Rebuilding derived outputs does not run training or inference.

The root also retains `requirements.txt`, the optional legacy `requirements-directml.txt`, and two execution-log summaries referenced by historical source records.

## Start here

- [Final research report](reports/FINAL_RESEARCH_REPORT.md) and [PDF](reports/FINAL_RESEARCH_REPORT.pdf)
- [Completion matrix](reports/FINAL_COMPLETION_MATRIX.md)
- [LLM evaluation results](reports/LLM_EVALUATION_REPORT.md)
- [Statistical analysis](reports/STATISTICAL_ANALYSIS_REPORT.md)
- [Marginal-gain analysis](reports/MARGINAL_GAIN_ANALYSIS.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Methodology](docs/EXPERIMENTAL_METHODOLOGY.md)
- [Figure catalogue](reports/FIGURE_CATALOGUE.md)
- [Data inventory](docs/DATA_INVENTORY.md)

## Environment and rebuild

The checked-in `requirements.txt` is the platform-neutral pinned dependency list. On macOS, create the isolated environment described in [setup notes](docs/MACOS_SETUP.md); the legacy DirectML requirements are for the historical Windows setup.

```sh
python3.12 -m venv .venv-mac
.venv-mac/bin/python -m pip install --upgrade pip
.venv-mac/bin/python -m pip install -r requirements.txt
.venv-mac/bin/python -m pip install 'accelerate>=1.0,<2.0'
```

From the repository root, run these derived-data commands in `.venv-mac`. They read saved records and regenerate analytical outputs; they do not train models or run inference.

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

The final report explains the experimental setup, analysis, supported conclusions, and limitations. Raw evidence is preserved; incomplete evaluations and literature-field gaps are reported explicitly.
