# Reproducibility and analysis workflow

## Evidence boundary

Raw records under `results/raw/`, `results/llm/`, `results/literature/`, `results/configurations/`, and saved evaluator outputs are the evidence sources. Raw data are read-only inputs to this workflow. The commands below rebuild derived indexes, summaries, visualizations, and inventory only; they do not import or invoke model training/inference functions.

## Recorded environments and provenance

The Windows DirectML environment and its outputs remain preserved as historical records. Later macOS analyses used `.venv-mac`, Python 3.12, PyTorch 2.15.0, and Apple MPS where selected; classical and RL runs used CPU. The LLM harness used lm-evaluation-harness 0.4.13, Transformers 5.22.1, and Accelerate 1.15.0. Exact package, device, checkpoint revision, data hash, and protocol fields are retained per record when available. Do not pool Windows/DirectML and Mac/MPS times as one controlled hardware comparison.

## Rebuild derived outputs

From the repository root:

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
.venv-mac/bin/python scripts/validate_project_outputs.py
```

`analyze_llm_full_eval.py` reads already saved evaluation tables and does not evaluate models. `plot_final_figures.py` reads processed or raw result tables and saves 25 figures as PNG and SVG under `results/figures/final/`. `audit_repository_inventory.py` writes `docs/DATA_INVENTORY.md` and machine-readable inventories. `analyze_strengthened_runs.py` excludes later repeated K-Means checkpoint-reuse rows, uses epoch-5-only dataset-size endpoints, pairs adjacent budgets within seed and setting, retains Double DQN `local_measurement` outputs, and documents the ResNet timing adjustment. Raw rows are preserved.

## Statistical reproducibility

Grouped summaries include (n), mean, standard deviation, median, minimum, maximum, and 95% t intervals where (n>1). Marginal tables pair checkpoints only inside the same model/seed/split/setting. Current output row counts are 4,864 summaries, 3,167 paired transitions, and 219 endpoint metric groups. These are analysis rows, not numbers of independent experiments.

## Integrity and completion

Run `scripts/validate_project_outputs.py` after rebuilding outputs. The 50-objective denominator remains unchanged. Current matrix status is 46/50: 20 conventional objectives, 11 full-test LLM endpoints, and 15 literature reviews. Four LLM endpoints and ancillary multi-model benchmark/trajectory subtasks remain incomplete. The run registry and canonical index are catalogs, not independent observation counts. See `docs/DATA_QUALITY_REGISTER.md` and `docs/PROJECT_AUDIT.md` for anomalies, exclusions, count reconciliation, and limitations. The pseudo-label extension log records the sequential continuation from saved epoch-5 checkpoints to the original 30-epoch schedule; it added no seeds or split changes.
