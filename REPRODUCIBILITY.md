# Reproducibility notes

## Environments

The original Windows project environment and DirectML artifacts are preserved. The expanded macOS work used isolated `.venv-mac/`, Python 3.12, and the package set recorded in `MACOS_EXECUTION_LOG.md`. PyTorch 2.15.0 used Apple MPS where selected; scikit-learn and Gymnasium runs used CPU. The LLM harness run used lm-evaluation-harness 0.4.13, Transformers 5.22.1, and Accelerate 1.15.0.

## Data, splits, and checkpoints

Raw data rows are under `results/raw/`; configuration JSON is under `results/configurations/`; model/optimizer checkpoints and environment snapshots are under `results/checkpoints/`; processing scripts preserve raw records in the canonical index. Dataset and split identities are recorded per run. Fashion-MNIST runs use fixed local splits and hold official test labels for endpoint evaluation. A requested 60K train run reserves 2K validation examples, yielding 58K optimization examples.

## Language-model evaluation

Six selected public checkpoints were run locally on the complete WikiText-2 raw-v1 test text at FP32, context 1024, stride 512, batch size one. `results/llm/macos_full_wikitext2_evaluations.csv` contains source revision, dataset text hash, token and byte counts, timing, precision, and memory notes. `results/llm/expanded_llm_evaluations.csv` records released Pythia checkpoint prefix evaluations and derived step-token metadata. Harness runs retain exact JSON configs, versions, aggregates, sample logs, validation split counts, and task versions in `results/llm/harness_full/` and `results/llm/harness/`.

## Rebuilding derived artifacts

From the project root:

```sh
.venv-mac/bin/python scripts/build_canonical_results.py
.venv-mac/bin/python scripts/analyze_strengthened_runs.py
.venv-mac/bin/python scripts/build_literature_evidence.py
.venv-mac/bin/python scripts/generate_completion_matrix.py
.venv-mac/bin/python scripts/validate_project_outputs.py
```

Do not overwrite raw results with processed summaries. Repeated measures within one seed are paired across budgets; they are not independent replicates. See `MACOS_SETUP.md`, `MACOS_EXECUTION_LOG.md`, `EXPERIMENTAL_METHODOLOGY.md`, and `FINAL_INDEPENDENT_AUDIT.md` for platform-specific limits and unresolved scope.
