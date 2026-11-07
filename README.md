# Compute and performance research project

This repository studies the marginal performance gain from additional computational expenditure. It preserves the question in the proposal and separates local experiments, model-family training trajectories, and literature-derived evidence.

## Current state

The project contains expanded three-seed multi-budget runs for selected conventional methods, six locally evaluated LLM configurations (50K WikiText-2 tokens each), and three LLM configurations evaluated on 100-example lm-evaluation-harness subsets. This remains an incomplete study; do not interpret the subset benchmark scores as full test-set results. The CPU-only `.venv` is preserved. DirectML runs use a separate environment and have operator fallback warnings. The NPU is detected but no execution provider was qualified. See [final research report](FINAL_RESEARCH_REPORT.md), [hardware benchmark report](HARDWARE_BENCHMARK_REPORT.md), [completion matrix](FINAL_COMPLETION_MATRIX.md), and [remaining work](REMAINING_WORK.md).

## Setup

Use Python 3.10 or newer in an environment with the packages in `requirements.txt`. Install the RL Box2D extras only when running LunarLander. Large language models require additional checkpoint storage and RAM/VRAM; start with Pythia-70M and inspect actual resources before increasing model size.

For the DirectML neural feasibility runner, create a separate environment and install `requirements-directml.txt`; `torch-directml` pins its compatible PyTorch version. Keep the main `.venv` for CPU models, RL, and LLM evaluation.

## Structure

- `configs/experiments.yaml`: target methods, budgets, seeds, and LLM checkpoint selection.
- `EXPERIMENT_DESIGN.md`, `LLM_RESEARCH_DESIGN.md`: protocol and comparability rules.
- `results/literature/large_model_literature.csv`: source-attributed published evidence; blanks are unavailable.
- `results/literature/literature_sources.md`: source registry and extraction caveats.
- `results/figures/`: plots generated from the measured pilots and compatible published evidence.
- `results/raw/`, `results/processed/`, `results/llm/`: pilot records, marginal analysis, and language-model evaluation.
- `results/hardware/`: detected capability inventory and CPU/DirectML microbenchmark.
- `FINAL_COMPLETION_MATRIX.md`, `REMAINING_WORK.md`: per-configuration status and concrete gaps.
- `scripts/analyze_results.py`: marginal-gain analysis for measured experiment CSVs.
- `scripts/run_missing_classical.py`, `scripts/run_missing_neural.py`, `scripts/run_missing_rl.py`: sequential one-budget feasibility runs for previously missing conventional models.
- `scripts/run_llm_evaluations.py`: resumable small-model/checkpoint WikiText-2 prefix evaluation; larger weight loads are marked resource-limited.
- `FINAL_RESEARCH_REPORT.md`: updated evidence, limitations, and conclusions; `research_report.md` is the earlier pilot report.

## Experiment record schema

The raw pilot schema records `run_id, paradigm, model, dataset, split_hash, seed, budget_unit, budget_value, train_seconds, cumulative_train_seconds, eval_seconds, metric_name, metric_value, train_steps, parameter_count, peak_ram_bytes, peak_vram_bytes, hardware, status, details`. These pilot rows do not contain a full validation/training metric history; the confirmatory schema should add those fields before expanding the experiments. Store inference measurements separately and use the actual model-specific budget unit (epochs, optimizer steps, algorithm iterations, or environment steps).

## Analysis

Run original pilots with `\.venv\Scripts\python.exe scripts/run_pilots.py`. Run the CPU classical and RL feasibility batches with `.venv\Scripts\python.exe scripts/run_missing_classical.py` and `.venv\Scripts\python.exe scripts/run_missing_rl.py`. Run neural DirectML feasibility batches with `.venv-dml\Scripts\python.exe scripts/run_missing_neural.py`. These scripts append rows to `results/raw/full_suite_results.csv`; they do not implement the full budget/seed schedule. Checkpoint evaluation cost is not its original pretraining cost.

`\.venv\Scripts\python.exe scripts/analyze_results.py results/raw/experiment_results.csv --output results/processed/marginal_results.csv` computes adjacent-budget improvements within each run/model/seed, uses cumulative wall time as the practical cost axis, and reports saturation flags at 0.5%, 1%, and 2% relative thresholds. Use `--lower-is-better` for loss or perplexity. `scripts/plot_pilots.py` plots only collected pilot data and compatible reported literature points. Neither utility trains models or creates missing results.
