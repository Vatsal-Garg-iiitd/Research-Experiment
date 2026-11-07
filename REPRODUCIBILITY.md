# Reproducibility guide

## Environment

The core project environment is `.venv` on Windows/Python 3.12. Install the pinned top-level packages with `python -m pip install -r requirements.txt`; check compatibility with `python -m pip check`. Optional DirectML and lm-eval environments are separate. LLM harness commands are recorded in `EXECUTION_LOG.md`; the successful Windows invocation uses ordered `PYTHONPATH` entries to reuse the tested project pandas and lm-eval dependencies.

## Data and runs

Fashion-MNIST downloads to `results/cache/` using torchvision. Keep the raw CSVs immutable when possible; experiment scripts append incremental records. Neural and RL checkpoints/configurations are under `results/checkpoints/` and `results/configurations/`. Some scripts resume from saved checkpoint state; inspect each script’s CLI and checkpoint logic before resuming a partial run. Do not assume every script supports resumption equally.

## Regeneration

From the project root, use:

```powershell
.\.venv\Scripts\python.exe scripts\build_canonical_results.py
.\.venv\Scripts\python.exe scripts\analyze_strengthened_runs.py
.\.venv\Scripts\python.exe scripts\extract_lm_harness_results.py
.\.venv\Scripts\python.exe scripts\analyze_llm_full_eval.py
.\.venv\Scripts\python.exe scripts\generate_completion_matrix.py
.\.venv\Scripts\python.exe scripts\validate_project_outputs.py
```

The canonical file preserves each source-row JSON and source path in addition to normalized common fields. Statistical scripts regenerate their summaries and plots from saved rows. Harness JSON and per-example outputs are under `results/llm/harness/`. Literature CSV is hand-curated with source notes and is not generated from local runs.

## Important limitations

These commands reproduce analyses from the existing artifacts; they do not complete missing model budgets, missing seeds, unavailable downstream full benchmark runs, missing NPU runs, or primary-source validation. The completion matrix and `REMAINING_WORK.md` define those outstanding items. Results from pilot/smoke subsets must remain labeled partial.
