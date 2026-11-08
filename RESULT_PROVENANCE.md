# Result provenance and canonical data guide

## Source hierarchy

Raw records in `results/raw/*.csv` and saved evaluator records in `results/llm/` are the primary machine-readable evidence. Per-run configurations are in `results/configurations/*.json`; checkpoints are listed under `results/checkpoints/`; execution details and known interruptions are in `EXECUTION_LOG.md` and `MACOS_EXECUTION_LOG.md`. Published facts are recorded in `results/literature/literature_field_evidence.csv` with evidence class, URL, locator, and notes. The file inventories list paths, sizes, columns/keys, missingness, status, and candidate identifiers.

## Canonical counts and meaning

- **430 rows:** historical execution-log milestone without a matching saved snapshot; exact membership cannot be reconstructed.
- **619 rows:** recoverable historical index at commit `afc488a`, from an earlier source state.
- **3,368 rows:** current common index of heterogeneous source records. It is not a count of independent experiments; repeated budgets and non-completed statuses are retained.
- **5,250 rows:** run/evidence registry across 29 source paths, a distinct catalog with a different unit of counting.
- **4,864 / 3,167 / 219:** current grouped metric summaries / paired within-seed marginal transitions / final held-out metric groups.

Older 430/619 canonical and 458/279/84 statistical counts refer to earlier source or analysis generations. The previous 2,612/3,614/2,417/169 snapshot predates the completed pseudo-label schedule. Historical totals are not added to current results; raw evidence is never replaced by derived totals.

## Rebuilding the data products

```sh
.venv-mac/bin/python scripts/build_canonical_results.py
.venv-mac/bin/python scripts/build_verified_dataset.py
.venv-mac/bin/python scripts/analyze_strengthened_runs.py
.venv-mac/bin/python scripts/analyze_llm_full_eval.py
.venv-mac/bin/python scripts/build_literature_evidence.py
.venv-mac/bin/python scripts/build_experiment_run_registry.py
```

`canonical_results.csv` is a common index that embeds each original source row as JSON and includes stable `canonical_id`/`evaluation_id`, model/dataset IDs, experiment/run identifiers, seed, configuration/checkpoint IDs, and protocol fields. `verified_canonical_results.csv` keeps an explicit eligibility/disposition column for transparency and assigns unique derived IDs to the excluded duplicate K-Means copies while retaining their original ID in `original_experiment_id`. LLM evaluation-only records have evaluation IDs rather than fabricated experiment IDs. This does not collapse distinct metrics, tasks, or budgets into one outcome. Protocol-specific processed tables remain the appropriate analytical inputs.

## Analytical treatments

- Fifteen later K-Means rows (source rows 32–36, 39–43, 46–50) repeat earlier experiment IDs and reuse the 50-update score at smaller budgets. Raw rows remain unchanged and later copies receive an excluded disposition.
- Mirror exports `replicated_learning_curves.csv` and `pseudolabel_learning_curves.csv` share all IDs with their richer result tables; they are alternate schemas, not new observations.
- The five-epoch dataset-size analysis uses only epoch-5 endpoints, avoiding duplicate 10K entries from 30-epoch endpoints.
- Double DQN `local_measurement` rows contain valid logged rewards and are retained in descriptive RL analysis with status unchanged.
- Pythia-160M/410M `main` aliases are not extra prefix trajectory checkpoints beyond explicit step143000.
- The 1,122.1625-second ResNet interval remains raw. Only the adjusted time-efficiency diagnostic subtracts this interval after epoch 2; its cause is unknown.

The machine-readable exclusion list is `results/processed/data_quality_exclusions.csv`; the broader issue register is `DATA_QUALITY_REGISTER.md` and `results/processed/data_quality_register.csv`.
