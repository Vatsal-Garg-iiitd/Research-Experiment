# Repository and data inventory

Inventory generated from `/Users/jagdishgarg/Downloads/Tcom_project`; 689 files after excluding Git internals, virtual environments, bytecode, and cache folders. Raw records are read-only inputs.

## File types

| Type | Files |
|---|---:|
| `.csv` | 53 |
| `.gz` | 4 |
| `.joblib` | 9 |
| `.json` | 153 |
| `.jsonl` | 15 |
| `.log` | 1 |
| `.md` | 30 |
| `.pdf` | 2 |
| `.pkl` | 30 |
| `.png` | 73 |
| `.pt` | 144 |
| `.py` | 40 |
| `.svg` | 25 |
| `.txt` | 2 |
| `.yaml` | 1 |
| `.zip` | 102 |
| `[no extension]` | 5 |

## Result tables

CSV row counts below exclude headers. Candidate identifier repetition is a screening signal, not an automatic duplicate verdict: repeated identifiers may represent legitimate checkpoint budgets. Full columns, missingness, status counts, and provenance fields are recorded in `results/processed/tabular_data_inventory.json`. JSON/JSONL record counts and key/identifier coverage are in `results/processed/json_data_inventory.json`.

| File | Rows | Columns | Candidate ID | Repeated ID excess | Missing cells |
|---|---:|---:|---|---:|---:|
| `results/hardware/backend_benchmark.csv` | 2 | 15 | — | 0 | 0 |
| `results/literature/large_model_literature.csv` | 15 | 27 | — | 0 | 142 |
| `results/literature/literature_field_evidence.csv` | 300 | 9 | — | 0 | 277 |
| `results/llm/expanded_llm_evaluations.csv` | 40 | 19 | — | 0 | 152 |
| `results/llm/full_wikitext2_evaluations.csv` | 6 | 28 | experiment_id | 0 | 12 |
| `results/llm/harness_full_summary.csv` | 3 | 19 | — | 0 | 0 |
| `results/llm/harness_summary.csv` | 9 | 13 | — | 0 | 0 |
| `results/llm/llm_checkpoint_results.csv` | 40 | 20 | — | 0 | 131 |
| `results/llm/macos_full_wikitext2_evaluations.csv` | 6 | 32 | experiment_id | 0 | 6 |
| `results/processed/canonical_results.csv` | 3362 | 35 | canonical_id | 0 | 23983 |
| `results/processed/conventional_completion_audit.csv` | 20 | 7 | — | 0 | 20 |
| `results/processed/data_quality_exclusions.csv` | 15 | 5 | experiment_id | 0 | 0 |
| `results/processed/data_quality_register.csv` | 21 | 5 | — | 0 | 5 |
| `results/processed/data_size_scaling_seed_results.csv` | 24 | 7 | experiment_id | 0 | 0 |
| `results/processed/data_size_scaling_summary.csv` | 8 | 12 | — | 0 | 0 |
| `results/processed/dbscan_size_scaling_summary.csv` | 14 | 5 | — | 0 | 0 |
| `results/processed/figure_catalogue.csv` | 25 | 8 | — | 0 | 0 |
| `results/processed/foreign_key_audit.csv` | 0 | 5 | — | 0 | 0 |
| `results/processed/full_suite_marginal_results.csv` | 33 | 27 | run_id | 13 | 198 |
| `results/processed/identifier_audit.csv` | 99 | 7 | — | 0 | 29 |
| `results/processed/identifier_findings.csv` | 1533 | 7 | — | 0 | 243 |
| `results/processed/identifier_reconciliation.csv` | 95 | 8 | — | 0 | 50 |
| `results/processed/literature_completion_audit.csv` | 15 | 16 | — | 0 | 15 |
| `results/processed/llm_full_eval_summary.csv` | 6 | 20 | — | 0 | 0 |
| `results/processed/ssl_algorithm_summary.csv` | 45 | 8 | — | 0 | 0 |
| `results/processed/strengthened_marginal_gains.csv` | 3167 | 22 | — | 0 | 4325 |
| `results/processed/strengthened_statistical_summaries.csv` | 4864 | 16 | — | 0 | 5138 |
| `results/processed/strengthened_test_statistics.csv` | 219 | 11 | — | 0 | 0 |
| `results/processed/verified_canonical_results.csv` | 3362 | 39 | canonical_id | 0 | 24094 |
| `results/raw/autoencoder_scaling.csv` | 210 | 23 | experiment_id | 0 | 210 |
| `results/raw/autoencoder_test_results.csv` | 18 | 12 | experiment_id | 0 | 6 |
| `results/raw/dbscan_parameter_sweep.csv` | 8 | 24 | experiment_id | 0 | 11 |
| `results/raw/dbscan_size_scaling.csv` | 9 | 20 | experiment_id | 0 | 4 |
| `results/raw/experiment_results.csv` | 26 | 20 | run_id | 16 | 26 |
| `results/raw/full_suite_results.csv` | 33 | 20 | run_id | 13 | 66 |
| `results/raw/label_propagation_scaling.csv` | 45 | 18 | experiment_id | 0 | 0 |
| `results/raw/pseudolabel_baseline_sweep.csv` | 900 | 31 | experiment_id | 0 | 900 |
| `results/raw/pseudolabel_learning_curves.csv` | 900 | 13 | experiment_id | 0 | 0 |
| `results/raw/pseudolabel_test_results.csv` | 60 | 14 | experiment_id | 0 | 0 |
| `results/raw/random_forest_scaling.csv` | 24 | 28 | experiment_id | 0 | 96 |
| `results/raw/random_forest_test_results.csv` | 6 | 12 | experiment_id | 0 | 0 |
| `results/raw/replicated_learning_curves.csv` | 420 | 13 | experiment_id | 0 | 0 |
| `results/raw/replicated_scaling_results.csv` | 420 | 41 | experiment_id | 0 | 4200 |
| `results/raw/replicated_scaling_test_results.csv` | 36 | 15 | experiment_id | 0 | 0 |
| `results/raw/rl_scaling_results.csv` | 132 | 29 | experiment_id | 0 | 132 |
| `results/raw/ssl_algorithms.csv` | 1350 | 25 | experiment_id | 0 | 1215 |
| `results/raw/ssl_algorithms_test.csv` | 90 | 10 | experiment_id | 0 | 0 |
| `results/raw/unsupervised_scaling.csv` | 51 | 26 | experiment_id | 15 | 30 |
| `results/raw/vit_tiny_scaling.csv` | 90 | 24 | experiment_id | 0 | 90 |
| `results/raw/vit_tiny_test_results.csv` | 3 | 15 | experiment_id | 0 | 0 |
