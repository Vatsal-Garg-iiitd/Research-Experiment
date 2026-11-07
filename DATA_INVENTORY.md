# Repository and data inventory

Inventory generated from `/Users/jagdishgarg/Downloads/Tcom_project`; 748 files after excluding Git internals, virtual environments, bytecode, and cache folders. Raw records are read-only inputs.

## File types

| Type | Files |
|---|---:|
| `.csv` | 70 |
| `.gz` | 4 |
| `.joblib` | 9 |
| `.json` | 153 |
| `.jsonl` | 15 |
| `.md` | 38 |
| `.pdf` | 3 |
| `.pkl` | 30 |
| `.png` | 95 |
| `.pt` | 144 |
| `.py` | 48 |
| `.svg` | 25 |
| `.txt` | 5 |
| `.yaml` | 2 |
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
| `results/processed/canonical_results.csv` | 2612 | 35 | canonical_id | 0 | 18733 |
| `results/processed/data_quality_exclusions.csv` | 15 | 5 | experiment_id | 0 | 0 |
| `results/processed/data_quality_register.csv` | 21 | 5 | — | 0 | 5 |
| `results/processed/data_size_scaling_seed_results.csv` | 24 | 7 | experiment_id | 0 | 0 |
| `results/processed/data_size_scaling_summary.csv` | 8 | 12 | — | 0 | 0 |
| `results/processed/dbscan_size_scaling_summary.csv` | 14 | 5 | — | 0 | 0 |
| `results/processed/figure_catalogue.csv` | 25 | 8 | — | 0 | 0 |
| `results/processed/foreign_key_audit.csv` | 0 | 5 | — | 0 | 0 |
| `results/processed/full_suite_marginal_results.csv` | 33 | 27 | run_id | 13 | 198 |
| `results/processed/identifier_audit.csv` | 101 | 7 | — | 0 | 29 |
| `results/processed/identifier_findings.csv` | 1388 | 7 | — | 0 | 153 |
| `results/processed/identifier_reconciliation.csv` | 95 | 8 | — | 0 | 50 |
| `results/processed/llm_full_eval_summary.csv` | 6 | 20 | — | 0 | 0 |
| `results/processed/pilot_marginal_results.csv` | 13 | 27 | run_id | 8 | 33 |
| `results/processed/ssl_algorithm_summary.csv` | 45 | 8 | — | 0 | 0 |
| `results/processed/strengthened_marginal_gains.csv` | 2417 | 22 | — | 0 | 3939 |
| `results/processed/strengthened_statistical_summaries.csv` | 3614 | 16 | — | 0 | 5138 |
| `results/processed/strengthened_test_statistics.csv` | 169 | 11 | — | 0 | 0 |
| `results/processed/verified_canonical_results.csv` | 2612 | 39 | canonical_id | 0 | 18844 |
| `results/raw/autoencoder_scaling.csv` | 210 | 23 | experiment_id | 0 | 210 |
| `results/raw/autoencoder_test_results.csv` | 18 | 12 | experiment_id | 0 | 6 |
| `results/raw/dbscan_parameter_sweep.csv` | 8 | 24 | experiment_id | 0 | 11 |
| `results/raw/dbscan_size_scaling.csv` | 9 | 20 | experiment_id | 0 | 4 |
| `results/raw/experiment_results.csv` | 26 | 20 | run_id | 16 | 26 |
| `results/raw/full_suite_results.csv` | 33 | 20 | run_id | 13 | 66 |
| `results/raw/label_propagation_scaling.csv` | 45 | 18 | experiment_id | 0 | 0 |
| `results/raw/pseudolabel_baseline_sweep.csv` | 150 | 31 | experiment_id | 0 | 150 |
| `results/raw/pseudolabel_learning_curves.csv` | 150 | 13 | experiment_id | 0 | 0 |
| `results/raw/pseudolabel_test_results.csv` | 30 | 14 | experiment_id | 0 | 0 |
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
| `results/snapshots/pre_canonical_schema_20261107/canonical_results.csv` | 604 | 20 | canonical_id | 0 | 1073 |
| `results/snapshots/pre_full_wikitext_accounting_repair_20261107/macos_full_wikitext2_evaluations.csv` | 6 | 29 | experiment_id | 0 | 0 |
| `results/snapshots/pre_literature_field_refresh_20261107/literature_field_evidence.csv` | 300 | 9 | — | 0 | 455 |
| `results/snapshots/pre_llm_trajectory_dedup_20261107/expanded_llm_evaluations.csv` | 51 | 19 | — | 0 | 174 |
| `results/snapshots/pre_replicated_example_count_repair_20261107/replicated_learning_curves.csv` | 420 | 11 | experiment_id | 0 | 0 |
| `results/snapshots/pre_replicated_example_count_repair_20261107/replicated_scaling_results.csv` | 420 | 40 | experiment_id | 0 | 4200 |
| `results/snapshots/pre_replicated_example_count_repair_20261107/replicated_scaling_test_results.csv` | 36 | 14 | experiment_id | 0 | 0 |
| `results/snapshots/pre_schema_repair_20261106/autoencoder_test_results.csv` | 12 | 11 | experiment_id | 0 | 0 |
| `results/snapshots/pre_ssl_csv_repair_20261107/ssl_algorithms.csv` | 1350 | 22 | experiment_id | 0 | 810 |
| `results/snapshots/pre_strengthening_20261106_150150/configs/model_registry.csv` | 20 | 8 | — | 0 | 0 |
| `results/snapshots/pre_strengthening_20261106_150150/results/hardware/backend_benchmark.csv` | 2 | 15 | — | 0 | 0 |
| `results/snapshots/pre_strengthening_20261106_150150/results/literature/large_model_literature.csv` | 15 | 27 | — | 0 | 142 |
| `results/snapshots/pre_strengthening_20261106_150150/results/llm/expanded_llm_evaluations.csv` | 34 | 19 | — | 0 | 140 |
| `results/snapshots/pre_strengthening_20261106_150150/results/llm/llm_checkpoint_results.csv` | 23 | 20 | — | 0 | 77 |
| `results/snapshots/pre_strengthening_20261106_150150/results/processed/full_suite_marginal_results.csv` | 33 | 27 | run_id | 13 | 198 |
| `results/snapshots/pre_strengthening_20261106_150150/results/processed/pilot_marginal_results.csv` | 13 | 27 | run_id | 8 | 33 |
| `results/snapshots/pre_strengthening_20261106_150150/results/raw/experiment_results.csv` | 26 | 20 | run_id | 16 | 26 |
| `results/snapshots/pre_strengthening_20261106_150150/results/raw/full_suite_results.csv` | 33 | 20 | run_id | 13 | 66 |
