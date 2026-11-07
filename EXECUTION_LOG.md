# Execution log

Times are local Asia/Kolkata where known. This log records verified actions and does not infer missing historical logs.

## 2026-11-06 — audit and preservation

- Read the new master execution prompt and inventoried the project excluding both Python environments.
- Created `results/snapshots/pre_strengthening_20261106_150150/` and copied existing reports, configs, scripts, raw/processed results, LLM and literature records, hardware files and figures; copied the proposal PDF and saved extracted text there.
- Searched project and checked attachment paths for `results.zip`; no supplied archive was found. Current workspace files are therefore the available evidence.
- Extracted the seven-page proposal. It asks whether increased compute improves accuracy, whether marginal accuracy gain diminishes, records accuracy and elapsed time over training levels, and separately analyzes comparable published results.
- Verified actual CSV counts: 26 historical conventional rows; 33 full-suite rows (20 pilot, 13 superseded); 34 expanded LLM rows; 15 literature rows. Current full-suite marginal file has 0 populated marginal-gain cells because it has only single-point active records per model.
- Installed `pypdf` in the existing project `.venv` only to read the source proposal; `pip check` passed afterward. No environment was replaced.
- Re-audited method labels and found the requested Label Propagation run actually used scikit-learn LabelSpreading; the ViT pilot is custom compact (3 layers, 96 width). Updated the matrix/report to identify these as proxies. The 160M/410M `main` LLM rows duplicate the step143000 checkpoint; matrix now calls out the alias.
- Historical data files remain preserved in the snapshot. No new experiment scores were entered during this audit segment.

## Prior execution evidence already in workspace

Detailed historical actions are represented by source scripts and raw result rows, not by a supplied run-log archive. See `results/raw/experiment_results.csv`, `results/raw/full_suite_results.csv`, LLM ledgers and `results/hardware/hardware_capabilities.md`. Do not infer unlogged events beyond those artifacts.

## 2026-11-06 — expanded experiments, evaluation and deliverables

- Executed replicated epochs 1–5 (seeds 11, 23, 37) for MLP, LeNet-style CNN, ResNet-18 and custom compact transformer; generated validation learning curves and independent official-test endpoints. See `results/raw/replicated_scaling_results.csv` and matching test-results CSV.
- Executed Random Forest warm-start tree budgets 10–400 across three seeds; MiniBatch K-Means and GMM algorithmic sweeps across three seeds; pseudo-labeling vs matched labeled-only baselines at five label fractions and epochs 1–5; AE/VAE epochs 1–5 across three seeds; DBSCAN eight-setting parameter sensitivity grid; RL DQN, Double DQN, A2C, PPO and SAC at 5K/10K/25K interactions across three seeds. Raw outputs and checkpoints are retained.
- Superseded the first AE/VAE linear-probe run after convergence warnings; reran using standardized features and a corrected convergence configuration. Both historical and corrected records remain, with old rows marked `superseded`.
- Completed CPU FP32 50K-token WikiText-2 raw test evaluations for Pythia-70M/160M/410M step143000, GPT-2 Small, OPT-125M and OPT-350M. Exact scores and runtime are in `results/llm/full_wikitext2_evaluations.csv`.
- Installed lm-evaluation-harness in an isolated environment. First downstream task attempt failed before model loading: Windows Application Control blocked the isolated pandas binary (`pandas_parser` DLL load). A second isolated environment reuses the working environment's packages and installs only lm-eval; a retry is in progress.
- The initial successful model evaluation then hit a Windows CP1252 console-print error after aggregated JSON was saved; reran with UTF-8 and completed cleanly. Successful 0-shot CPU FP32 harness evaluations: Pythia-70M step143000 (100 each HellaSwag/PIQA/ARC-Easy; 97.48s), GPT-2 Small (100 each; 72.61s), and OPT-125M (100 each; 180.38s). Full task results and sampled examples are under `results/llm/harness/`; `results/llm/harness_summary.csv` contains 9 model-task rows after deduplicating Pythia's failed-print output.
- Generated language-model report and size/loss/perplexity/throughput plots; regenerated completion matrices with 20 conventional, 15 LLM and 15 literature rows. Added hardware, literature validation and final research reports, canonical result builder, and output validator.
- The statistical analysis pipeline was run after corrected AE/VAE records; it produced 458 grouped summaries, 279 within-seed adjacent-budget increments and 84 held-out endpoint summaries. It does not assert general saturation or cross-paradigm metric equivalence.
- No scores were invented or inserted for missing experiments. Exact incompleteness and the observed proxy implementations are recorded in `REMAINING_WORK.md` and the matrices.
- Canonical export contains 430 local result rows. The first automated validation exposed a malformed v2 AE/VAE test CSV header (the actual `probe_n_iter` field lacked a header); made a dated snapshot, repaired the header while preserving all 12 rows, regenerated the analysis, and reran validation successfully: zero CSV errors and zero warnings. Core `.venv` `pip check` passed.

## 2026-11-06 — follow-up prompt execution

- Read the latest master execution prompt. Its scope excludes further hardware investigations; followed that constraint and made no additional hardware measurements or provider changes.
- Inspected the named reports, canonical data, scripts, configs, raw CSV files and checkpoint directories. Preserved prior outputs and retained the strict 0/50 complete-objective count.
- Implemented and executed actual sklearn LabelPropagation with 3 seeds and maximum iteration budgets 1, 5, 10, 30, 100 on fixed 2,500-sample Fashion-MNIST transductive splits (2,000 train nodes, 500 test nodes, 10% labels, standardized PCA-32 features, 10-neighbor graph). Convergence warnings were generated and retained in terminal run history; rows record actual iterations/convergence status. Output: `results/raw/label_propagation_scaling.csv`.
- Added the LabelPropagation rows to canonical export and updated the research/remaining-work reports and reproducibility guide. Legacy LabelSpreading pilot remains distinct and unchanged.

- Implemented and executed Mean Teacher, FixMatch and MixMatch on shared Fashion-MNIST split: 5 label fractions x 3 seeds x 3 epoch checkpoints per method (135 seed-budget rows) plus 45 final test endpoints. DirectML failed on a FixMatch boolean-mask autograd op and MixMatch indexed MixUp op; FixMatch was corrected with a float mask and MixMatch ran on CPU, both recorded. Results: esults/raw/ssl_algorithms.csv, ssl_algorithms_test.csv; summary and plots generated.

## 2026-11-07 — current project-wide inventory

- Inventoried the active project tree excluding virtual environments, bytecode caches, and preserved snapshots; file counts by extension:  4, .csv 33, .gitignore 1, .gz 4, .joblib 9, .json 127, .jsonl 12, .md 22, .pdf 1, .png 44, .pt 108, .py 27, .txt 2, .yaml 1, .zip 37.
- Read the final reports, implementation gaps, completion matrix, remaining-work ledger, methodology, statistics, marginal gains, LLM, literature, hardware record, execution log and reproducibility guide. Checked raw/processed CSV headers, model registry, experiment scripts, and checkpoint directories.
- Reconciled prompt-specific status: LabelPropagation is its own implementation; historical LabelSpreading is separate. LabelPropagation convergence was reached under tolerance 1e-3 in 197–438 iterations; no remaining proxy claim should refer to it.


- Implemented and executed Mean Teacher, FixMatch and MixMatch on shared Fashion-MNIST split: 5 label fractions x 3 seeds x 3 epoch checkpoints per method (135 seed-budget rows) plus 45 final test endpoints. DirectML failed on a FixMatch boolean-mask autograd op and MixMatch indexed MixUp op; FixMatch was corrected with a float mask and MixMatch ran on CPU, both recorded. Results: results/raw/ssl_algorithms.csv, ssl_algorithms_test.csv; summary and plots generated.

- Added separate DBSCAN data-size scaling at n=2,500/5,000/10,000, fixed eps/min_samples/PCA settings, three seeds (9 runs total); saved runtime, ARI, NMI, silhouette, cluster count and noise fraction. Results: results/raw/dbscan_size_scaling.csv. This is labeled data-size compute scaling, separate from parameter sensitivity.

- Canonical schema extended with metric direction, budget unit, run type, checkpoint revision, source and explicit empty training/evaluation compute fields; LLM WikiText and harness smoke records included (local harness rows distinguished by run_type/source_type). Backed up prior canonical data to results/snapshots/pre_canonical_schema_20261107/. Canonical now has 619 rows.
