# Project audit

Audit date: 2026-11-06. Scope: current workspace, source files, raw/processed CSVs, configs, reports, figures, available logs and original proposal PDF.

## Research objective

The seven-page proposal, `ResearchProp_Grp119_sec_C (1).pdf`, asks whether model accuracy improves with additional computational effort and whether marginal gains become smaller. It proposes a small-scale experiment across training levels, records accuracy and wall time, and separately contextualizes results from papers where conditions are comparable. The current expanded project includes four conventional learning paradigms, LLM checkpoint evaluation, literature, and hardware work beyond the initial proposal.

## Inventory and evidence

- Source proposal: present; a text extraction is stored in the dated snapshot directory.
- `results.zip`: not present in the project or the checked Codex attachments. The current project tree itself contains the experiment scripts and outputs; no archive logs were available for cross-checking.
- Scripts: `scripts/run_pilots.py`, `run_missing_classical.py`, `run_missing_neural.py`, `run_missing_rl.py`, `run_llm_evaluations.py`, `benchmark_backends.py`, `analyze_results.py`, `plot_pilots.py`, and `finalize_study_status.py`.
- Raw conventional results: `results/raw/experiment_results.csv` (26 rows) and `results/raw/full_suite_results.csv` (33 rows; 20 current pilots and 13 superseded retries/versions).
- Processed results: pilot and full-suite marginal CSVs. Current full-suite marginal CSV has no calculable marginal values because each current one-budget model entry has only one active point.
- LLM: 34 rows in expanded evaluations (completed, failed, resource-limited, superseded); 23 rows in the Pythia checkpoint ledger including historical entries.
- Literature: 15 configurations in `large_model_literature.csv`; field-by-field source verification is incomplete.
- Hardware: Windows machine report and one small CPU/DirectML MLP microbenchmark. DirectML model pilots ran on Radeon 860M with some CPU fallbacks. NPU inference was not demonstrated; ROCm was not installed under the detected driver state.
- Proposal and original files were copied before modifications to `results/snapshots/pre_strengthening_20261106_150150/`.

## Model completion evidence

All 20 conventional registry entries have at least one saved feasibility record, but these are not all exact implementations of the requested names: the ViT entry is a custom 3-layer/96-wide compact transformer; the Label Propagation entry used scikit-learn LabelSpreading. See `EXPERIMENT_COMPLETION_MATRIX.md` and `IMPLEMENTATION_GAPS.md`.

Six of 15 LLM configurations have completed 50K-token WikiText-2 raw test evaluations: Pythia-70M/160M/410M, GPT-2 Small, OPT-125M and OPT-350M. Three configurations (Pythia-70M step143000, GPT-2 Small and OPT-125M) also have lm-evaluation-harness 0-shot scores on 100 examples each for HellaSwag, PIQA and ARC-Easy. These are diagnostic subsets, not full benchmark scores. Pythia-70M has eight historical checkpoint points; Pythia-160M and Pythia-410M have three distinct points each. `main` duplicates `step143000`, not an additional checkpoint. GPT-2 Medium has an attempted failed weight retrieval; other configs lack a long local evaluation.

The 15 literature configurations are literature-only, not local reproductions. Some quantitative evidence is source-reported; some derived values are labeled in notes. The ledger has gaps in exact page/table locators and metadata; it is not yet a validated complete literature dataset.

## Integrity and status

No placeholder score was found in the current raw local result tables. Failed and superseded attempts are retained. The MixMatch v1 row is superseded by the v2 run. Reported perplexities are calculated from measured cross-entropy; selected literature token totals are derived and annotated. This audit does not independently re-verify every literature number against original papers.

**Overall status: incomplete exploratory study.** The raw evidence does not support calling the requested full study complete, making cross-model significance claims, or claiming general diminishing returns.
