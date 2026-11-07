# Implementation gaps

## Implemented and exercised

- Original project scripts and 20-entry model registry preserved; snapshots stored in `results/snapshots/`.
- Replicated neural epoch sweeps, Random Forest warm-start sweep, unsupervised optimization curves, pseudo-label baseline ablation, AE/VAE curves, DBSCAN parameter grid and RL checkpoint sweeps save incremental raw CSVs and model checkpoints.
- Separate CPU and DirectML environments. An isolated lm-evaluation-harness environment successfully ran three model/task subset evaluations. Windows console encoding was configured UTF-8 for the successful runs.
- LLM 50K-token WikiText-2 test evaluation, harness JSON extraction and language-model plot/report generation.
- Canonical result index builder, conventional statistical analysis and final completion matrix generation scripts.
- Output validation script parses project CSV results and checks required deliverables and numeric finiteness. Latest run: zero errors and zero warnings.

## Remaining implementation work

- Multi-budget/multi-seed implementations for Mean Teacher, FixMatch and MixMatch remain missing; their registered code/pilot status is not evidence of a full implementation study. LabelPropagation now has its own actual multi-seed iteration sweep; historical LabelSpreading remains a distinct pilot.
- Replace compact ViT proxy with a standard ViT-Tiny definition if strict architecture fidelity is required.
- Extend epoch and RL step schedules, and include full training-set or justified larger-data runs.
- Canonical schema normalizes core fields and retains each raw record as JSON, but not every historical CSV has identical metric semantics. The validator does not yet check all matrix/report counts, checkpoint completeness, source locators, or plot regeneration provenance.
- Conventional inference timing, FLOPs, GPU-placement/profile telemetry, and NPU runtime execution are incomplete.
- Full LLM coverage and checkpoint-series analyses are incomplete; harness subsets are deliberately smoke subsets, not benchmark-grade runs.
- Literature source extraction and independent locator/value verification are incomplete.
- The lm-eval isolated environment has a Windows Application Control blocked pandas wheel if used alone; successful isolated execution uses ordered package paths to reuse the known-working project's pandas while leaving the base environment unchanged.
