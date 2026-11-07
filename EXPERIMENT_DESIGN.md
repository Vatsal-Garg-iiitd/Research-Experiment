# Experiment design

**Execution status:** the full schedules below remain the target protocol. All 20 conventional methods now have one-seed feasibility evaluation records; the full multi-budget/multi-seed sweeps are pending. See `FINAL_COMPLETION_MATRIX.md` and `REMAINING_WORK.md`.

## Research question

How does model performance change as computational expenditure increases, and does marginal improvement diminish at higher budgets? This preserves the proposal's question while extending the analysis to conventional ML, language-model training trajectories, and published large-model studies.

## Evidence classes

Every result must be tagged as one of: (1) locally measured, (2) copied from a source, or (3) calculated from source-reported inputs. Literature results are not reproductions. Missing values stay missing. Wall time is only compared within the same hardware/software run; literature compute is kept in its reported unit and source.

## Local experimental design

The target architecture registry covers 20 algorithms across supervised, unsupervised, semi-supervised, and reinforcement learning. Every adapter must declare its task metric, budget unit, training/evaluation timings, parameter count, seed, hardware, peak memory when observable, and evaluation status. Epoch budgets are 1, 3, 5, 10, 20, and 30 for iterative supervised learners. K-means/GMM/DBSCAN use algorithm-specific iteration or neighborhood budgets; RL uses environment interactions and gradient updates. Non-iterative estimators are recorded as one fit budget and are not given fictitious epochs.

Use fixed train/validation/test splits and preserve the same split across budgets and seeds. Choose preprocessing and hyperparameters on training/validation only; consult test metrics only at preregistered final checkpoints. Semi-supervised runs share labeled indices at each label fraction (5%, 10%, 25%) across methods. RL reports mean episodic return and uncertainty across seeds, not classification accuracy. Train/evaluation/inference costs are separate.

## Analysis

For a higher-is-better metric, gain is `score[i] - score[i-1]`; for loss/perplexity, improvement is `metric[i-1] - metric[i]`. Cost deltas use cumulative measured seconds within a run. Efficiency is metric-specific gain per second. Report seed mean, standard deviation, and 95% t intervals where replicate runs exist. Use paired comparisons only when seeds/splits align. Do not infer significance from paper-to-paper comparisons.

Saturation is a descriptive flag, not a universal property: define the outcome and threshold before a run. The default is less than 1% relative improvement in the target metric over two consecutive budget increments, with absolute-threshold sensitivity analyses and uncertainty reported. This threshold is an operational convention, not evidence that the remaining gain has no value.

## Current feasibility and status

The workspace initially contained only the seven-page proposal PDF. The runtime reports 16 logical processors, 25 GB RAM (about 6 GB available after dependencies were installed), and approximately 866 GB free on C:. PyTorch reports no CUDA device. Small pilots for the four conventional paradigms and Pythia-70M WikiText-2 evaluation are in the raw CSVs. These are single-seed exploratory pilots, not the full suite. The remaining methods and full accelerator evaluation remain pending.

## Reproducibility

Pin versions before a full run, store config and source revision with every run, seed Python/NumPy/frameworks, record actual device/hardware and package versions, and checkpoint only resumable iterative models. Dataset versions, preprocessing, split hashes, tokenizer, prompts, few-shot examples, and evaluator version belong in run metadata. Never combine incompatible benchmark protocols into one score.
