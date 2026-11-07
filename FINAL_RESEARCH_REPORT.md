# Computational scaling study: current evidence and completion status

## Abstract

This is an interim, reproducibility-focused research record updated 2026-11-07. Additional multi-seed sweeps, full-corpus checkpoint evaluations, and one complete zero-shot harness run are now saved. The strict configuration-level matrix remains **0/50 full study objectives complete**: completed pilots and expanded partials do not satisfy every prespecified objective. Results support descriptive, method-specific observations, not a universal scaling law.

## Research question and measurement

How does task performance change as computational expenditure increases, and do marginal gains decline? Budgets remain native to each method: epochs and examples for supervised learning; trees; optimization iterations; labeled fractions; environment interactions; and published checkpoint steps/tokens. They are not collapsed into one cross-paradigm compute axis. Local checkpoint inference is distinct from pretraining compute.

## Local experiments

- Fashion-MNIST supervised models use fixed seeded splits with train, validation, and untouched official test data. MLP, LeNet-style CNN, ResNet-18, and a custom compact transformer have three-seed 30-epoch curves on the 10K training subset. MLP and LeNet also have three-seed five-epoch runs at requested 30K and 60K train sizes; the 60K request uses 58K optimization examples plus 2K validation examples. The official test set is unchanged.
- Random Forest curves now extend warm-start complexity through 1,000 trees. MiniBatch K-Means, Gaussian Mixture, DBSCAN, and LabelPropagation retain separate native budgets and protocols. DBSCAN’s sample-size curve is a cost comparison, not iterative training.
- Pseudo-labeling and labeled-only baselines cover five label fractions. Mean Teacher, FixMatch, and MixMatch cover 1%, 5%, 10%, 20%, and 40% labels over three seeds and 30 epochs. These are grayscale Fashion-MNIST adaptations, not canonical benchmark replications.
- Autoencoder and VAE curves cover 30 epochs and three seeds, with reconstruction and standardized linear-probe measures kept separate.
- RL curves include DQN, Double DQN, A2C, PPO, PPO-Pendulum, and SAC. Checkpoints span requested 5K–250K interactions; SAC and Double DQN include five seeds. Budget overshoot and one legacy non-exact Double DQN resume are documented in `MACOS_EXECUTION_LOG.md`.
- Local neural and LLM work ran on Apple M3 with PyTorch MPS where supported; classical methods and RL ran on CPU. Detailed environment and setup notes are in `MACOS_SETUP.md` and `REPRODUCIBILITY.md`.

## Selected results

At five epochs, mean held-out Fashion-MNIST accuracy across seeds was MLP 0.8537 (30K requested) and 0.8689 (58K actual), and LeNet 0.8784 (30K) and 0.8989 (58K). These are descriptive fixed-split results; the MLP’s separate lower-sized run is non-monotone, so the observations do not establish a data-scaling law. Endpoint intervals and per-seed measurements are in `STATISTICAL_ANALYSIS_REPORT.md` and `results/processed/data_size_scaling_summary.csv`.

Six published checkpoints were evaluated on the full WikiText-2 raw-v1 test text with shared 1,024-token context and 512-token stride. Perplexities were Pythia-70M 44.9334, 160M 25.6365, 410M 15.4583, GPT-2 Small 24.9794, OPT-125M 25.8534, and OPT-350M 20.7992. This is matched local inference on one corpus, not matched original training compute; tokenizer and context effects remain relevant.

Pythia-70M was evaluated with lm-evaluation-harness 0.4.13, zero-shot and without a limit, on validation splits: HellaSwag n=10,042 accuracy .2667; PIQA n=1,838 .5985; ARC-Easy n=2,376 .3754. Earlier 100-example checks for GPT-2 Small and OPT-125M are smoke runs only. Results and sample logs are under `results/llm/harness_full/` and `results/llm/harness/` respectively.

## Marginal gains and saturation

The regenerated pipeline reports 3,614 grouped validation summaries, 2,407 paired within-seed increments, and 169 held-out endpoint summaries separated by final budget/checkpoint. It flags consecutive small gains at several descriptive thresholds. These screens do not demonstrate saturation: curves are finite, some results are non-monotone, and seed counts are small. Full tables and figures are linked from `STATISTICAL_ANALYSIS_REPORT.md` and `MARGINAL_GAIN_ANALYSIS.md`.

## Literature

Fifteen published configurations have a field-level evidence ledger in `results/literature/literature_field_evidence.csv`. Extracted values carry evidence classes, primary source URLs, and table/section locators. Several fields remain unreported, estimated, or unresolved. Published runs are not locally reproduced, and cross-paper metrics/protocols are not treated as directly comparable. See `LITERATURE_VALIDATION_REPORT.md`.

## Conclusions and limits

The records show that performance changes with native budgets and that several small local curves flatten or fluctuate. They do not establish universal diminishing returns, model-family superiority, robust population-level uncertainty, or comparable efficiency across paradigms. Much of the work remains a partial study against the original scope; the strict 50-objective matrix preserves that status. `FINAL_INDEPENDENT_AUDIT.md`, `EXPERIMENT_RUN_REGISTRY.csv`, and `FINAL_COMPLETION_MATRIX.md` provide the artifact-level audit trail.
