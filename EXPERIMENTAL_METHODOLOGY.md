# Experimental methodology and evidence boundary

## Current status

The registered conventional schedules are complete and the analysis is finalized. The pseudo-label and matched labeled-only runs resumed from saved epoch-5 checkpoints to the original 30-epoch target, preserving the same seeds and splits. This document describes the measured protocols and analysis. The strict objective matrix is 35/50: 20 conventional local studies and 15 literature-only reviews are complete under separate criteria; 15 local LLM objectives remain incomplete. See `FINAL_COMPLETION_MATRIX.md` for statuses.

## Evidence hierarchy

1. Raw experiment records and their run configurations are primary evidence.
2. Checkpoints, execution logs, evaluator JSON, and per-sample logs support provenance and completion status.
3. Processed CSVs and reports are derived artifacts and are regenerated from raw evidence.
4. Literature observations are cited external evidence and are never labeled as local reproductions.

Raw experimental files are preserved. A repeated source export is not treated as an additional independent observation. The current canonical index contains 3,368 heterogeneous source rows; it includes repeated budgets and superseded/partial statuses and is not an independent-run count.

## Datasets and splits

Most conventional-learning curves use Fashion-MNIST. The principal learning split uses 10,000 training and 2,000 validation examples, identified by split ID `24574ee00a20ab31`, plus the official test split for final endpoints. The requested 60,000-size condition reserves 2,000 examples for validation, resulting in 58,000 optimization examples. The 30K and 58K size comparison is restricted to five-epoch endpoints. DBSCAN has both density-parameter and nested fit-size sweeps; the fit-size results use PCA-32 and transductive evaluation where recorded. RL environments are CartPole-v1 and Pendulum-v1. Full language-model evaluation uses WikiText-2 raw-v1 test text; harness validation tasks are recorded separately.

## Model identity and scope

The repository contains both the custom compact three-layer, width-96 transformer proxy and a separate 12-block, width-192 ViT-Tiny-dimension VisionTransformer adapted to one-channel 28x28 inputs with 4x4 patches. The latter is not represented as a default 224px ImageNet model. Mean Teacher, FixMatch, MixMatch, and pseudo-labeling are Fashion-MNIST adaptations. Current corrected Label Propagation runs use sklearn LabelPropagation; legacy LabelSpreading rows are superseded. PPO-Pendulum is a recorded extension; the original RL scope remains DQN, Double DQN, A2C, PPO, and SAC.

## Native budgets and metrics

Epochs, RF tree counts, K-Means partial-fit updates, GMM warm-start EM fit calls, DBSCAN sample/parameter settings, graph iteration ceilings, SSL labeled fractions/epochs, RL environment steps, released LLM checkpoint steps, and evaluation tokens/examples remain distinct. Classification accuracy/F1, clustering metrics, reconstruction losses, RL returns, language-model perplexity/loss, and literature benchmarks are analyzed separately. No epoch equivalents or pooled accuracy scores are fabricated.

## Statistical analysis

Descriptive tables report valid (n), mean, standard deviation, median, minimum, maximum, and 95% Student-t intervals when multiple independent seeds exist. Repeated checkpoints are paired within seed, split, model, and setting. Marginal improvement is sign-corrected for lower-is-better metrics and normalized by native budget or positive elapsed-time increments only. A screen for two consecutive small relative gains is descriptive; it is not a formal saturation test. No cross-paradigm p-values or causal inference are claimed.

## Protocol-specific notes

- Supervised and representation-learning curves: 30 epochs for core 10K conditions, three seeds; MLP and LeNet have 30K/58K five-epoch size endpoints.
- Random Forest: warm-start additions through 1,000 trees; tree count is the complexity budget.
- Unsupervised: K-Means uses partial-fit updates; GMM uses warm-start EM calls; DBSCAN uses data-size and parameter grids. Fifteen later K-Means duplicate checkpoint-reuse rows are excluded from curves but remain in raw data.
- Semi-supervised: Mean Teacher, FixMatch, MixMatch, pseudo-labeling, and the matched labeled-only baseline cover 1%, 5%, 10%, 20%, and 40% labels, 30 epochs, and three seeds.
- RL: 5K–250K requested interactions, checkpoints at 5K/10K/25K/50K/100K/250K, 20 deterministic evaluation episodes per saved checkpoint. Double DQN has five seeds and status `local_measurement`; these valid recorded rewards are included with status preserved. PPO-Pendulum is outside the original algorithm count.
- LLM full corpus: eleven checkpoints, WikiText-2 raw-v1 test, FP32, Apple M3 MPS, context 1,024, stride 512, batch one. Pythia models score 285,830 tokens; GPT-2/OPT score 283,286. Prefix scaling and zero-shot harness runs are separate protocols; checkpoint evaluation is not pretraining.
- Literature: 300 field records over 15 configurations carry reported/derived/estimated/uncertain labels, source URLs, and locators. Unknown fields remain unknown.

## Known anomalies and exclusions

The ResNet-18 seed-11 epoch-2-to-3 elapsed interval is 1,122.1625 seconds. It is retained as wall-clock evidence and excluded only from adjusted efficiency analysis; records cannot establish the underlying cause. The K-Means repeated rows, mirror exports, and Pythia checkpoint aliases are detailed in `DATA_QUALITY_REGISTER.md`. Historical Windows/DirectML timing is not pooled with newer Apple MPS timing.

## Completion and limitations

Completion criteria were not relaxed after results. The denominator remains 50: all 20 conventional schedules and all 15 literature-only reviews pass their respective audits, while the 15 local LLM objectives remain partial or unevaluated. Literature completion denotes source-based review, not local experiments or reproduction. Evidence supports selected descriptive comparisons, but not broad saturation claims, family rankings, or training-compute comparisons for locally evaluated public LLMs.
