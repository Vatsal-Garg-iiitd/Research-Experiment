# Computational scaling study: current evidence and completion status

## Abstract

This report summarizes the actual experiments and audits saved in this project as of 2026-11-06. The study has expanded beyond its original pilots with multi-seed, multi-budget runs for several conventional methods and 50,000-token language-model evaluations for six checkpoints/configurations. Under the strict configuration-level definition in `FINAL_COMPLETION_MATRIX.md`, 0 of 50 full study objectives are complete (0%); the missing scope is not hidden by pilot or subset results. Results below are interim evidence, not a completed study.

## Research question

How does task performance change as computational expenditure increases, and do marginal gains decline? Budget units remain native to each paradigm: epochs/trees, iterative optimization, labeled fraction plus epochs, environment interactions, and language-model training-token metadata. Different task metrics are not pooled.

## Methods and reproducibility

Fashion-MNIST neural and classical runs use a fixed 10,000-image training subset and 2,000-image validation subset, with the official 10,000-image test set reserved for endpoint evaluation. Most new sweeps use seeds 11, 23 and 37. Neural runs used DirectML in a separate PyTorch 2.4.1 environment; Adam emitted an operator CPU-fallback warning. Classical models and RL used CPU. Detailed configuration and row-level records are stored under `results/configurations/`, `results/raw/` and checkpoints under `results/checkpoints/`. See `EXPERIMENTAL_METHODOLOGY.md` for method-specific caveats.

## Completed expanded local runs

- MLP, LeNet-style CNN, ResNet-18 and a **custom compact 3-layer, width-96 transformer**: epochs 1–5, three seeds, validation learning curves and one held-out test endpoint per seed. The transformer is not standard ViT-Tiny.
- Random Forest: warm-start tree counts 10, 25, 50, 100, 200, 400; three seeds and held-out endpoint results.
- MiniBatch K-Means and Gaussian mixture: algorithm-specific iterative budgets and three seeds, with validation ARI/NMI/silhouette and GMM likelihood.
- Pseudo-labeling versus a matched labeled-only baseline: 1%, 5%, 10%, 20%, 40% labeled data; epochs 1–5; three seeds; held-out endpoint results.
- Mean Teacher, FixMatch and MixMatch: each ran at 1%, 5%, 10%, 20%, 40% labeled fractions; three seeds; epochs 1–3; final official-test endpoints. Compact MLP recipe: Mean Teacher EMA decay .99; FixMatch confidence threshold .95; MixMatch two-view averaging, sharpening temperature .5 and MixUp alpha .75. Photometric augmentations are adapted for grayscale inputs. MixMatch ran on CPU due to a DirectML indexing failure; backend is recorded per row. This is not a canonical CIFAR/ImageNet SSL reproduction.
- DQN, Double DQN, A2C, PPO and SAC: three seeds and requested 5K, 10K, 25K environment-step checkpoints; 20 episode evaluations per checkpoint. Some rollout algorithms overshoot requested steps; actual counts are recorded. CartPole and Pendulum results remain separate.
- Actual sklearn LabelPropagation: initial maximum iteration budgets 1–100 and controlled extended budgets 100–2,000 on three seeded fixed transductive Fashion-MNIST subsets, 10% labeled training nodes, PCA-32 features and 10-neighbor graph. At tolerance 1e-3, convergence occurred after 374, 438 and 197 iterations respectively. The 100/250 caps sometimes stopped before convergence. Diagnostic graph for each seed was connected, symmetric after union, and had no isolated nodes. Accuracy and timing stabilized by convergence. This is distinct from the historical LabelSpreading pilot.
- DBSCAN: retained the 8-setting epsilon/min-samples parameter sensitivity grid and added a separate fixed-parameter data-size cost curve at 2,500/5,000/10,000 samples, three seeds, PCA-32 features, eps=2.88036 and min_samples=5. This measures data-size scaling, not iterative training.
- Autoencoder and VAE: epochs 1–5 and three seeds. A first probe variant had convergence warnings and is retained as superseded; corrected standardized linear-probe runs and test endpoints are separately recorded.
- Six local LM checkpoints/configurations: Pythia-70M/160M/410M at step 143000, GPT-2 Small main, OPT-125M and OPT-350M, each evaluated on 50,000 WikiText-2 raw test tokens. These are inference evaluations, not pretraining runs.

## Selected measured endpoints

Fashion-MNIST official-test accuracy means across three seeds: MLP 0.8279 (SD 0.0060), LeNet-style CNN 0.8368 (SD 0.0034), Random Forest 0.8524 (SD 0.0009). These intervals are conditional on the fixed 10k/2k development split; they do not describe variation across new data splits. Autoencoder test reconstruction MSE was 0.0196 (SD 0.0003), with linear-probe accuracy 0.7917 (SD 0.0033); VAE MSE 0.0292 (SD 0.0001), probe accuracy 0.7838 (SD 0.0043). See `STATISTICAL_ANALYSIS_REPORT.md` for the complete endpoints and intervals.

On the matched WikiText-2 evaluations, loss/perplexity were: Pythia-70M 3.9036/49.58, Pythia-160M 3.3766/29.27, Pythia-410M 2.8753/17.73, GPT-2 Small 3.4796/32.45, OPT-125M 3.4103/30.27, OPT-350M 3.1930/24.36. All were CPU FP32. GPT-2 used a shorter context (1,024) than Pythia/OPT (2,048), so this is not a pure parameter-count comparison. The benchmark is the WikiText-2 test split, not WikiText-103.

An lm-evaluation-harness 0-shot run evaluated 100 examples per task for Pythia-70M step143000, GPT-2 Small and OPT-125M on HellaSwag, PIQA and ARC-Easy. Raw accuracy / normalized accuracy: Pythia-70M HellaSwag .33/.41, PIQA .60/.60, ARC-Easy .40/.33; GPT-2 Small .36/.44, .60/.63, .38/.34; OPT-125M .38/.43, .66/.60, .46/.45. Sampling standard errors are roughly 0.047–0.050 with n=100. These task subsets are smoke diagnostics and are not comparable to full benchmark reports. Full raw harness JSON and samples are retained under `results/llm/harness/`.

## Marginal gains and saturation

The analysis pipeline computes within-seed adjacent-budget changes for completed rows and native metric directions. It generated 279 adjacent-budget increments and 458 grouped summaries from its included sweep sources. Curves are short and some metrics are non-monotonic; no general diminishing-return or saturation claim is warranted. No cross-paradigm scalar compute axis is asserted. See `MARGINAL_GAIN_ANALYSIS.md`.

## Hardware

The Ryzen AI 7 350 system exposes Radeon 860M through tested DirectML, while the working PyTorch environment is CPU-only. A microbenchmark showed faster inference throughput but slower median batch latency/model load and fewer training steps per second on DirectML for the specific small MLP. Adam had a CPU fallback warning. The NPU was detected but no provider-backed inference was run. See `HARDWARE_BENCHMARK_REPORT.md`.

## Literature

Fifteen published configurations are recorded, but the source-by-source field audit and precise source locators remain partial. Literature values are kept separate from local results; missing values remain blank. See `LITERATURE_VALIDATION_REPORT.md`.

## Conclusions supported now

The expanded records establish actual multi-budget curves for a subset of conventional methods and show that task performance can change non-monotonically with budget/configuration. The small fixed-split results support descriptive comparisons within each protocol. Six pretrained LM checkpoints have actual local perplexity measurements on the same text split and similar token count. Existing evidence does not establish universal diminishing returns, comparative paradigm superiority, hardware-wide speedups, or robust population-level confidence.

## Remaining limitations

The 20-model suite is not uniformly complete: Mean Teacher, FixMatch and MixMatch now have three-seed, five-fraction curves, but only through three epochs and using compact recipes rather than canonical benchmark settings. Unsupervised autoencoder coverage differs from clustering; the custom transformer remains a proxy for standard ViT-Tiny. LabelPropagation has finite-budget multi-seed results and convergence analysis but remains a limited graph-propagation study. Other neural epoch ceilings stop at 5 and RL at 25K; three-seed runs use a 10k training subset. Only six of 15 LLM configurations have long local WikiText evaluations, and detailed Pythia checkpoint analysis is limited. Harness execution covers only three configurations and 100 examples/task. Literature source verification is incomplete. These facts are recorded in `REMAINING_WORK.md` and `FINAL_COMPLETION_MATRIX.md`.
