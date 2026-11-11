# macOS experiment and analysis provenance

This summary documents the macOS environment represented in the retained results. Exact per-run device, backend, seed, budget, and package fields should be taken from the corresponding result/configuration records.

## Environment

- Apple M3, arm64, macOS 26.6.2, 16 GiB RAM.
- Python 3.12.11 in `.venv-mac`; PyTorch 2.15.0 with MPS available.
- Neural Fashion-MNIST schedules used MPS when recorded; classical and RL runs used CPU. Language-model runs record their own CPU/MPS settings.
- Historical Windows/DirectML measurements remain separate and are not pooled with macOS timings.

## Verified study coverage

- Supervised neural and representation-learning curves: 30 epochs, three seeds for the registered 10K conditions; MLP and LeNet also have matched five-epoch 30K/58K data-size endpoints.
- Random Forest: warm-start tree schedule through 1,000 trees, three seeds.
- Unsupervised methods: K-Means, GMM, DBSCAN, Autoencoder, and VAE at native method-specific budgets.
- Semi-supervised methods: five label fractions and three seeds; Mean Teacher, FixMatch, and MixMatch through 30 epochs. Pseudo-Labeling and its matched labeled-only baseline resumed from saved epoch-5 checkpoints through epoch 30 for the same seeds and fractions. Run command, timestamps, source revision, and exit status are in `results/run_logs/`.
- Reinforcement learning: DQN, Double DQN, A2C, PPO, and SAC at recorded requested budgets and seed counts; PPO-Pendulum is an extension.
- LLM checkpoint inference and harness evaluations are separate evaluation records; they do not represent local pretraining.

All 20 conventional schedules and all 15 literature-only objectives pass their respective audits. The 50-objective matrix is 46/50 under the per-model full-test endpoint rule; four LLM endpoints and multi-model benchmark/trajectory subtasks remain incomplete. Literature completion indicates source-based review and does not claim local reproduction. `docs/PROJECT_AUDIT.md`, `reports/EXPERIMENT_COMPLETION_MATRIX.md`, and the final report contain result-level details and limitations.

## Analysis

The analysis-only workflow and output verification are documented in `docs/REPRODUCIBILITY.md`. Rebuilding those artifacts does not invoke model training or inference.
