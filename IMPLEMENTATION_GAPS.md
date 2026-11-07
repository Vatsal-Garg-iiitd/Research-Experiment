# Implementation gaps and verified status

This file records implementation limitations after the experimental phase closed. It is not a request to add experiments or extend scope.

| Area | Existing evidence | Gap / reporting treatment |
|---|---|---|
| Supervised architectures | MLP, LeNet-style CNN, ResNet-18, a separate ViT-Tiny-dimension implementation, compact transformer proxy and Random Forest outputs exist. | ViT-Tiny-dimension model adapts one-channel 28x28 inputs and 4x4 patches; the three-layer/96-width proxy is not the ViT-Tiny scope model. Full registered objective remains incomplete. |
| Unsupervised | K-Means, GMM, DBSCAN, Autoencoder, VAE, and LabelPropagation records exist. | Native budget definitions differ; no epoch-equivalent is assigned. Fifteen duplicated K-Means rows are excluded analytically and retained raw. |
| Semi-supervised | Pseudo-labeling, Mean Teacher, FixMatch, MixMatch, and LabelPropagation evidence exists. | Fashion-MNIST adaptations are not canonical SSL benchmark replications. Legacy LabelSpreading is superseded and not labeled LabelPropagation. |
| Reinforcement learning | DQN, Double DQN, A2C, PPO, SAC and an additional PPO-Pendulum extension are present. | PPO-Pendulum is outside original five-method scope. Double DQN rows are marked `local_measurement` and analyzed with that provenance preserved. Environment/seed coverage and logged wall time constrain conclusions. |
| Local language models | Six full WikiText-2 evaluations; Pythia prefixes; a Pythia-70M full zero-shot run; smoke and restricted runs. | No local pretraining. Many planned sizes/configurations are unavailable or resource-limited. Smoke subsets remain clearly separate. |
| Published evidence | 300 field-level rows over 15 configurations. | 104 fields remain uncertain; estimated values are explicitly classified and no missing compute is imputed. |
| Cost instrumentation | Per-run wall time and selected memory/backend fields are retained. | Energy and actual FLOPs are incomplete. ResNet timing anomaly is preserved and qualified. Windows/DirectML and Mac/MPS times are not pooled. |

The unchanged strict completion score is 0/50. This is a completion-criteria status, not a count of executed experiments. See `FINAL_COMPLETION_MATRIX.md`, `PROJECT_AUDIT.md`, and the final report.
