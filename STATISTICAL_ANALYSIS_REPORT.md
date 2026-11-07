# Statistical analysis of strengthened runs

Analyses use saved local measurements from MLP/LeNet/ResNet/custom-ViT epoch sweeps, a Random Forest tree curve, MiniBatch K-Means/GMM optimization curves, AE/VAE reconstruction and linear-probe sweeps, pseudo-label/labeled-only ablation, RL budget curves, and a separate DBSCAN density-parameter grid. Most learning curves use a fixed 10k-train/2k-validation Fashion-MNIST subset; RL uses CartPole/Pendulum and DBSCAN has transductive validation. These are exploratory partial results, not a completed 20-model confirmatory study.

- Validation summary rows: 458.
- Paired within-seed adjacent-budget marginal increments: 279.
- Held-out test endpoint summary rows: 84.
- Confidence intervals: two-sided 95% Student t intervals across seeds; n is three for the new sweeps. Intervals are wide with n=3 and do not establish generalization.
- Budget observations within a seed are repeated measures. Marginal increments are calculated within seed first; no checkpoint is treated as an independent seed.
- Test results are aggregated only at final checkpoints; validation data drives curves. No p-values or cross-model significance claims are reported.
- Initial AE/VAE v1 probe rows are retained but superseded after convergence warnings; analyses use the standardized v2 probe.

## Final held-out test endpoints

| Run group | Metric | Seeds | Mean | SD | 95% CI |
|---|---|---:|---:|---:|---|
| MLP | test_accuracy | 3 | 0.8279 | 0.0060 | [0.8132, 0.8427] |
| MLP | test_loss | 3 | 0.4911 | 0.0165 | [0.4501, 0.5321] |
| MLP | macro_precision | 3 | 0.8350 | 0.0048 | [0.8230, 0.8470] |
| MLP | macro_recall | 3 | 0.8279 | 0.0060 | [0.8132, 0.8427] |
| MLP | macro_f1 | 3 | 0.8232 | 0.0091 | [0.8007, 0.8457] |
| CNN / LeNet | test_accuracy | 3 | 0.8368 | 0.0034 | [0.8284, 0.8452] |
| CNN / LeNet | test_loss | 3 | 0.4438 | 0.0029 | [0.4365, 0.4511] |
| CNN / LeNet | macro_precision | 3 | 0.8437 | 0.0055 | [0.8300, 0.8574] |
| CNN / LeNet | macro_recall | 3 | 0.8368 | 0.0034 | [0.8284, 0.8452] |
| CNN / LeNet | macro_f1 | 3 | 0.8322 | 0.0081 | [0.8122, 0.8522] |
| ResNet-18 | test_accuracy | 3 | 0.8400 | 0.0116 | [0.8113, 0.8688] |
| ResNet-18 | test_loss | 3 | 0.4664 | 0.0190 | [0.4191, 0.5137] |
| ResNet-18 | macro_precision | 3 | 0.8531 | 0.0011 | [0.8504, 0.8559] |
| ResNet-18 | macro_recall | 3 | 0.8400 | 0.0116 | [0.8113, 0.8688] |
| ResNet-18 | macro_f1 | 3 | 0.8411 | 0.0085 | [0.8200, 0.8622] |
| Vision Transformer (compact ViT-Tiny-style) | test_accuracy | 3 | 0.8065 | 0.0035 | [0.7978, 0.8152] |
| Vision Transformer (compact ViT-Tiny-style) | test_loss | 3 | 0.5404 | 0.0139 | [0.5058, 0.5749] |
| Vision Transformer (compact ViT-Tiny-style) | macro_precision | 3 | 0.8127 | 0.0042 | [0.8022, 0.8233] |
| Vision Transformer (compact ViT-Tiny-style) | macro_recall | 3 | 0.8065 | 0.0035 | [0.7978, 0.8152] |
| Vision Transformer (compact ViT-Tiny-style) | macro_f1 | 3 | 0.8012 | 0.0050 | [0.7889, 0.8136] |
| Random Forest | test_accuracy | 3 | 0.8524 | 0.0009 | [0.8503, 0.8546] |
| Random Forest | test_log_loss | 3 | 0.4662 | 0.0033 | [0.4580, 0.4745] |
| Random Forest | macro_precision | 3 | 0.8510 | 0.0010 | [0.8486, 0.8534] |
| Random Forest | macro_recall | 3 | 0.8524 | 0.0009 | [0.8503, 0.8546] |
| Random Forest | macro_f1 | 3 | 0.8510 | 0.0009 | [0.8488, 0.8532] |
| MLP / labeled_only / 0.01 | test_accuracy | 3 | 0.6575 | 0.0152 | [0.6198, 0.6952] |
| MLP / labeled_only / 0.01 | test_loss | 3 | 1.4999 | 0.1241 | [1.1916, 1.8082] |
| MLP / labeled_only / 0.01 | macro_precision | 3 | 0.6935 | 0.0048 | [0.6816, 0.7054] |
| MLP / labeled_only / 0.01 | macro_recall | 3 | 0.6575 | 0.0152 | [0.6198, 0.6952] |
| MLP / labeled_only / 0.01 | macro_f1 | 3 | 0.6526 | 0.0140 | [0.6179, 0.6873] |
| MLP / pseudo_label / 0.01 | test_accuracy | 3 | 0.6369 | 0.0105 | [0.6108, 0.6630] |
| MLP / pseudo_label / 0.01 | test_loss | 3 | 3.4565 | 0.3077 | [2.6922, 4.2208] |
| MLP / pseudo_label / 0.01 | macro_precision | 3 | 0.7042 | 0.0351 | [0.6171, 0.7913] |
| MLP / pseudo_label / 0.01 | macro_recall | 3 | 0.6369 | 0.0105 | [0.6108, 0.6630] |
| MLP / pseudo_label / 0.01 | macro_f1 | 3 | 0.6086 | 0.0194 | [0.5604, 0.6569] |
| MLP / labeled_only / 0.05 | test_accuracy | 3 | 0.7718 | 0.0005 | [0.7706, 0.7731] |
| MLP / labeled_only / 0.05 | test_loss | 3 | 0.8468 | 0.0439 | [0.7378, 0.9558] |
| MLP / labeled_only / 0.05 | macro_precision | 3 | 0.7797 | 0.0023 | [0.7741, 0.7853] |
| MLP / labeled_only / 0.05 | macro_recall | 3 | 0.7718 | 0.0005 | [0.7706, 0.7731] |
| MLP / labeled_only / 0.05 | macro_f1 | 3 | 0.7722 | 0.0017 | [0.7681, 0.7763] |
| MLP / pseudo_label / 0.05 | test_accuracy | 3 | 0.7844 | 0.0074 | [0.7659, 0.8029] |
| MLP / pseudo_label / 0.05 | test_loss | 3 | 0.9774 | 0.0855 | [0.7651, 1.1897] |
| MLP / pseudo_label / 0.05 | macro_precision | 3 | 0.7899 | 0.0077 | [0.7707, 0.8091] |
| MLP / pseudo_label / 0.05 | macro_recall | 3 | 0.7844 | 0.0074 | [0.7659, 0.8029] |
| MLP / pseudo_label / 0.05 | macro_f1 | 3 | 0.7832 | 0.0074 | [0.7648, 0.8017] |
| MLP / labeled_only / 0.1 | test_accuracy | 3 | 0.8045 | 0.0041 | [0.7943, 0.8147] |
| MLP / labeled_only / 0.1 | test_loss | 3 | 0.6149 | 0.0096 | [0.5910, 0.6389] |
| MLP / labeled_only / 0.1 | macro_precision | 3 | 0.8073 | 0.0041 | [0.7971, 0.8175] |
| MLP / labeled_only / 0.1 | macro_recall | 3 | 0.8045 | 0.0041 | [0.7943, 0.8147] |
| MLP / labeled_only / 0.1 | macro_f1 | 3 | 0.8035 | 0.0031 | [0.7958, 0.8111] |
| MLP / pseudo_label / 0.1 | test_accuracy | 3 | 0.8073 | 0.0038 | [0.7978, 0.8168] |
| MLP / pseudo_label / 0.1 | test_loss | 3 | 0.6656 | 0.0211 | [0.6132, 0.7180] |
| MLP / pseudo_label / 0.1 | macro_precision | 3 | 0.8084 | 0.0033 | [0.8003, 0.8165] |
| MLP / pseudo_label / 0.1 | macro_recall | 3 | 0.8073 | 0.0038 | [0.7978, 0.8168] |
| MLP / pseudo_label / 0.1 | macro_f1 | 3 | 0.8054 | 0.0046 | [0.7939, 0.8168] |
| MLP / labeled_only / 0.2 | test_accuracy | 3 | 0.8161 | 0.0017 | [0.8119, 0.8203] |
| MLP / labeled_only / 0.2 | test_loss | 3 | 0.5367 | 0.0007 | [0.5350, 0.5384] |
| MLP / labeled_only / 0.2 | macro_precision | 3 | 0.8183 | 0.0022 | [0.8128, 0.8238] |
| MLP / labeled_only / 0.2 | macro_recall | 3 | 0.8161 | 0.0017 | [0.8119, 0.8203] |
| MLP / labeled_only / 0.2 | macro_f1 | 3 | 0.8123 | 0.0044 | [0.8014, 0.8232] |
| MLP / pseudo_label / 0.2 | test_accuracy | 3 | 0.8147 | 0.0021 | [0.8094, 0.8200] |
| MLP / pseudo_label / 0.2 | test_loss | 3 | 0.5664 | 0.0027 | [0.5598, 0.5730] |
| MLP / pseudo_label / 0.2 | macro_precision | 3 | 0.8161 | 0.0007 | [0.8143, 0.8179] |
| MLP / pseudo_label / 0.2 | macro_recall | 3 | 0.8147 | 0.0021 | [0.8094, 0.8200] |
| MLP / pseudo_label / 0.2 | macro_f1 | 3 | 0.8103 | 0.0047 | [0.7986, 0.8219] |
| MLP / labeled_only / 0.4 | test_accuracy | 3 | 0.8176 | 0.0050 | [0.8051, 0.8300] |
| MLP / labeled_only / 0.4 | test_loss | 3 | 0.5177 | 0.0090 | [0.4953, 0.5400] |
| MLP / labeled_only / 0.4 | macro_precision | 3 | 0.8185 | 0.0045 | [0.8073, 0.8298] |
| MLP / labeled_only / 0.4 | macro_recall | 3 | 0.8176 | 0.0050 | [0.8051, 0.8300] |
| MLP / labeled_only / 0.4 | macro_f1 | 3 | 0.8141 | 0.0051 | [0.8015, 0.8267] |
| MLP / pseudo_label / 0.4 | test_accuracy | 3 | 0.8147 | 0.0040 | [0.8047, 0.8247] |
| MLP / pseudo_label / 0.4 | test_loss | 3 | 0.5328 | 0.0152 | [0.4949, 0.5707] |
| MLP / pseudo_label / 0.4 | macro_precision | 3 | 0.8161 | 0.0032 | [0.8082, 0.8240] |
| MLP / pseudo_label / 0.4 | macro_recall | 3 | 0.8147 | 0.0040 | [0.8047, 0.8247] |
| MLP / pseudo_label / 0.4 | macro_f1 | 3 | 0.8107 | 0.0075 | [0.7921, 0.8292] |
| Autoencoder | test_reconstruction_mse | 3 | 0.0196 | 0.0003 | [0.0189, 0.0202] |
| Autoencoder | test_linear_probe_accuracy | 3 | 0.7917 | 0.0033 | [0.7836, 0.7998] |
| Variational Autoencoder | test_reconstruction_mse | 3 | 0.0292 | 0.0001 | [0.0291, 0.0294] |
| Variational Autoencoder | test_linear_probe_accuracy | 3 | 0.7838 | 0.0043 | [0.7732, 0.7944] |
| DQN | mean_return | 3 | 34.5000 | 21.8198 | [-19.7033, 88.7033] |
| A2C | mean_return | 3 | 442.5500 | 83.8238 | [234.3200, 650.7800] |
| PPO | mean_return | 3 | 405.3000 | 134.4327 | [71.3506, 739.2494] |
| SAC | mean_return | 3 | -168.3397 | 14.2510 | [-203.7411, -132.9382] |
| Double DQN | mean_return | 3 | 146.2667 | 19.2313 | [98.4934, 194.0399] |

## Interpretation limits

The MLP, LeNet, ResNet-18, compact ViT-style, Random Forest, AE/VAE and RL curves use three training seeds on fixed splits/subsets; uncertainty is conditional on those splits and does not quantify dataset-sampling uncertainty. K-Means/GMM ARI varies non-monotonically across updates/iterations and seeds; no saturation model is fit. DBSCAN is a deterministic parameter grid, not a seed-level budget curve. Pseudo-label vs labeled-only comparisons are paired by seed and labeled subset; current results vary with fraction and seed, so no universal SSL benefit is claimed. RL uses 20 deterministic evaluation episodes per checkpoint and three training seeds; within-run episode dispersion is separately retained. Historical pilot curves in `experiment_results.csv` include repeated test-set looks and remain exploratory.
