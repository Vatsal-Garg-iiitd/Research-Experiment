# Statistical analysis of strengthened runs

Analyses use saved local measurements from MLP/LeNet/ResNet/custom-ViT epoch sweeps, a Random Forest tree curve, MiniBatch K-Means/GMM optimization curves, AE/VAE reconstruction and linear-probe sweeps, pseudo-label/labeled-only ablation, RL budget curves, and a separate DBSCAN density-parameter grid. Most learning curves use a fixed 10k-train/2k-validation Fashion-MNIST subset; RL uses CartPole/Pendulum and DBSCAN has transductive validation. These are exploratory partial results, not a completed 20-model confirmatory study.

- Validation summary rows: 3614.
- Paired within-seed adjacent-budget marginal increments: 2407.
- Held-out test endpoint summary rows: 169.
- Summaries report mean, median, standard deviation, seed count, and two-sided 95% Student t intervals where n > 1. Intervals are wide with n=3 and do not establish generalization.
- Budget observations within a seed are repeated measures. Marginal increments are calculated within seed first; no checkpoint is treated as an independent seed.
- Two consecutive favorable relative gains below the stated screen: 0.5% threshold: 55 model/setting/seed series met the two-increment screen; 1% threshold: 80 model/setting/seed series met the two-increment screen; 2% threshold: 96 model/setting/seed series met the two-increment screen. These are descriptive flags; short curves and three-seed uncertainty do not establish general saturation.
- Test results are aggregated only at final checkpoints; validation data drives curves. No p-values or cross-model significance claims are reported.
- Initial AE/VAE v1 probe rows are retained but superseded after convergence warnings; analyses use the standardized v2 probe.

## Final held-out test endpoints

| Run group | Metric | Seeds | Mean | Median | SD | 95% CI |
|---|---|---:|---:|---:|---:|---|
| MLP / 24574ee00a20ab31 / 5 | test_accuracy | 3 | 0.8279 | 0.8247 | 0.0060 | [0.8132, 0.8427] |
| MLP / 24574ee00a20ab31 / 5 | test_loss | 3 | 0.4911 | 0.4989 | 0.0165 | [0.4501, 0.5321] |
| MLP / 24574ee00a20ab31 / 5 | macro_precision | 3 | 0.8350 | 0.8349 | 0.0048 | [0.8230, 0.8470] |
| MLP / 24574ee00a20ab31 / 5 | macro_recall | 3 | 0.8279 | 0.8247 | 0.0060 | [0.8132, 0.8427] |
| MLP / 24574ee00a20ab31 / 5 | macro_f1 | 3 | 0.8232 | 0.8228 | 0.0091 | [0.8007, 0.8457] |
| CNN / LeNet / 24574ee00a20ab31 / 5 | test_accuracy | 3 | 0.8368 | 0.8350 | 0.0034 | [0.8284, 0.8452] |
| CNN / LeNet / 24574ee00a20ab31 / 5 | test_loss | 3 | 0.4438 | 0.4446 | 0.0029 | [0.4365, 0.4511] |
| CNN / LeNet / 24574ee00a20ab31 / 5 | macro_precision | 3 | 0.8437 | 0.8433 | 0.0055 | [0.8300, 0.8574] |
| CNN / LeNet / 24574ee00a20ab31 / 5 | macro_recall | 3 | 0.8368 | 0.8350 | 0.0034 | [0.8284, 0.8452] |
| CNN / LeNet / 24574ee00a20ab31 / 5 | macro_f1 | 3 | 0.8322 | 0.8287 | 0.0081 | [0.8122, 0.8522] |
| ResNet-18 / 24574ee00a20ab31 / 5 | test_accuracy | 3 | 0.8400 | 0.8458 | 0.0116 | [0.8113, 0.8688] |
| ResNet-18 / 24574ee00a20ab31 / 5 | test_loss | 3 | 0.4664 | 0.4711 | 0.0190 | [0.4191, 0.5137] |
| ResNet-18 / 24574ee00a20ab31 / 5 | macro_precision | 3 | 0.8531 | 0.8532 | 0.0011 | [0.8504, 0.8559] |
| ResNet-18 / 24574ee00a20ab31 / 5 | macro_recall | 3 | 0.8400 | 0.8458 | 0.0116 | [0.8113, 0.8688] |
| ResNet-18 / 24574ee00a20ab31 / 5 | macro_f1 | 3 | 0.8411 | 0.8459 | 0.0085 | [0.8200, 0.8622] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 5 | test_accuracy | 3 | 0.8065 | 0.8068 | 0.0035 | [0.7978, 0.8152] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 5 | test_loss | 3 | 0.5404 | 0.5364 | 0.0139 | [0.5058, 0.5749] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 5 | macro_precision | 3 | 0.8127 | 0.8150 | 0.0042 | [0.8022, 0.8233] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 5 | macro_recall | 3 | 0.8065 | 0.8068 | 0.0035 | [0.7978, 0.8152] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 5 | macro_f1 | 3 | 0.8012 | 0.8007 | 0.0050 | [0.7889, 0.8136] |
| MLP / c1427ea14824cbdc / 5 | test_accuracy | 3 | 0.8537 | 0.8556 | 0.0081 | [0.8336, 0.8737] |
| MLP / c1427ea14824cbdc / 5 | test_loss | 3 | 0.4073 | 0.4023 | 0.0158 | [0.3680, 0.4466] |
| MLP / c1427ea14824cbdc / 5 | macro_precision | 3 | 0.8609 | 0.8618 | 0.0028 | [0.8541, 0.8678] |
| MLP / c1427ea14824cbdc / 5 | macro_recall | 3 | 0.8537 | 0.8556 | 0.0081 | [0.8336, 0.8737] |
| MLP / c1427ea14824cbdc / 5 | macro_f1 | 3 | 0.8537 | 0.8524 | 0.0069 | [0.8364, 0.8709] |
| CNN / LeNet / c1427ea14824cbdc / 5 | test_accuracy | 3 | 0.8784 | 0.8768 | 0.0035 | [0.8699, 0.8870] |
| CNN / LeNet / c1427ea14824cbdc / 5 | test_loss | 3 | 0.3355 | 0.3426 | 0.0148 | [0.2987, 0.3724] |
| CNN / LeNet / c1427ea14824cbdc / 5 | macro_precision | 3 | 0.8828 | 0.8830 | 0.0004 | [0.8818, 0.8838] |
| CNN / LeNet / c1427ea14824cbdc / 5 | macro_recall | 3 | 0.8784 | 0.8768 | 0.0035 | [0.8699, 0.8870] |
| CNN / LeNet / c1427ea14824cbdc / 5 | macro_f1 | 3 | 0.8774 | 0.8772 | 0.0051 | [0.8647, 0.8900] |
| MLP / 4116dae0312b5981 / 5 | test_accuracy | 3 | 0.8689 | 0.8692 | 0.0008 | [0.8669, 0.8710] |
| MLP / 4116dae0312b5981 / 5 | test_loss | 3 | 0.3615 | 0.3617 | 0.0026 | [0.3550, 0.3681] |
| MLP / 4116dae0312b5981 / 5 | macro_precision | 3 | 0.8711 | 0.8719 | 0.0018 | [0.8666, 0.8757] |
| MLP / 4116dae0312b5981 / 5 | macro_recall | 3 | 0.8689 | 0.8692 | 0.0008 | [0.8669, 0.8710] |
| MLP / 4116dae0312b5981 / 5 | macro_f1 | 3 | 0.8683 | 0.8682 | 0.0014 | [0.8647, 0.8719] |
| CNN / LeNet / 4116dae0312b5981 / 5 | test_accuracy | 3 | 0.8989 | 0.8975 | 0.0035 | [0.8903, 0.9075] |
| CNN / LeNet / 4116dae0312b5981 / 5 | test_loss | 3 | 0.2856 | 0.2908 | 0.0092 | [0.2629, 0.3084] |
| CNN / LeNet / 4116dae0312b5981 / 5 | macro_precision | 3 | 0.8994 | 0.8985 | 0.0029 | [0.8923, 0.9066] |
| CNN / LeNet / 4116dae0312b5981 / 5 | macro_recall | 3 | 0.8989 | 0.8975 | 0.0035 | [0.8903, 0.9075] |
| CNN / LeNet / 4116dae0312b5981 / 5 | macro_f1 | 3 | 0.8978 | 0.8956 | 0.0040 | [0.8878, 0.9078] |
| MLP / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8612 | 0.8626 | 0.0041 | [0.8511, 0.8713] |
| MLP / 24574ee00a20ab31 / 30 | test_loss | 3 | 0.4218 | 0.4184 | 0.0120 | [0.3921, 0.4515] |
| MLP / 24574ee00a20ab31 / 30 | macro_precision | 3 | 0.8637 | 0.8644 | 0.0013 | [0.8605, 0.8669] |
| MLP / 24574ee00a20ab31 / 30 | macro_recall | 3 | 0.8612 | 0.8626 | 0.0041 | [0.8511, 0.8713] |
| MLP / 24574ee00a20ab31 / 30 | macro_f1 | 3 | 0.8614 | 0.8632 | 0.0035 | [0.8526, 0.8701] |
| CNN / LeNet / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8891 | 0.8887 | 0.0011 | [0.8863, 0.8918] |
| CNN / LeNet / 24574ee00a20ab31 / 30 | test_loss | 3 | 0.3813 | 0.3811 | 0.0148 | [0.3445, 0.4181] |
| CNN / LeNet / 24574ee00a20ab31 / 30 | macro_precision | 3 | 0.8909 | 0.8921 | 0.0021 | [0.8857, 0.8961] |
| CNN / LeNet / 24574ee00a20ab31 / 30 | macro_recall | 3 | 0.8891 | 0.8887 | 0.0011 | [0.8863, 0.8918] |
| CNN / LeNet / 24574ee00a20ab31 / 30 | macro_f1 | 3 | 0.8891 | 0.8888 | 0.0014 | [0.8857, 0.8926] |
| ResNet-18 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8646 | 0.8611 | 0.0078 | [0.8453, 0.8839] |
| ResNet-18 / 24574ee00a20ab31 / 30 | test_loss | 3 | 0.6132 | 0.6000 | 0.0234 | [0.5552, 0.6713] |
| ResNet-18 / 24574ee00a20ab31 / 30 | macro_precision | 3 | 0.8675 | 0.8675 | 0.0056 | [0.8536, 0.8813] |
| ResNet-18 / 24574ee00a20ab31 / 30 | macro_recall | 3 | 0.8646 | 0.8611 | 0.0078 | [0.8453, 0.8839] |
| ResNet-18 / 24574ee00a20ab31 / 30 | macro_f1 | 3 | 0.8645 | 0.8605 | 0.0072 | [0.8466, 0.8825] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8401 | 0.8431 | 0.0070 | [0.8228, 0.8574] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 30 | test_loss | 3 | 0.5063 | 0.5091 | 0.0291 | [0.4339, 0.5787] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 30 | macro_precision | 3 | 0.8456 | 0.8456 | 0.0026 | [0.8391, 0.8521] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 30 | macro_recall | 3 | 0.8401 | 0.8431 | 0.0070 | [0.8228, 0.8574] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 30 | macro_f1 | 3 | 0.8407 | 0.8437 | 0.0053 | [0.8275, 0.8539] |
| ViT-Tiny / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.7621 | 0.7890 | 0.0616 | [0.6090, 0.9152] |
| ViT-Tiny / 24574ee00a20ab31 / 30 | test_loss | 3 | 0.6395 | 0.5788 | 0.1459 | [0.2771, 1.0019] |
| ViT-Tiny / 24574ee00a20ab31 / 30 | macro_precision | 3 | 0.7682 | 0.8000 | 0.0629 | [0.6120, 0.9244] |
| ViT-Tiny / 24574ee00a20ab31 / 30 | macro_recall | 3 | 0.7621 | 0.7890 | 0.0616 | [0.6090, 0.9152] |
| ViT-Tiny / 24574ee00a20ab31 / 30 | macro_f1 | 3 | 0.7571 | 0.7909 | 0.0710 | [0.5806, 0.9336] |
| Random Forest /  / 400 | test_accuracy | 3 | 0.8524 | 0.8526 | 0.0009 | [0.8503, 0.8546] |
| Random Forest /  / 400 | test_log_loss | 3 | 0.4662 | 0.4644 | 0.0033 | [0.4580, 0.4745] |
| Random Forest /  / 400 | macro_precision | 3 | 0.8510 | 0.8511 | 0.0010 | [0.8486, 0.8534] |
| Random Forest /  / 400 | macro_recall | 3 | 0.8524 | 0.8526 | 0.0009 | [0.8503, 0.8546] |
| Random Forest /  / 400 | macro_f1 | 3 | 0.8510 | 0.8511 | 0.0009 | [0.8488, 0.8532] |
| Random Forest /  / 1000 | test_accuracy | 3 | 0.8534 | 0.8530 | 0.0010 | [0.8509, 0.8560] |
| Random Forest /  / 1000 | test_log_loss | 3 | 0.4598 | 0.4597 | 0.0002 | [0.4593, 0.4602] |
| Random Forest /  / 1000 | macro_precision | 3 | 0.8521 | 0.8517 | 0.0010 | [0.8495, 0.8547] |
| Random Forest /  / 1000 | macro_recall | 3 | 0.8534 | 0.8530 | 0.0010 | [0.8509, 0.8560] |
| Random Forest /  / 1000 | macro_f1 | 3 | 0.8521 | 0.8517 | 0.0010 | [0.8496, 0.8546] |
| MLP / labeled_only / 0.01 /  / 5 | test_accuracy | 3 | 0.6575 | 0.6519 | 0.0152 | [0.6198, 0.6952] |
| MLP / labeled_only / 0.01 /  / 5 | test_loss | 3 | 1.4999 | 1.5203 | 0.1241 | [1.1916, 1.8082] |
| MLP / labeled_only / 0.01 /  / 5 | macro_precision | 3 | 0.6935 | 0.6950 | 0.0048 | [0.6816, 0.7054] |
| MLP / labeled_only / 0.01 /  / 5 | macro_recall | 3 | 0.6575 | 0.6519 | 0.0152 | [0.6198, 0.6952] |
| MLP / labeled_only / 0.01 /  / 5 | macro_f1 | 3 | 0.6526 | 0.6581 | 0.0140 | [0.6179, 0.6873] |
| MLP / pseudo_label / 0.01 /  / 5 | test_accuracy | 3 | 0.6369 | 0.6330 | 0.0105 | [0.6108, 0.6630] |
| MLP / pseudo_label / 0.01 /  / 5 | test_loss | 3 | 3.4565 | 3.3167 | 0.3077 | [2.6922, 4.2208] |
| MLP / pseudo_label / 0.01 /  / 5 | macro_precision | 3 | 0.7042 | 0.7209 | 0.0351 | [0.6171, 0.7913] |
| MLP / pseudo_label / 0.01 /  / 5 | macro_recall | 3 | 0.6369 | 0.6330 | 0.0105 | [0.6108, 0.6630] |
| MLP / pseudo_label / 0.01 /  / 5 | macro_f1 | 3 | 0.6086 | 0.6161 | 0.0194 | [0.5604, 0.6569] |
| MLP / labeled_only / 0.05 /  / 5 | test_accuracy | 3 | 0.7718 | 0.7719 | 0.0005 | [0.7706, 0.7731] |
| MLP / labeled_only / 0.05 /  / 5 | test_loss | 3 | 0.8468 | 0.8415 | 0.0439 | [0.7378, 0.9558] |
| MLP / labeled_only / 0.05 /  / 5 | macro_precision | 3 | 0.7797 | 0.7810 | 0.0023 | [0.7741, 0.7853] |
| MLP / labeled_only / 0.05 /  / 5 | macro_recall | 3 | 0.7718 | 0.7719 | 0.0005 | [0.7706, 0.7731] |
| MLP / labeled_only / 0.05 /  / 5 | macro_f1 | 3 | 0.7722 | 0.7729 | 0.0017 | [0.7681, 0.7763] |
| MLP / pseudo_label / 0.05 /  / 5 | test_accuracy | 3 | 0.7844 | 0.7825 | 0.0074 | [0.7659, 0.8029] |
| MLP / pseudo_label / 0.05 /  / 5 | test_loss | 3 | 0.9774 | 1.0095 | 0.0855 | [0.7651, 1.1897] |
| MLP / pseudo_label / 0.05 /  / 5 | macro_precision | 3 | 0.7899 | 0.7926 | 0.0077 | [0.7707, 0.8091] |
| MLP / pseudo_label / 0.05 /  / 5 | macro_recall | 3 | 0.7844 | 0.7825 | 0.0074 | [0.7659, 0.8029] |
| MLP / pseudo_label / 0.05 /  / 5 | macro_f1 | 3 | 0.7832 | 0.7801 | 0.0074 | [0.7648, 0.8017] |
| MLP / labeled_only / 0.1 /  / 5 | test_accuracy | 3 | 0.8045 | 0.8053 | 0.0041 | [0.7943, 0.8147] |
| MLP / labeled_only / 0.1 /  / 5 | test_loss | 3 | 0.6149 | 0.6134 | 0.0096 | [0.5910, 0.6389] |
| MLP / labeled_only / 0.1 /  / 5 | macro_precision | 3 | 0.8073 | 0.8084 | 0.0041 | [0.7971, 0.8175] |
| MLP / labeled_only / 0.1 /  / 5 | macro_recall | 3 | 0.8045 | 0.8053 | 0.0041 | [0.7943, 0.8147] |
| MLP / labeled_only / 0.1 /  / 5 | macro_f1 | 3 | 0.8035 | 0.8046 | 0.0031 | [0.7958, 0.8111] |
| MLP / pseudo_label / 0.1 /  / 5 | test_accuracy | 3 | 0.8073 | 0.8062 | 0.0038 | [0.7978, 0.8168] |
| MLP / pseudo_label / 0.1 /  / 5 | test_loss | 3 | 0.6656 | 0.6715 | 0.0211 | [0.6132, 0.7180] |
| MLP / pseudo_label / 0.1 /  / 5 | macro_precision | 3 | 0.8084 | 0.8069 | 0.0033 | [0.8003, 0.8165] |
| MLP / pseudo_label / 0.1 /  / 5 | macro_recall | 3 | 0.8073 | 0.8062 | 0.0038 | [0.7978, 0.8168] |
| MLP / pseudo_label / 0.1 /  / 5 | macro_f1 | 3 | 0.8054 | 0.8052 | 0.0046 | [0.7939, 0.8168] |
| MLP / labeled_only / 0.2 /  / 5 | test_accuracy | 3 | 0.8161 | 0.8155 | 0.0017 | [0.8119, 0.8203] |
| MLP / labeled_only / 0.2 /  / 5 | test_loss | 3 | 0.5367 | 0.5367 | 0.0007 | [0.5350, 0.5384] |
| MLP / labeled_only / 0.2 /  / 5 | macro_precision | 3 | 0.8183 | 0.8175 | 0.0022 | [0.8128, 0.8238] |
| MLP / labeled_only / 0.2 /  / 5 | macro_recall | 3 | 0.8161 | 0.8155 | 0.0017 | [0.8119, 0.8203] |
| MLP / labeled_only / 0.2 /  / 5 | macro_f1 | 3 | 0.8123 | 0.8129 | 0.0044 | [0.8014, 0.8232] |
| MLP / pseudo_label / 0.2 /  / 5 | test_accuracy | 3 | 0.8147 | 0.8135 | 0.0021 | [0.8094, 0.8200] |
| MLP / pseudo_label / 0.2 /  / 5 | test_loss | 3 | 0.5664 | 0.5666 | 0.0027 | [0.5598, 0.5730] |
| MLP / pseudo_label / 0.2 /  / 5 | macro_precision | 3 | 0.8161 | 0.8157 | 0.0007 | [0.8143, 0.8179] |
| MLP / pseudo_label / 0.2 /  / 5 | macro_recall | 3 | 0.8147 | 0.8135 | 0.0021 | [0.8094, 0.8200] |
| MLP / pseudo_label / 0.2 /  / 5 | macro_f1 | 3 | 0.8103 | 0.8105 | 0.0047 | [0.7986, 0.8219] |
| MLP / labeled_only / 0.4 /  / 5 | test_accuracy | 3 | 0.8176 | 0.8173 | 0.0050 | [0.8051, 0.8300] |
| MLP / labeled_only / 0.4 /  / 5 | test_loss | 3 | 0.5177 | 0.5214 | 0.0090 | [0.4953, 0.5400] |
| MLP / labeled_only / 0.4 /  / 5 | macro_precision | 3 | 0.8185 | 0.8172 | 0.0045 | [0.8073, 0.8298] |
| MLP / labeled_only / 0.4 /  / 5 | macro_recall | 3 | 0.8176 | 0.8173 | 0.0050 | [0.8051, 0.8300] |
| MLP / labeled_only / 0.4 /  / 5 | macro_f1 | 3 | 0.8141 | 0.8112 | 0.0051 | [0.8015, 0.8267] |
| MLP / pseudo_label / 0.4 /  / 5 | test_accuracy | 3 | 0.8147 | 0.8142 | 0.0040 | [0.8047, 0.8247] |
| MLP / pseudo_label / 0.4 /  / 5 | test_loss | 3 | 0.5328 | 0.5285 | 0.0152 | [0.4949, 0.5707] |
| MLP / pseudo_label / 0.4 /  / 5 | macro_precision | 3 | 0.8161 | 0.8147 | 0.0032 | [0.8082, 0.8240] |
| MLP / pseudo_label / 0.4 /  / 5 | macro_recall | 3 | 0.8147 | 0.8142 | 0.0040 | [0.8047, 0.8247] |
| MLP / pseudo_label / 0.4 /  / 5 | macro_f1 | 3 | 0.8107 | 0.8124 | 0.0075 | [0.7921, 0.8292] |
| MeanTeacher / 0.01 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.6519 | 0.6452 | 0.0143 | [0.6164, 0.6874] |
| FixMatch / 0.01 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.6619 | 0.6626 | 0.0032 | [0.6541, 0.6698] |
| MixMatch / 0.01 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.6586 | 0.6643 | 0.0110 | [0.6314, 0.6859] |
| MeanTeacher / 0.05 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7713 | 0.7745 | 0.0069 | [0.7542, 0.7884] |
| FixMatch / 0.05 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7644 | 0.7627 | 0.0096 | [0.7405, 0.7882] |
| MixMatch / 0.05 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7634 | 0.7680 | 0.0086 | [0.7419, 0.7848] |
| MeanTeacher / 0.1 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7802 | 0.7835 | 0.0084 | [0.7594, 0.8010] |
| FixMatch / 0.1 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7644 | 0.7664 | 0.0208 | [0.7127, 0.8162] |
| MixMatch / 0.1 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7572 | 0.7584 | 0.0036 | [0.7482, 0.7661] |
| MeanTeacher / 0.2 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7890 | 0.7885 | 0.0086 | [0.7677, 0.8103] |
| FixMatch / 0.2 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7855 | 0.7851 | 0.0018 | [0.7811, 0.7900] |
| MixMatch / 0.2 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7784 | 0.7785 | 0.0020 | [0.7734, 0.7833] |
| MeanTeacher / 0.4 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7940 | 0.7936 | 0.0016 | [0.7900, 0.7979] |
| FixMatch / 0.4 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7926 | 0.7941 | 0.0030 | [0.7852, 0.8000] |
| MixMatch / 0.4 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7784 | 0.7773 | 0.0084 | [0.7577, 0.7992] |
| MeanTeacher / 0.01 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.6576 | 0.6585 | 0.0125 | [0.6265, 0.6887] |
| FixMatch / 0.01 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.6627 | 0.6580 | 0.0118 | [0.6333, 0.6921] |
| MixMatch / 0.01 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.6575 | 0.6592 | 0.0107 | [0.6309, 0.6841] |
| MeanTeacher / 0.05 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.7851 | 0.7856 | 0.0060 | [0.7703, 0.7999] |
| FixMatch / 0.05 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.7821 | 0.7763 | 0.0116 | [0.7533, 0.8110] |
| MixMatch / 0.05 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.7879 | 0.7919 | 0.0080 | [0.7681, 0.8076] |
| MeanTeacher / 0.1 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8070 | 0.8083 | 0.0037 | [0.7977, 0.8163] |
| FixMatch / 0.1 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8059 | 0.8126 | 0.0123 | [0.7754, 0.8364] |
| MixMatch / 0.1 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8095 | 0.8130 | 0.0069 | [0.7922, 0.8268] |
| MeanTeacher / 0.2 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8213 | 0.8222 | 0.0028 | [0.8143, 0.8283] |
| FixMatch / 0.2 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8232 | 0.8234 | 0.0006 | [0.8216, 0.8248] |
| MixMatch / 0.2 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8302 | 0.8280 | 0.0058 | [0.8157, 0.8447] |
| MeanTeacher / 0.4 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8416 | 0.8418 | 0.0004 | [0.8405, 0.8427] |
| FixMatch / 0.4 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8381 | 0.8385 | 0.0040 | [0.8282, 0.8479] |
| MixMatch / 0.4 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8372 | 0.8379 | 0.0040 | [0.8272, 0.8473] |
| Autoencoder / 24574ee00a20ab31 / 5 | test_reconstruction_mse | 3 | 0.0196 | 0.0195 | 0.0003 | [0.0189, 0.0202] |
| Autoencoder / 24574ee00a20ab31 / 5 | test_linear_probe_accuracy | 3 | 0.7917 | 0.7910 | 0.0033 | [0.7836, 0.7998] |
| Variational Autoencoder / 24574ee00a20ab31 / 5 | test_reconstruction_mse | 3 | 0.0292 | 0.0292 | 0.0001 | [0.0291, 0.0294] |
| Variational Autoencoder / 24574ee00a20ab31 / 5 | test_linear_probe_accuracy | 3 | 0.7838 | 0.7853 | 0.0043 | [0.7732, 0.7944] |
| Autoencoder / 24574ee00a20ab31 / 30 | test_reconstruction_mse | 3 | 0.0119 | 0.0119 | 0.0002 | [0.0115, 0.0123] |
| Autoencoder / 24574ee00a20ab31 / 30 | test_linear_probe_accuracy | 3 | 0.8120 | 0.8126 | 0.0016 | [0.8079, 0.8161] |
| Variational Autoencoder / 24574ee00a20ab31 / 30 | test_reconstruction_mse | 3 | 0.0204 | 0.0205 | 0.0003 | [0.0195, 0.0212] |
| Variational Autoencoder / 24574ee00a20ab31 / 30 | test_linear_probe_accuracy | 3 | 0.7962 | 0.7976 | 0.0027 | [0.7895, 0.8030] |
| DQN / CartPole-v1 | mean_return | 3 | 34.5000 | 43.0500 | 21.8198 | [-19.7033, 88.7033] |
| A2C / CartPole-v1 | mean_return | 3 | 442.5500 | 484.3000 | 83.8238 | [234.3200, 650.7800] |
| PPO / CartPole-v1 | mean_return | 3 | 405.3000 | 465.0500 | 134.4327 | [71.3506, 739.2494] |
| SAC / Pendulum-v1 | mean_return | 5 | -171.1897 | -174.1756 | 10.8446 | [-184.6551, -157.7243] |
| Double DQN / CartPole-v1 | mean_return | 5 | 190.2400 | 166.1000 | 63.6508 | [111.2071, 269.2729] |
| PPO-Pendulum / Pendulum-v1 | mean_return | 3 | -1260.9494 | -1208.2417 | 139.2603 | [-1606.8911, -915.0076] |

## Interpretation limits

The MLP, LeNet, ResNet-18, compact ViT-style, Random Forest, AE/VAE and RL curves use three training seeds on fixed splits/subsets; uncertainty is conditional on those splits and does not quantify dataset-sampling uncertainty. K-Means/GMM ARI varies non-monotonically across updates/iterations and seeds; no saturation model is fit. DBSCAN is a deterministic parameter grid, not a seed-level budget curve. Pseudo-label vs labeled-only comparisons are paired by seed and labeled subset; current results vary with fraction and seed, so no universal SSL benefit is claimed. RL uses 20 deterministic evaluation episodes per checkpoint and three training seeds; within-run episode dispersion is separately retained. Historical pilot curves in `experiment_results.csv` include repeated test-set looks and remain exploratory.
