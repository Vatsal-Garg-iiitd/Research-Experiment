# Statistical analysis of strengthened runs

Analyses use verified local measurements from MLP/LeNet/ResNet/transformer epoch sweeps, Random Forest, K-Means/GMM, AE/VAE, pseudo-labeling and its labeled-only comparator, other SSL methods, RL, and DBSCAN. Most learning curves use the fixed Fashion-MNIST 10K-train/2K-validation split; RL uses CartPole/Pendulum. All 20 registered conventional schedules pass the coverage audit. This is a descriptive multi-paradigm study, not a confirmatory cross-paradigm comparison. Pseudo-labeling and its matched baseline resumed from saved epoch-5 checkpoints to the original 30-epoch target; all other results were pre-existing records.

- Validation metric/budget summary rows: 4864.
- Paired within-seed adjacent-budget marginal transitions: 3167.
- Held-out endpoint metric groups: 219.
- Grouped summaries report mean, median, minimum, maximum, standard deviation, valid seed count, and two-sided 95% Student t intervals where n > 1. Intervals are imprecise with n=3 and describe seed variation conditional on the fixed split.
- Budget observations within a seed are repeated measures. Marginal increments pair only within model, seed, split, and setting; no checkpoint is treated as an independent seed.
- The dataset-size endpoint summary uses epoch 5 only. This prevents the 10K epoch-30 endpoints from being pooled into the matched 10K/30K/58K comparison.
- Fifteen later K-Means rows duplicate experiment IDs and reuse the 50-update state at earlier budgets. They remain in raw data but are excluded from the derived K-Means curves.
- Double DQN rows are restored to completed status and local_measurement source type in normalized outputs; success_rate is unavailable because the raw field was omitted.
- Two consecutive favorable relative gains below the stated screen: 0.5% threshold: 77 model/setting/seed series met the two-increment screen; 1% threshold: 101 model/setting/seed series met the two-increment screen; 2% threshold: 110 model/setting/seed series met the two-increment screen. These are descriptive flags; short curves and small seed counts do not establish general saturation.
- Final test endpoints are grouped by model, metric, split, and final budget; validation curves are analyzed separately. No p-values or cross-model significance claims are reported.
- Initial AE/VAE v1 probe rows are retained but superseded after convergence warnings; analyses use the standardized v2 probe.

## Final held-out test endpoints

| Run group | Metric | Seeds | Mean | Median | Min | Max | SD | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP / 24574ee00a20ab31 / 5 | test_accuracy | 3 | 0.8279 | 0.8247 | 0.8243 | 0.8348 | 0.0060 | [0.8132, 0.8427] |
| MLP / 24574ee00a20ab31 / 5 | test_loss | 3 | 0.4911 | 0.4989 | 0.4721 | 0.5022 | 0.0165 | [0.4501, 0.5321] |
| MLP / 24574ee00a20ab31 / 5 | macro_precision | 3 | 0.8350 | 0.8349 | 0.8302 | 0.8399 | 0.0048 | [0.8230, 0.8470] |
| MLP / 24574ee00a20ab31 / 5 | macro_recall | 3 | 0.8279 | 0.8247 | 0.8243 | 0.8348 | 0.0060 | [0.8132, 0.8427] |
| MLP / 24574ee00a20ab31 / 5 | macro_f1 | 3 | 0.8232 | 0.8228 | 0.8144 | 0.8325 | 0.0091 | [0.8007, 0.8457] |
| CNN / LeNet / 24574ee00a20ab31 / 5 | test_accuracy | 3 | 0.8368 | 0.8350 | 0.8347 | 0.8407 | 0.0034 | [0.8284, 0.8452] |
| CNN / LeNet / 24574ee00a20ab31 / 5 | test_loss | 3 | 0.4438 | 0.4446 | 0.4406 | 0.4463 | 0.0029 | [0.4365, 0.4511] |
| CNN / LeNet / 24574ee00a20ab31 / 5 | macro_precision | 3 | 0.8437 | 0.8433 | 0.8384 | 0.8494 | 0.0055 | [0.8300, 0.8574] |
| CNN / LeNet / 24574ee00a20ab31 / 5 | macro_recall | 3 | 0.8368 | 0.8350 | 0.8347 | 0.8407 | 0.0034 | [0.8284, 0.8452] |
| CNN / LeNet / 24574ee00a20ab31 / 5 | macro_f1 | 3 | 0.8322 | 0.8287 | 0.8265 | 0.8414 | 0.0081 | [0.8122, 0.8522] |
| ResNet-18 / 24574ee00a20ab31 / 5 | test_accuracy | 3 | 0.8400 | 0.8458 | 0.8267 | 0.8476 | 0.0116 | [0.8113, 0.8688] |
| ResNet-18 / 24574ee00a20ab31 / 5 | test_loss | 3 | 0.4664 | 0.4711 | 0.4454 | 0.4826 | 0.0190 | [0.4191, 0.5137] |
| ResNet-18 / 24574ee00a20ab31 / 5 | macro_precision | 3 | 0.8531 | 0.8532 | 0.8520 | 0.8542 | 0.0011 | [0.8504, 0.8559] |
| ResNet-18 / 24574ee00a20ab31 / 5 | macro_recall | 3 | 0.8400 | 0.8458 | 0.8267 | 0.8476 | 0.0116 | [0.8113, 0.8688] |
| ResNet-18 / 24574ee00a20ab31 / 5 | macro_f1 | 3 | 0.8411 | 0.8459 | 0.8313 | 0.8461 | 0.0085 | [0.8200, 0.8622] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 5 | test_accuracy | 3 | 0.8065 | 0.8068 | 0.8029 | 0.8099 | 0.0035 | [0.7978, 0.8152] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 5 | test_loss | 3 | 0.5404 | 0.5364 | 0.5289 | 0.5558 | 0.0139 | [0.5058, 0.5749] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 5 | macro_precision | 3 | 0.8127 | 0.8150 | 0.8078 | 0.8154 | 0.0042 | [0.8022, 0.8233] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 5 | macro_recall | 3 | 0.8065 | 0.8068 | 0.8029 | 0.8099 | 0.0035 | [0.7978, 0.8152] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 5 | macro_f1 | 3 | 0.8012 | 0.8007 | 0.7966 | 0.8064 | 0.0050 | [0.7889, 0.8136] |
| MLP / c1427ea14824cbdc / 5 | test_accuracy | 3 | 0.8537 | 0.8556 | 0.8448 | 0.8606 | 0.0081 | [0.8336, 0.8737] |
| MLP / c1427ea14824cbdc / 5 | test_loss | 3 | 0.4073 | 0.4023 | 0.3946 | 0.4250 | 0.0158 | [0.3680, 0.4466] |
| MLP / c1427ea14824cbdc / 5 | macro_precision | 3 | 0.8609 | 0.8618 | 0.8578 | 0.8631 | 0.0028 | [0.8541, 0.8678] |
| MLP / c1427ea14824cbdc / 5 | macro_recall | 3 | 0.8537 | 0.8556 | 0.8448 | 0.8606 | 0.0081 | [0.8336, 0.8737] |
| MLP / c1427ea14824cbdc / 5 | macro_f1 | 3 | 0.8537 | 0.8524 | 0.8475 | 0.8612 | 0.0069 | [0.8364, 0.8709] |
| CNN / LeNet / c1427ea14824cbdc / 5 | test_accuracy | 3 | 0.8784 | 0.8768 | 0.8761 | 0.8824 | 0.0035 | [0.8699, 0.8870] |
| CNN / LeNet / c1427ea14824cbdc / 5 | test_loss | 3 | 0.3355 | 0.3426 | 0.3185 | 0.3455 | 0.0148 | [0.2987, 0.3724] |
| CNN / LeNet / c1427ea14824cbdc / 5 | macro_precision | 3 | 0.8828 | 0.8830 | 0.8823 | 0.8830 | 0.0004 | [0.8818, 0.8838] |
| CNN / LeNet / c1427ea14824cbdc / 5 | macro_recall | 3 | 0.8784 | 0.8768 | 0.8761 | 0.8824 | 0.0035 | [0.8699, 0.8870] |
| CNN / LeNet / c1427ea14824cbdc / 5 | macro_f1 | 3 | 0.8774 | 0.8772 | 0.8723 | 0.8825 | 0.0051 | [0.8647, 0.8900] |
| MLP / 4116dae0312b5981 / 5 | test_accuracy | 3 | 0.8689 | 0.8692 | 0.8680 | 0.8696 | 0.0008 | [0.8669, 0.8710] |
| MLP / 4116dae0312b5981 / 5 | test_loss | 3 | 0.3615 | 0.3617 | 0.3588 | 0.3641 | 0.0026 | [0.3550, 0.3681] |
| MLP / 4116dae0312b5981 / 5 | macro_precision | 3 | 0.8711 | 0.8719 | 0.8690 | 0.8724 | 0.0018 | [0.8666, 0.8757] |
| MLP / 4116dae0312b5981 / 5 | macro_recall | 3 | 0.8689 | 0.8692 | 0.8680 | 0.8696 | 0.0008 | [0.8669, 0.8710] |
| MLP / 4116dae0312b5981 / 5 | macro_f1 | 3 | 0.8683 | 0.8682 | 0.8669 | 0.8698 | 0.0014 | [0.8647, 0.8719] |
| CNN / LeNet / 4116dae0312b5981 / 5 | test_accuracy | 3 | 0.8989 | 0.8975 | 0.8963 | 0.9028 | 0.0035 | [0.8903, 0.9075] |
| CNN / LeNet / 4116dae0312b5981 / 5 | test_loss | 3 | 0.2856 | 0.2908 | 0.2751 | 0.2911 | 0.0092 | [0.2629, 0.3084] |
| CNN / LeNet / 4116dae0312b5981 / 5 | macro_precision | 3 | 0.8994 | 0.8985 | 0.8972 | 0.9027 | 0.0029 | [0.8923, 0.9066] |
| CNN / LeNet / 4116dae0312b5981 / 5 | macro_recall | 3 | 0.8989 | 0.8975 | 0.8963 | 0.9028 | 0.0035 | [0.8903, 0.9075] |
| CNN / LeNet / 4116dae0312b5981 / 5 | macro_f1 | 3 | 0.8978 | 0.8956 | 0.8954 | 0.9024 | 0.0040 | [0.8878, 0.9078] |
| MLP / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8612 | 0.8626 | 0.8566 | 0.8644 | 0.0041 | [0.8511, 0.8713] |
| MLP / 24574ee00a20ab31 / 30 | test_loss | 3 | 0.4218 | 0.4184 | 0.4118 | 0.4350 | 0.0120 | [0.3921, 0.4515] |
| MLP / 24574ee00a20ab31 / 30 | macro_precision | 3 | 0.8637 | 0.8644 | 0.8622 | 0.8645 | 0.0013 | [0.8605, 0.8669] |
| MLP / 24574ee00a20ab31 / 30 | macro_recall | 3 | 0.8612 | 0.8626 | 0.8566 | 0.8644 | 0.0041 | [0.8511, 0.8713] |
| MLP / 24574ee00a20ab31 / 30 | macro_f1 | 3 | 0.8614 | 0.8632 | 0.8573 | 0.8637 | 0.0035 | [0.8526, 0.8701] |
| CNN / LeNet / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8891 | 0.8887 | 0.8882 | 0.8903 | 0.0011 | [0.8863, 0.8918] |
| CNN / LeNet / 24574ee00a20ab31 / 30 | test_loss | 3 | 0.3813 | 0.3811 | 0.3666 | 0.3962 | 0.0148 | [0.3445, 0.4181] |
| CNN / LeNet / 24574ee00a20ab31 / 30 | macro_precision | 3 | 0.8909 | 0.8921 | 0.8885 | 0.8921 | 0.0021 | [0.8857, 0.8961] |
| CNN / LeNet / 24574ee00a20ab31 / 30 | macro_recall | 3 | 0.8891 | 0.8887 | 0.8882 | 0.8903 | 0.0011 | [0.8863, 0.8918] |
| CNN / LeNet / 24574ee00a20ab31 / 30 | macro_f1 | 3 | 0.8891 | 0.8888 | 0.8879 | 0.8907 | 0.0014 | [0.8857, 0.8926] |
| ResNet-18 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8646 | 0.8611 | 0.8592 | 0.8735 | 0.0078 | [0.8453, 0.8839] |
| ResNet-18 / 24574ee00a20ab31 / 30 | test_loss | 3 | 0.6132 | 0.6000 | 0.5995 | 0.6402 | 0.0234 | [0.5552, 0.6713] |
| ResNet-18 / 24574ee00a20ab31 / 30 | macro_precision | 3 | 0.8675 | 0.8675 | 0.8619 | 0.8731 | 0.0056 | [0.8536, 0.8813] |
| ResNet-18 / 24574ee00a20ab31 / 30 | macro_recall | 3 | 0.8646 | 0.8611 | 0.8592 | 0.8735 | 0.0078 | [0.8453, 0.8839] |
| ResNet-18 / 24574ee00a20ab31 / 30 | macro_f1 | 3 | 0.8645 | 0.8605 | 0.8602 | 0.8729 | 0.0072 | [0.8466, 0.8825] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8401 | 0.8431 | 0.8321 | 0.8450 | 0.0070 | [0.8228, 0.8574] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 30 | test_loss | 3 | 0.5063 | 0.5091 | 0.4759 | 0.5340 | 0.0291 | [0.4339, 0.5787] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 30 | macro_precision | 3 | 0.8456 | 0.8456 | 0.8429 | 0.8481 | 0.0026 | [0.8391, 0.8521] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 30 | macro_recall | 3 | 0.8401 | 0.8431 | 0.8321 | 0.8450 | 0.0070 | [0.8228, 0.8574] |
| Vision Transformer (compact ViT-Tiny-style) / 24574ee00a20ab31 / 30 | macro_f1 | 3 | 0.8407 | 0.8437 | 0.8346 | 0.8438 | 0.0053 | [0.8275, 0.8539] |
| ViT-Tiny / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.7621 | 0.7890 | 0.6916 | 0.8057 | 0.0616 | [0.6090, 0.9152] |
| ViT-Tiny / 24574ee00a20ab31 / 30 | test_loss | 3 | 0.6395 | 0.5788 | 0.5337 | 0.8059 | 0.1459 | [0.2771, 1.0019] |
| ViT-Tiny / 24574ee00a20ab31 / 30 | macro_precision | 3 | 0.7682 | 0.8000 | 0.6958 | 0.8088 | 0.0629 | [0.6120, 0.9244] |
| ViT-Tiny / 24574ee00a20ab31 / 30 | macro_recall | 3 | 0.7621 | 0.7890 | 0.6916 | 0.8057 | 0.0616 | [0.6090, 0.9152] |
| ViT-Tiny / 24574ee00a20ab31 / 30 | macro_f1 | 3 | 0.7571 | 0.7909 | 0.6755 | 0.8049 | 0.0710 | [0.5806, 0.9336] |
| Random Forest /  / 400 | test_accuracy | 3 | 0.8524 | 0.8526 | 0.8515 | 0.8532 | 0.0009 | [0.8503, 0.8546] |
| Random Forest /  / 400 | test_log_loss | 3 | 0.4662 | 0.4644 | 0.4642 | 0.4701 | 0.0033 | [0.4580, 0.4745] |
| Random Forest /  / 400 | macro_precision | 3 | 0.8510 | 0.8511 | 0.8500 | 0.8519 | 0.0010 | [0.8486, 0.8534] |
| Random Forest /  / 400 | macro_recall | 3 | 0.8524 | 0.8526 | 0.8515 | 0.8532 | 0.0009 | [0.8503, 0.8546] |
| Random Forest /  / 400 | macro_f1 | 3 | 0.8510 | 0.8511 | 0.8501 | 0.8518 | 0.0009 | [0.8488, 0.8532] |
| Random Forest /  / 1000 | test_accuracy | 3 | 0.8534 | 0.8530 | 0.8527 | 0.8546 | 0.0010 | [0.8509, 0.8560] |
| Random Forest /  / 1000 | test_log_loss | 3 | 0.4598 | 0.4597 | 0.4596 | 0.4600 | 0.0002 | [0.4593, 0.4602] |
| Random Forest /  / 1000 | macro_precision | 3 | 0.8521 | 0.8517 | 0.8514 | 0.8533 | 0.0010 | [0.8495, 0.8547] |
| Random Forest /  / 1000 | macro_recall | 3 | 0.8534 | 0.8530 | 0.8527 | 0.8546 | 0.0010 | [0.8509, 0.8560] |
| Random Forest /  / 1000 | macro_f1 | 3 | 0.8521 | 0.8517 | 0.8514 | 0.8532 | 0.0010 | [0.8496, 0.8546] |
| MLP / labeled_only / 0.01 /  / 5 | test_accuracy | 3 | 0.6575 | 0.6519 | 0.6459 | 0.6747 | 0.0152 | [0.6198, 0.6952] |
| MLP / labeled_only / 0.01 /  / 5 | test_loss | 3 | 1.4999 | 1.5203 | 1.3668 | 1.6125 | 0.1241 | [1.1916, 1.8082] |
| MLP / labeled_only / 0.01 /  / 5 | macro_precision | 3 | 0.6935 | 0.6950 | 0.6881 | 0.6973 | 0.0048 | [0.6816, 0.7054] |
| MLP / labeled_only / 0.01 /  / 5 | macro_recall | 3 | 0.6575 | 0.6519 | 0.6459 | 0.6747 | 0.0152 | [0.6198, 0.6952] |
| MLP / labeled_only / 0.01 /  / 5 | macro_f1 | 3 | 0.6526 | 0.6581 | 0.6367 | 0.6630 | 0.0140 | [0.6179, 0.6873] |
| MLP / pseudo_label / 0.01 /  / 5 | test_accuracy | 3 | 0.6369 | 0.6330 | 0.6289 | 0.6488 | 0.0105 | [0.6108, 0.6630] |
| MLP / pseudo_label / 0.01 /  / 5 | test_loss | 3 | 3.4565 | 3.3167 | 3.2436 | 3.8093 | 0.3077 | [2.6922, 4.2208] |
| MLP / pseudo_label / 0.01 /  / 5 | macro_precision | 3 | 0.7042 | 0.7209 | 0.6639 | 0.7278 | 0.0351 | [0.6171, 0.7913] |
| MLP / pseudo_label / 0.01 /  / 5 | macro_recall | 3 | 0.6369 | 0.6330 | 0.6289 | 0.6488 | 0.0105 | [0.6108, 0.6630] |
| MLP / pseudo_label / 0.01 /  / 5 | macro_f1 | 3 | 0.6086 | 0.6161 | 0.5866 | 0.6232 | 0.0194 | [0.5604, 0.6569] |
| MLP / labeled_only / 0.05 /  / 5 | test_accuracy | 3 | 0.7718 | 0.7719 | 0.7713 | 0.7723 | 0.0005 | [0.7706, 0.7731] |
| MLP / labeled_only / 0.05 /  / 5 | test_loss | 3 | 0.8468 | 0.8415 | 0.8058 | 0.8931 | 0.0439 | [0.7378, 0.9558] |
| MLP / labeled_only / 0.05 /  / 5 | macro_precision | 3 | 0.7797 | 0.7810 | 0.7771 | 0.7811 | 0.0023 | [0.7741, 0.7853] |
| MLP / labeled_only / 0.05 /  / 5 | macro_recall | 3 | 0.7718 | 0.7719 | 0.7713 | 0.7723 | 0.0005 | [0.7706, 0.7731] |
| MLP / labeled_only / 0.05 /  / 5 | macro_f1 | 3 | 0.7722 | 0.7729 | 0.7703 | 0.7735 | 0.0017 | [0.7681, 0.7763] |
| MLP / pseudo_label / 0.05 /  / 5 | test_accuracy | 3 | 0.7844 | 0.7825 | 0.7781 | 0.7926 | 0.0074 | [0.7659, 0.8029] |
| MLP / pseudo_label / 0.05 /  / 5 | test_loss | 3 | 0.9774 | 1.0095 | 0.8806 | 1.0422 | 0.0855 | [0.7651, 1.1897] |
| MLP / pseudo_label / 0.05 /  / 5 | macro_precision | 3 | 0.7899 | 0.7926 | 0.7812 | 0.7959 | 0.0077 | [0.7707, 0.8091] |
| MLP / pseudo_label / 0.05 /  / 5 | macro_recall | 3 | 0.7844 | 0.7825 | 0.7781 | 0.7926 | 0.0074 | [0.7659, 0.8029] |
| MLP / pseudo_label / 0.05 /  / 5 | macro_f1 | 3 | 0.7832 | 0.7801 | 0.7779 | 0.7917 | 0.0074 | [0.7648, 0.8017] |
| MLP / labeled_only / 0.1 /  / 5 | test_accuracy | 3 | 0.8045 | 0.8053 | 0.8001 | 0.8082 | 0.0041 | [0.7943, 0.8147] |
| MLP / labeled_only / 0.1 /  / 5 | test_loss | 3 | 0.6149 | 0.6134 | 0.6061 | 0.6252 | 0.0096 | [0.5910, 0.6389] |
| MLP / labeled_only / 0.1 /  / 5 | macro_precision | 3 | 0.8073 | 0.8084 | 0.8027 | 0.8107 | 0.0041 | [0.7971, 0.8175] |
| MLP / labeled_only / 0.1 /  / 5 | macro_recall | 3 | 0.8045 | 0.8053 | 0.8001 | 0.8082 | 0.0041 | [0.7943, 0.8147] |
| MLP / labeled_only / 0.1 /  / 5 | macro_f1 | 3 | 0.8035 | 0.8046 | 0.8000 | 0.8058 | 0.0031 | [0.7958, 0.8111] |
| MLP / pseudo_label / 0.1 /  / 5 | test_accuracy | 3 | 0.8073 | 0.8062 | 0.8042 | 0.8116 | 0.0038 | [0.7978, 0.8168] |
| MLP / pseudo_label / 0.1 /  / 5 | test_loss | 3 | 0.6656 | 0.6715 | 0.6422 | 0.6831 | 0.0211 | [0.6132, 0.7180] |
| MLP / pseudo_label / 0.1 /  / 5 | macro_precision | 3 | 0.8084 | 0.8069 | 0.8061 | 0.8121 | 0.0033 | [0.8003, 0.8165] |
| MLP / pseudo_label / 0.1 /  / 5 | macro_recall | 3 | 0.8073 | 0.8062 | 0.8042 | 0.8116 | 0.0038 | [0.7978, 0.8168] |
| MLP / pseudo_label / 0.1 /  / 5 | macro_f1 | 3 | 0.8054 | 0.8052 | 0.8009 | 0.8101 | 0.0046 | [0.7939, 0.8168] |
| MLP / labeled_only / 0.2 /  / 5 | test_accuracy | 3 | 0.8161 | 0.8155 | 0.8148 | 0.8180 | 0.0017 | [0.8119, 0.8203] |
| MLP / labeled_only / 0.2 /  / 5 | test_loss | 3 | 0.5367 | 0.5367 | 0.5360 | 0.5374 | 0.0007 | [0.5350, 0.5384] |
| MLP / labeled_only / 0.2 /  / 5 | macro_precision | 3 | 0.8183 | 0.8175 | 0.8166 | 0.8208 | 0.0022 | [0.8128, 0.8238] |
| MLP / labeled_only / 0.2 /  / 5 | macro_recall | 3 | 0.8161 | 0.8155 | 0.8148 | 0.8180 | 0.0017 | [0.8119, 0.8203] |
| MLP / labeled_only / 0.2 /  / 5 | macro_f1 | 3 | 0.8123 | 0.8129 | 0.8077 | 0.8164 | 0.0044 | [0.8014, 0.8232] |
| MLP / pseudo_label / 0.2 /  / 5 | test_accuracy | 3 | 0.8147 | 0.8135 | 0.8135 | 0.8172 | 0.0021 | [0.8094, 0.8200] |
| MLP / pseudo_label / 0.2 /  / 5 | test_loss | 3 | 0.5664 | 0.5666 | 0.5637 | 0.5690 | 0.0027 | [0.5598, 0.5730] |
| MLP / pseudo_label / 0.2 /  / 5 | macro_precision | 3 | 0.8161 | 0.8157 | 0.8156 | 0.8170 | 0.0007 | [0.8143, 0.8179] |
| MLP / pseudo_label / 0.2 /  / 5 | macro_recall | 3 | 0.8147 | 0.8135 | 0.8135 | 0.8172 | 0.0021 | [0.8094, 0.8200] |
| MLP / pseudo_label / 0.2 /  / 5 | macro_f1 | 3 | 0.8103 | 0.8105 | 0.8055 | 0.8149 | 0.0047 | [0.7986, 0.8219] |
| MLP / labeled_only / 0.4 /  / 5 | test_accuracy | 3 | 0.8176 | 0.8173 | 0.8127 | 0.8227 | 0.0050 | [0.8051, 0.8300] |
| MLP / labeled_only / 0.4 /  / 5 | test_loss | 3 | 0.5177 | 0.5214 | 0.5074 | 0.5242 | 0.0090 | [0.4953, 0.5400] |
| MLP / labeled_only / 0.4 /  / 5 | macro_precision | 3 | 0.8185 | 0.8172 | 0.8149 | 0.8236 | 0.0045 | [0.8073, 0.8298] |
| MLP / labeled_only / 0.4 /  / 5 | macro_recall | 3 | 0.8176 | 0.8173 | 0.8127 | 0.8227 | 0.0050 | [0.8051, 0.8300] |
| MLP / labeled_only / 0.4 /  / 5 | macro_f1 | 3 | 0.8141 | 0.8112 | 0.8112 | 0.8200 | 0.0051 | [0.8015, 0.8267] |
| MLP / pseudo_label / 0.4 /  / 5 | test_accuracy | 3 | 0.8147 | 0.8142 | 0.8109 | 0.8189 | 0.0040 | [0.8047, 0.8247] |
| MLP / pseudo_label / 0.4 /  / 5 | test_loss | 3 | 0.5328 | 0.5285 | 0.5202 | 0.5498 | 0.0152 | [0.4949, 0.5707] |
| MLP / pseudo_label / 0.4 /  / 5 | macro_precision | 3 | 0.8161 | 0.8147 | 0.8138 | 0.8197 | 0.0032 | [0.8082, 0.8240] |
| MLP / pseudo_label / 0.4 /  / 5 | macro_recall | 3 | 0.8147 | 0.8142 | 0.8109 | 0.8189 | 0.0040 | [0.8047, 0.8247] |
| MLP / pseudo_label / 0.4 /  / 5 | macro_f1 | 3 | 0.8107 | 0.8124 | 0.8025 | 0.8171 | 0.0075 | [0.7921, 0.8292] |
| MLP / labeled_only / 0.01 /  / 30 | test_accuracy | 3 | 0.6544 | 0.6458 | 0.6362 | 0.6813 | 0.0238 | [0.5954, 0.7134] |
| MLP / labeled_only / 0.01 /  / 30 | test_loss | 3 | 2.0985 | 2.1394 | 1.8615 | 2.2947 | 0.2194 | [1.5534, 2.6436] |
| MLP / labeled_only / 0.01 /  / 30 | macro_precision | 3 | 0.6904 | 0.6913 | 0.6871 | 0.6928 | 0.0030 | [0.6831, 0.6977] |
| MLP / labeled_only / 0.01 /  / 30 | macro_recall | 3 | 0.6544 | 0.6458 | 0.6362 | 0.6813 | 0.0238 | [0.5954, 0.7134] |
| MLP / labeled_only / 0.01 /  / 30 | macro_f1 | 3 | 0.6502 | 0.6522 | 0.6283 | 0.6701 | 0.0209 | [0.5981, 0.7022] |
| MLP / pseudo_label / 0.01 /  / 30 | test_accuracy | 3 | 0.6389 | 0.6405 | 0.6313 | 0.6450 | 0.0070 | [0.6216, 0.6563] |
| MLP / pseudo_label / 0.01 /  / 30 | test_loss | 3 | 9.2239 | 9.6875 | 7.9418 | 10.0424 | 1.1245 | [6.4306, 12.0172] |
| MLP / pseudo_label / 0.01 /  / 30 | macro_precision | 3 | 0.6991 | 0.7189 | 0.6564 | 0.7219 | 0.0370 | [0.6072, 0.7909] |
| MLP / pseudo_label / 0.01 /  / 30 | macro_recall | 3 | 0.6389 | 0.6405 | 0.6313 | 0.6450 | 0.0070 | [0.6216, 0.6563] |
| MLP / pseudo_label / 0.01 /  / 30 | macro_f1 | 3 | 0.6107 | 0.6055 | 0.5945 | 0.6320 | 0.0193 | [0.5628, 0.6586] |
| MLP / labeled_only / 0.05 /  / 30 | test_accuracy | 3 | 0.7695 | 0.7690 | 0.7665 | 0.7730 | 0.0033 | [0.7614, 0.7776] |
| MLP / labeled_only / 0.05 /  / 30 | test_loss | 3 | 1.3419 | 1.3432 | 1.2564 | 1.4262 | 0.0849 | [1.1310, 1.5529] |
| MLP / labeled_only / 0.05 /  / 30 | macro_precision | 3 | 0.7766 | 0.7768 | 0.7754 | 0.7775 | 0.0011 | [0.7739, 0.7793] |
| MLP / labeled_only / 0.05 /  / 30 | macro_recall | 3 | 0.7695 | 0.7690 | 0.7665 | 0.7730 | 0.0033 | [0.7614, 0.7776] |
| MLP / labeled_only / 0.05 /  / 30 | macro_f1 | 3 | 0.7701 | 0.7686 | 0.7684 | 0.7734 | 0.0028 | [0.7631, 0.7771] |
| MLP / pseudo_label / 0.05 /  / 30 | test_accuracy | 3 | 0.7839 | 0.7802 | 0.7780 | 0.7935 | 0.0084 | [0.7631, 0.8047] |
| MLP / pseudo_label / 0.05 /  / 30 | test_loss | 3 | 2.5806 | 2.6145 | 2.2402 | 2.8870 | 0.3248 | [1.7738, 3.3873] |
| MLP / pseudo_label / 0.05 /  / 30 | macro_precision | 3 | 0.7909 | 0.7888 | 0.7823 | 0.8016 | 0.0098 | [0.7665, 0.8154] |
| MLP / pseudo_label / 0.05 /  / 30 | macro_recall | 3 | 0.7839 | 0.7802 | 0.7780 | 0.7935 | 0.0084 | [0.7631, 0.8047] |
| MLP / pseudo_label / 0.05 /  / 30 | macro_f1 | 3 | 0.7820 | 0.7796 | 0.7729 | 0.7935 | 0.0105 | [0.7559, 0.8080] |
| MLP / labeled_only / 0.1 /  / 30 | test_accuracy | 3 | 0.8054 | 0.8059 | 0.8023 | 0.8081 | 0.0029 | [0.7982, 0.8127] |
| MLP / labeled_only / 0.1 /  / 30 | test_loss | 3 | 1.0545 | 1.0458 | 1.0372 | 1.0803 | 0.0228 | [0.9978, 1.1111] |
| MLP / labeled_only / 0.1 /  / 30 | macro_precision | 3 | 0.8076 | 0.8083 | 0.8037 | 0.8108 | 0.0036 | [0.7986, 0.8166] |
| MLP / labeled_only / 0.1 /  / 30 | macro_recall | 3 | 0.8054 | 0.8059 | 0.8023 | 0.8081 | 0.0029 | [0.7982, 0.8127] |
| MLP / labeled_only / 0.1 /  / 30 | macro_f1 | 3 | 0.8054 | 0.8061 | 0.8025 | 0.8075 | 0.0026 | [0.7990, 0.8118] |
| MLP / pseudo_label / 0.1 /  / 30 | test_accuracy | 3 | 0.8087 | 0.8098 | 0.8049 | 0.8114 | 0.0034 | [0.8003, 0.8171] |
| MLP / pseudo_label / 0.1 /  / 30 | test_loss | 3 | 1.5726 | 1.5453 | 1.5237 | 1.6489 | 0.0669 | [1.4063, 1.7389] |
| MLP / pseudo_label / 0.1 /  / 30 | macro_precision | 3 | 0.8111 | 0.8120 | 0.8083 | 0.8130 | 0.0025 | [0.8049, 0.8173] |
| MLP / pseudo_label / 0.1 /  / 30 | macro_recall | 3 | 0.8087 | 0.8098 | 0.8049 | 0.8114 | 0.0034 | [0.8003, 0.8171] |
| MLP / pseudo_label / 0.1 /  / 30 | macro_f1 | 3 | 0.8085 | 0.8096 | 0.8055 | 0.8106 | 0.0027 | [0.8018, 0.8153] |
| MLP / labeled_only / 0.2 /  / 30 | test_accuracy | 3 | 0.8214 | 0.8224 | 0.8191 | 0.8228 | 0.0020 | [0.8164, 0.8265] |
| MLP / labeled_only / 0.2 /  / 30 | test_loss | 3 | 0.8208 | 0.8158 | 0.7856 | 0.8611 | 0.0380 | [0.7264, 0.9153] |
| MLP / labeled_only / 0.2 /  / 30 | macro_precision | 3 | 0.8219 | 0.8234 | 0.8188 | 0.8235 | 0.0027 | [0.8152, 0.8285] |
| MLP / labeled_only / 0.2 /  / 30 | macro_recall | 3 | 0.8214 | 0.8224 | 0.8191 | 0.8228 | 0.0020 | [0.8164, 0.8265] |
| MLP / labeled_only / 0.2 /  / 30 | macro_f1 | 3 | 0.8207 | 0.8211 | 0.8185 | 0.8223 | 0.0019 | [0.8159, 0.8254] |
| MLP / pseudo_label / 0.2 /  / 30 | test_accuracy | 3 | 0.8252 | 0.8256 | 0.8187 | 0.8313 | 0.0063 | [0.8095, 0.8409] |
| MLP / pseudo_label / 0.2 /  / 30 | test_loss | 3 | 1.0352 | 1.0542 | 0.9672 | 1.0843 | 0.0608 | [0.8841, 1.1864] |
| MLP / pseudo_label / 0.2 /  / 30 | macro_precision | 3 | 0.8260 | 0.8261 | 0.8211 | 0.8307 | 0.0048 | [0.8140, 0.8379] |
| MLP / pseudo_label / 0.2 /  / 30 | macro_recall | 3 | 0.8252 | 0.8256 | 0.8187 | 0.8313 | 0.0063 | [0.8095, 0.8409] |
| MLP / pseudo_label / 0.2 /  / 30 | macro_f1 | 3 | 0.8237 | 0.8251 | 0.8163 | 0.8298 | 0.0069 | [0.8067, 0.8408] |
| MLP / labeled_only / 0.4 /  / 30 | test_accuracy | 3 | 0.8379 | 0.8390 | 0.8354 | 0.8394 | 0.0022 | [0.8325, 0.8434] |
| MLP / labeled_only / 0.4 /  / 30 | test_loss | 3 | 0.5369 | 0.5323 | 0.5304 | 0.5481 | 0.0097 | [0.5128, 0.5611] |
| MLP / labeled_only / 0.4 /  / 30 | macro_precision | 3 | 0.8398 | 0.8396 | 0.8381 | 0.8418 | 0.0019 | [0.8352, 0.8445] |
| MLP / labeled_only / 0.4 /  / 30 | macro_recall | 3 | 0.8379 | 0.8390 | 0.8354 | 0.8394 | 0.0022 | [0.8325, 0.8434] |
| MLP / labeled_only / 0.4 /  / 30 | macro_f1 | 3 | 0.8371 | 0.8371 | 0.8341 | 0.8400 | 0.0029 | [0.8298, 0.8444] |
| MLP / pseudo_label / 0.4 /  / 30 | test_accuracy | 3 | 0.8397 | 0.8414 | 0.8359 | 0.8419 | 0.0033 | [0.8315, 0.8480] |
| MLP / pseudo_label / 0.4 /  / 30 | test_loss | 3 | 0.5962 | 0.5951 | 0.5881 | 0.6053 | 0.0086 | [0.5748, 0.6176] |
| MLP / pseudo_label / 0.4 /  / 30 | macro_precision | 3 | 0.8404 | 0.8409 | 0.8374 | 0.8428 | 0.0028 | [0.8335, 0.8472] |
| MLP / pseudo_label / 0.4 /  / 30 | macro_recall | 3 | 0.8397 | 0.8414 | 0.8359 | 0.8419 | 0.0033 | [0.8315, 0.8480] |
| MLP / pseudo_label / 0.4 /  / 30 | macro_f1 | 3 | 0.8379 | 0.8382 | 0.8335 | 0.8420 | 0.0042 | [0.8274, 0.8484] |
| MeanTeacher / 0.01 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.6519 | 0.6452 | 0.6422 | 0.6683 | 0.0143 | [0.6164, 0.6874] |
| FixMatch / 0.01 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.6619 | 0.6626 | 0.6585 | 0.6647 | 0.0032 | [0.6541, 0.6698] |
| MixMatch / 0.01 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.6586 | 0.6643 | 0.6460 | 0.6656 | 0.0110 | [0.6314, 0.6859] |
| MeanTeacher / 0.05 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7713 | 0.7745 | 0.7634 | 0.7760 | 0.0069 | [0.7542, 0.7884] |
| FixMatch / 0.05 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7644 | 0.7627 | 0.7557 | 0.7747 | 0.0096 | [0.7405, 0.7882] |
| MixMatch / 0.05 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7634 | 0.7680 | 0.7534 | 0.7687 | 0.0086 | [0.7419, 0.7848] |
| MeanTeacher / 0.1 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7802 | 0.7835 | 0.7707 | 0.7864 | 0.0084 | [0.7594, 0.8010] |
| FixMatch / 0.1 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7644 | 0.7664 | 0.7427 | 0.7842 | 0.0208 | [0.7127, 0.8162] |
| MixMatch / 0.1 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7572 | 0.7584 | 0.7531 | 0.7600 | 0.0036 | [0.7482, 0.7661] |
| MeanTeacher / 0.2 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7890 | 0.7885 | 0.7807 | 0.7978 | 0.0086 | [0.7677, 0.8103] |
| FixMatch / 0.2 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7855 | 0.7851 | 0.7840 | 0.7875 | 0.0018 | [0.7811, 0.7900] |
| MixMatch / 0.2 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7784 | 0.7785 | 0.7763 | 0.7803 | 0.0020 | [0.7734, 0.7833] |
| MeanTeacher / 0.4 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7940 | 0.7936 | 0.7926 | 0.7957 | 0.0016 | [0.7900, 0.7979] |
| FixMatch / 0.4 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7926 | 0.7941 | 0.7892 | 0.7946 | 0.0030 | [0.7852, 0.8000] |
| MixMatch / 0.4 / 24574ee00a20ab31 / 3 | test_accuracy | 3 | 0.7784 | 0.7773 | 0.7707 | 0.7873 | 0.0084 | [0.7577, 0.7992] |
| MeanTeacher / 0.01 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.6576 | 0.6585 | 0.6446 | 0.6696 | 0.0125 | [0.6265, 0.6887] |
| FixMatch / 0.01 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.6627 | 0.6580 | 0.6540 | 0.6762 | 0.0118 | [0.6333, 0.6921] |
| MixMatch / 0.01 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.6575 | 0.6592 | 0.6460 | 0.6672 | 0.0107 | [0.6309, 0.6841] |
| MeanTeacher / 0.05 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.7851 | 0.7856 | 0.7789 | 0.7908 | 0.0060 | [0.7703, 0.7999] |
| FixMatch / 0.05 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.7821 | 0.7763 | 0.7746 | 0.7955 | 0.0116 | [0.7533, 0.8110] |
| MixMatch / 0.05 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.7879 | 0.7919 | 0.7787 | 0.7930 | 0.0080 | [0.7681, 0.8076] |
| MeanTeacher / 0.1 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8070 | 0.8083 | 0.8028 | 0.8099 | 0.0037 | [0.7977, 0.8163] |
| FixMatch / 0.1 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8059 | 0.8126 | 0.7917 | 0.8133 | 0.0123 | [0.7754, 0.8364] |
| MixMatch / 0.1 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8095 | 0.8130 | 0.8015 | 0.8140 | 0.0069 | [0.7922, 0.8268] |
| MeanTeacher / 0.2 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8213 | 0.8222 | 0.8181 | 0.8235 | 0.0028 | [0.8143, 0.8283] |
| FixMatch / 0.2 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8232 | 0.8234 | 0.8225 | 0.8237 | 0.0006 | [0.8216, 0.8248] |
| MixMatch / 0.2 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8302 | 0.8280 | 0.8258 | 0.8368 | 0.0058 | [0.8157, 0.8447] |
| MeanTeacher / 0.4 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8416 | 0.8418 | 0.8411 | 0.8419 | 0.0004 | [0.8405, 0.8427] |
| FixMatch / 0.4 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8381 | 0.8385 | 0.8339 | 0.8418 | 0.0040 | [0.8282, 0.8479] |
| MixMatch / 0.4 / 24574ee00a20ab31 / 30 | test_accuracy | 3 | 0.8372 | 0.8379 | 0.8329 | 0.8409 | 0.0040 | [0.8272, 0.8473] |
| Autoencoder / 24574ee00a20ab31 / 5 | test_reconstruction_mse | 3 | 0.0196 | 0.0195 | 0.0193 | 0.0198 | 0.0003 | [0.0189, 0.0202] |
| Autoencoder / 24574ee00a20ab31 / 5 | test_linear_probe_accuracy | 3 | 0.7917 | 0.7910 | 0.7889 | 0.7953 | 0.0033 | [0.7836, 0.7998] |
| Variational Autoencoder / 24574ee00a20ab31 / 5 | test_reconstruction_mse | 3 | 0.0292 | 0.0292 | 0.0292 | 0.0293 | 0.0001 | [0.0291, 0.0294] |
| Variational Autoencoder / 24574ee00a20ab31 / 5 | test_linear_probe_accuracy | 3 | 0.7838 | 0.7853 | 0.7790 | 0.7871 | 0.0043 | [0.7732, 0.7944] |
| Autoencoder / 24574ee00a20ab31 / 30 | test_reconstruction_mse | 3 | 0.0119 | 0.0119 | 0.0117 | 0.0120 | 0.0002 | [0.0115, 0.0123] |
| Autoencoder / 24574ee00a20ab31 / 30 | test_linear_probe_accuracy | 3 | 0.8120 | 0.8126 | 0.8101 | 0.8132 | 0.0016 | [0.8079, 0.8161] |
| Variational Autoencoder / 24574ee00a20ab31 / 30 | test_reconstruction_mse | 3 | 0.0204 | 0.0205 | 0.0200 | 0.0206 | 0.0003 | [0.0195, 0.0212] |
| Variational Autoencoder / 24574ee00a20ab31 / 30 | test_linear_probe_accuracy | 3 | 0.7962 | 0.7976 | 0.7931 | 0.7980 | 0.0027 | [0.7895, 0.8030] |
| DQN / CartPole-v1 | mean_return | 3 | 34.5000 | 43.0500 | 9.7000 | 50.7500 | 21.8198 | [-19.7033, 88.7033] |
| A2C / CartPole-v1 | mean_return | 3 | 442.5500 | 484.3000 | 346.0500 | 497.3000 | 83.8238 | [234.3200, 650.7800] |
| PPO / CartPole-v1 | mean_return | 3 | 405.3000 | 465.0500 | 251.3500 | 499.5000 | 134.4327 | [71.3506, 739.2494] |
| SAC / Pendulum-v1 | mean_return | 5 | -171.1897 | -174.1756 | -184.7029 | -158.6500 | 10.8446 | [-184.6551, -157.7243] |
| Double DQN / CartPole-v1 | mean_return | 5 | 190.2400 | 166.1000 | 127.7000 | 278.1500 | 63.6508 | [111.2071, 269.2729] |
| PPO-Pendulum / Pendulum-v1 | mean_return | 3 | -1260.9494 | -1208.2417 | -1418.8701 | -1155.7363 | 139.2603 | [-1606.8911, -915.0076] |

## Interpretation limits

The MLP, LeNet, ResNet-18, compact ViT-style, Random Forest, AE/VAE and RL curves use three training seeds on fixed splits/subsets; uncertainty is conditional on those splits and does not quantify dataset-sampling uncertainty. K-Means/GMM ARI varies non-monotonically across updates/iterations and seeds; no saturation model is fit. DBSCAN is a deterministic parameter grid, not a seed-level budget curve. Pseudo-label vs labeled-only comparisons are paired by seed and labeled subset; current results vary with fraction and seed, so no universal SSL benefit is claimed. RL uses 20 deterministic evaluation episodes per checkpoint and three training seeds; within-run episode dispersion is separately retained. Historical pilot curves in `experiment_results.csv` include repeated test-set looks and remain exploratory.
