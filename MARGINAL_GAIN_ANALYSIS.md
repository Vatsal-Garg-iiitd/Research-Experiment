# Marginal-gain analysis

All marginal values below are computed only for the new saved multi-budget runs. Improvement is positive in the favorable direction (higher accuracy/ARI is better). `gain / budget` uses the native budget; time efficiency uses the measured increment in cumulative training time. These ratios are descriptive and task-specific.

- MLP/LeNet: epoch budgets; validation accuracy.
- Random Forest: cumulative warm-start tree count; validation accuracy.
- MiniBatch K-Means: actual `partial_fit` updates; validation ARI.
- Gaussian Mixture: repeated warm-start fit calls, one EM iteration each; validation ARI.
- Pseudo-labeling/labeled-only: epochs, per seed and label fraction; validation accuracy.
- Autoencoder/VAE: epoch increments for reconstruction MSE (lower is better) and downstream linear-probe accuracy (higher is better).
- RL: environment interactions; CartPole and Pendulum remain separate tasks.
- No cross-paradigm pooling. Saturation is not declared from these short curves.

## Seed-level marginal data

| Model | Setting | Seed | Budget interval | Improvement | Gain / budget | Additional seconds | Gain / second | Relative improvement |
|---|---|---:|---|---:|---:|---:|---:|---:|
| MLP |  | 11 | 1.0–2.0 epochs | 0.05800 | 0.05800 | 0.8143 | 0.07123 | 7.703% |
| MLP |  | 11 | 2.0–3.0 epochs | 0.02450 | 0.02450 | 0.5683 | 0.04311 | 3.021% |
| MLP |  | 11 | 3.0–4.0 epochs | -0.00550 | -0.00550 | 0.6139 | -0.00896 | -0.658% |
| MLP |  | 11 | 4.0–5.0 epochs | 0.00900 | 0.00900 | 0.5706 | 0.01577 | 1.084% |
| MLP |  | 23 | 1.0–2.0 epochs | 0.01650 | 0.01650 | 0.5070 | 0.03254 | 2.086% |
| MLP |  | 23 | 2.0–3.0 epochs | 0.02050 | 0.02050 | 0.4627 | 0.04430 | 2.539% |
| MLP |  | 23 | 3.0–4.0 epochs | 0.02350 | 0.02350 | 0.4954 | 0.04744 | 2.838% |
| MLP |  | 23 | 4.0–5.0 epochs | -0.01300 | -0.01300 | 0.4763 | -0.02729 | -1.527% |
| MLP |  | 37 | 1.0–2.0 epochs | 0.03600 | 0.03600 | 0.5066 | 0.07106 | 4.545% |
| MLP |  | 37 | 2.0–3.0 epochs | 0.00050 | 0.00050 | 0.4821 | 0.00104 | 0.060% |
| MLP |  | 37 | 3.0–4.0 epochs | 0.01600 | 0.01600 | 0.4844 | 0.03303 | 1.931% |
| MLP |  | 37 | 4.0–5.0 epochs | 0.00450 | 0.00450 | 0.4758 | 0.00946 | 0.533% |
| CNN / LeNet |  | 11 | 1.0–2.0 epochs | 0.08100 | 0.08100 | 1.6730 | 0.04842 | 10.991% |
| CNN / LeNet |  | 11 | 2.0–3.0 epochs | 0.02650 | 0.02650 | 1.7593 | 0.01506 | 3.240% |
| CNN / LeNet |  | 11 | 3.0–4.0 epochs | 0.01350 | 0.01350 | 1.7700 | 0.00763 | 1.599% |
| CNN / LeNet |  | 11 | 4.0–5.0 epochs | -0.00350 | -0.00350 | 1.7836 | -0.00196 | -0.408% |
| CNN / LeNet |  | 23 | 1.0–2.0 epochs | 0.03100 | 0.03100 | 1.4313 | 0.02166 | 4.098% |
| CNN / LeNet |  | 23 | 2.0–3.0 epochs | 0.03850 | 0.03850 | 1.3851 | 0.02780 | 4.889% |
| CNN / LeNet |  | 23 | 3.0–4.0 epochs | 0.03550 | 0.03550 | 1.4119 | 0.02514 | 4.298% |
| CNN / LeNet |  | 23 | 4.0–5.0 epochs | -0.01150 | -0.01150 | 1.4558 | -0.00790 | -1.335% |
| CNN / LeNet |  | 37 | 1.0–2.0 epochs | 0.04300 | 0.04300 | 1.3873 | 0.03100 | 5.647% |
| CNN / LeNet |  | 37 | 2.0–3.0 epochs | 0.02300 | 0.02300 | 1.3913 | 0.01653 | 2.859% |
| CNN / LeNet |  | 37 | 3.0–4.0 epochs | 0.02550 | 0.02550 | 1.3819 | 0.01845 | 3.082% |
| CNN / LeNet |  | 37 | 4.0–5.0 epochs | 0.00200 | 0.00200 | 1.3976 | 0.00143 | 0.234% |
| ResNet-18 |  | 11 | 1.0–2.0 epochs | 0.05200 | 0.05200 | 12.6772 | 0.00410 | 6.468% |
| ResNet-18 |  | 11 | 2.0–3.0 epochs | 0.01100 | 0.01100 | 1122.1625 | 0.00001 | 1.285% |
| ResNet-18 |  | 11 | 3.0–4.0 epochs | -0.02600 | -0.02600 | 8.4248 | -0.00309 | -2.999% |
| ResNet-18 |  | 11 | 4.0–5.0 epochs | 0.02500 | 0.02500 | 9.4050 | 0.00266 | 2.973% |
| ResNet-18 |  | 23 | 1.0–2.0 epochs | 0.03300 | 0.03300 | 8.6593 | 0.00381 | 4.342% |
| ResNet-18 |  | 23 | 2.0–3.0 epochs | 0.06450 | 0.06450 | 8.3415 | 0.00773 | 8.134% |
| ResNet-18 |  | 23 | 3.0–4.0 epochs | 0.01100 | 0.01100 | 6.6793 | 0.00165 | 1.283% |
| ResNet-18 |  | 23 | 4.0–5.0 epochs | -0.00600 | -0.00600 | 6.7278 | -0.00089 | -0.691% |
| ResNet-18 |  | 37 | 1.0–2.0 epochs | 0.00250 | 0.00250 | 6.7331 | 0.00037 | 0.307% |
| ResNet-18 |  | 37 | 2.0–3.0 epochs | 0.00650 | 0.00650 | 6.6311 | 0.00098 | 0.796% |
| ResNet-18 |  | 37 | 3.0–4.0 epochs | 0.02500 | 0.02500 | 6.6840 | 0.00374 | 3.036% |
| ResNet-18 |  | 37 | 4.0–5.0 epochs | -0.00450 | -0.00450 | 6.6871 | -0.00067 | -0.530% |
| Vision Transformer (compact ViT-Tiny-style) |  | 11 | 1.0–2.0 epochs | 0.08600 | 0.08600 | 7.2105 | 0.01193 | 12.045% |
| Vision Transformer (compact ViT-Tiny-style) |  | 11 | 2.0–3.0 epochs | -0.01300 | -0.01300 | 7.1054 | -0.00183 | -1.625% |
| Vision Transformer (compact ViT-Tiny-style) |  | 11 | 3.0–4.0 epochs | 0.02750 | 0.02750 | 7.1218 | 0.00386 | 3.494% |
| Vision Transformer (compact ViT-Tiny-style) |  | 11 | 4.0–5.0 epochs | 0.01750 | 0.01750 | 6.9525 | 0.00252 | 2.149% |
| Vision Transformer (compact ViT-Tiny-style) |  | 23 | 1.0–2.0 epochs | 0.01750 | 0.01750 | 6.5160 | 0.00269 | 2.333% |
| Vision Transformer (compact ViT-Tiny-style) |  | 23 | 2.0–3.0 epochs | 0.04100 | 0.04100 | 6.6001 | 0.00621 | 5.342% |
| Vision Transformer (compact ViT-Tiny-style) |  | 23 | 3.0–4.0 epochs | 0.02400 | 0.02400 | 6.5763 | 0.00365 | 2.968% |
| Vision Transformer (compact ViT-Tiny-style) |  | 23 | 4.0–5.0 epochs | -0.01000 | -0.01000 | 6.5587 | -0.00152 | -1.201% |
| Vision Transformer (compact ViT-Tiny-style) |  | 37 | 1.0–2.0 epochs | 0.02400 | 0.02400 | 6.6171 | 0.00363 | 3.237% |
| Vision Transformer (compact ViT-Tiny-style) |  | 37 | 2.0–3.0 epochs | 0.03100 | 0.03100 | 6.6474 | 0.00466 | 4.050% |
| Vision Transformer (compact ViT-Tiny-style) |  | 37 | 3.0–4.0 epochs | 0.01700 | 0.01700 | 6.5876 | 0.00258 | 2.134% |
| Vision Transformer (compact ViT-Tiny-style) |  | 37 | 4.0–5.0 epochs | 0.00900 | 0.00900 | 6.5903 | 0.00137 | 1.106% |
| Random Forest |  | 11 | 10.0–25.0 trees | 0.02400 | 0.00160 | 0.3439 | 0.06979 | 2.852% |
| Random Forest |  | 11 | 25.0–50.0 trees | 0.00600 | 0.00024 | 0.7006 | 0.00856 | 0.693% |
| Random Forest |  | 11 | 50.0–100.0 trees | 0.00500 | 0.00010 | 0.8308 | 0.00602 | 0.574% |
| Random Forest |  | 11 | 100.0–200.0 trees | -0.00250 | -0.00002 | 1.2058 | -0.00207 | -0.285% |
| Random Forest |  | 11 | 200.0–400.0 trees | 0.00500 | 0.00003 | 2.3103 | 0.00216 | 0.572% |
| Random Forest |  | 23 | 10.0–25.0 trees | 0.01900 | 0.00127 | 0.2308 | 0.08233 | 2.265% |
| Random Forest |  | 23 | 25.0–50.0 trees | 0.01150 | 0.00046 | 0.3555 | 0.03235 | 1.340% |
| Random Forest |  | 23 | 50.0–100.0 trees | 0.00400 | 0.00008 | 0.6831 | 0.00586 | 0.460% |
| Random Forest |  | 23 | 100.0–200.0 trees | 0.00200 | 0.00002 | 1.3142 | 0.00152 | 0.229% |
| Random Forest |  | 23 | 200.0–400.0 trees | -0.00050 | -0.00000 | 2.2176 | -0.00023 | -0.057% |
| Random Forest |  | 37 | 10.0–25.0 trees | 0.01650 | 0.00110 | 0.2188 | 0.07542 | 1.960% |
| Random Forest |  | 37 | 25.0–50.0 trees | 0.01400 | 0.00056 | 0.3693 | 0.03791 | 1.631% |
| Random Forest |  | 37 | 50.0–100.0 trees | 0.00100 | 0.00002 | 0.6755 | 0.00148 | 0.115% |
| Random Forest |  | 37 | 100.0–200.0 trees | -0.00150 | -0.00002 | 1.1766 | -0.00127 | -0.172% |
| Random Forest |  | 37 | 200.0–400.0 trees | 0.00100 | 0.00001 | 2.3189 | 0.00043 | 0.115% |
| MiniBatch K-Means |  | 11 | 1.0–5.0 partial_fit_updates | 0.01003 | 0.00251 | 0.0280 | 0.35853 | 2.717% |
| MiniBatch K-Means |  | 11 | 5.0–10.0 partial_fit_updates | 0.00391 | 0.00078 | 0.0300 | 0.13027 | 1.030% |
| MiniBatch K-Means |  | 11 | 10.0–20.0 partial_fit_updates | 0.00189 | 0.00019 | 0.0593 | 0.03193 | 0.494% |
| MiniBatch K-Means |  | 11 | 20.0–50.0 partial_fit_updates | 0.00048 | 0.00002 | 0.1817 | 0.00262 | 0.124% |
| Gaussian Mixture Model |  | 11 | 1.0–3.0 warm_start_em_iterations | 0.00498 | 0.00249 | 0.0226 | 0.22020 | 1.212% |
| Gaussian Mixture Model |  | 11 | 3.0–5.0 warm_start_em_iterations | 0.00669 | 0.00334 | 0.0239 | 0.27997 | 1.607% |
| Gaussian Mixture Model |  | 11 | 5.0–10.0 warm_start_em_iterations | -0.01709 | -0.00342 | 0.0536 | -0.31913 | -4.043% |
| Gaussian Mixture Model |  | 11 | 10.0–20.0 warm_start_em_iterations | -0.01560 | -0.00156 | 0.0985 | -0.15830 | -3.844% |
| MiniBatch K-Means |  | 23 | 1.0–5.0 partial_fit_updates | 0.01326 | 0.00331 | 0.0281 | 0.47190 | 3.351% |
| MiniBatch K-Means |  | 23 | 5.0–10.0 partial_fit_updates | -0.00121 | -0.00024 | 0.0285 | -0.04246 | -0.296% |
| MiniBatch K-Means |  | 23 | 10.0–20.0 partial_fit_updates | 0.00031 | 0.00003 | 0.0675 | 0.00464 | 0.077% |
| MiniBatch K-Means |  | 23 | 20.0–50.0 partial_fit_updates | 0.00071 | 0.00002 | 0.2178 | 0.00327 | 0.175% |
| Gaussian Mixture Model |  | 23 | 1.0–3.0 warm_start_em_iterations | -0.00024 | -0.00012 | 0.0311 | -0.00774 | -0.072% |
| Gaussian Mixture Model |  | 23 | 3.0–5.0 warm_start_em_iterations | 0.00576 | 0.00288 | 0.0294 | 0.19593 | 1.736% |
| Gaussian Mixture Model |  | 23 | 5.0–10.0 warm_start_em_iterations | 0.02839 | 0.00568 | 0.0754 | 0.37651 | 8.404% |
| Gaussian Mixture Model |  | 23 | 10.0–20.0 warm_start_em_iterations | -0.00241 | -0.00024 | 0.2184 | -0.01105 | -0.659% |
| MiniBatch K-Means |  | 37 | 1.0–5.0 partial_fit_updates | 0.01563 | 0.00391 | 0.0540 | 0.28940 | 4.443% |
| MiniBatch K-Means |  | 37 | 5.0–10.0 partial_fit_updates | -0.00224 | -0.00045 | 0.0685 | -0.03269 | -0.609% |
| MiniBatch K-Means |  | 37 | 10.0–20.0 partial_fit_updates | -0.00011 | -0.00001 | 0.1334 | -0.00084 | -0.031% |
| MiniBatch K-Means |  | 37 | 20.0–50.0 partial_fit_updates | -0.00131 | -0.00004 | 0.4152 | -0.00316 | -0.359% |
| Gaussian Mixture Model |  | 37 | 1.0–3.0 warm_start_em_iterations | -0.00087 | -0.00043 | 0.0320 | -0.02707 | -0.239% |
| Gaussian Mixture Model |  | 37 | 3.0–5.0 warm_start_em_iterations | 0.00140 | 0.00070 | 0.0333 | 0.04207 | 0.389% |
| Gaussian Mixture Model |  | 37 | 5.0–10.0 warm_start_em_iterations | 0.00107 | 0.00021 | 0.0730 | 0.01459 | 0.294% |
| Gaussian Mixture Model |  | 37 | 10.0–20.0 warm_start_em_iterations | 0.01487 | 0.00149 | 0.1362 | 0.10916 | 4.095% |
| Labeled-only baseline | frac=0.01 | 11 | 1.0–2.0 epochs | -0.01450 | -0.01450 | 0.5323 | -0.02724 | -2.176% |
| Labeled-only baseline | frac=0.01 | 11 | 2.0–3.0 epochs | 0.00350 | 0.00350 | 0.5204 | 0.00673 | 0.537% |
| Labeled-only baseline | frac=0.01 | 11 | 3.0–4.0 epochs | 0.00050 | 0.00050 | 0.5038 | 0.00099 | 0.076% |
| Labeled-only baseline | frac=0.01 | 11 | 4.0–5.0 epochs | -0.00200 | -0.00200 | 0.4901 | -0.00408 | -0.305% |
| Pseudo-Labeling | frac=0.01 | 11 | 1.0–2.0 epochs | 0.00150 | 0.00150 | 1.2329 | 0.00122 | 0.222% |
| Pseudo-Labeling | frac=0.01 | 11 | 2.0–3.0 epochs | -0.01150 | -0.01150 | 1.0176 | -0.01130 | -1.699% |
| Pseudo-Labeling | frac=0.01 | 11 | 3.0–4.0 epochs | 0.00300 | 0.00300 | 0.9296 | 0.00323 | 0.451% |
| Pseudo-Labeling | frac=0.01 | 11 | 4.0–5.0 epochs | -0.01100 | -0.01100 | 0.9541 | -0.01153 | -1.645% |
| Labeled-only baseline | frac=0.05 | 11 | 1.0–2.0 epochs | 0.00950 | 0.00950 | 0.4778 | 0.01988 | 1.227% |
| Labeled-only baseline | frac=0.05 | 11 | 2.0–3.0 epochs | -0.00550 | -0.00550 | 0.4947 | -0.01112 | -0.702% |
| Labeled-only baseline | frac=0.05 | 11 | 3.0–4.0 epochs | 0.00350 | 0.00350 | 0.4785 | 0.00731 | 0.450% |
| Labeled-only baseline | frac=0.05 | 11 | 4.0–5.0 epochs | -0.00500 | -0.00500 | 0.4859 | -0.01029 | -0.640% |
| Pseudo-Labeling | frac=0.05 | 11 | 1.0–2.0 epochs | 0.02400 | 0.02400 | 0.9156 | 0.02621 | 3.158% |
| Pseudo-Labeling | frac=0.05 | 11 | 2.0–3.0 epochs | 0.00100 | 0.00100 | 0.8943 | 0.00112 | 0.128% |
| Pseudo-Labeling | frac=0.05 | 11 | 3.0–4.0 epochs | 0.00300 | 0.00300 | 0.8711 | 0.00344 | 0.382% |
| Pseudo-Labeling | frac=0.05 | 11 | 4.0–5.0 epochs | -0.00750 | -0.00750 | 0.8599 | -0.00872 | -0.952% |
| Labeled-only baseline | frac=0.1 | 11 | 1.0–2.0 epochs | 0.03450 | 0.03450 | 0.4484 | 0.07694 | 4.475% |
| Labeled-only baseline | frac=0.1 | 11 | 2.0–3.0 epochs | 0.00350 | 0.00350 | 0.4425 | 0.00791 | 0.435% |
| Labeled-only baseline | frac=0.1 | 11 | 3.0–4.0 epochs | 0.01300 | 0.01300 | 0.4321 | 0.03008 | 1.607% |
| Labeled-only baseline | frac=0.1 | 11 | 4.0–5.0 epochs | -0.00500 | -0.00500 | 0.4325 | -0.01156 | -0.608% |
| Pseudo-Labeling | frac=0.1 | 11 | 1.0–2.0 epochs | 0.05100 | 0.05100 | 0.8644 | 0.05900 | 6.832% |
| Pseudo-Labeling | frac=0.1 | 11 | 2.0–3.0 epochs | 0.00800 | 0.00800 | 0.8601 | 0.00930 | 1.003% |
| Pseudo-Labeling | frac=0.1 | 11 | 3.0–4.0 epochs | 0.00950 | 0.00950 | 0.8439 | 0.01126 | 1.179% |
| Pseudo-Labeling | frac=0.1 | 11 | 4.0–5.0 epochs | 0.00400 | 0.00400 | 0.8366 | 0.00478 | 0.491% |
| Labeled-only baseline | frac=0.2 | 11 | 1.0–2.0 epochs | 0.07950 | 0.07950 | 0.3955 | 0.20101 | 10.913% |
| Labeled-only baseline | frac=0.2 | 11 | 2.0–3.0 epochs | 0.00800 | 0.00800 | 0.3967 | 0.02017 | 0.990% |
| Labeled-only baseline | frac=0.2 | 11 | 3.0–4.0 epochs | 0.00200 | 0.00200 | 0.3908 | 0.00512 | 0.245% |
| Labeled-only baseline | frac=0.2 | 11 | 4.0–5.0 epochs | 0.01550 | 0.01550 | 0.3907 | 0.03967 | 1.895% |
| Pseudo-Labeling | frac=0.2 | 11 | 1.0–2.0 epochs | 0.07200 | 0.07200 | 0.7417 | 0.09708 | 9.730% |
| Pseudo-Labeling | frac=0.2 | 11 | 2.0–3.0 epochs | -0.00350 | -0.00350 | 0.7270 | -0.00481 | -0.431% |
| Pseudo-Labeling | frac=0.2 | 11 | 3.0–4.0 epochs | 0.00300 | 0.00300 | 0.7138 | 0.00420 | 0.371% |
| Pseudo-Labeling | frac=0.2 | 11 | 4.0–5.0 epochs | 0.01800 | 0.01800 | 0.7298 | 0.02466 | 2.218% |
| Labeled-only baseline | frac=0.4 | 11 | 1.0–2.0 epochs | 0.04150 | 0.04150 | 0.3019 | 0.13747 | 5.500% |
| Labeled-only baseline | frac=0.4 | 11 | 2.0–3.0 epochs | 0.00500 | 0.00500 | 0.2978 | 0.01679 | 0.628% |
| Labeled-only baseline | frac=0.4 | 11 | 3.0–4.0 epochs | 0.02200 | 0.02200 | 0.2900 | 0.07586 | 2.747% |
| Labeled-only baseline | frac=0.4 | 11 | 4.0–5.0 epochs | 0.01100 | 0.01100 | 0.3027 | 0.03634 | 1.337% |
| Pseudo-Labeling | frac=0.4 | 11 | 1.0–2.0 epochs | 0.04800 | 0.04800 | 0.5554 | 0.08643 | 6.473% |
| Pseudo-Labeling | frac=0.4 | 11 | 2.0–3.0 epochs | 0.01500 | 0.01500 | 0.5613 | 0.02672 | 1.900% |
| Pseudo-Labeling | frac=0.4 | 11 | 3.0–4.0 epochs | 0.02050 | 0.02050 | 0.5424 | 0.03780 | 2.548% |
| Pseudo-Labeling | frac=0.4 | 11 | 4.0–5.0 epochs | 0.01000 | 0.01000 | 0.5521 | 0.01811 | 1.212% |
| Labeled-only baseline | frac=0.01 | 23 | 1.0–2.0 epochs | 0.00900 | 0.00900 | 0.4914 | 0.01832 | 1.342% |
| Labeled-only baseline | frac=0.01 | 23 | 2.0–3.0 epochs | 0.01000 | 0.01000 | 0.4956 | 0.02018 | 1.472% |
| Labeled-only baseline | frac=0.01 | 23 | 3.0–4.0 epochs | -0.00600 | -0.00600 | 0.4949 | -0.01212 | -0.870% |
| Labeled-only baseline | frac=0.01 | 23 | 4.0–5.0 epochs | 0.00500 | 0.00500 | 0.4971 | 0.01006 | 0.732% |
| Pseudo-Labeling | frac=0.01 | 23 | 1.0–2.0 epochs | -0.00700 | -0.00700 | 0.9156 | -0.00764 | -1.060% |
| Pseudo-Labeling | frac=0.01 | 23 | 2.0–3.0 epochs | -0.00900 | -0.00900 | 0.9205 | -0.00978 | -1.377% |
| Pseudo-Labeling | frac=0.01 | 23 | 3.0–4.0 epochs | 0.00150 | 0.00150 | 0.9077 | 0.00165 | 0.233% |
| Pseudo-Labeling | frac=0.01 | 23 | 4.0–5.0 epochs | -0.00950 | -0.00950 | 0.9222 | -0.01030 | -1.471% |
| Labeled-only baseline | frac=0.05 | 23 | 1.0–2.0 epochs | 0.01850 | 0.01850 | 0.4494 | 0.04116 | 2.392% |
| Labeled-only baseline | frac=0.05 | 23 | 2.0–3.0 epochs | -0.00800 | -0.00800 | 0.4575 | -0.01749 | -1.010% |
| Labeled-only baseline | frac=0.05 | 23 | 3.0–4.0 epochs | 0.00100 | 0.00100 | 0.4694 | 0.00213 | 0.128% |
| Labeled-only baseline | frac=0.05 | 23 | 4.0–5.0 epochs | -0.00100 | -0.00100 | 0.4424 | -0.00226 | -0.127% |
| Pseudo-Labeling | frac=0.05 | 23 | 1.0–2.0 epochs | 0.02750 | 0.02750 | 0.8619 | 0.03191 | 3.588% |
| Pseudo-Labeling | frac=0.05 | 23 | 2.0–3.0 epochs | 0.00450 | 0.00450 | 0.8886 | 0.00506 | 0.567% |
| Pseudo-Labeling | frac=0.05 | 23 | 3.0–4.0 epochs | -0.00750 | -0.00750 | 0.8606 | -0.00872 | -0.939% |
| Pseudo-Labeling | frac=0.05 | 23 | 4.0–5.0 epochs | 0.01350 | 0.01350 | 0.8630 | 0.01564 | 1.707% |
| Labeled-only baseline | frac=0.1 | 23 | 1.0–2.0 epochs | 0.02800 | 0.02800 | 0.4466 | 0.06270 | 3.585% |
| Labeled-only baseline | frac=0.1 | 23 | 2.0–3.0 epochs | 0.00050 | 0.00050 | 0.4493 | 0.00111 | 0.062% |
| Labeled-only baseline | frac=0.1 | 23 | 3.0–4.0 epochs | 0.01200 | 0.01200 | 0.4393 | 0.02732 | 1.482% |
| Labeled-only baseline | frac=0.1 | 23 | 4.0–5.0 epochs | 0.00250 | 0.00250 | 0.4397 | 0.00569 | 0.304% |
| Pseudo-Labeling | frac=0.1 | 23 | 1.0–2.0 epochs | 0.02500 | 0.02500 | 0.8141 | 0.03071 | 3.201% |
| Pseudo-Labeling | frac=0.1 | 23 | 2.0–3.0 epochs | 0.00500 | 0.00500 | 0.8573 | 0.00583 | 0.620% |
| Pseudo-Labeling | frac=0.1 | 23 | 3.0–4.0 epochs | 0.00650 | 0.00650 | 0.8444 | 0.00770 | 0.801% |
| Pseudo-Labeling | frac=0.1 | 23 | 4.0–5.0 epochs | 0.00050 | 0.00050 | 0.8568 | 0.00058 | 0.061% |
| Labeled-only baseline | frac=0.2 | 23 | 1.0–2.0 epochs | 0.03600 | 0.03600 | 0.4115 | 0.08748 | 4.627% |
| Labeled-only baseline | frac=0.2 | 23 | 2.0–3.0 epochs | 0.00950 | 0.00950 | 0.4072 | 0.02333 | 1.167% |
| Labeled-only baseline | frac=0.2 | 23 | 3.0–4.0 epochs | 0.01000 | 0.01000 | 0.4024 | 0.02485 | 1.214% |
| Labeled-only baseline | frac=0.2 | 23 | 4.0–5.0 epochs | 0.00250 | 0.00250 | 0.3951 | 0.00633 | 0.300% |
| Pseudo-Labeling | frac=0.2 | 23 | 1.0–2.0 epochs | 0.04150 | 0.04150 | 0.7303 | 0.05683 | 5.511% |
| Pseudo-Labeling | frac=0.2 | 23 | 2.0–3.0 epochs | 0.02750 | 0.02750 | 0.7199 | 0.03820 | 3.461% |
| Pseudo-Labeling | frac=0.2 | 23 | 3.0–4.0 epochs | 0.01150 | 0.01150 | 0.7172 | 0.01604 | 1.399% |
| Pseudo-Labeling | frac=0.2 | 23 | 4.0–5.0 epochs | -0.00150 | -0.00150 | 0.7243 | -0.00207 | -0.180% |
| Labeled-only baseline | frac=0.4 | 23 | 1.0–2.0 epochs | 0.05750 | 0.05750 | 0.2964 | 0.19399 | 7.791% |
| Labeled-only baseline | frac=0.4 | 23 | 2.0–3.0 epochs | 0.00200 | 0.00200 | 0.2873 | 0.00696 | 0.251% |
| Labeled-only baseline | frac=0.4 | 23 | 3.0–4.0 epochs | 0.02500 | 0.02500 | 0.2882 | 0.08674 | 3.135% |
| Labeled-only baseline | frac=0.4 | 23 | 4.0–5.0 epochs | 0.00850 | 0.00850 | 0.2839 | 0.02994 | 1.033% |
| Pseudo-Labeling | frac=0.4 | 23 | 1.0–2.0 epochs | 0.08450 | 0.08450 | 0.5475 | 0.15435 | 11.893% |
| Pseudo-Labeling | frac=0.4 | 23 | 2.0–3.0 epochs | 0.01650 | 0.01650 | 0.5250 | 0.03143 | 2.075% |
| Pseudo-Labeling | frac=0.4 | 23 | 3.0–4.0 epochs | 0.00850 | 0.00850 | 0.5450 | 0.01560 | 1.047% |
| Pseudo-Labeling | frac=0.4 | 23 | 4.0–5.0 epochs | 0.00200 | 0.00200 | 0.5418 | 0.00369 | 0.244% |
| Labeled-only baseline | frac=0.01 | 37 | 1.0–2.0 epochs | 0.00500 | 0.00500 | 0.4869 | 0.01027 | 0.750% |
| Labeled-only baseline | frac=0.01 | 37 | 2.0–3.0 epochs | 0.00400 | 0.00400 | 0.5162 | 0.00775 | 0.596% |
| Labeled-only baseline | frac=0.01 | 37 | 3.0–4.0 epochs | -0.00300 | -0.00300 | 0.4833 | -0.00621 | -0.444% |
| Labeled-only baseline | frac=0.01 | 37 | 4.0–5.0 epochs | 0.00450 | 0.00450 | 0.4784 | 0.00941 | 0.669% |
| Pseudo-Labeling | frac=0.01 | 37 | 1.0–2.0 epochs | -0.00100 | -0.00100 | 0.8993 | -0.00111 | -0.147% |
| Pseudo-Labeling | frac=0.01 | 37 | 2.0–3.0 epochs | -0.00850 | -0.00850 | 0.8875 | -0.00958 | -1.250% |
| Pseudo-Labeling | frac=0.01 | 37 | 3.0–4.0 epochs | -0.00100 | -0.00100 | 0.8982 | -0.00111 | -0.149% |
| Pseudo-Labeling | frac=0.01 | 37 | 4.0–5.0 epochs | 0.00200 | 0.00200 | 0.9006 | 0.00222 | 0.298% |
| Labeled-only baseline | frac=0.05 | 37 | 1.0–2.0 epochs | 0.01050 | 0.01050 | 0.4622 | 0.02272 | 1.343% |
| Labeled-only baseline | frac=0.05 | 37 | 2.0–3.0 epochs | -0.00850 | -0.00850 | 0.4608 | -0.01845 | -1.073% |
| Labeled-only baseline | frac=0.05 | 37 | 3.0–4.0 epochs | 0.00850 | 0.00850 | 0.4623 | 0.01838 | 1.084% |
| Labeled-only baseline | frac=0.05 | 37 | 4.0–5.0 epochs | -0.00500 | -0.00500 | 0.4635 | -0.01079 | -0.631% |
| Pseudo-Labeling | frac=0.05 | 37 | 1.0–2.0 epochs | 0.02400 | 0.02400 | 0.8577 | 0.02798 | 3.145% |
| Pseudo-Labeling | frac=0.05 | 37 | 2.0–3.0 epochs | -0.00400 | -0.00400 | 0.8609 | -0.00465 | -0.508% |
| Pseudo-Labeling | frac=0.05 | 37 | 3.0–4.0 epochs | 0.01400 | 0.01400 | 0.8753 | 0.01599 | 1.788% |
| Pseudo-Labeling | frac=0.05 | 37 | 4.0–5.0 epochs | -0.00350 | -0.00350 | 0.8782 | -0.00399 | -0.439% |
| Labeled-only baseline | frac=0.1 | 37 | 1.0–2.0 epochs | 0.01150 | 0.01150 | 0.4417 | 0.02604 | 1.464% |
| Labeled-only baseline | frac=0.1 | 37 | 2.0–3.0 epochs | 0.00850 | 0.00850 | 0.4537 | 0.01873 | 1.066% |
| Labeled-only baseline | frac=0.1 | 37 | 3.0–4.0 epochs | 0.00750 | 0.00750 | 0.4501 | 0.01666 | 0.931% |
| Labeled-only baseline | frac=0.1 | 37 | 4.0–5.0 epochs | 0.00650 | 0.00650 | 0.4418 | 0.01471 | 0.800% |
| Pseudo-Labeling | frac=0.1 | 37 | 1.0–2.0 epochs | 0.05350 | 0.05350 | 0.9456 | 0.05658 | 7.072% |
| Pseudo-Labeling | frac=0.1 | 37 | 2.0–3.0 epochs | 0.00100 | 0.00100 | 0.8739 | 0.00114 | 0.123% |
| Pseudo-Labeling | frac=0.1 | 37 | 3.0–4.0 epochs | 0.00300 | 0.00300 | 0.8708 | 0.00344 | 0.370% |
| Pseudo-Labeling | frac=0.1 | 37 | 4.0–5.0 epochs | 0.01450 | 0.01450 | 0.7967 | 0.01820 | 1.781% |
| Labeled-only baseline | frac=0.2 | 37 | 1.0–2.0 epochs | 0.03100 | 0.03100 | 0.4022 | 0.07708 | 3.972% |
| Labeled-only baseline | frac=0.2 | 37 | 2.0–3.0 epochs | -0.00550 | -0.00550 | 0.4083 | -0.01347 | -0.678% |
| Labeled-only baseline | frac=0.2 | 37 | 3.0–4.0 epochs | 0.01900 | 0.01900 | 0.3998 | 0.04752 | 2.357% |
| Labeled-only baseline | frac=0.2 | 37 | 4.0–5.0 epochs | 0.00300 | 0.00300 | 0.3941 | 0.00761 | 0.364% |
| Pseudo-Labeling | frac=0.2 | 37 | 1.0–2.0 epochs | 0.03750 | 0.03750 | 0.7365 | 0.05091 | 5.007% |
| Pseudo-Labeling | frac=0.2 | 37 | 2.0–3.0 epochs | 0.01200 | 0.01200 | 0.7384 | 0.01625 | 1.526% |
| Pseudo-Labeling | frac=0.2 | 37 | 3.0–4.0 epochs | 0.02000 | 0.02000 | 0.7365 | 0.02716 | 2.505% |
| Pseudo-Labeling | frac=0.2 | 37 | 4.0–5.0 epochs | 0.00750 | 0.00750 | 0.7257 | 0.01034 | 0.916% |
| Labeled-only baseline | frac=0.4 | 37 | 1.0–2.0 epochs | 0.01650 | 0.01650 | 0.2909 | 0.05672 | 2.172% |
| Labeled-only baseline | frac=0.4 | 37 | 2.0–3.0 epochs | 0.03250 | 0.03250 | 0.3016 | 0.10776 | 4.188% |
| Labeled-only baseline | frac=0.4 | 37 | 3.0–4.0 epochs | 0.01200 | 0.01200 | 0.2877 | 0.04171 | 1.484% |
| Labeled-only baseline | frac=0.4 | 37 | 4.0–5.0 epochs | 0.01600 | 0.01600 | 0.2878 | 0.05559 | 1.950% |
| Pseudo-Labeling | frac=0.4 | 37 | 1.0–2.0 epochs | 0.03800 | 0.03800 | 0.5324 | 0.07137 | 5.163% |
| Pseudo-Labeling | frac=0.4 | 37 | 2.0–3.0 epochs | 0.05400 | 0.05400 | 0.5370 | 0.10056 | 6.977% |
| Pseudo-Labeling | frac=0.4 | 37 | 3.0–4.0 epochs | 0.00050 | 0.00050 | 0.5357 | 0.00093 | 0.060% |
| Pseudo-Labeling | frac=0.4 | 37 | 4.0–5.0 epochs | 0.00450 | 0.00450 | 0.5377 | 0.00837 | 0.543% |
| DQN | CartPole-v1 | 11 | 5000.0–10000.0 environment_steps | 8.45000 | 0.00169 | 3.6163 | 2.33666 | 49.560% |
| DQN | CartPole-v1 | 11 | 10000.0–25000.0 environment_steps | 17.55000 | 0.00117 | 10.2201 | 1.71720 | 68.824% |
| DQN | CartPole-v1 | 23 | 5000.0–10000.0 environment_steps | 7.60000 | 0.00152 | 3.4910 | 2.17702 | 24.756% |
| DQN | CartPole-v1 | 23 | 10000.0–25000.0 environment_steps | 12.45000 | 0.00083 | 10.0586 | 1.23775 | 32.507% |
| DQN | CartPole-v1 | 37 | 5000.0–10000.0 environment_steps | 0.45000 | 0.00009 | 3.3936 | 0.13260 | 4.945% |
| DQN | CartPole-v1 | 37 | 10000.0–25000.0 environment_steps | 0.15000 | 0.00001 | 10.2989 | 0.01456 | 1.571% |
| A2C | CartPole-v1 | 11 | 5000.0–10000.0 environment_steps | 284.35000 | 0.05687 | 2.7606 | 103.00435 | 200.246% |
| A2C | CartPole-v1 | 11 | 10000.0–25000.0 environment_steps | 57.95000 | 0.00386 | 8.4537 | 6.85495 | 13.592% |
| A2C | CartPole-v1 | 23 | 5000.0–10000.0 environment_steps | 158.30000 | 0.03166 | 3.2738 | 48.35337 | 67.707% |
| A2C | CartPole-v1 | 23 | 10000.0–25000.0 environment_steps | 105.20000 | 0.00701 | 8.7505 | 12.02212 | 26.830% |
| A2C | CartPole-v1 | 37 | 5000.0–10000.0 environment_steps | 355.55000 | 0.07111 | 2.9790 | 119.35094 | 434.924% |
| A2C | CartPole-v1 | 37 | 10000.0–25000.0 environment_steps | -91.25000 | -0.00608 | 8.1550 | -11.18939 | -20.867% |
| PPO | CartPole-v1 | 11 | 5000.0–10000.0 environment_steps | 190.05000 | 0.03801 | 3.7671 | 50.45032 | 95.671% |
| PPO | CartPole-v1 | 11 | 10000.0–25000.0 environment_steps | 76.35000 | 0.00509 | 11.4665 | 6.65853 | 19.642% |
| PPO | CartPole-v1 | 23 | 5000.0–10000.0 environment_steps | 71.65000 | 0.01433 | 3.7545 | 19.08401 | 35.888% |
| PPO | CartPole-v1 | 23 | 10000.0–25000.0 environment_steps | -19.95000 | -0.00133 | 11.0592 | -1.80394 | -7.353% |
| PPO | CartPole-v1 | 37 | 5000.0–10000.0 environment_steps | 144.75000 | 0.02895 | 3.7667 | 38.42851 | 97.639% |
| PPO | CartPole-v1 | 37 | 10000.0–25000.0 environment_steps | 206.50000 | 0.01377 | 12.0000 | 17.20831 | 70.478% |
| SAC | Pendulum-v1 | 11 | 5000.0–10000.0 environment_steps | 237.73819 | 0.04755 | 46.9384 | 5.06490 | 54.638% |
| SAC | Pendulum-v1 | 11 | 10000.0–25000.0 environment_steps | 12.67749 | 0.00085 | 139.0208 | 0.09119 | 6.423% |
| SAC | Pendulum-v1 | 23 | 5000.0–10000.0 environment_steps | 63.02264 | 0.01260 | 45.9007 | 1.37302 | 28.021% |
| SAC | Pendulum-v1 | 23 | 10000.0–25000.0 environment_steps | 3.23764 | 0.00022 | 139.8545 | 0.02315 | 2.000% |
| SAC | Pendulum-v1 | 37 | 5000.0–10000.0 environment_steps | 123.32973 | 0.02467 | 45.9561 | 2.68364 | 42.208% |
| SAC | Pendulum-v1 | 37 | 10000.0–25000.0 environment_steps | 7.19906 | 0.00048 | 138.2998 | 0.05205 | 4.263% |
| Autoencoder | downstream linear probe | 11 | 1.0–2.0 epochs | 0.00350 | 0.00350 | 0.4369 | 0.00801 | 0.445% |
| Autoencoder | downstream linear probe | 11 | 2.0–3.0 epochs | 0.01700 | 0.01700 | 0.4029 | 0.04219 | 2.152% |
| Autoencoder | downstream linear probe | 11 | 3.0–4.0 epochs | 0.00400 | 0.00400 | 0.3803 | 0.01052 | 0.496% |
| Autoencoder | downstream linear probe | 11 | 4.0–5.0 epochs | 0.00100 | 0.00100 | 0.3852 | 0.00260 | 0.123% |
| Autoencoder |  | 11 | 1.0–2.0 epochs | 0.01315 | 0.01315 | 0.4369 | 0.03009 | 32.627% |
| Autoencoder |  | 11 | 2.0–3.0 epochs | 0.00391 | 0.00391 | 0.4029 | 0.00970 | 14.400% |
| Autoencoder |  | 11 | 3.0–4.0 epochs | 0.00150 | 0.00150 | 0.3803 | 0.00395 | 6.471% |
| Autoencoder |  | 11 | 4.0–5.0 epochs | 0.00219 | 0.00219 | 0.3852 | 0.00569 | 10.078% |
| Autoencoder | downstream linear probe | 23 | 1.0–2.0 epochs | 0.01300 | 0.01300 | 0.3711 | 0.03504 | 1.667% |
| Autoencoder | downstream linear probe | 23 | 2.0–3.0 epochs | 0.00750 | 0.00750 | 0.3896 | 0.01925 | 0.946% |
| Autoencoder | downstream linear probe | 23 | 3.0–4.0 epochs | 0.00350 | 0.00350 | 0.3898 | 0.00898 | 0.437% |
| Autoencoder | downstream linear probe | 23 | 4.0–5.0 epochs | 0.01300 | 0.01300 | 0.3964 | 0.03280 | 1.617% |
| Autoencoder |  | 23 | 1.0–2.0 epochs | 0.01382 | 0.01382 | 0.3711 | 0.03724 | 34.133% |
| Autoencoder |  | 23 | 2.0–3.0 epochs | 0.00343 | 0.00343 | 0.3896 | 0.00879 | 12.846% |
| Autoencoder |  | 23 | 3.0–4.0 epochs | 0.00198 | 0.00198 | 0.3898 | 0.00507 | 8.499% |
| Autoencoder |  | 23 | 4.0–5.0 epochs | 0.00137 | 0.00137 | 0.3964 | 0.00345 | 6.434% |
| Autoencoder | downstream linear probe | 37 | 1.0–2.0 epochs | 0.02750 | 0.02750 | 0.3959 | 0.06946 | 3.551% |
| Autoencoder | downstream linear probe | 37 | 2.0–3.0 epochs | 0.00750 | 0.00750 | 0.3907 | 0.01920 | 0.935% |
| Autoencoder | downstream linear probe | 37 | 3.0–4.0 epochs | 0.00250 | 0.00250 | 0.3774 | 0.00662 | 0.309% |
| Autoencoder | downstream linear probe | 37 | 4.0–5.0 epochs | 0.00350 | 0.00350 | 0.3660 | 0.00956 | 0.431% |
| Autoencoder |  | 37 | 1.0–2.0 epochs | 0.01508 | 0.01508 | 0.3959 | 0.03809 | 36.135% |
| Autoencoder |  | 37 | 2.0–3.0 epochs | 0.00273 | 0.00273 | 0.3907 | 0.00700 | 10.261% |
| Autoencoder |  | 37 | 3.0–4.0 epochs | 0.00195 | 0.00195 | 0.3774 | 0.00516 | 8.149% |
| Autoencoder |  | 37 | 4.0–5.0 epochs | 0.00263 | 0.00263 | 0.3660 | 0.00718 | 11.955% |
| Variational Autoencoder | downstream linear probe | 11 | 1.0–2.0 epochs | 0.00050 | 0.00050 | 0.4778 | 0.00105 | 0.064% |
| Variational Autoencoder | downstream linear probe | 11 | 2.0–3.0 epochs | 0.01250 | 0.01250 | 0.5028 | 0.02486 | 1.594% |
| Variational Autoencoder | downstream linear probe | 11 | 3.0–4.0 epochs | 0.00000 | 0.00000 | 0.4844 | 0.00000 | 0.000% |
| Variational Autoencoder | downstream linear probe | 11 | 4.0–5.0 epochs | 0.01150 | 0.01150 | 0.4902 | 0.02346 | 1.444% |
| Variational Autoencoder |  | 11 | 1.0–2.0 epochs | 0.01402 | 0.01402 | 0.4778 | 0.02935 | 24.721% |
| Variational Autoencoder |  | 11 | 2.0–3.0 epochs | 0.00731 | 0.00731 | 0.5028 | 0.01454 | 17.120% |
| Variational Autoencoder |  | 11 | 3.0–4.0 epochs | 0.00534 | 0.00534 | 0.4844 | 0.01102 | 15.086% |
| Variational Autoencoder |  | 11 | 4.0–5.0 epochs | 0.00086 | 0.00086 | 0.4902 | 0.00175 | 2.847% |
| Variational Autoencoder | downstream linear probe | 23 | 1.0–2.0 epochs | 0.00150 | 0.00150 | 0.5413 | 0.00277 | 0.193% |
| Variational Autoencoder | downstream linear probe | 23 | 2.0–3.0 epochs | 0.00900 | 0.00900 | 0.5057 | 0.01780 | 1.154% |
| Variational Autoencoder | downstream linear probe | 23 | 3.0–4.0 epochs | -0.00100 | -0.00100 | 0.4824 | -0.00207 | -0.127% |
| Variational Autoencoder | downstream linear probe | 23 | 4.0–5.0 epochs | 0.00550 | 0.00550 | 0.4832 | 0.01138 | 0.698% |
| Variational Autoencoder |  | 23 | 1.0–2.0 epochs | 0.01721 | 0.01721 | 0.5413 | 0.03178 | 29.173% |
| Variational Autoencoder |  | 23 | 2.0–3.0 epochs | 0.00769 | 0.00769 | 0.5057 | 0.01521 | 18.412% |
| Variational Autoencoder |  | 23 | 3.0–4.0 epochs | 0.00215 | 0.00215 | 0.4824 | 0.00446 | 6.314% |
| Variational Autoencoder |  | 23 | 4.0–5.0 epochs | 0.00274 | 0.00274 | 0.4832 | 0.00566 | 8.567% |
| Variational Autoencoder | downstream linear probe | 37 | 1.0–2.0 epochs | 0.01100 | 0.01100 | 0.4774 | 0.02304 | 1.408% |
| Variational Autoencoder | downstream linear probe | 37 | 2.0–3.0 epochs | 0.01400 | 0.01400 | 0.4886 | 0.02865 | 1.767% |
| Variational Autoencoder | downstream linear probe | 37 | 3.0–4.0 epochs | 0.00100 | 0.00100 | 0.4691 | 0.00213 | 0.124% |
| Variational Autoencoder | downstream linear probe | 37 | 4.0–5.0 epochs | -0.00600 | -0.00600 | 0.4834 | -0.01241 | -0.743% |
| Variational Autoencoder |  | 37 | 1.0–2.0 epochs | 0.01281 | 0.01281 | 0.4774 | 0.02684 | 23.185% |
| Variational Autoencoder |  | 37 | 2.0–3.0 epochs | 0.00553 | 0.00553 | 0.4886 | 0.01132 | 13.028% |
| Variational Autoencoder |  | 37 | 3.0–4.0 epochs | 0.00482 | 0.00482 | 0.4691 | 0.01028 | 13.062% |
| Variational Autoencoder |  | 37 | 4.0–5.0 epochs | 0.00282 | 0.00282 | 0.4834 | 0.00583 | 8.777% |
