# Final Completion Matrix

Updated 2026-11-07 from verified result files. `pilot` means evidence exists but the full registered objective is not complete. Double DQN has `completed` experiment status and `local_measurement` provenance source type after raw-column alignment. Local checkpoint evaluation of an LLM does not mean local pretraining.

**Strict objective completion: 0/50 configurations (0%).** The denominator remains 20 conventional models + 15 specified LLM configurations + 15 literature configurations. No definitions were relaxed. A configuration counts complete only when its registered study objective is met; pilots, smoke/subset evaluations, and partial literature rows do not count. This does not mean no work was completed; evidence volume, partially completed objectives, and fully complete objectives are distinct.

| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |
|---|---|---|---|---:|---|---|---|
| MLP | yes | yes | 10K train: epochs 1–30; 30K/58K train: epochs 1–5; 3 seeds; 10K/30K/60K requested-size endpoints | 3 | Apple M3 / PyTorch MPS | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | pilot |
| LeNet-style CNN | yes | yes | 10K train: epochs 1–30; 30K/58K train: epochs 1–5; 3 seeds; 10K/30K/60K requested-size endpoints | 3 | Apple M3 / PyTorch MPS | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | pilot |
| ResNet-18 | yes | yes | 10K train: epochs 1–30; 30K/58K train: epochs 1–5; 3 seeds; 10K/30K/60K requested-size endpoints | 3 | Apple M3 / PyTorch MPS | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | pilot |
| ViT-Tiny | yes | yes | 10K Fashion-MNIST: epochs 1–30; 3 seeds; 5,353,738 parameters; 28x28 grayscale / 4x4 patches | 3 | Apple M3 / PyTorch MPS | `results/raw/vit_tiny_scaling.csv`, `results/raw/vit_tiny_test_results.csv` | pilot |
| Random Forest | yes | yes | 10–1,000 warm-start trees; 3 seeds | 3 | CPU | `results/raw/random_forest_scaling.csv`, `results/raw/random_forest_test_results.csv` | pilot |
| K-Means | yes | yes | algorithm-specific iterative budgets | 3 | CPU | `results/raw/unsupervised_scaling.csv` | pilot |
| Gaussian Mixture Model | yes | yes | algorithm-specific iterative budgets | 3 | CPU | `results/raw/unsupervised_scaling.csv` | pilot |
| DBSCAN | yes | yes | 2.5K,5K,10K samples x 3 seeds (data-size cost curve); separate 8-setting parameter sensitivity | 3 | CPU | `results/raw/dbscan_parameter_sweep.csv`, `results/raw/dbscan_size_scaling.csv` | pilot |
| Autoencoder | yes | yes | epochs 1–30; 3 seeds; standardized linear probe | 3 | Apple M3 / PyTorch MPS | `results/raw/autoencoder_scaling.csv`, `results/raw/autoencoder_test_results.csv` | pilot |
| Variational Autoencoder | yes | yes | epochs 1–30; 3 seeds; standardized linear probe | 3 | Apple M3 / PyTorch MPS | `results/raw/autoencoder_scaling.csv`, `results/raw/autoencoder_test_results.csv` | pilot |
| Pseudo-Labeling | yes | yes | epochs 1–5; label fractions 1–40%; 3 seeds | 3 | Apple M3 / PyTorch MPS | `results/raw/pseudolabel_baseline_sweep.csv`, `results/raw/pseudolabel_test_results.csv` | pilot |
| Mean Teacher | yes | yes | epochs 1–30; labeled fractions 1,5,10,20,40%; 3 seeds | 3 | Apple M3 / PyTorch MPS | `results/raw/ssl_algorithms.csv`, `results/raw/ssl_algorithms_test.csv` | pilot |
| FixMatch | yes | yes | epochs 1–30; labeled fractions 1,5,10,20,40%; 3 seeds | 3 | Apple M3 / PyTorch MPS | `results/raw/ssl_algorithms.csv`, `results/raw/ssl_algorithms_test.csv` | pilot |
| MixMatch | yes | yes | epochs 1–30; labeled fractions 1,5,10,20,40%; 3 seeds | 3 | Apple M3 / PyTorch MPS | `results/raw/ssl_algorithms.csv`, `results/raw/ssl_algorithms_test.csv` | pilot |
| Label Propagation | yes | yes | max iterations 1–2,000; tol 1e-3; 3 seeds; convergence observed | 3 | CPU sklearn; actual LabelPropagation | `results/raw/label_propagation_scaling.csv` | pilot |
| DQN | yes | yes | 5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds) | 3 | CPU | `results/raw/rl_scaling_results.csv` | pilot |
| Double DQN | yes | yes | 5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds) | 5 | CPU | `results/raw/rl_scaling_results.csv` | pilot |
| A2C | yes | yes | 5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds) | 3 | CPU | `results/raw/rl_scaling_results.csv` | pilot |
| PPO | yes | yes | 5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds) | 3 | CPU | `results/raw/rl_scaling_results.csv` | pilot |
| SAC | yes | yes | 5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds) | 5 | CPU | `results/raw/rl_scaling_results.csv` | pilot |

### Additional recorded extensions (outside the 50-objective denominator)

| Model | Environment | Budget | Seeds | Result location | Status |
|---|---|---|---:|---|---|
| Compact transformer proxy | Fashion-MNIST | 10K split, epochs 1–30 | 3 | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | Supplementary 3-layer, width-96 proxy; separate from the ViT-Tiny-scope implementation |
| PPO-Pendulum | Pendulum-v1 | 5K–250K requested environment steps | 3 | `results/raw/rl_scaling_results.csv` | Recorded RL extension; not an original sixth method |

## LLM configurations (15)

| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |
|---|---|---|---|---:|---|---|---|
| Pythia-70M | checkpoint/config available | yes, local inference | 8 released checkpoints (0–143K); full WikiText-2 test; full HellaSwag/PIQA/ARC-Easy zero-shot | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | pilot |
| Pythia-160M | checkpoint/config available | yes, local inference | 8 released checkpoints (0–143K); full WikiText-2 test | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | pilot |
| Pythia-410M | checkpoint/config available | yes, local inference | step 0, 10K, 143K prefix checkpoints; full WikiText-2 test | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | pilot |
| Pythia-1B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource_limited |
| Pythia-1.4B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource_limited |
| Pythia-2.8B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource_limited |
| Pythia-6.9B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource_limited |
| Pythia-12B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource_limited |
| GPT-2 Small | checkpoint/config available | yes, local inference | main; full WikiText-2 test + 100/task harness smoke subset | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | pilot |
| GPT-2 Medium | checkpoint/config available | failed local checkpoint attempt | weight retrieval/evaluation incomplete | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | failed |
| GPT-2 Large | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource_limited |
| GPT-2 XL | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource_limited |
| OPT-125M | checkpoint/config available | yes, local inference | main; full WikiText-2 test + 100/task harness smoke subset | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | pilot |
| OPT-350M | checkpoint/config available | yes, local inference | main; full WikiText-2 test | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | pilot |
| OPT-1.3B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource_limited |

## Literature configurations (15)

| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |
|---|---|---|---|---:|---|---|---|
| GPT-3 175B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| Gopher 280B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| Chinchilla 70B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| PaLM 540B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| OPT-13B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| OPT-30B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| OPT-66B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| OPT-175B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | 992 NVIDIA A100 80GB GPUs | `results/literature/large_model_literature.csv` | literature_only |
| LLaMA-7B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| LLaMA-13B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| LLaMA-33B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| LLaMA-65B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| BLOOM-176B | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | NVIDIA A100 80GB | `results/literature/large_model_literature.csv` | literature_only |
| PaLM-62B (795B tokens) | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
| PaLM-62B continued (1.325T tokens) | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | not fully extracted | `results/literature/large_model_literature.csv` | literature_only |
