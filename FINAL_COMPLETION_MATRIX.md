# Final Completion Matrix

Updated 2026-11-08 from raw-result coverage checks. Conventional status requires all registered budgets, seeds, and final evaluations verified by `scripts/audit_conventional_completion.py`. Double DQN retains its source-mapping correction in derived records only; raw CSV rows are preserved. Local checkpoint evaluation of an LLM does not mean local pretraining.

**Strict objective completion: 20/50 configurations (40%).** The denominator remains 20 conventional models + 15 specified LLM configurations + 15 literature configurations. The original criteria and denominator are unchanged. This count distinguishes fully completed registered objectives from collected evidence and partial evaluations.

| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |
|---|---|---|---|---:|---|---|---|
| MLP | yes | yes | 10K train: epochs 1–30; 30K/58K train: epochs 1–5; 3 seeds; 10K/30K/60K requested-size endpoints | 3 | Apple M3 / PyTorch MPS | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | completed |
| LeNet-style CNN | yes | yes | 10K train: epochs 1–30; 30K/58K train: epochs 1–5; 3 seeds; 10K/30K/60K requested-size endpoints | 3 | Apple M3 / PyTorch MPS | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | completed |
| ResNet-18 | yes | yes | 10K train: epochs 1–30; 30K/58K train: epochs 1–5; 3 seeds; 10K/30K/60K requested-size endpoints | 3 | Apple M3 / PyTorch MPS | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | completed |
| ViT-Tiny | yes | yes | 10K Fashion-MNIST: epochs 1–30; 3 seeds; 5,353,738 parameters; 28x28 grayscale / 4x4 patches | 3 | Apple M3 / PyTorch MPS | `results/raw/vit_tiny_scaling.csv`, `results/raw/vit_tiny_test_results.csv` | completed |
| Random Forest | yes | yes | 10–1,000 warm-start trees; 3 seeds | 3 | CPU | `results/raw/random_forest_scaling.csv`, `results/raw/random_forest_test_results.csv` | completed |
| K-Means | yes | yes | algorithm-specific iterative budgets | 3 | CPU | `results/raw/unsupervised_scaling.csv` | completed |
| Gaussian Mixture Model | yes | yes | algorithm-specific iterative budgets | 3 | CPU | `results/raw/unsupervised_scaling.csv` | completed |
| DBSCAN | yes | yes | 2.5K,5K,10K samples x 3 seeds (data-size cost curve); separate 8-setting parameter sensitivity | 3 | CPU | `results/raw/dbscan_parameter_sweep.csv`, `results/raw/dbscan_size_scaling.csv` | completed |
| Autoencoder | yes | yes | epochs 1–30; 3 seeds; standardized linear probe | 3 | Apple M3 / PyTorch MPS | `results/raw/autoencoder_scaling.csv`, `results/raw/autoencoder_test_results.csv` | completed |
| Variational Autoencoder | yes | yes | epochs 1–30; 3 seeds; standardized linear probe | 3 | Apple M3 / PyTorch MPS | `results/raw/autoencoder_scaling.csv`, `results/raw/autoencoder_test_results.csv` | completed |
| Pseudo-Labeling | yes | yes | epochs 1–30; label fractions 1–40%; 3 seeds; matched labeled-only baseline | 3 | Apple M3 / PyTorch MPS | `results/raw/pseudolabel_baseline_sweep.csv`, `results/raw/pseudolabel_test_results.csv` | completed |
| Mean Teacher | yes | yes | epochs 1–30; labeled fractions 1,5,10,20,40%; 3 seeds | 3 | Apple M3 / PyTorch MPS | `results/raw/ssl_algorithms.csv`, `results/raw/ssl_algorithms_test.csv` | completed |
| FixMatch | yes | yes | epochs 1–30; labeled fractions 1,5,10,20,40%; 3 seeds | 3 | Apple M3 / PyTorch MPS | `results/raw/ssl_algorithms.csv`, `results/raw/ssl_algorithms_test.csv` | completed |
| MixMatch | yes | yes | epochs 1–30; labeled fractions 1,5,10,20,40%; 3 seeds | 3 | Apple M3 / PyTorch MPS | `results/raw/ssl_algorithms.csv`, `results/raw/ssl_algorithms_test.csv` | completed |
| Label Propagation | yes | yes | max iterations 1–2,000; tol 1e-3; 3 seeds; convergence observed | 3 | CPU sklearn; actual LabelPropagation | `results/raw/label_propagation_scaling.csv` | completed |
| DQN | yes | yes | 5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds) | 3 | CPU | `results/raw/rl_scaling_results.csv` | completed |
| Double DQN | yes | yes | 5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds) | 5 | CPU | `results/raw/rl_scaling_results.csv` | completed |
| A2C | yes | yes | 5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds) | 3 | CPU | `results/raw/rl_scaling_results.csv` | completed |
| PPO | yes | yes | 5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds) | 3 | CPU | `results/raw/rl_scaling_results.csv` | completed |
| SAC | yes | yes | 5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds) | 5 | CPU | `results/raw/rl_scaling_results.csv` | completed |

### Additional recorded extensions (outside the 50-objective denominator)

| Model | Environment | Budget | Seeds | Result location | Status |
|---|---|---|---:|---|---|
| Compact transformer proxy | Fashion-MNIST | 10K split, epochs 1–30 | 3 | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | Supplementary 3-layer, width-96 proxy; separate from the ViT-Tiny-scope implementation |
| PPO-Pendulum | Pendulum-v1 | 5K–250K requested environment steps | 3 | `results/raw/rl_scaling_results.csv` | Recorded RL extension; not an original sixth method |

## LLM configurations (15)

| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |
|---|---|---|---|---:|---|---|---|
| Pythia-70M | checkpoint/config available | yes, local inference | 8 released checkpoints (0–143K); full WikiText-2 test; full HellaSwag/PIQA/ARC-Easy zero-shot | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | partial evaluation |
| Pythia-160M | checkpoint/config available | yes, local inference | 8 released checkpoints (0–143K); full WikiText-2 test | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | partial evaluation |
| Pythia-410M | checkpoint/config available | yes, local inference | step 0, 10K, 143K prefix checkpoints; full WikiText-2 test | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | partial evaluation |
| Pythia-1B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource-limited / not evaluated |
| Pythia-1.4B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource-limited / not evaluated |
| Pythia-2.8B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource-limited / not evaluated |
| Pythia-6.9B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource-limited / not evaluated |
| Pythia-12B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource-limited / not evaluated |
| GPT-2 Small | checkpoint/config available | yes, local inference | main; full WikiText-2 test + 100/task harness smoke subset | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | partial evaluation |
| GPT-2 Medium | checkpoint/config available | failed local checkpoint attempt | weight retrieval/evaluation incomplete | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | failed |
| GPT-2 Large | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource-limited / not evaluated |
| GPT-2 XL | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource-limited / not evaluated |
| OPT-125M | checkpoint/config available | yes, local inference | main; full WikiText-2 test + 100/task harness smoke subset | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | partial evaluation |
| OPT-350M | checkpoint/config available | yes, local inference | main; full WikiText-2 test | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | partial evaluation |
| OPT-1.3B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | resource-limited / not evaluated |

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
