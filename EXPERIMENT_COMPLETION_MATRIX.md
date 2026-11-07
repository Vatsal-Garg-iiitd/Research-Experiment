# Final Completion Matrix

Updated 2026-11-07 from result files. `pilot` means actual runs exist but the requested full study for that entry is incomplete. Local checkpoint evaluation of an LLM does not mean local pretraining.

**Strict objective completion: 0/50 configurations (0%).** The denominator is 20 conventional models + 15 specified LLM configurations + 15 literature configurations. A configuration counts complete only when its requested study objective is met; pilot/smoke/subset evaluations and partially verified literature rows do not count. This does not mean no work was completed; it distinguishes partial evidence from full requested scope.

| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |
|---|---|---|---|---:|---|---|---|
| MLP | yes | yes | epochs 1–5 | 3 | AMD Radeon 860M / DirectML (with operator CPU fallback) | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | pilot |
| LeNet-style CNN | yes | yes | epochs 1–5 | 3 | AMD Radeon 860M / DirectML (with operator CPU fallback) | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | pilot |
| ResNet-18 | yes | yes | epochs 1–5 | 3 | AMD Radeon 860M / DirectML (with operator CPU fallback) | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | pilot |
| ViT-Tiny (custom compact 3-layer, 96-wide proxy) | yes | yes | epochs 1–5 | 3 | AMD Radeon 860M / DirectML (with operator CPU fallback) | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | pilot |
| Random Forest | yes | yes | 10,25,50,100,200,400 trees | 3 | CPU | `results/raw/random_forest_scaling.csv`, `results/raw/random_forest_test_results.csv` | pilot |
| K-Means | yes | yes | algorithm-specific iterative budgets | 3 | CPU | `results/raw/unsupervised_scaling.csv` | pilot |
| Gaussian Mixture Model | yes | yes | algorithm-specific iterative budgets | 3 | CPU | `results/raw/unsupervised_scaling.csv` | pilot |
| DBSCAN | yes | yes | 2.5K,5K,10K samples x 3 seeds (data-size cost curve); separate 8-setting parameter sensitivity | 3 | CPU | `results/raw/dbscan_parameter_sweep.csv`, `results/raw/dbscan_size_scaling.csv` | pilot |
| Autoencoder | yes | yes | epochs 1–5; 3 seeds | 3 | AMD Radeon 860M / DirectML | `results/raw/autoencoder_scaling.csv`, `results/raw/autoencoder_test_results.csv` | pilot |
| Variational Autoencoder | yes | yes | epochs 1–5; 3 seeds | 3 | AMD Radeon 860M / DirectML | `results/raw/autoencoder_scaling.csv`, `results/raw/autoencoder_test_results.csv` | pilot |
| Pseudo-Labeling | yes | yes | epochs 1–5; label fractions 1–40%; 3 seeds | 3 | AMD Radeon 860M / DirectML | `results/raw/pseudolabel_baseline_sweep.csv`, `results/raw/pseudolabel_test_results.csv` | pilot |
| Mean Teacher | yes | yes | epochs 1–3; labeled fractions 1,5,10,20,40%; 3 seeds | 3 | DirectML for MeanTeacher/FixMatch; CPU MixMatch | `results/raw/ssl_algorithms.csv`, `results/raw/ssl_algorithms_test.csv` | pilot |
| FixMatch | yes | yes | epochs 1–3; labeled fractions 1,5,10,20,40%; 3 seeds | 3 | DirectML for MeanTeacher/FixMatch; CPU MixMatch | `results/raw/ssl_algorithms.csv`, `results/raw/ssl_algorithms_test.csv` | pilot |
| MixMatch | yes | yes | epochs 1–3; labeled fractions 1,5,10,20,40%; 3 seeds | 3 | DirectML for MeanTeacher/FixMatch; CPU MixMatch | `results/raw/ssl_algorithms.csv`, `results/raw/ssl_algorithms_test.csv` | pilot |
| Label Propagation | yes | yes | max iterations 1–2,000; tol 1e-3; 3 seeds; convergence observed | 3 | CPU sklearn; actual LabelPropagation | `results/raw/label_propagation_scaling.csv` | pilot |
| DQN | yes | yes | 5K,10K,25K environment steps; 3 seeds | 3 | CPU | `results/raw/rl_scaling_results.csv` | pilot |
| Double DQN | yes | yes | 5K,10K,25K environment steps; 3 seeds | 3 | CPU | `results/raw/rl_scaling_results.csv` | pilot |
| A2C | yes | yes | 5K,10K,25K environment steps; 3 seeds | 3 | CPU | `results/raw/rl_scaling_results.csv` | pilot |
| PPO | yes | yes | 5K,10K,25K environment steps; 3 seeds | 3 | CPU | `results/raw/rl_scaling_results.csv` | pilot |
| SAC | yes | yes | 5K,10K,25K environment steps; 3 seeds | 3 | CPU | `results/raw/rl_scaling_results.csv` | pilot |

## LLM configurations (15)

| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |
|---|---|---|---|---:|---|---|---|
| Pythia-70M | checkpoint/config available | yes, local inference | step143000; 50K WikiText-2 tokens + 100/task harness subset | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | pilot |
| Pythia-160M | checkpoint/config available | yes, local inference | step143000; 50K tokens | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | pilot |
| Pythia-410M | checkpoint/config available | yes, local inference | step143000; 50K tokens | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | pilot |
| Pythia-1B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | resource_limited |
| Pythia-1.4B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | resource_limited |
| Pythia-2.8B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | resource_limited |
| Pythia-6.9B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | resource_limited |
| Pythia-12B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | resource_limited |
| GPT-2 Small | checkpoint/config available | yes, local inference | main; 50K WikiText-2 tokens + 100/task harness subset | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | pilot |
| GPT-2 Medium | checkpoint/config available | attempt/status in prior ledger only | weight retrieval/evaluation incomplete | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | resource_limited |
| GPT-2 Large | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | resource_limited |
| GPT-2 XL | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | resource_limited |
| OPT-125M | checkpoint/config available | yes, local inference | main; 50K WikiText-2 tokens + 100/task harness subset | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | pilot |
| OPT-350M | checkpoint/config available | yes, local inference | main; 50K WikiText-2 tokens | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | pilot |
| OPT-1.3B | checkpoint/config available | attempt/status in prior ledger only | no long local evaluation | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | resource_limited |

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
