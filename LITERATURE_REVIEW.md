# Literature review: scaling, compute, and evaluation

## Scaling language models

Large language-model studies provide influential evidence that quality and model/data scale are related, but that relationship depends on the training recipe and evaluation. GPT-3 reports a 175B model and a family trained for 300B tokens. Gopher studies scales up to 280B. Chinchilla’s compute-optimal analysis reports a 70B model trained on substantially more data than Gopher under a comparable compute budget. These findings motivate measuring both model size and data/compute; parameter count alone is not a training-cost measurement.

PaLM reports a 540B dense Transformer and separately evaluates a 62B model family. OPT releases decoder-only models from 125M to 175B and documents system and carbon information. LLaMA reports models from 7B to 65B with different token budgets by size. BLOOM documents a 176B multilingual model and its distributed training environment. These papers use different corpora, tokenizer choices, training-token definitions, and task evaluations. Their results are valuable context but cannot be stacked into one controlled curve without harmonizing protocols.

## Local project relationship

This repository evaluates released Pythia, GPT-2, and OPT checkpoints; it does not reproduce original pretraining. The six full WikiText-2 evaluations share text, context, stride, precision, and local device protocol, but have family-specific tokenizers and training histories. Pythia checkpoint-prefix trajectories use a restricted text prefix and derive token counts from published step metadata. A full Pythia-70M validation-split zero-shot evaluation is a separate result class. None estimates original pretraining cost.

## Evidence quality

The field ledger includes 15 published configurations and 300 fields classified as 172 reported, 8 derived, 16 estimated, and 104 uncertain. It stores source links and source locators. Literature estimates are distinguished from source-reported quantities. Training FLOPs, accelerator-hours, wall time, data volume, and energy are not treated as equivalent. See `LITERATURE_VALIDATION_REPORT.md` for primary-source spot checks and limitations.

## Primary references

1. Brown et al. (2020), [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
2. Rae et al. (2021), [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446).
3. Hoffmann et al. (2022), [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
4. Chowdhery et al. (2022), [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311).
5. Zhang et al. (2022), [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068).
6. Touvron et al. (2023), [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971).
7. Scao et al. (2023), [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://www.jmlr.org/papers/v25/23-0581.html).

These are primary works for headline model facts. Exact field claims and table/section locations are recorded in `results/literature/literature_field_evidence.csv`; unresolved fields should remain marked uncertain.
