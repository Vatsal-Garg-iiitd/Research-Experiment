# Literature sources and extraction ledger

This file records primary sources and rules for the accompanying CSV. CSV blanks are classified as uncertain or unavailable in the field-evidence ledger, never interpreted as zero. Model parameter counts are source-reported; calculated or differently defined token totals are identified in row notes. Literature comparisons are descriptive only.

| ID | Source | DOI / arXiv | Relevant evidence and locator |
|---|---|---|---|
| KAPLAN2020 | Kaplan et al., *Scaling Laws for Neural Language Models* (2020) | [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) | Abstract and scaling analysis: cross-entropy loss follows empirical power laws in model size, data, and compute. |
| BROWN2020 | Brown et al., *Language Models are Few-Shot Learners* (2020) | [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) | Table 2.1 for GPT-3 sizes; Section 2.3 and Table 2.2 for 300B token training mixture and heterogeneous dataset passes. |
| RAE2021 | Rae et al., *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (2021) | [arXiv:2112.11446](https://arxiv.org/abs/2112.11446) | Gopher model configurations, training and 152-task evaluation; per-field locators still need final page/table verification. |
| HOFFMANN2022 | Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022) | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) | Abstract and Section 3: Chinchilla is 70B parameters, 1.4T tokens, with compute budget matched to Gopher; study analyzes >400 models. |
| ZHANG2022 | Zhang et al., *OPT: Open Pre-trained Transformer Language Models* (2022) | [arXiv:2205.01068](https://arxiv.org/abs/2205.01068) | Abstract: OPT suite 125M-175B; model-size and training information in Sections 2-3; infrastructure logbook is linked by authors. |
| CHOWDHERY2022 | Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* (2022) | [arXiv:2204.02311](https://arxiv.org/abs/2204.02311) | Abstract and Appendix Tables 22/31: PaLM-540B uses 780B tokens and 2.56e24 training FLOPs (with rematerialization); Table 31 contains a PaLM-62B continuation pair with tokens, compute, and paper-defined average-task scores. |
| BIGSCIENCE2022 | BigScience Workshop, *BLOOM: A 176B-Parameter Open-Access Multilingual Language Model* (2022) | [arXiv:2211.05100](https://arxiv.org/abs/2211.05100) | Sections 3.1 and 3.4: ROOTS corpus and training setup; paper identifies 176B parameters. |
| BIGSCIENCECARBON2022 | Luccioni et al., *Estimating the Carbon Footprint of BLOOM* (2022) | [arXiv:2211.02001](https://arxiv.org/abs/2211.02001) | Table 1 reports BLOOM training duration, accelerator-hours, energy, hardware, and grid carbon intensity. Secondary primary project accounting source. |
| TOUVRON2023 | Touvron et al., *LLaMA: Open and Efficient Foundation Language Models* (2023) | [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) | Section 2.1/Table 2 reports model parameters and tokens; Table 3 reports zero-shot common-sense benchmark scores. |
| BIDERMAN2023 | Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling* (2023) | [arXiv:2304.01373](https://arxiv.org/abs/2304.01373) | Abstract and checkpoint release: 16 models from 70M to 12B, same data order, 154 public checkpoints per model. |
| PYTHIA_MODEL_CARD | EleutherAI, *Pythia-70M model card* | [Hugging Face model card](https://huggingface.co/EleutherAI/pythia-70m) | Training procedure: 143,000 steps, 299,892,736,000 tokens; revision `step143000` is the main checkpoint; exact total parameter count is 70,426,624. |
| PYTHIA160_MODEL_CARD | EleutherAI, *Pythia-160M model card* | [Hugging Face model card](https://huggingface.co/EleutherAI/pythia-160m) | Model-card training procedure reports 299,892,736,000 tokens and checkpoints through `step143000`; used to annotate the locally evaluated `main` revision. |
| PYTHIA410_MODEL_CARD | EleutherAI, *Pythia-410M model card* | [Hugging Face model card](https://huggingface.co/EleutherAI/pythia-410m) | Model-card training procedure reports 299,892,736,000 tokens and checkpoints through `step143000`; used to annotate the locally evaluated `main` revision. |

## Quantitative status

The final source ledger contains 15 configurations from seven families, each with 20 field-level records. All 15 literature-only objectives pass `scripts/audit_literature_completion.py`: each entry has source identity, source-located core scale/token evidence, and an evidence class plus locator for every field. The corresponding audit is `results/processed/literature_completion_audit.csv`. Coverage includes 174 reported, 6 derived, 16 estimated, 88 uncertain, and 16 unavailable field records. Some configurations lack reported compute, duration, or compatible downstream scores; these gaps remain explicitly classified and are not filled with fabricated values. BLOOM's 366B processed tokens are labeled calculated as 341B ROOTS tokens plus 25B repeated tokens. Locators use paper sections, tables, or figures; where evidence is not resolved, the field is classified uncertain/unavailable. This is a completed source review, not a uniform quantitative synthesis or local reproduction.

## Interpretation cautions

1. Training FLOPs inferred via `6 * parameters * tokens` are estimates and are kept distinct from directly reported compute; estimate status, formula, and assumptions are recorded in the field ledger.
2. Hardware-days, GPU-hours, energy, and dollar estimates use different definitions and must remain separate.
3. Benchmark scores depend on prompt, shots, dataset version, tokenizer, and scoring method; only the same paper protocol is directly comparable.
4. Training mixture repeats make a single epoch count ill-defined for GPT-3; do not report an epoch count from its total tokens alone.
