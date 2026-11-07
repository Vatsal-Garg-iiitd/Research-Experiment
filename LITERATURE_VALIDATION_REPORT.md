# Literature evidence validation report

## Scope and evidence rules

The literature ledger covers 15 published configurations across GPT-3, Gopher, Chinchilla, PaLM, OPT, LLaMA, and BLOOM. It contains 300 model-field rows (20 fields per configuration) in `results/literature/literature_field_evidence.csv`. Evidence classes distinguish directly reported values (`reported`), calculations from reported inputs (`derived`), secondary estimates (`estimated`), and unresolved or insufficiently verified fields (`uncertain`). An uncertain field is not interpreted as zero or as proof that a paper omitted it. Local experiments and published evidence remain separate.

| Evidence class | Rows | Interpretation |
|---|---:|---|
| Reported | 172 | Directly located in a cited primary paper, official technical report, or model documentation. |
| Derived | 8 | Reproducible calculation from reported inputs; source and method are in the field note. |
| Estimated | 16 | Explicit estimates, generally from secondary sources; not presented as paper-reported. |
| Uncertain | 104 | Not resolved from saved evidence; may be unavailable, unextracted, or unclear. |

Every ledger row includes model, field, value, evidence class, source URL, locator, and notes. Source names and link mapping are in `results/literature/literature_sources.md`. The automated integrity check confirms 300 unique model/field keys and non-empty URL/locator fields. This structural check does not substitute for a fresh line-by-line review of every cited paper.

## Primary-source spot verification

The core claims central to the comparison are consistent with the cited primary sources:

- GPT-3 is reported as a 175B-parameter model; the paper’s Table 2.1 states that the model series, including GPT-3, was trained for 300B tokens. [Brown et al. (2020), arXiv:2005.14165](https://arxiv.org/abs/2005.14165), Table 2.1.
- Gopher is the 280B model in the paper’s scale range. [Rae et al. (2021), arXiv:2112.11446](https://arxiv.org/abs/2112.11446).
- Chinchilla is reported as 70B parameters and is described as using four times more data than Gopher at the same compute budget. [Hoffmann et al. (2022), arXiv:2203.15556](https://arxiv.org/abs/2203.15556).
- PaLM is reported as a 540B dense Transformer; the paper also analyzes smaller configurations. [Chowdhery et al. (2022), arXiv:2204.02311](https://arxiv.org/abs/2204.02311).
- OPT is a model suite from 125M through 175B parameters. [Zhang et al. (2022), arXiv:2205.01068](https://arxiv.org/abs/2205.01068).
- LLaMA reports model sizes from 7B through 65B. [Touvron et al. (2023), arXiv:2302.13971](https://arxiv.org/abs/2302.13971).
- BLOOM is reported as a 176B-parameter open-access multilingual model. Its paper and technical reporting describe a 384-GPU A100 training system; duration and total energy/carbon values should be taken from the specific technical or carbon-footprint source named in the ledger, not inferred from parameter count. [Scao et al. (2023), JMLR](https://www.jmlr.org/papers/v25/23-0581.html); [BLOOM carbon-footprint analysis](https://jmlr.org/papers/volume24/23-0069/23-0069.pdf).

These spot checks verify stated headline identities, not every benchmark cell or training-runtime estimate. Benchmark values are reported only with the source attribution and protocol note available in the field ledger. If a later paper reproduces an earlier paper’s score, the ledger notes that attribution rather than presenting the score as a new measurement.

## Comparability and unresolved fields

Published parameter counts and token totals are often directly documented, but training duration, optimizer details, hardware utilization, energy, FLOPs, evaluation prompts, and comparable held-out benchmark metrics have uneven coverage. FLOPs, accelerator-hours, wall-clock duration, and carbon emissions are different quantities. No conversion is implied without explicit assumptions, and estimated fields are not promoted to reported values.

Cross-paper benchmark outcomes are not a controlled scaling curve: models use different training corpora, tokenization, context lengths, architectures, compute budgets, and evaluation protocols. PaLM 62B checkpoints trained for different token counts remain separate configurations. The ledger is an evidence audit, not a standardized re-evaluation or causal compute analysis.

## Conclusion

The literature records support selected factual comparisons about published model scale and documented training setup. They do not supply a complete, uniform compute-performance dataset for the project’s research question. The 104 uncertain fields remain explicit limitations; no missing hardware, training compute, duration, or score has been fabricated.
