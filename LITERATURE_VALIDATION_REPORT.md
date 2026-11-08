# Literature evidence validation report

## Scope and evidence rules

The literature ledger covers 15 published configurations across GPT-3, Gopher, Chinchilla, PaLM, OPT, LLaMA, and BLOOM. It contains 300 model-field rows (20 fields per configuration) in `results/literature/literature_field_evidence.csv`. Evidence classes distinguish directly reported values (`reported`), calculations from reported inputs (`derived`), secondary estimates (`estimated`), unresolved fields (`uncertain`), and fields with no value in a reviewed source location (`unavailable`). Blank or unavailable values are not interpreted as zero. Local experiments and published evidence remain separate.

| Evidence class | Rows | Interpretation |
|---|---:|---|
| Reported | 174 | Directly located in a cited primary paper, official technical report, or model documentation. |
| Derived | 6 | Reproducible calculation from reported inputs; source and method are in the field note. |
| Estimated | 16 | Explicit estimates, generally from secondary sources; not presented as paper-reported. |
| Uncertain | 88 | Not resolved from available evidence; not treated as source-confirmed absence. |
| Unavailable | 16 | No verified value appears in the cited source location reviewed for that field. |

Every ledger row includes model, field, value, evidence class, source URL, locator, and notes. Source names and link mapping are in `results/literature/literature_sources.md`. The automated integrity check confirms 300 unique model/field keys, non-empty URL/locator fields, and consistency between each field row's source ID and its configuration-level source record. Primary-source spot checks below cover headline facts; the per-field verification notes retain the specific locator and the unresolved items. This audit does not claim that every benchmark protocol is harmonized or that every uncertain field has a known value.

## Configuration-level completion audit

The following table is generated from `results/processed/literature_completion_audit.csv`. For each entry, the completion rule checks a unique 20-field schema, a source identity, source URL and locator on every field, reported parameter count, reported or transparently derived token count, and explicit classification of all unresolved blanks. Benchmark evidence is separately labeled as a reported compatible score or an explicit score gap. Counts describe evidence availability; uncertain and unavailable fields remain genuine limitations. All 15 entries pass this literature-review criterion. The field-level values, locators, and notes remain the detailed evidence record.

| Configuration | Source ID | Reported | Derived | Estimated | Uncertain | Unavailable | Benchmark evidence | Status |
|---|---|---:|---:|---:|---:|---:|---|---|
| GPT-3 175B | `BROWN2020` | 14 | 0 | 1 | 5 | 0 | reported_score | completed_literature_only |
| Gopher 280B | `RAE2021` | 12 | 0 | 1 | 7 | 0 | reported_score | completed_literature_only |
| Chinchilla 70B | `HOFFMANN2022` | 13 | 0 | 1 | 5 | 1 | reported_score | completed_literature_only |
| PaLM 540B | `CHOWDHERY2022` | 15 | 0 | 1 | 4 | 0 | reported_score | completed_literature_only |
| OPT-13B | `ZHANG2022` | 8 | 1 | 1 | 7 | 3 | score_gap_explicitly_classified | completed_literature_only |
| OPT-30B | `ZHANG2022` | 8 | 1 | 1 | 7 | 3 | score_gap_explicitly_classified | completed_literature_only |
| OPT-66B | `ZHANG2022` | 8 | 1 | 1 | 7 | 3 | score_gap_explicitly_classified | completed_literature_only |
| OPT-175B | `ZHANG2022` | 10 | 1 | 2 | 7 | 0 | score_gap_explicitly_classified | completed_literature_only |
| LLaMA-7B | `TOUVRON2023` | 12 | 0 | 1 | 6 | 1 | reported_score | completed_literature_only |
| LLaMA-13B | `TOUVRON2023` | 12 | 0 | 1 | 6 | 1 | reported_score | completed_literature_only |
| LLaMA-33B | `TOUVRON2023` | 12 | 0 | 1 | 6 | 1 | reported_score | completed_literature_only |
| LLaMA-65B | `TOUVRON2023` | 14 | 0 | 1 | 5 | 0 | reported_score | completed_literature_only |
| BLOOM-176B | `BIGSCIENCE2022` | 10 | 2 | 1 | 6 | 1 | score_gap_explicitly_classified | completed_literature_only |
| PaLM-62B (795B tokens) | `CHOWDHERY2022` | 13 | 0 | 1 | 5 | 1 | reported_score | completed_literature_only |
| PaLM-62B continued (1.325T tokens) | `CHOWDHERY2022` | 13 | 0 | 1 | 5 | 1 | reported_score | completed_literature_only |

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

PaLM compute fields were reconciled to their source units: PaLM-540B is reported at 2.56×10²⁴ FLOPs in Appendix Table 22; PaLM-62B at 795B tokens is reported at 3.08×10²³ FLOPs in the same table; and the continued 62B run's 492.9 zettaFLOPs in Appendix Table 31 is converted to 4.929×10²³ FLOPs. The last value is a unit conversion of source-reported compute, not a 6PT estimate. Project-derived 6PT estimates remain separately labeled `estimated`.

## Comparability and unresolved fields

Published parameter counts and token totals are often directly documented, but training duration, optimizer details, hardware utilization, energy, FLOPs, evaluation prompts, and comparable held-out benchmark metrics have uneven coverage. FLOPs, accelerator-hours, wall-clock duration, and carbon emissions are different quantities. No conversion is implied without explicit assumptions, and estimated fields are not promoted to reported values.

Cross-paper benchmark outcomes are not a controlled scaling curve: models use different training corpora, tokenization, context lengths, architectures, compute budgets, and evaluation protocols. PaLM 62B checkpoints trained for different token counts remain separate configurations. The ledger is an evidence audit, not a standardized re-evaluation or causal compute analysis.

## Conclusion

The literature records support selected factual comparisons about published model scale and documented training setup. They do not supply a complete, uniform compute-performance dataset for the project’s research question. Uncertain and unavailable fields remain explicit limitations; no missing hardware, training compute, duration, or score has been fabricated.
