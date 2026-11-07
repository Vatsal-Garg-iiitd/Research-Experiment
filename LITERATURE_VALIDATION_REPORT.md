# Literature validation report

The literature ledger contains 15 published configurations across seven model families. They are published observations only and were not reproduced locally. `results/literature/literature_field_evidence.csv` contains 300 rows: 20 requested fields for each configuration. Each field is labeled `reported`, `derived`, `estimated`, or `uncertain`, and stores a source URL, source locator, and attribution note where available.

| Evidence class | Fields | Interpretation |
|---|---:|---|
| Reported | 172 | Direct value found in the cited primary paper or documented source |
| Derived | 8 | Reproducible calculation from cited reported values, such as a compute conversion |
| Estimated | 16 | Secondary estimate, including documented training-time estimates |
| Uncertain | 104 | Field unresolved, not extracted, or not demonstrated to be absent from the source |

The primary evidence sources include [GPT-3](https://arxiv.org/abs/2005.14165), [Gopher](https://arxiv.org/abs/2112.11446), [Chinchilla](https://arxiv.org/abs/2203.15556), [PaLM](https://arxiv.org/abs/2204.02311), [OPT](https://arxiv.org/abs/2205.01068), [LLaMA](https://arxiv.org/abs/2302.13971), and [BLOOM](https://www.jmlr.org/papers/v25/23-0581.html). BLOOM carbon and duration details additionally cite the [BLOOM carbon footprint analysis](https://arxiv.org/abs/2211.02001). Detailed source mapping and known caveats are in `results/literature/literature_sources.md` and per-field locators are in the CSV.

The strongest coverage is for published model size, training-token totals, batch, sequence length, optimizer, selected learning rates, accelerator system, and published benchmark observations. Coverage varies by model and field. Epoch counts, training/validation losses, per-configuration training duration, and evaluation protocols are often not consistently reported. Some benchmark values are cross-listed by a later paper and are attributed as such; PaLM 62B task-average results are kept distinct from individual benchmark metrics. Several runtime estimates come from later sources rather than the original study.

These sources use different datasets, tokenization, context lengths, training recipes, evaluation prompts, and metric definitions. Keep local experiments separate from published observations. Do not infer a causal training-compute effect from cross-paper comparisons, treat an uncertain field as zero, or describe the literature ledger as a complete standardized benchmark. Full page-level verification and an apples-to-apples benchmark synthesis remain open work.
