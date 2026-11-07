# Literature review and extraction protocol

The proposal identifies a gap between reporting final accuracy and quantifying the marginal gain attributable to additional compute. Prior work gives important but different forms of evidence:

* Kaplan et al. fit empirical power-law relations between language-model cross-entropy, parameters, data, and training compute, and study compute allocation. This motivates scaling fits but does not make cross-paper benchmark scores interchangeable.
* Hoffmann et al. trained more than 400 models and argued that model and token counts should scale together under a compute budget. Their Chinchilla result demonstrates that a smaller, more thoroughly trained model can outperform larger undertrained models at a similar training-compute allocation.
* Biderman et al.'s Pythia suite controls data order across 16 model sizes and publishes 154 checkpoints each, making it especially useful for studying training progression rather than only endpoint scaling.
* Brown et al. document GPT-3's 300B-token training mixture and report some component datasets repeated while others were not exhausted. Thus “epochs” are not a single well-defined corpus pass.
* OPT and BLOOM publish open training/accounting details; these support selected resource comparisons, with hardware and cost definitions retained verbatim.
* LLaMA reports several sizes trained on trillions of tokens and emphasizes the usefulness of smaller models trained on more data. Its evaluations remain protocol-specific.

The CSV is an evidence ledger, not a harmonized leaderboard. Parameter and token counts are retained in original units. Compute estimates are excluded unless a paper reports them or the derived field states its formula and assumptions. Page/table/section locators and direct source URLs are included per row. A missing cell means unavailable/not extracted, never zero.

## Included sources

See `results/literature/literature_sources.md` for exact source URLs and extraction notes. The source set includes Kaplan et al. (2020), Brown et al. (2020), Rae et al. (2021), Hoffmann et al. (2022), Zhang et al. (2022), Chowdhery et al. (2022), BigScience Workshop (2022), Touvron et al. (2023), and Biderman et al. (2023). Numeric claims in the CSV include a locator and should be checked against the cited paper before publication.

## Comparability limits

Model families differ in corpus composition, tokenization, architecture, filtering, optimization, hardware and evaluation setup. The CSV supports descriptive questions such as whether parameter and token allocations differ; it cannot establish causal performance-per-FLOP rankings across papers. Benchmark results are intentionally separated by metric and protocol. Literature comparisons are descriptive, with no claims of statistical significance.

## Completion status

The ledger currently has 15 model-configuration rows, but it is a selective extraction rather than the fully verified literature matrix requested for the final study. Some rows report parameters/tokens and selected benchmark data while compute, hardware, duration, epoch count, and train/validation loss remain blank or need closer page/table validation. Those blanks are intentional; the literature work remains incomplete. See `REMAINING_WORK.md` and `results/literature/literature_sources.md`.
