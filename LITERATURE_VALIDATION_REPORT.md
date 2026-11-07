# Literature validation report

The project ledger contains 15 configurations across GPT-3, Gopher, Chinchilla, PaLM, OPT, LLaMA and BLOOM in `results/literature/large_model_literature.csv`; provenance notes are in `results/literature/literature_sources.md`. They are literature-only observations, not local reproductions.

The ledger records selected primary-paper or model-card values, with blanks for unavailable or unextracted fields. Several values have attribution notes distinguishing original-paper reports from scores cross-listed in a later comparison table. BLOOM token total is marked calculated; PaLM compute conversions are noted; no values are filled with zero. Existing field locators are mostly section/table references, not verified page-level locators for every configuration. The requested field-by-field verification of training hardware, duration, epochs, train/validation loss and directly comparable benchmark scores across all 15 rows remains incomplete. Therefore the ledger is not a complete quantitative evidence synthesis and must not be used for unqualified cross-paper causal comparisons.

The source ledger includes Kaplan scaling-law work and primary papers/model cards. A rigorous scaling analysis still needs standardized extracted fields and protocol-aware comparability rules. No local-reproduction claim is made.
