# Identifier audit

Identifier scopes are source file plus identifier type unless a cross-file key is explicitly normalized. A run_id intentionally repeats over budget observations; it must map to one model, seed, and configuration within a source. `evaluation_id` and `canonical_id` identify individual analytical records. Seed values repeat across runs by design and are not globally unique.

- Tables scanned: 51.
- Identifier-field summaries: 99.
- Reconciliation entries: 95.
- Identifier/model/run integrity errors: 0.
- Unresolved foreign-key paths: 0.

Known corrections, generated evaluation IDs, and affected rows are listed in `IDENTIFIER_RECONCILIATION.csv`. The full machine-readable identifier scan is `results/processed/identifier_audit.csv`; findings are `results/processed/identifier_findings.csv`; file-path references are in `results/processed/foreign_key_audit.csv`. Raw files are not edited.

## Important distinctions

- Repeated `run_id` values across epochs/budgets are legitimate when model, seed, and configuration remain consistent.
- The only duplicate `experiment_id` within a raw source is the later K-Means append; it is excluded from analysis and given a unique derived ID.
- LLM evaluation records without source experiment IDs receive stable row-scoped `evaluation_id` values; they are not fabricated as training experiments.
- Thirty Double DQN CSV rows are one cell short. Derived outputs realign the configuration/checkpoint/timestamp/details fields using their values, leaving the omitted success_rate unavailable. The original malformed rows are preserved.
