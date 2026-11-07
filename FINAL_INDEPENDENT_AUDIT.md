# Final independent audit

Audit date: 2026-11-07. This is a source-linked consistency audit of the project artifacts and generated records. It verifies stated counts and protocols against saved CSV/JSON sources; it is not an independent replication of experiments or a complete re-review of every cited paper.

## Cross-file checks

| Check | Result |
|---|---|
| Strict objective denominator | 50 entries; 0 classified fully complete, consistent across both generated matrices |
| Row-level run registry | 3,720 rows from 29 source paths in `EXPERIMENT_RUN_REGISTRY.csv` |
| Canonical local result index | 2,612 rows with unique canonical IDs; raw files retained as sources of record |
| Strengthened statistical summaries | 3,614 grouped validation summaries; 2,407 paired within-seed increments; 169 held-out endpoint summaries separated by final epoch/tree budget |
| Literature evidence ledger | 300 unique model/field rows across 15 configurations; reported 172, derived 8, estimated 16, uncertain 104 |
| Full WikiText-2 evaluations | Six completed records; common text hash; FP32, context 1024, stride 512 |
| Full zero-shot harness | Three validation tasks; no `--limit`; 0-shot; counts matched saved result JSON and sample logs |
| Python source compilation | `python -m compileall -q scripts` completed successfully in `.venv-mac` |

The Pythia-70M harness aggregate JSON records HellaSwag n=10,042, PIQA n=1,838, and ARC-Easy n=2,376. The full result CSV retains 285,830 scored tokens for Pythia models and 283,286 for GPT-2/OPT checkpoints, with a shared raw-text hash. Literature evidence classes and field uniqueness were checked by the output validator.

## Validation results

The final run of `.venv-mac/bin/python scripts/validate_project_outputs.py` completed with **0 CSV/integrity errors and 0 warnings**. It parses all non-snapshot result CSVs and checks numeric finiteness, required deliverables, canonical ID uniqueness, full-WikiText protocol/hash, harness task/few-shot/limit settings, literature evidence row counts and uniqueness, registry source paths, strict matrix status, and regenerated analysis counts. Test endpoint aggregation was corrected during this audit to keep epoch-5 and epoch-30 evaluations (and different Random Forest tree counts) from being counted as additional independent seeds.

No standalone test directory or pytest configuration is present. The project’s experiment scripts have been exercised through the recorded training, evaluation, and small platform smoke runs. Compilation and the output validator are not substitutes for a complete test suite.

## Remaining limits

The strict 0/50 count remains appropriate: multiple registered conventional methods, LLM configurations, and literature fields lack full scope. The full Pythia-410M checkpoint trajectory was not completed; step 0 was saved before the next remote revision stalled, while existing step 10K and 143K observations remain. Endpoint checkpoint selection is not uniform across all runs. The audit checks record consistency and source paths but does not reproduce every experiment, verify every remote checkpoint byte-for-byte, or resolve the 104 uncertain literature fields.
