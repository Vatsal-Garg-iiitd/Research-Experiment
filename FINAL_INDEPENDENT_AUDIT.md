# Final independent audit

Audit date: 2026-11-07. This is a consistency review of repository artifacts against saved CSV/JSON/JSONL records, source scripts, evaluator outputs, and historical commits. It is not an independent reproduction of training runs or a complete line-by-line review of every literature field.

## Verified status

| Check | Final state |
|---|---|
| Strict completion | 0/50, unchanged registered denominator and criteria |
| Current canonical local index | 2,612 unique canonical IDs; heterogeneous rows, not independent runs |
| Source composition | 2,548 selected conventional raw records plus 64 selected LLM records |
| Statistical summary groups | 3,614 |
| Paired within-seed transitions | 2,417 |
| Held-out endpoint metric groups | 169 |
| Unsupervised analytical curves | 15 K-Means rows + 21 GMM rows after excluding 15 later duplicate K-Means checkpoint-reuse rows |
| SSL | 1,350 validation rows and 90 test rows for Mean Teacher/FixMatch/MixMatch |
| RL | 132 rows: five original algorithms plus 18 PPO-Pendulum extension rows |
| Full WikiText-2 evaluations | 6 completed full-test checkpoint evaluations with shared test text hash and matched context/stride/precision/device protocol |
| Full zero-shot harness | 3 unrestricted 0-shot Pythia-70M validation tasks; counts checked against aggregate JSON and sample logs |
| ViT-Tiny-scope evidence | 90 completed 30-epoch validation rows, 3 epoch-30 test endpoints, genuine 12-block/192-width dimensions adapted to grayscale Fashion-MNIST |
| Literature evidence | 300 unique model-field rows, 15 configurations, 172 reported / 8 derived / 16 estimated / 104 uncertain |
| Figures | 25 catalogued PNG figures and 25 vector SVG figures |

The older 430, 619, 458, 279, and 84 values refer to earlier analysis/export generations, not current totals. The 619-row index is recoverable from commit `afc488a`; the 430-row milestone has no matching saved export snapshot. See `PROJECT_AUDIT.md` for definitions and provenance.

## Corrections made in derived analysis

- Excluded the 15 repeated K-Means records from budget curves without editing raw files.
- Restricted data-size scaling to epoch-5 endpoints, preventing epoch-30 10K tests from entering the matched 5/30/58K comparison.
- Realigned the 30 Double DQN rows that were one cell short; preserved raw records, restored `completed` status and `local_measurement` source type in derived outputs; left success_rate unavailable.
- Normalized legacy Double DQN `source_type` path values in the canonical index while retaining the raw rows in embedded JSON.
- Preserved the 1,122.1625-second ResNet interval in raw elapsed time; adjusted efficiency analysis removes it after epoch 2 and states why.
- Kept mirror CSV exports, LLM checkpoint aliases, smoke tests, full-corpus evaluation, checkpoint-prefix evaluations, and literature observations as separate evidence classes.

## Validation

The current validator is expected to check raw CSV parsing and finite values, deliverable presence, canonical ID uniqueness, LLM protocol/hash consistency, harness task counts, literature ledger structure, run-registry references, five-epoch-only data-size endpoints, figure catalogue paths, report sections, PDF signature/size, and output row counts. Run `.venv-mac/bin/python scripts/validate_project_outputs.py` after rebuilding derived data. This validation is not a model test suite and invokes no training or inference.

## Limits

Raw execution cannot be reconstructed for every historical row. The report does not claim a universal scaling law, causal model-family efficiency, full local pretraining, complete literature coverage, or uniform endpoint-selection protocol. 104 literature fields remain uncertain. Some device, energy, compute, and process-level timing details are unavailable.
