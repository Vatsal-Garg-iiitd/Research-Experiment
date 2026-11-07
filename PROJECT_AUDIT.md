# Project audit

Audit updated 2026-11-07 from the working tree, raw data, generated outputs, configuration records, checkpoint inventory, and execution log. The current project source `research-experiment` remote matched the starting HEAD `afc488a46f318c7b264eb44d8b7d4eb78536a4de`. The workspace contained pre-existing modifications before this audit; they were preserved. A compressed workspace snapshot excluding `.git` and virtual environments is stored at `../Tcom_project-audit-backup-20261107T140412Z.tar.gz`.

## Research objective

The proposal asks how performance changes with added computational effort and whether marginal gains diminish. The project also contains conventional ML, RL, semi-supervised, representation-learning, language-model, hardware, and literature evidence. Budgets and metrics remain native to each method. Published checkpoint inference is not original pretraining.

## Evidence inventory

- Raw results are under `results/raw/`; per-run configuration JSON under `results/configurations/`; checkpoints under `results/checkpoints/`; original and repaired snapshots under `results/snapshots/`.
- `EXPERIMENT_RUN_REGISTRY.csv` indexes 3,720 result/evidence rows across raw experiment CSVs, LLM CSVs, and the literature ledger. `results/processed/canonical_results.csv` is the common local-result index with raw record JSON retained.
- The regenerated strengthened analysis contains 3,614 validation summaries, 2,407 within-seed paired marginal increments, and 169 held-out endpoint groupings separated by final budget/checkpoint. These counts describe included measurements, not independent configurations.
- Six public model checkpoints have completed full WikiText-2 raw-v1 test-text evaluation on common context/stride. Pythia-70M has an unbounded zero-shot validation-split harness run. Two earlier 100-example/task harness runs remain smoke tests.
- The literature ledger contains 15 published configurations and 300 field-level evidence rows. 104 fields remain uncertain; sources and locators are stored per field. These are not local reproductions.

## Integrity and interpretation

Raw experiment files and pre-repair snapshots were retained. Repairs include schema union for SSL CSV rows, actual-example counts for supervised runs, unioned token-byte accounting for full WikiText-2, deduplication of successful identical Pythia checkpoint evaluations after a snapshot, and exact separation of smoke versus full harness scopes. `scripts/validate_project_outputs.py` checks CSV parseability, numeric finiteness, required deliverables, key analysis counts, literature field uniqueness/classes, full WikiText-2 protocol consistency, harness task/limit settings, and registry source paths.

Endpoint evaluation does not mean every endpoint was chosen by consistent validation-based selection. Training timing across different devices is not directly comparable. Some records include selected fixed-budget endpoints, shared fixed splits, rollout overshoot, and one non-exact legacy Double DQN continuation; those limitations are documented in run-level details and `MACOS_EXECUTION_LOG.md`.

**Overall status: incomplete exploratory evidence.** The strict objective matrix is still 0/50. Available results do not support calling the proposed full study complete, causal cross-paper comparisons, cross-paradigm efficiency claims, or universal diminishing returns. See `FINAL_INDEPENDENT_AUDIT.md` for final post-generation checks.
