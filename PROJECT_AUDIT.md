# Final project audit

Audit date: 8 November 2026. The local `main` branch was compared with fetched `research-experiment/main`; both pointed to `d775ad18611b95c516800d646a2870b367796456` before this finalization pass. The working tree contained a documented semi-supervised continuation and analysis additions. Raw records, historical evidence, configurations, and checkpoints were preserved. The pseudo-labeling and matched labeled-only studies resumed from saved epoch-5 checkpoints and completed the already registered 30-epoch schedule with the same five label fractions and three seeds.

## Verified result counts

| Count | Meaning |
|---:|---|
| 3,362 | Canonical heterogeneous index rows with source path and row locators; not a count of independent experiments. |
| 4,864 | Grouped metric summaries by model, setting, metric, and budget. |
| 3,167 | Paired adjacent-budget marginal transitions within matched seed/configuration series. |
| 219 | Held-out metric endpoint groups. |
| 5,250 | Evidence registry rows across 29 source paths; a source catalog, not unique runs. |
| 20/50 | Strict registered objectives complete: all 20 conventional schedules verified; 15 local LLM configurations remain partial/not evaluated; 15 literature entries remain literature-only. |

Earlier counts (including 430/619 canonical rows, 458/279/84 analysis rows, 2,612/3,614/2,417/169, and 3,720 registry rows) describe older source snapshots or processing generations. They are not additive and are superseded by the reproducible current outputs. The original 50-objective denominator and criteria remain unchanged.

## Data integrity and analytical treatment

- The conventional coverage audit checks source rows, seeds, budgets, configuration references, and required endpoints. Its output is `results/processed/conventional_completion_audit.csv`; all 20 registered conventional objectives pass.
- The pseudo-label continuation adds 750 epoch rows per exported curve table and 30 epoch-30 held-out evaluations. Existing epoch-5 evaluations and checkpoints remain available. Seed-matched endpoint comparisons are mixed and do not show a consistent pseudo-labeling advantage.
- Fifteen duplicate K-Means rows repeat IDs and reuse a final model state at earlier budgets. They remain in the raw file but are excluded from the K-Means analytical curve.
- `replicated_learning_curves.csv` mirrors the richer supervised result table; `pseudolabel_learning_curves.csv` mirrors the baseline sweep. Mirrors are not independent runs.
- The Double DQN raw CSV has one missing field per row, shifting trailing values. Derived builders realign the fields by types and paths, retain the raw source JSON, and leave unavailable `success_rate` blank.
- ResNet-18 seed 11 has a verified wall-clock interval of 1,122.1625285 seconds between epochs 2 and 3. Its cause is unresolved. The raw measurement is preserved; only an adjusted efficiency diagnostic subtracts it after epoch 2.
- Historical Windows/DirectML and later Apple MPS timings remain separate. No cross-platform speed ranking is asserted.

## Scope and completion

The 20 conventional objectives cover supervised learning, unsupervised learning, semi-supervised learning, and RL using their method-specific budgets. The matrix distinguishes the ViT-Tiny-dimension VisionTransformer from the separate three-layer width-96 transformer proxy. Current Label Propagation rows use the actual LabelPropagation estimator; superseded LabelSpreading records remain distinct. Five original RL methods are present; PPO-Pendulum is a separately identified extension, not a sixth required algorithm.

The 15 local LLM configurations are not all evaluated. Six public checkpoints have full WikiText-2 test-set evaluations under a shared local protocol; checkpoint-prefix trajectories, selected endpoints, full zero-shot Pythia-70M tasks, smoke subsets, failed attempts, and unevaluated models remain separately identified. Public checkpoint inference is not local pretraining.

The literature ledger has 15 model configurations × 20 fields. Evidence classes are `reported`, `derived`, `estimated`, `uncertain`, and `unavailable`. PaLM compute conversions and 6-parameter-times-token estimates are labeled explicitly; unreported values are never treated as zero. Primary source locators are recorded in the field ledger.

## Reproducible outputs

The analysis pipeline generated 25 PNG/SVG figure pairs and regenerated the statistical tables, completion audit, canonical indexes, run registry, literature evidence, and report source. See `DATA_INVENTORY.md`, `DATA_QUALITY_REGISTER.md`, `IDENTIFIER_AUDIT.md`, `FIGURE_CATALOGUE.md`, and `REPRODUCIBILITY.md`. Validation checks are recorded in the final audit summary and can be rerun with `scripts/validate_project_outputs.py`.

The conclusions remain descriptive and task-specific. The evidence does not establish a universal diminishing-returns law, a causal ranking of compute efficiency across paradigms, or complete local evaluation of the LLM scope.
