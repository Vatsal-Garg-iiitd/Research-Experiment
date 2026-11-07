# Final project audit

Audit date: 2026-11-07. The local `main` checkout and fetched GitHub `main` were both at `d5fdc732f9527e20ee687a27daded3145670741e` at audit start. The working tree was clean. A fresh pre-edit backup is stored at `../Tcom_project-pre-final-audit-20261107.tar.gz`. Raw records, configurations, checkpoints, and existing research artifacts were preserved. Current analysis uses saved evidence only.

## Count reconciliation

| Reported count | Meaning and status |
|---|---|
| 430 canonical rows | Historical execution-log milestone. No matching persisted snapshot was found, so exact row membership is unavailable. It is not the current count. |
| 619 canonical rows | Recoverable historical canonical export at commit `afc488a`; it indexed an earlier, smaller source set. |
| 2,612 canonical rows | Current lossless heterogeneous index: 2,548 selected raw conventional rows plus 64 selected LLM rows. Repeated budgets, statuses, and partials mean it is not a unique-run count. |
| 458 summaries / 279 increments / 84 endpoints | Earlier statistical-analysis generation retained in historical documentation. It predates later source expansion and is not the current pipeline output. Exact old grouping membership is unavailable in the current workspace. |
| 3,614 summaries / 2,417 increments / 169 endpoint groups | Current regenerated statistical output. Summary rows are metric × model/setting/budget groups; increments pair adjacent budgets within a seed/setting; endpoint rows are metric groups at selected held-out budgets. These are not counts of raw trials or distinct experiments. |
| 3,720 registry rows | Current run/evidence registry; row-level index across source paths, not the same population as the canonical local-result index. |

The current raw sources have repeated exports and statuses. `replicated_learning_curves.csv` mirrors all 420 experiment IDs in `replicated_scaling_results.csv`; `pseudolabel_learning_curves.csv` mirrors all 150 IDs in `pseudolabel_baseline_sweep.csv`. Each pair represents alternate schemas for the same runs. The canonical result builder uses the richer result tables only.

## Data integrity findings

- `unsupervised_scaling.csv` has 51 rows: 30 K-Means and 21 GMM. Fifteen later K-Means rows duplicate earlier experiment IDs and carry the final 50-update score at budget 1/5/10/20/50. The later rows remain in raw data and are excluded from curves. The analytical file contains 15 K-Means observations (three seeds × five budgets) and 21 GMM observations.
- Dataset-size summaries now use only five-epoch endpoints. The earlier generated table accidentally mixed epoch-5 and epoch-30 test endpoints at 10K, inflating that group to six rows; this has been corrected. The resulting matched comparison has three seeds per model/size condition.
- `strengthened_statistical_summaries.csv` reports grouped means, medians, standard deviations, minima, maxima, valid seed counts, and 95% t intervals where justified. `strengthened_marginal_gains.csv` preserves paired seed transitions and positive-cost ratios. Detailed exclusions and anomalies are in `DATA_QUALITY_REGISTER.md` and `results/processed/data_quality_register.csv`.
- The raw Double DQN CSV has 30 rows with 28 fields for 29 headers because success_rate was omitted. Builders realign later values by their type and path signatures; the raw records remain unchanged and success_rate remains unavailable. The canonical index restores `completed` status, `local_measurement` source_type, and configuration/checkpoint references while retaining the malformed source row in raw_record_json.

## Scope reconciliation

### Supervised transformer identity

The current 20-model registry’s ViT-Tiny entry is backed by a separate `results/raw/vit_tiny_scaling.csv` file with 90 completed validation rows (30 epochs × three seeds) and three completed epoch-30 test rows. The implementation uses torchvision `VisionTransformer` dimensions 12 blocks × 192 width × 3 heads × 768 MLP and 5,353,738 parameters, adapted to 28×28 grayscale inputs and 4×4 patches. `replicated_scaling_results.csv` contains a separate three-layer, width-96 transformer proxy. The matrix now assigns the ViT-Tiny objective to the 12-block implementation and lists the proxy as an additional record outside the 50-objective denominator.

### Semi-supervised learning

The current SSL table contains 1,350 completed validation records: three algorithms × five label fractions × 30 epochs × three seeds. There are 90 terminal test rows. The five-fraction, three-seed structure is confirmed from the raw files. Label Propagation is the actual corrected sklearn estimator; superseded LabelSpreading rows are kept separate.

### Reinforcement learning

The raw RL file has 132 rows: DQN/A2C/PPO each have 18 rows (three seeds × six requested checkpoints); Double DQN and SAC each have 30 (five seeds × six checkpoints). All five original algorithms are present. PPO-Pendulum contributes 18 additional rows and is an extension, not a sixth original-scope method. Double DQN’s 30 rows have `completed` experiment status and `local_measurement` source type after schema alignment; the omitted success_rate remains unavailable. Records include environment IDs, requested/actual steps, evaluation episode returns, elapsed times, and seed/split metadata. See run JSON for algorithm configuration and `MACOS_EXECUTION_LOG.md` for overshoot and resume qualifications.

### Language models

Six full WikiText-2 evaluations are distinct from 50K-token historical prefix evaluations, selected Pythia checkpoint trajectories, full validation-split zero-shot harness evaluations, 100-example smoke checks, and failed/resource-limited records. The six full evaluations share the raw-text hash, FP32, MPS, context 1,024, stride 512, and batch one. Pythia models score 285,830 tokens; GPT-2/OPT score 283,286 with their own tokenizers. Local evaluation time is not original pretraining compute. Pythia-70M, 160M, and 410M are evaluated on a prefix trajectory at different checkpoint coverage; `main` aliases for 160M/410M are not new checkpoints.

### Timing

ResNet-18 seed 11 records a 1,122.1625285-second epoch-2-to-3 wall-clock interval. Timestamp spacing supports that elapsed measurement, but does not prove compute duration or identify a pause/interruption cause. The raw value is retained and visible; only the adjusted efficiency diagnostic subtracts it after epoch 2. Historical Windows/DirectML records and newer Mac/MPS records are not pooled as a hardware comparison.

## Literature ledger

The ledger has 15 model configurations × 20 fields = 300 rows, unique by model/field. Class counts are 172 reported, 8 derived, 16 estimated, and 104 uncertain. Primary sources are linked in `LITERATURE_VALIDATION_REPORT.md` and per-field locators in the ledger. Estimated values are not labeled as reported. Cross-paper compute units remain distinct; uncertainties and gaps are not imputed.

## Completion and validation

Both completion matrices retain 0/50 under the original criteria. They distinguish work performed from full objective completion. Figure outputs include 25 generated figures, each PNG and vector SVG, and a catalogue. `DATA_INVENTORY.md`, JSON inventories, the statistical tables, marginal-gain table, data-quality register, and final Markdown/PDF report provide reproducible audit artifacts. The final validator checks file presence, raw tabular integrity, row/identifier consistency, full LLM evaluation protocol, harness result counts, literature schema, run-registry references, strict completion count, and output row counts.

## Interpretation

The evidence is descriptive and method-specific. It does not establish universal diminishing returns, causal compute-efficiency differences between model families, or complete coverage of planned objectives. It supports transparent reporting of the finite observed trajectories and their limits.
