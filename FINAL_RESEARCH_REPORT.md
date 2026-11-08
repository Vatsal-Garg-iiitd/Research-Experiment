# How Much Accuracy Is Worth the Additional Computing Cost?

## 1. Title page

**How Much Accuracy Is Worth the Additional Computing Cost? Investigating the Trade-off Between AI Performance and Computational Resources**

Final research audit and results report

Research project, Group 119, Section C

Evidence cut-off: 8 November 2026
Repository: `Vatsal-Garg-iiitd/Research-Experiment`

## 2. Abstract

This report audits supervised, unsupervised, semi-supervised, reinforcement-learning (RL), language-model, and literature evidence. The one unfinished registered semi-supervised schedule was continued from saved epoch-5 checkpoints to epoch 30 for the same configurations, seeds, and data splits; no additional methods, seeds, datasets, or evaluation protocols were introduced. All remaining statistics and figures are rebuilt from saved measurements. The strict project framework records **20 of 50 registered objectives complete**: the 20 conventional objectives meet their registered schedule, seed, and endpoint checks; 15 local LLM configurations remain partially evaluated; and 15 literature configurations are source-based rather than local experiments. The canonical index contains 3,362 heterogeneous source rows, not 3,362 independent experiments. Current analyses contain 4,864 grouped metric summaries, 3,167 within-seed marginal transitions, and 219 held-out metric groups. Fifteen duplicate K-Means records are excluded from its analytical trajectory because they reuse experiment IDs and copy a final checkpoint score into earlier budgets; raw values remain untouched. Results are task-specific, include reversals and plateaus, and do not establish a general diminishing-returns law or a cross-paradigm efficiency ranking. Whether additional budget is worthwhile depends on task, metric, model, evaluation protocol, and stopping point; evidence is insufficient for a universal threshold or causal claim.

## 3. Keywords

Computational cost; marginal performance; scaling; supervised learning; semi-supervised learning; reinforcement learning; language models; reproducibility; research audit.

## 4. Introduction

Model quality is commonly reported without a commensurate account of the resources used to obtain it. A useful comparison must specify the resource being spent, the outcome metric, the evaluation set, and the method by which additional budget changes the result. This project asks how performance changes as computational expenditure increases and whether marginal improvements diminish.

The repository is an exploratory multi-paradigm collection rather than one controlled experiment. Its evidence includes distinct tasks, budgets, devices, and metrics. We therefore analyze each family in its native units and treat the synthesis as a set of bounded case studies.

## 5. Research motivation

Training and evaluation can consume substantial compute, time, memory, and energy. Reporting marginal quality per additional resource can help decide when a run is useful to extend. However, wall time is hardware- and environment-dependent, parameter count is not training compute, environment steps are not optimizer updates, and benchmark accuracy is not interchangeable with reconstruction error or language-model perplexity. These distinctions motivate a protocol-specific audit rather than a single pooled score.

## 6. Research problem

The central problem is to determine whether observed gains justify the added measured budget. The evidence must support paired within-configuration comparisons and preserve negative gains, uncertainty, incomplete curves, and timing anomalies. The study cannot infer costs that were not recorded or treat public checkpoint evaluation as a measurement of original pretraining.

## 7. Research questions

1. How do task-specific outcomes change over the saved native budget curves?
2. What marginal improvement is observed between adjacent budgets within a seed and configuration?
3. Do these finite observations support diminishing returns or saturation?
4. What conclusions remain unresolved because runs are partial, protocols differ, or resource measurements are missing?

## 8. Research objectives

The original scope spans 20 conventional methods, 15 LLM configurations, and 15 literature configurations. The unchanged strict all-or-nothing matrix records 20/50 complete. The evidence audit verifies that all conventional schedules, seeds, and endpoints are present. The remaining 30 objectives are not complete: local LLM coverage is partial and literature records are not local replications.

## 9. Background and related work

Published scaling studies characterize relationships among model size, training tokens, loss, and downstream performance under particular data and optimization regimes. GPT-3 studied few-shot behavior across scales; Gopher reported a 280B model and broad task suite; Chinchilla argued that model size and training tokens should scale together under a compute budget; PaLM reported 540B and 62B configurations; OPT released models through 175B; LLaMA reported four sizes through 65B; and BLOOM documented a 176B multilingual model. These are literature observations, not local replications. They differ in corpus, tokenizer, architecture, training compute definition, benchmark protocols, and reporting coverage. See the evidence classifications in `results/literature/literature_field_evidence.csv` and the citations in Section 34.

## 10. Theoretical foundations

For a higher-is-better metric (y), the adjacent gain is (\Delta y_i=y_i-y_{i-1}). For a lower-is-better loss (L), favorable improvement is (L_{i-1}-L_i). With native budget (b), a budget-normalized gain is (\Delta y/\Delta b). When positive, the measured time cost per unit improvement is (\Delta t/\Delta y). These ratios are meaningful only within a matched series and with a valid measured time increment. A non-positive gain does not have a finite positive “cost per gain”; it is reported as a reversal or zero gain.

## 11. Experimental methodology

This is a post hoc audit with one documented continuation of a previously incomplete, registered semi-supervised schedule. The pseudo-labeling and matched labeled-only runs resume from preserved epoch-5 checkpoints and stop at the pre-existing 30-epoch target, retaining the original 3 seeds and 5 label fractions. All other analyses use existing records. Raw CSV, JSON, JSONL, checkpoint/configuration, and log artifacts remain source evidence. Analysis scripts rebuild the common index, summaries, marginal tables, inventory, and figures without training or evaluation. Superseded, failed, incomplete, smoke, and alternate protocol records are retained and separated by status. Duplicate exports with shared experiment IDs are not pooled.

## 12. Experimental setup

Neural Fashion-MNIST work includes historical Windows/DirectML records and later Apple M3/PyTorch MPS records. Classical methods and RL use CPU records. Platform timings are not pooled as a controlled hardware comparison. Per-run hardware/backend, split identity, seeds, and configuration paths are recorded where available. The old and newer environments remain separately documented in `MACOS_EXECUTION_LOG.md`, `MACOS_SETUP.md`, and `HARDWARE_BENCHMARK_REPORT.md`.

## 13. Datasets

Most conventional learning curves use Fashion-MNIST with a fixed local 10,000-example training subset and 2,000-example validation subset; official test data are used for selected final endpoints. Requested 60,000-example runs reserve 2,000 examples for validation and optimize on 58,000. DBSCAN uses PCA-32 features and a transductive validation protocol for some sweeps. RL uses CartPole-v1 and Pendulum-v1. Language-model full evaluation uses WikiText-2 raw-v1 test text; zero-shot task results use the validation splits named in the harness record. Published literature rows use paper-specific datasets.

## 14. Model architectures

The five supervised scope entries are MLP, LeNet-style CNN, ResNet-18, ViT-Tiny, and Random Forest. A genuine ViT-Tiny-dimension VisionTransformer run uses 28×28 grayscale Fashion-MNIST inputs, 4×4 patches, 12 blocks, 192 hidden width, three attention heads, 768-wide MLP, one-channel projection, and a 10-class head (5,353,738 parameters). It is an input-resolution/channel adaptation of canonical ViT-Tiny dimensions. The separate custom three-layer, width-96 transformer is a project proxy and is not counted as the ViT-Tiny implementation. Unsupervised methods include MiniBatch K-Means, Gaussian Mixture, DBSCAN, Autoencoder, and VAE. Semi-supervised methods include pseudo-labeling, Mean Teacher, FixMatch, MixMatch, and actual scikit-learn LabelPropagation. Some earlier LabelSpreading records are superseded and are not renamed as LabelPropagation. RL includes DQN, Double DQN, A2C, PPO, and SAC; PPO-Pendulum is an additional recorded extension, distinct from the five-method original scope.

## 15. Computational budget definitions

Budgets remain method-specific: epochs/examples for neural supervised and representation models; tree count for Random Forest; partial-fit updates for MiniBatch K-Means; warm-start EM fit calls for GMM; fitted sample count and density parameters for DBSCAN; maximum iterations for LabelPropagation; label fraction and epochs for SSL; environment interactions and gradient updates for RL; released checkpoint step/training-token metadata for Pythia trajectories; and evaluation tokens/examples for inference. No epoch equivalent is invented for non-epoch methods. Training compute for pretrained public LLMs is not measured locally.

## 16. Evaluation metrics

Classification accuracy, macro precision/recall/F1, and cross-entropy are reported separately. Clustering uses ARI, NMI, silhouette, and where present log-likelihood. Reconstruction uses MSE/objective loss; representation quality uses a separate linear-probe accuracy. RL uses episodic return and its within-evaluation episode dispersion. Language modeling uses cross-entropy/perplexity and bits per byte where recorded. Literature benchmarks retain their published definitions. No cross-metric composite is calculated.

## 17. Statistical methodology

For repeated runs, means, sample standard deviations, medians, minima, maxima, valid-seed counts, and two-sided 95% Student-t confidence intervals are produced where (n>1). For (n=1), no interval is asserted. These intervals describe seed variability conditional on the fixed data split and are not population-level dataset uncertainty. Marginal transitions pair adjacent budgets within the same model, seed, split, method, label fraction, and source series. A single seed’s checkpoints are not independent replicates. No p-values or cross-model significance tests are claimed.

## 18. Experimental results: evidence overview

The current canonical local index contains 3,362 source rows across conventional and selected LLM result files. This heterogeneous index is not a count of independent experiments and includes repeated checkpoints, superseded records, and mirrored exports. Prior count snapshots refer to different export generations and aggregation levels; they must not be added together. The current reproducible analysis reports 4,864 grouped metric summaries, 3,167 paired within-seed marginal transitions, and 219 held-out metric groups. These counts were regenerated after adding the verified 30-epoch pseudo-labeling and labeled-only continuations.

## 19. Supervised learning analysis

MLP, LeNet-style CNN, ResNet-18, the compact transformer proxy, and the ViT-Tiny-dimension model have 30-epoch validation curves at the fixed 10K train subset, three seeds each. MLP and LeNet also have five-epoch endpoints at 30K and 58K actual train sizes. At five epochs, MLP test accuracy averages 0.8537 (SD 0.0081; 95% t CI [0.8336, 0.8737]) at 30K and 0.8689 (SD 0.0008; CI [0.8669, 0.8710]) at 58K. LeNet averages 0.8784 (SD 0.0035; CI [0.8699, 0.8870]) and 0.8989 (SD 0.0035; CI [0.8903, 0.9075]), respectively. Each size uses three matched seeds. The two points are consistent with better observed accuracy at the larger size, but cannot establish a general data-scaling law. The genuine ViT-Tiny-dimension model’s 30-epoch official test accuracy averages 0.7621 (SD 0.0616; 95% t CI [0.6090, 0.9152], min 0.6916, max 0.8057; n=3). This wide interval and range indicate substantial seed variability in this implementation; the result does not support a stable model ranking. Validation trajectories and time plots are Figures 1–5 and 20. Test endpoints at 30 epochs are distinct from the matched five-epoch data-size comparison.

The ResNet timing observation is preserved at 1,122.1625 seconds for the epoch-2-to-3 interval for seed 11. Adjacent timestamps show an approximately 18m45s gap; the next epoch interval is about 9.7s. The observation is a genuine logged wall-clock interval, but available logs cannot distinguish compute from process suspension, interruption, or accounting delay. It is retained in the raw-time plot and excluded from the adjusted efficiency diagnostic. Neither value is silently deleted or asserted to be pure device compute. Figure 20 plots supervised validation accuracy against cumulative measured training time after the same limited adjustment.

## 20. Unsupervised learning analysis

The raw unsupervised table has 51 rows. Fifteen later K-Means rows repeat earlier experiment IDs and stamp budgets 1, 5, 10, and 20 with the score of the 50-update model state. They are preserved but excluded from the analytical curve; 15 K-Means observations across three seeds remain. The GMM records contain 21 rows across three seeds and distinct iteration budgets. Figures 6–7 plot ARI against native iterative units. Figure 8 shows DBSCAN’s fit-size sweep; ARI is near zero in most listed settings and its size points are not treated as an iterative learning curve. Figures 9–10 keep AE/VAE reconstruction MSE separate from downstream linear-probe accuracy. Non-neural methods have no invented epoch counts.

## 21. Semi-supervised learning analysis

Pseudo-labeling and its matched labeled-only baseline each cover five label fractions, three seeds, and 30 epochs; historical epoch-5 test endpoints remain preserved alongside new epoch-30 endpoints. At epoch 30, pseudo-labeling mean test accuracies by label fraction are 0.6389, 0.7839, 0.8087, 0.8252, and 0.8397 (n=3 each), while the matched labeled-only means are 0.6544, 0.7695, 0.8054, 0.8214, and 0.8379. Paired seed-level differences (pseudo-label minus labeled-only) are -0.0155 (SD 0.0300; 95% t CI [-0.0900, 0.0590]), +0.0144 (0.0109; [-0.0128, 0.0416]), +0.0033 (0.0007; [0.0017, 0.0049]), +0.0038 (0.0082; [-0.0165, 0.0240]), and +0.0018 (0.0011; [-0.0010, 0.0046]) at 1%, 5%, 10%, 20%, and 40% labels respectively. Only the observed 10% paired interval excludes zero; with three seeds and one fixed split, this is not evidence of a general benefit. Mean Teacher, FixMatch, and MixMatch each have five label fractions × 30 epochs × three seeds, with one terminal test evaluation per model/fraction/seed. Figure 11 compares endpoint test accuracy by label fraction; Figure 12 compares validation curves at 10% labels; Figure 21 compares pseudo-labeling with its matched labeled-only baseline. Label Propagation has 30 current completed rows (three seeds × iteration ceilings) and 15 superseded legacy LabelSpreading/implementation rows; it is reported as LabelPropagation only for the current implementation. Figure 22 plots linear-probe representation quality against measured training time, separately from reconstruction loss.

## 22. Reinforcement learning analysis

The raw RL file contains 132 rows across six recorded methods. DQN, A2C, and PPO have three seeds at 5K, 10K, 25K, 50K, 100K, and 250K interactions. SAC and Double DQN have five seeds at these checkpoints. The Double DQN rows have `completed` experiment status and `local_measurement` source type after derived-schema alignment. The raw CSV omitted the success_rate cell, so that field remains unavailable; all other shifted fields are realigned in derived outputs without modifying the raw rows. PPO-Pendulum is the additional sixth method, an extension rather than a change to the original five-algorithm scope. CartPole and Pendulum results are plotted separately in Figures 13–14 because their reward definitions and ranges differ. Checkpoint evaluation uses 20 deterministic episodes per record; training and evaluation environments are separate. Requested step budgets, actual steps, overshoot, and a legacy resumed run are detailed in the execution log. The finite and noisy curves do not identify a universally best algorithm or a general saturation point.

## 23. Language-model analysis

Six public checkpoints were evaluated on the complete WikiText-2 raw-v1 test text on Apple M3 MPS, FP32, context 1,024, stride 512, batch size one. Pythia models scored 285,830 tokens; GPT-2/OPT scored 283,286 because tokenizers differ. Perplexities were Pythia-70M 44.9334, 160M 25.6365, 410M 15.4583, GPT-2 Small 24.9794, OPT-125M 25.8534, and OPT-350M 20.7992. These are single local endpoint evaluations; model-family comparison is limited by tokenizer, training data, and checkpoint differences. Figure 15 presents this matched local evaluation separately from pretraining compute.

Checkpoint-prefix scaling is a different protocol: Pythia-70M and 160M each have eight selected released checkpoints evaluated on a 2,048-token prefix of the first 30,000 raw characters using CPU FP32; Pythia-410M has selected 0, 10K, and 143K points. Selected training-token counts derive from Pythia’s per-step metadata and are not measured local compute. Figures 16–17 show the two trajectories, including step 0. They are checkpoint evaluations, not original pretraining runs. Full zero-shot harness evaluation used Pythia-70M step143000, lm-evaluation-harness 0.4.13, FP32 MPS, batch 1, 0-shot/no limit on validation splits: HellaSwag 10,042 examples at 0.2667 accuracy; PIQA 1,838 at 0.5985; ARC-Easy 2,376 at 0.3754. Figure 18 shows task results separately. Earlier 100-example/task Pythia-70M, GPT-2 Small, and OPT-125M runs are smoke tests; Figure 25 presents these restricted scores in its own panel. Figures 23–24 show full-test inference time and loss against model size; they are not training-compute plots. A multi-model full-split zero-shot score-versus-size figure cannot be produced: only Pythia-70M has the full validation-split run. The six full evaluations, prefix curves, benchmark run, old 50K-token tests, smoke tests, failed/resource-limited records, and superseded aliases remain separate in the LLM ledgers.

## 24. Published literature comparison

The literature ledger contains 15 published configurations and 300 requested model-field rows (20 fields per configuration). Evidence classes are 174 reported, 6 derived, 16 estimated, 88 uncertain, and 16 unavailable. Uncertain fields lack a verified value; unavailable values were not found in the reviewed source location. Neither is treated as zero. Training tokens, parameter totals, hardware, duration, compute, datasets, and benchmarks are not forced into a common unit. FLOPs and GPU-hours are never conflated without a stated conversion. Literature-only configurations are not counted as locally completed experiments.

## 25. Marginal performance improvement analysis

The regenerated table contains 3,167 adjacent within-seed transitions. Each row records native budget delta, favorable-direction score change, relative change, additional logged seconds where available, gain per native budget unit, gain per second, and seconds per positive unit gain. Positive, zero, and negative changes are retained. Time-normalized values are not reported where the time delta is nonpositive or missing. Table-level summaries and row-level results are in `results/processed/strengthened_marginal_gains.csv`; the complete discussion is in `MARGINAL_GAIN_ANALYSIS.md`.

## 26. Accuracy-versus-compute trade-off

Within comparable local runs, marginal ratios can characterize a specific checkpoint extension, but they do not price compute universally. Epoch seconds vary with device/backend and host load; RL seconds depend on environment and implementation; and no original LLM pretraining times are available for local comparison. ResNet’s anomalous interval demonstrates why measured elapsed wall time must not be equated automatically with accelerator work. The most defensible operational output is a task- and protocol-specific curve plus its measured incremental cost, not a single cross-project exchange rate.

## 27. Diminishing returns analysis

The pipeline screens for two consecutive nonnegative relative improvements below 0.5%, 1%, or 2% within a series. A screen can flag small observed increments but does not establish saturation: sequences are short, threshold choice is conventional, some curves are noisy or negative, and (n=3) often gives wide uncertainty. An isolated interval or negative gain is not evidence of a general diminishing-returns law. Current evidence cannot establish the central diminishing-returns hypothesis across paradigms. It supports only local descriptive statements about finite observed trajectories.

## 28. Cross-paradigm discussion

Figure 19 displays percent change from each series’ first recorded point in separate panels for supervised accuracy, semi-supervised accuracy, CartPole return, and reconstruction MSE. It uses an ordered native-budget index rather than a shared cost axis. This normalization shows within-series direction only; it does not make percentage changes comparable across metrics, baselines, or budgets. A robust cross-paradigm cost-efficiency ranking is not supported.

## 29. Limitations

The project remains incomplete against its 50-objective framework: 20 conventional objectives are complete, while local LLM coverage is partial and literature entries are not local replications. Many conventional curves have three seeds and fixed splits; SSL recipes are task adaptations; RL rewards are noisy and seed counts differ; model-selection/test protocols vary across historical outputs. Energy and peak-memory coverage is incomplete. Local full-test LLM evidence covers six checkpoints, whereas many planned model sizes were not evaluated. The literature ledger has 88 uncertain and 16 unavailable fields. The registered experimental phase is complete; these limits are reported without filling missing evidence.

## 30. Threats to validity

Internal validity is limited by heterogeneous code paths, platform changes, historical superseded runs, wall-clock anomalies, and incomplete logs. Construct validity is limited by treating time as a resource proxy when compute/energy are not measured and by using task-specific metrics. External validity is limited by Fashion-MNIST subsets, small seed counts, one RL environment per action regime, and one local WikiText-2 protocol. Reproducibility risks include evolving dependencies, checkpoint aliases, incomplete original hardware metadata, and data transformations whose configurations must be read with each source record.

## 31. Reproducibility

From the project root, rebuild derived records and reports without invoking model execution:

```sh
.venv-mac/bin/python scripts/build_canonical_results.py
.venv-mac/bin/python scripts/build_verified_dataset.py
.venv-mac/bin/python scripts/analyze_strengthened_runs.py
.venv-mac/bin/python scripts/analyze_llm_full_eval.py
.venv-mac/bin/python scripts/build_literature_evidence.py
.venv-mac/bin/python scripts/audit_conventional_completion.py
.venv-mac/bin/python scripts/generate_completion_matrix.py
.venv-mac/bin/python scripts/build_experiment_run_registry.py
.venv-mac/bin/python scripts/plot_final_figures.py
.venv-mac/bin/python scripts/audit_repository_inventory.py
.venv-mac/bin/python scripts/audit_identifiers.py
.venv-mac/bin/python scripts/build_final_report_pdf.py
.venv-mac/bin/python scripts/validate_project_outputs.py
```

These commands read existing records and regenerate indices, statistics, figures, identifier reconciliation, and inventory. They do not train or infer. Raw data and source snapshots are not rewritten. See `REPRODUCIBILITY.md` for environment and provenance detail.

## 32. Conclusion

Additional native budget sometimes coincides with better measured performance, especially over early supervised epochs and larger MLP/LeNet training subsets; later increments can be small, variable, or negative. The current evidence does not identify a universal amount of accuracy worth a unit of compute and does not prove general diminishing returns. The defensible conclusion is that the trade-off is conditional on task, model, protocol, metric, hardware, and the chosen endpoint. Broader causal or universal claims remain unresolved.

## 33. Future research directions

Future work, if separately authorized, could pre-register comparable budgets, replicate matched conditions with more seeds, record energy and device utilization, standardize validation/test selection, and evaluate representative LLM checkpoints under a fully matched protocol. These are recommendations only; this audit performs none of them and does not complete missing runs.

## 34. References

1. Brown et al. (2020). “Language Models are Few-Shot Learners.” [arXiv:2005.14165](https://arxiv.org/abs/2005.14165), especially model sizes and training tokens in Table 2.1.
2. Rae et al. (2021). “Scaling Language Models: Methods, Analysis & Insights from Training Gopher.” [arXiv:2112.11446](https://arxiv.org/abs/2112.11446).
3. Hoffmann et al. (2022). “Training Compute-Optimal Large Language Models.” [arXiv:2203.15556](https://arxiv.org/abs/2203.15556).
4. Chowdhery et al. (2022). “PaLM: Scaling Language Modeling with Pathways.” [arXiv:2204.02311](https://arxiv.org/abs/2204.02311).
5. Zhang et al. (2022). “OPT: Open Pre-trained Transformer Language Models.” [arXiv:2205.01068](https://arxiv.org/abs/2205.01068).
6. Touvron et al. (2023). “LLaMA: Open and Efficient Foundation Language Models.” [arXiv:2302.13971](https://arxiv.org/abs/2302.13971).
7. Scao et al. (2023). “BLOOM: A 176B-Parameter Open-Access Multilingual Language Model.” [JMLR 24(1)](https://www.jmlr.org/papers/v25/23-0581.html).
8. WikiText-2 raw-v1, Salesforce Research/Hugging Face dataset repository, revision recorded in `results/llm/macos_full_wikitext2_evaluations.csv`.

Field-level source URLs, locators, and evidence classes are in `results/literature/literature_field_evidence.csv` and `results/literature/literature_sources.md`. Primary-paper field verification is summarized in `LITERATURE_VALIDATION_REPORT.md`.

## 35. Appendix: experimental configurations

The experiment run registry and `results/configurations/*.json` record run-level model, seed, data split, budget, backend, and source references where available. The main fixed Fashion-MNIST split is identified by split ID `24574ee00a20ab31`. Primary budgets are: 1–30 epochs for main supervised/AE/VAE/SSL curves; 10–1,000 trees for Random Forest; native update/iteration schedules for clustering and LabelPropagation; 5K–250K requested environment steps for RL; and protocol-specific checkpoint and evaluation lengths for LLMs. Label fractions for the multi-method SSL study are 1%, 5%, 10%, 20%, and 40%. Detailed hyperparameters remain in per-run JSON rather than being reconstructed from memory. Missing configuration evidence is marked unavailable in the inventory and audit register. `results/processed/canonical_results.csv` preserves a common source index; `results/processed/verified_canonical_results.csv` adds explicit protocol-level eligibility and exclusion reasons without combining unlike metrics. `IDENTIFIER_AUDIT.md` and `IDENTIFIER_RECONCILIATION.csv` provide source-scoped uniqueness, foreign-key checks, and all derived identifier corrections.

## 36. Appendix: completion status and additional figures

The strict completion denominator remains 50: 20 conventional, 15 LLM, and 15 literature objectives. Twenty objectives are verified complete; the remaining 30 are partial local LLM evaluations and literature-only comparisons, not local replications. Evidence volume, partial objectives, and complete objectives are distinct quantities. Figure paths, sources, scripts, interpretation, and limitations for 25 figures are listed in `FIGURE_CATALOGUE.md`; PNG and vector SVG files are under `results/figures/final/`. Inventory and row-level schema/missingness summaries are in `DATA_INVENTORY.md` and `results/processed/tabular_data_inventory.json`. Identifier and foreign-key audit outputs are in `IDENTIFIER_AUDIT.md` and `results/processed/`.
