# Methodology gaps and limits

The following limitations apply to the saved evidence and are reflected in the final report. They are not resolved by post hoc analysis and do not trigger new runs.

- **Completion criterion:** the original 50-objective all-or-nothing framework remains 0/50. Partial evidence is not promoted to complete.
- **Replicates:** most supervised, SSL, representation, and core RL conditions have three seeds; Double DQN and SAC have five. Intervals are conditional on fixed splits and small-sample uncertainty is substantial.
- **Validation and testing:** historical endpoint selection is not completely uniform. Some analyses use held-out endpoints at designated budgets; validation curves are treated separately. Data-size scaling is restricted to matched epoch-5 endpoints.
- **Architecture and algorithm fidelity:** the compact transformer is a 3-layer/96-width proxy. SSL recipes are Fashion-MNIST adaptations. Historical LabelSpreading rows do not count as LabelPropagation.
- **Unsupervised comparability:** K-Means updates, GMM warm-start EM calls, DBSCAN fit sizes/parameters, graph iteration ceilings, and reconstruction epochs are distinct computational budgets.
- **RL variability:** reward distributions are noisy; tasks use different scales; seed counts differ; elapsed time depends on runtime and host. PPO-Pendulum is an extension.
- **LLM protocol:** local full-test checkpoint evaluation, restricted checkpoint-prefix curves, full validation harness results, smoke tests, and superseded/failed rows are different evidence classes. Local inference does not reveal original training compute.
- **Hardware/resource accounting:** wall time is not energy or accelerator FLOPs. The 1,122.16-second ResNet interval is unexplained; historical Windows/DirectML and newer Mac/MPS timings are separate.
- **Literature:** model identities and selected training facts were spot-checked against primary sources. 104 of 300 field rows remain uncertain; benchmark protocols are not harmonized.

The current analysis scripts expose these limitations and retain raw values. See `DATA_QUALITY_REGISTER.md`, `LITERATURE_VALIDATION_REPORT.md`, and `FINAL_RESEARCH_REPORT.md`.
