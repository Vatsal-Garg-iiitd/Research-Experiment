# Research design and final evidence state

## Research question

How much measured performance improvement accompanies additional computational budget, and do observed marginal gains diminish? Budgets and metrics are method-specific. The project does not treat epochs, optimizer steps, tree counts, environment interactions, inference time, and published FLOPs as interchangeable.

## Original scope

The strict completion framework has 50 objectives: 20 conventional methods (supervised, unsupervised, semi-supervised, and RL), 15 specified local LLM configurations, and 15 literature configurations. The fifth original RL set is DQN, Double DQN, A2C, PPO, SAC. PPO-Pendulum is a recorded extension. Literature entries are source-based observations, not local model runs.

## Evidence and completion criteria

Raw CSV/JSON/JSONL records, model/configuration files, evaluator outputs, and execution logs define the evidence. The original objective types remain distinct: 20 conventional local studies, 15 local LLM configurations, and 15 literature reviews. The current status is 46/50 under the per-configuration endpoint rule: all 20 conventional objectives and 15 literature objectives are complete, as are 11 LLM configurations with full-test endpoints. Four LLM configurations lack endpoints; groupwise LLM benchmark/trajectory subtasks remain partial. Literature review completion is not local model reproduction. Current status by objective is in `reports/FINAL_COMPLETION_MATRIX.md`.

## Current dataset and protocol summary

The main supervised and representation-learning dataset is Fashion-MNIST with a fixed 10K train/2K validation split and official test endpoints. Core neural curves extend to 30 epochs over three seeds. MLP/LeNet data-size checkpoints use 5 epochs at 10K/30K/58K actual train examples. Random Forest uses warm-start tree counts; unsupervised algorithms retain native units; SSL includes five label fractions; RL extends through requested 250K environment steps; selected public LLM checkpoints have separate full-corpus and checkpoint-prefix inference protocols. See `docs/EXPERIMENTAL_METHODOLOGY.md` for implementation and metric details.

## Analysis

Seed means, standard deviations, medians, minima/maxima, and t intervals are computed from independent aligned seed runs. Adjacent budget changes pair within seed and setting. No p-values are reported. Saturation flags are exploratory and do not establish a general law. Results are kept within metric families; cross-paradigm display uses separate normalized panels only.

## Final status

Experimental phase closed. No additional experiment, training, inference, benchmarking, or data collection is included in this final audit. Data-quality exclusions and anomalies are recorded in `docs/DATA_QUALITY_REGISTER.md`; raw data remain preserved. See `reports/FINAL_RESEARCH_REPORT.md` for results and limitations.
