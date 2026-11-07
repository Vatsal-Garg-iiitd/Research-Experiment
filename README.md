# Computational scaling research project

This repository studies how model performance changes as computational budgets increase, and whether marginal gains decline. It keeps local experiments, evaluations of published checkpoints, and literature-derived evidence separate. Budget units stay specific to each method; results from different tasks are not pooled onto one compute scale.

## Current status

This is an expanded but incomplete exploratory study. The strict matrix counts **0 of 50 requested configurations as fully complete**: 20 conventional methods, 15 LLM configurations, and 15 literature configurations. Many substantial experiments are complete, but pilots, partial schedules, and literature-only observations do not satisfy the full registered objective. See the [completion matrix](FINAL_COMPLETION_MATRIX.md) and [independent audit](FINAL_INDEPENDENT_AUDIT.md).

### Completed work

- **Conventional methods:** multi-seed and multi-budget runs include 30-epoch MLP, LeNet, ResNet-18, compact-transformer, AE/VAE, and SSL curves; MLP/LeNet five-epoch data-size comparisons; Random Forest through 1,000 trees; clustering and LabelPropagation sweeps; DBSCAN parameter and data-size studies; and RL schedules through 250K requested interactions. SAC and Double DQN have five seeds. The transformer is a compact proxy, and SSL recipes are grayscale task adaptations.
- **Language models:** six published checkpoints have full WikiText-2 raw-v1 test evaluations using FP32, Apple MPS, context 1,024, and stride 512. Pythia-70M and Pythia-160M each have eight released-checkpoint prefix evaluations. Pythia-410M has selected prefix checkpoints. A full zero-shot Pythia-70M harness run covers the HellaSwag, PIQA, and ARC-Easy validation splits; earlier GPT-2 Small and OPT-125M harness runs are 100-example-per-task smoke checks. No model was pretrained locally.
- **Literature evidence:** 15 published configurations have a 300-row field ledger with source URLs, locators, and reported/derived/estimated/uncertain classes. The ledger is not a standardized cross-paper benchmark; 104 fields remain uncertain.
- **Audit artifacts:** the project includes a 3,720-row [run registry](EXPERIMENT_RUN_REGISTRY.csv), a 2,612-row canonical result index, regenerated statistical summaries, figures, and a validator. The final validator reported zero errors and zero warnings. See the [final research report](FINAL_RESEARCH_REPORT.md), [macOS execution log](MACOS_EXECUTION_LOG.md), and [remaining work](REMAINING_WORK.md).

### Selected observations

At five epochs, mean held-out Fashion-MNIST accuracy across three seeds was 0.8537 for MLP at the requested 30K training size (30K optimization examples) and 0.8689 at the requested 60K size (58K optimization examples plus 2K validation). LeNet means were 0.8784 and 0.8989, respectively. These fixed-split results are descriptive; they do not establish a general data-scaling law.

Full WikiText-2 test perplexities were Pythia-70M 44.9334, Pythia-160M 25.6365, Pythia-410M 15.4583, GPT-2 Small 24.9794, OPT-125M 25.8534, and OPT-350M 20.7992. These are local evaluations of public checkpoints, not matched measurements of original training compute.

The full zero-shot Pythia-70M harness run evaluated 10,042 HellaSwag, 1,838 PIQA, and 2,376 ARC-Easy validation examples with no task limit. The results and protocol are in the [LLM evaluation report](LLM_EVALUATION_REPORT.md).

## Setup

For the verified Apple M3 macOS environment, see [macOS setup](MACOS_SETUP.md). The isolated `.venv-mac` uses PyTorch MPS where supported; scikit-learn and RL runs use CPU. Install the project requirements with Python 3.10 or newer. The recorded harness run also uses Accelerate, installed as documented in the setup guide.

The original Windows and DirectML artifacts are retained. On Windows, use the separate DirectML environment and `requirements-directml.txt`; do not mix its timings with macOS or CPU timings as a controlled hardware comparison. The NPU has not been benchmarked.

## Repository map

- `configs/`: target experiment design and model registry.
- `results/configurations/`: per-run configuration JSON.
- `results/raw/`: row-level conventional experiment results.
- `results/llm/`: full-corpus evaluations, checkpoint trajectories, harness aggregates, and sample logs.
- `results/literature/`: published configuration ledger, source notes, and field-level evidence.
- `results/processed/`: canonical result index and generated statistical summaries.
- `results/figures/`: generated result plots.
- `results/checkpoints/`: local training checkpoints; these are not needed to read the saved results.
- `EXPERIMENT_RUN_REGISTRY.csv`: row-level index of results and literature evidence.
- `FINAL_COMPLETION_MATRIX.md`, `FINAL_RESEARCH_REPORT.md`, and `FINAL_INDEPENDENT_AUDIT.md`: scope, evidence, and limitations.

## Rebuild analysis outputs

From the project root on the verified Mac:

```sh
.venv-mac/bin/python scripts/build_canonical_results.py
.venv-mac/bin/python scripts/analyze_strengthened_runs.py
.venv-mac/bin/python scripts/analyze_llm_full_eval.py
.venv-mac/bin/python scripts/build_literature_evidence.py
.venv-mac/bin/python scripts/generate_completion_matrix.py
.venv-mac/bin/python scripts/build_experiment_run_registry.py
.venv-mac/bin/python scripts/validate_project_outputs.py
```

The scripts rebuild derived artifacts from saved records; they do not fill in missing experiment runs. Checkpoints and validation curves are method-specific, and held-out endpoints are not uniformly selected with the same validation-based rule across every run. Consult `EXPERIMENTAL_METHODOLOGY.md` before interpreting the summaries.
