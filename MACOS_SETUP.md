# macOS setup

## Verified machine and environment

Verified on macOS 26.6.2, Apple M3, arm64, 16 GiB RAM, Python 3.12. The project data and checkpoints were already present; Fashion-MNIST loads from `results/cache/`. At setup, 115 GiB was available on the workspace volume. PyTorch 2.15.0 reports MPS built and available.

The macOS environment is isolated at `.venv-mac/`; existing Windows environments are preserved. Install packages using the repository's pinned, platform-neutral `requirements.txt` (do not install `requirements-directml.txt` on macOS). The recorded lm-evaluation-harness run also required Accelerate, installed in this environment as `accelerate==1.15.0`:

```sh
python3.12 -m venv .venv-mac
.venv-mac/bin/python -m pip install --upgrade pip
.venv-mac/bin/python -m pip install -r requirements.txt
.venv-mac/bin/python -m pip install 'accelerate>=1.0,<2.0'
.venv-mac/bin/python -m pip check
```

The tested package set is recorded in `MACOS_EXECUTION_LOG.md`. Neural scripts choose Apple MPS when available and fall back to CPU through `scripts/backend.py`. Classical scikit-learn and RL runs remain CPU based. No GPU/NPU comparison or power benchmark is part of this workflow.

## Smoke checks

```sh
.venv-mac/bin/python -m compileall -q scripts
.venv-mac/bin/python scripts/validate_project_outputs.py
```

A small MPS training step, scikit-learn clustering example, Gymnasium CartPole step, and Fashion-MNIST load were executed successfully on the verified setup. Selected Pythia, GPT-2, and OPT checkpoints were downloaded for explicitly scoped local inference and evaluation; these checkpoints are not locally pretrained results. See `MACOS_EXECUTION_LOG.md` for exact completed model scope and resource observations.

## Resumable runs

Experiment checkpoints and downloaded datasets stay local and are ignored by Git. For example:

```sh
.venv-mac/bin/python scripts/run_vit_tiny.py --seeds 11 23 37 --train-size 10000 --max-epochs 30
.venv-mac/bin/python scripts/run_replicated_scaling.py --models lenet resnet18 vit_compact --seeds 11 23 37 --train-size 10000
.venv-mac/bin/python scripts/run_ssl_algorithms.py --seeds 11 23 37 --fractions .01 .05 .10 .20 .40 --max-epochs 30
```

Neural checkpoints are written after each epoch, and epoch-level records are appended to raw CSVs. A resumed run skips epochs already stored in its checkpoint. Do not run duplicate jobs against the same checkpoint/CSV concurrently. Inspect validation curves before continuing past any plateau or overfit behavior. Use official test results only as final endpoints; do not select budgets using test metrics.

LLM models and harness benchmarks can consume substantial memory and disk. Evaluate one checkpoint at a time and keep precision/context settings in the result record. A full zero-shot Pythia-70M validation run completed without `--limit`; older 100-example harness records remain smoke tests, not full benchmark scores.
