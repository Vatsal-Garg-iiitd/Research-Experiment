# LLM evaluation report

All local measurements evaluate published checkpoints; no language model was pretrained in this project. Checkpoint evaluation, benchmark accuracy, and published training-token metadata are separate quantities.

## Evaluation inventory and reconciliation

The complete WikiText-2 raw-v1 test evaluation now contains 11 completed checkpoint rows in `macos_full_wikitext2_evaluations.csv`: Apple M3 MPS, PyTorch 2.15.0, FP32, context 1,024, stride 512, batch size 1. The original six evaluations are joined by full-test evaluations for GPT-2 Medium, GPT-2 Large, Pythia-1B, Pythia-1.4B, and OPT-1.3B. All are local checkpoint inference, not original pretraining. GPT-2 XL was interrupted during checkpoint download at the user's request before weights loaded or any test tokens were scored; no result is reported for it.

`expanded_llm_evaluations.csv` contains 40 records with status counts: 24 completed, 7 superseded, 8 resource-limited, and 1 failed. The 24 completed rows include duplicate `main` aliases for Pythia-160M/410M; they are not extra points beyond explicit step143000. Pythia-70M and 160M each have eight distinct released checkpoint steps in the current prefix series. Pythia-410M has selected 0, 10K, and 143K points. These prefix evaluations use CPU FP32 and 2,048 model tokens from the first 30,000 raw characters. Token counts shown as training progress are model-card-derived metadata, not local training compute.

The older `full_wikitext2_evaluations.csv` contains six completed 50K-token CPU FP32 evaluations and is a restricted prefix protocol. It must not be substituted for or pooled with the full-test MPS results. The `harness_full_summary.csv` contains three unrestricted 0-shot validation tasks for Pythia-70M, evaluated on MPS FP32, batch 1. `harness_summary.csv` contains nine rows from 100-example/task checks (three tasks each for Pythia-70M, GPT-2 Small, and OPT-125M); these are smoke tests. The sample limits and hardware differ, so those rows are not presented as full benchmark comparisons. The `expanded_llm_evaluations.csv` failed/resource-limited/superseded statuses are retained as evidence of scope limits and are not included as completed evaluations.

## Matched WikiText-2 test evaluation

All 11 completed checkpoints were evaluated on the complete WikiText-2 raw-v1 test text in FP32 on Apple M3 MPS with context 1,024 and stride 512. Each corpus token is scored once, except the initial context token. Raw test corpus and byte accounting are shared across models; GPT-2 and OPT models use their own tokenizers, so scored token counts differ. Five additional full-test evaluations were completed from the prior six; no new benchmark tasks or pretraining were conducted. See `results/llm/macos_full_wikitext2_evaluations.csv`.

| Model | Checkpoint | Parameters | Scored tokens | Cross entropy | Perplexity | Eval seconds |
|---|---|---:|---:|---:|---:|---:|
| Pythia-70M | step143000 | 70,426,624 | 285,830 | 3.8052 | 44.9334 | 38.54 |
| GPT-2 Small | main | 124,439,808 | 283,286 | 3.2181 | 24.9794 | 177.11 |
| OPT-125M | main | 125,239,808 | 283,286 | 3.2524 | 25.8534 | 158.38 |
| Pythia-160M | step143000 | 162,322,944 | 285,830 | 3.2440 | 25.6365 | 167.93 |
| OPT-350M | main | 331,000,000 | 283,286 | 3.0349 | 20.7992 | 420.28 |
| GPT-2 Medium | main | 354,823,168 | 283,286 | 2.9030 | 18.2287 | 375.73 |
| Pythia-410M | step143000 | 405,334,016 | 285,830 | 2.7381 | 15.4583 | 463.60 |
| GPT-2 Large | main | 774,030,080 | 283,286 | 2.7359 | 15.4238 | 1108.80 |
| Pythia-1B | step143000 | 1,011,382,272 | 285,830 | 2.5449 | 12.7416 | 1435.18 |
| OPT-1.3B | main | 1,315,000,000 | 283,286 | 2.6227 | 13.7731 | 1734.36 |
| Pythia-1.4B | step143000 | 1,414,647,808 | 285,830 | 2.4392 | 11.4635 | 1763.61 |

Bits per byte uses the union of UTF-8 byte-offset spans for scored tokens to avoid double-counting overlapping tokenizer offsets. The 50K-token evaluations recorded earlier are smaller prefix evaluations and are retained separately. The three larger planned checkpoints Pythia-2.8B/6.9B/12B were not run locally because FP32 weights exceed this Mac's safe MPS memory budget; GPT-2 XL has no score because its download was interrupted before inference. These four configurations remain incomplete under the unchanged strict objective criteria. The 50K and full-corpus results must not be mixed. Eval time and FP32 weight-size estimates are not measurements of original pretraining cost. End-of-evaluation process RSS is recorded; reliable peak RSS is unavailable for these MPS runs.

## Pythia checkpoint trajectory

The separate prefix trajectory evaluates released Pythia checkpoints on the same first 30,000 raw characters using CPU FP32. Values below are cross entropy at the 2,048-token model prefix; `—` means that point was not evaluated. Token counts are derived from the Pythia model card’s final tokens per step except step143000, whose final training-token count is source-reported. These are published metadata, not measured local pretraining compute.

| Model | step 0 | 1K | 5K | 10K | 20K | 50K | 100K | 143K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pythia-70M | 11.0185 | 4.8243 | 3.7401 | 3.6676 | 3.6107 | 3.4690 | 3.5632 | 3.5621 |
| Pythia-160M | 11.1059 | 4.6557 | 3.3751 | 3.2116 | 3.1042 | 2.9765 | 2.9419 | 3.0102 |
| Pythia-410M | 11.0904 | — | — | 2.8836 | — | — | — | 2.5387 |

Alternate 410M checkpoint retrieval was stopped after step 0 because the next revision did not load promptly; the existing 10K and 143K records remain. The curves are descriptive evidence for one text prefix and one released checkpoint series, not a fitted or replicated scaling law. See `results/llm/expanded_llm_evaluations.csv`.

## lm-evaluation-harness

An unbounded 0-shot run evaluated Pythia-70M step143000 on validation splits for HellaSwag, PIQA, and ARC-Easy. It used lm-evaluation-harness 0.4.13, Transformers 5.22.1, FP32, Apple M3 MPS, and batch size 1, with no `--limit`. The harness reported:

| Task (validation split) | Examples | Accuracy | Normalized accuracy | Accuracy SE |
|---|---:|---:|---:|---:|
| HellaSwag | 10,042 | 0.2667 | 0.2738 | 0.0044 |
| PIQA | 1,838 | 0.5985 | 0.5919 | 0.0114 |
| ARC-Easy | 2,376 | 0.3754 | 0.3523 | 0.0099 |

The extracted table is `results/llm/harness_full_summary.csv`; full aggregate JSON and per-sample logs are under `results/llm/harness_full/pythia70m_step143000/`. The earlier 100-example checks for GPT-2 Small and OPT-125M remain smoke runs under `results/llm/harness/` and are not full benchmark results. No direct comparison across models is made from that smoke subset.

## Figures

- `results/figures/llm_model_size_vs_perplexity.png`
- `results/figures/llm_model_size_vs_loss.png`
- `results/figures/llm_model_size_vs_throughput.png`

## Interpretation limits

Loss and perplexity are fixed-corpus checkpoint evaluations, not seed replicates. Checkpoint-level local evaluation does not show how much compute was used to create those checkpoints unless published training metadata is separately cited. Harness standard errors describe task-example sampling, not training-seed variability. These measurements do not establish universal scaling behavior or compare the original training efficiency of model families.
