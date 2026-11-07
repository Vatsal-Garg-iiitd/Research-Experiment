# LLM Evaluation Report

Updated from `results\llm\full_wikitext2_evaluations.csv`. All listed scores are local checkpoint inference measurements; they are not pretraining runs.

## Completed local evaluations

6 configurations completed a matched WikiText-2 raw-v1 test evaluation: Pythia-70M (step143000), Pythia-160M (step143000), Pythia-410M (step143000), GPT-2 Small (main), OPT-125M (main), OPT-350M (main).

The requested input budget was 50,000 tokenizer tokens for each run. Scored-token counts are 49,951–49,975 because causal next-token loss excludes each non-overlapping chunk’s first token. Pythia and OPT used 2,048-token context windows; GPT-2 Small used 1,024. That context difference is a comparability limitation. Runs used CPU FP32. The table reports model loading time, evaluation time, observed throughput, process peak RAM and the recorded checkpoint metadata. Local evaluation time is not a proxy for original pretraining compute.

## Results

| Model | Checkpoint | Parameters | Scored tokens | Cross entropy | Perplexity | Eval seconds | Tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Pythia-70M | step143000 | 70,426,624 | 49975 | 3.9036 | 49.581 | 23.34 | 2141.0 |
| Pythia-160M | step143000 | 162,322,944 | 49975 | 3.3766 | 29.271 | 62.93 | 794.1 |
| Pythia-410M | step143000 | 405,334,016 | 49975 | 2.8753 | 17.732 | 182.85 | 273.3 |
| GPT-2 Small | main | 124,439,808 | 49951 | 3.4796 | 32.445 | 58.84 | 848.9 |
| OPT-125M | main | 125,239,808 | 49975 | 3.4103 | 30.275 | 48.40 | 1032.6 |
| OPT-350M | main | 331,000,000 | 49975 | 3.1930 | 24.361 | 136.62 | 365.8 |

## Downstream harness evaluation

lm-evaluation-harness 0.4.13 ran 0-shot CPU FP32 evaluations on 100 examples per task for Pythia-70M step143000, GPT-2 Small, and OPT-125M. These are smoke subsets and must not be presented as full benchmark scores. The result JSONs and per-sample logs are under `results/llm/harness/`; the extracted table is `results/llm/harness_summary.csv`.

| Model | HellaSwag acc / normalized | PIQA acc / normalized | ARC-Easy acc / normalized | Total eval seconds |
|---|---:|---:|---:|---:|
| openai-community/gpt2 | 0.36 / 0.44 | 0.6 / 0.63 | 0.38 / 0.34 | 72.61219999999958 |
| facebook/opt-125m | 0.38 / 0.43 | 0.66 / 0.6 | 0.46 / 0.45 | 180.38221200000044 |
| EleutherAI/pythia-70m | 0.33 / 0.41 | 0.6 / 0.6 | 0.4 / 0.33 | 50.01474390000112 |

## Figures

- `results/figures/llm_model_size_vs_perplexity.png`
- `results/figures/llm_model_size_vs_loss.png`
- `results/figures/llm_model_size_vs_throughput.png`

No confidence intervals are reported for the WikiText comparisons: these are deterministic evaluations of distinct fixed checkpoints, not repeated seed runs. Harness standard errors are dataset-subset sampling estimates for n=100, not seed variability.
