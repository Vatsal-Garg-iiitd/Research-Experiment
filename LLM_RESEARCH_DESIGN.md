# LLM evaluation design and final status

## Research question

Describe how local checkpoint evaluation metrics vary with released model size or checkpoint step while keeping inference cost distinct from original pretraining compute. No LLM was pretrained in this project.

## Planned coverage and actual evidence

The 15-config target comprises Pythia-70M/160M/410M/1B/1.4B/2.8B/6.9B/12B, GPT-2 Small/Medium/Large/XL, and OPT-125M/350M/1.3B. Eleven checkpoints (Pythia-70M/160M/410M/1B/1.4B, GPT-2 Small/Medium/Large, OPT-125M/350M/1.3B) have full WikiText-2 raw-v1 test-text evaluations. Pythia-70M and 160M have eight selected released checkpoint-prefix points each; Pythia-410M has selected steps 0, 10K, and 143K. Pythia-70M also has a full, unrestricted 0-shot validation-harness evaluation on HellaSwag, PIQA, and ARC-Easy. GPT-2 Small and OPT-125M have earlier 100-example/task smoke checks. Pythia-2.8B/6.9B/12B exceed the safe local FP32 MPS memory budget; GPT-2 XL was interrupted during download before inference. Remaining configurations are not evaluated under all registered protocols as recorded in `FINAL_COMPLETION_MATRIX.md` and `results/llm/expanded_llm_evaluations.csv`.

## Protocol separation

1. **Full WikiText-2 raw-v1 test:** eleven completed runs; FP32, Apple M3 MPS, context 1,024, stride 512, batch one. The common text SHA-256, scored-token counts, perplexity, NLL, and evaluation seconds are recorded in `macos_full_wikitext2_evaluations.csv`.
2. **Historical 50K-token evaluation:** six CPU FP32 runs, restricted budget, context 2,048. Kept separate from the full test set.
3. **Checkpoint-prefix trajectory:** Pythia selected released checkpoints, CPU FP32, one 2,048-token sequence from the first 30,000 raw text characters. Token counts per step are published metadata, not local training compute. `main` aliases for 160M/410M are not extra trajectory points.
4. **Full zero-shot harness:** Pythia-70M step143000, lm-evaluation-harness 0.4.13, Transformers 5.22.1, FP32 MPS, batch one, no task limit, validation splits.
5. **Smoke tests:** 100 examples per task for Pythia-70M/GPT-2 Small/OPT-125M on CPU FP32; not full benchmark results.
6. **Superseded, failed, and resource-limited rows:** retained with status and excluded from completed-result claims.

## Measurement and comparison

Full-corpus NLL and perplexity are computed from token-level losses, with tokenizer-specific token counts retained. Bits per byte uses UTF-8 byte-span accounting for cross-tokenizer context but remains a single-corpus local inference measure. Parameter count is not a compute budget. Published training-step/token metadata are not elapsed training measurements. Evaluation time is compared only within the same local protocol and device, with tokenizer token-count differences noted. Full benchmark model-size comparisons are unavailable because only one model has a full validation-split harness run; the smoke subset is shown separately and never labeled a complete benchmark.

## Completion and limits

The LLM portion remains partial under the strict 50-objective framework. The 11 full WikiText-2 evaluations, selected Pythia trajectories, one full harness model, and three smoke-run models do not complete the 15-configuration target. This design note documents current evidence only and does not authorize more evaluation.
