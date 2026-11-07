# LLM research design

**Execution status:** six of the 15 requested model configurations have local WikiText-2 prefix evaluations, including eight Pythia-70M checkpoint points. The full checkpoint sweeps, downstream benchmarks, and remaining configurations are incomplete; see `FINAL_COMPLETION_MATRIX.md` and `REMAINING_WORK.md`.

## Primary question

Measure how language-model loss and downstream metrics vary with tokens and compute, keeping parameter scaling separate from training progression. Checkpoint evaluation cost is not original pretraining cost.

## Candidate coverage

The requested eight Pythia scales (70M through 12B), four GPT-2 scales, and three OPT scales form the 15-configuration evaluation target. Pythia's published suite has 154 checkpoints per model; select 10 logarithmically spaced training stages including early, intermediate, and final checkpoints. Record the actual checkpoint step and token count from model metadata. A Pythia-70M final checkpoint was loaded sequentially on CPU for a short WikiText-2 pilot. The remaining checkpoints were not attempted because the machine is CPU-only and had about 6 GB available RAM after package installation.

For each successfully evaluated checkpoint record parameter count, step, tokens, source-reported train/validation loss when available, locally measured WikiText-103 loss/perplexity, benchmark task/protocol, load/evaluation duration, peak RAM/VRAM if measurable, and status. Use a fixed tokenizer, dataset revision, context length, harness version, and zero-shot protocol for primary comparisons. A small common task suite can use PIQA, HellaSwag, and ARC-Easy only if the same harness revision and task configs support every selected model. Do not compare raw scores from unrelated papers as a single leaderboard.

Tokens and steps are the primary progression axes. Epochs are omitted unless corpus size and sampling policy establish an actual data pass. Training FLOPs, duration, and energy are reported only if available from a source or measurement. The 6ND approximation can be added as a separate calculated column only when N and D are known and the dense-transformer assumption is stated.

## Budget and evaluation safety

Load models sequentially. Before attempting a checkpoint, estimate weight memory from parameter count and dtype plus runtime overhead; skip with a reason if the measured available memory is insufficient. The 6.9B/12B Pythia and multi-billion GPT-2/OPT configurations are not assumed feasible. Use source-reported training trajectory values for those rows when local inference cannot be performed, keeping local evaluation fields null.

## Analysis

Plot loss and perplexity against tokens only for comparable checkpoints in a family/protocol. Plot downstream task scores separately by task. Fit power laws only when there are enough distinct observations and a plausible scaling regime; report fitted parameters, residuals, uncertainty, and sensitivity to exclusions. A final checkpoint alone cannot establish diminishing returns. Marginal gain versus compute requires intermediate measurements and an explicit, consistent cost measure.
