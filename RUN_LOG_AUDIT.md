# Run-log audit and non-overlap procedure

## Existing logging formats

The repository contains two Markdown execution histories (`EXECUTION_LOG.md`
and `MACOS_EXECUTION_LOG.md`) and timestamped JSONL per-example outputs for
language-model harness runs. Earlier conventional job console streams were not
archived, so they cannot be reconstructed after the fact. The final pseudo-label
continuation has a timestamped combined stdout/stderr log. The JSONL files are
model-evaluation evidence, not process logs.

Training scripts write experiment records to shared append-only CSV files. Their
writers flush each row, and model scripts print progress to stdout. Several
scripts target the same CSV, so concurrent invocations could interleave output
or race on headers and checkpoints. The Markdown histories are chronological
summaries rather than captured terminal transcripts; repeated events are
distinguished by source row IDs, checkpoint IDs, and timestamps where recorded.

## Procedure for additional runs

All further training commands must be invoked sequentially through
`scripts/run_logged.py`. It acquires one repository-wide exclusive run lock,
merges stdout and stderr into a unique UTC-timestamped file under
`results/run_logs/`, prefixes each output line with an ISO-8601 UTC timestamp,
records the command, Python executable, source revision, start/end time, and
exit code, and refuses to start while another wrapped run holds the lock. The
experiment scripts remain responsible for row-level timing, run IDs, seeds,
and configuration references in their CSV outputs. A failed run keeps its log
and any completed rows; resumable scripts may continue from verified saved
checkpoints in a later, separately logged invocation.

The wrapper serializes only commands launched through it; do not launch a
second training command directly while a wrapped experiment is running. Raw
result records and original checkpoints must not be edited to improve log
presentation.
