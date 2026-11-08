# Limitations and scope boundaries

The registered conventional schedules are complete; this file records limits on interpretation rather than unfinished runs to be silently carried forward.

## Unresolved evidence limits

- The strict completion matrix remains 20/50. Fifteen local LLM objectives are partial or unevaluated; 15 literature entries are published evidence, not local experiments.
- Six public checkpoints have complete WikiText-2 test evaluations, but checkpoint trajectories and zero-shot tasks cover only selected models. No local model pretraining was performed. Restricted smoke subsets are not complete benchmarks.
- Transformer and semi-supervised implementations are Fashion-MNIST task adaptations. The custom three-layer width-96 transformer proxy is separate from the ViT-Tiny-dimension implementation.
- Published literature fields are unevenly reported. Unavailable and uncertain values remain unfilled, and estimates are explicitly labeled.
- Energy and direct training FLOPs are unavailable for most local experiments. The 1,122.1625285-second ResNet interval is a valid wall-clock observation with unresolved cause; adjusted efficiency figures are diagnostic only.
- Small seed counts and fixed splits limit uncertainty estimates. Short curves and threshold flags cannot establish a general saturation law.
- Windows/DirectML and macOS/MPS timing measurements are not a controlled hardware comparison.

## Conclusions not supported

The results do not establish universal diminishing returns, causal compute-efficiency rankings across paradigms, a single accuracy-per-compute score, or generalization beyond recorded tasks and protocols. Any extension beyond this evidence would require a separately scoped study.

The repository contains a reproducible analysis workflow, 25 PNG/SVG figure pairs, a canonical data index, data-quality and identifier audits, a field-level literature ledger, and the final Markdown/PDF report.
