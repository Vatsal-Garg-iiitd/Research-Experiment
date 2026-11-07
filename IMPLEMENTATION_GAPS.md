# Implementation gaps

## Implemented and exercised

- Existing proposal code and artifacts preserved; additional platform setup, result repair snapshots, model checkpoints, and run notes are saved under the project.
- Resumable multi-budget code exists for neural supervised models, Random Forest, clustering, pseudo-labeling and SSL, AE/VAE, LabelPropagation, DBSCAN, and RL. New output schemas retain actual examples/steps and support adding columns without corrupting older rows.
- macOS backend selection and MPS-compatible execution are implemented for neural scripts. Classical and RL experiments use CPU. Historical DirectML runs remain separately identified.
- Full-corpus WikiText-2 checkpoint evaluator and unbounded zero-shot harness execution are implemented and exercised. Pythia checkpoint trajectories use a common short prefix; no model pretraining is claimed.
- Literature extraction creates a field-level evidence ledger with source URLs, locators, attribution, and evidence classes.
- Canonical result index, strengthened statistical analysis, completion matrix, run registry, and output-validation scripts are present. Final independent validation is in progress and its exact outcome will be recorded in `FINAL_INDEPENDENT_AUDIT.md`.

## Remaining implementation work

- Full matched multi-budget/seed coverage across the 20 conventional entries remains incomplete. The compact transformer is not standard ViT-Tiny, and SSL recipes are task adaptations.
- The validator checks CSV parseability, numeric finiteness, key output counts, and selected full-WikiText, harness, literature-ledger, registry, and matrix invariants. It does not verify every narrative result against every source row or verify third-party checkpoint contents.
- Conventional inference timing, FLOPs, controlled MPS/CPU comparison, energy measurement, and NPU runtime execution are incomplete.
- Eight requested LLM configurations lack full-corpus local evaluation. Only Pythia-70M has a full zero-shot task run; GPT-2 Small and OPT-125M have smoke harness subsets.
- Literature page-level extraction and independent locator/value review remain incomplete for uncertain fields.
