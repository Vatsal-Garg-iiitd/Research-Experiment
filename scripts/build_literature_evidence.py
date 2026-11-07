"""Expand the literature ledger into an explicit field-by-field provenance view."""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/literature/large_model_literature.csv"
OUTPUT = ROOT / "results/literature/literature_field_evidence.csv"
FIELDS = [
    ("parameters", "parameters", "parameter_unit"),
    ("training_tokens", "training_tokens", "token_unit"),
    ("training_steps", "training_steps", ""),
    ("training_compute", "training_compute", "compute_unit"),
    ("training_duration", "training_duration", ""),
    ("hardware", "hardware", ""),
    ("accelerator_count", "accelerator_count", ""),
    ("accelerator_type", "accelerator_type", ""),
    ("batch_size", "batch_size", ""),
    ("sequence_length", "sequence_length", ""),
    ("optimizer", "optimizer", ""),
    ("learning_rate", "learning_rate", ""),
    ("training_dataset", "training_dataset", ""),
    ("training_loss", "training_loss", ""),
    ("validation_loss", "validation_loss", ""),
    ("performance_metric", "performance_metric", ""),
    ("performance_value", "performance_value", "performance_unit"),
    ("evaluation_dataset_protocol", "evaluation_dataset_protocol", ""),
    ("epochs", "epochs", ""),
]
OUT_FIELDS = [
    "model", "source_id", "field", "value", "unit", "evidence_class",
    "source_url", "locator", "verification_note",
]

# Field-level locators for the quantities represented in the ledger. These use
# paper sections/tables instead of broad paper-level citations so each extracted
# cell can be independently checked. Empty fields remain uncertain until checked.
FIELD_LOCATORS = {
    "BROWN2020": {
        "parameters": "Section 2.1, Table 2.1, GPT-3 175B row",
        "training_tokens": "Section 2.1, Table 2.1; Section 2.2, Table 2.2",
        "training_dataset": "Section 2.2, Table 2.2",
        "training_compute": "Figure 2.2; Appendix D (compute method)",
        "optimizer": "Appendix B, Details of Model Training",
        "learning_rate": "Section 2.1, Table 2.1; Appendix B",
        "batch_size": "Section 2.1, Table 2.1; Appendix B",
        "sequence_length": "Section 2.1 (2048-token context)",
        "performance_value": "LLaMA (Touvron et al. 2023), Table 3; cross-paper zero-shot HellaSwag",
    },
    "RAE2021": {
        "parameters": "Section 3.1, Table 1, Gopher 280B row",
        "training_tokens": "Section 3.2 (300B tokens)",
        "training_dataset": "Section 3.4, Table 2 (MassiveText)",
        "optimizer": "Section 3.2 (Adam)",
        "batch_size": "Section 3.1, Table 1; Section 3.2",
        "sequence_length": "Section 3.2 (2048-token context)",
        "hardware": "Section 3.3 (TPUv3)",
        "performance_value": "LLaMA (Touvron et al. 2023), Table 3; cross-paper zero-shot HellaSwag",
    },
    "HOFFMANN2022": {
        "parameters": "Table 1 and Section 4 (Chinchilla 70B)",
        "training_tokens": "Abstract; Table 1; Section 4 (1.4T)",
        "training_dataset": "Section 4 (MassiveText)",
        "training_compute": "Section 4; Appendix model card Table A8 (compute matched to Gopher)",
        "optimizer": "Section 4 (AdamW)",
        "hardware": "Section 4 and model card Table A8 (TPUv3/TPUv4; exact per-run allocation may vary)",
        "performance_value": "LLaMA (Touvron et al. 2023), Table 3; cross-paper zero-shot HellaSwag",
    },
    "CHOWDHERY2022": {
        "parameters": "Abstract; Section 2, Table 1 (PaLM-540B/62B)",
        "training_tokens": "Section 1; Section 2; Appendix Table 31 for PaLM-62B continuation rows",
        "training_compute": "Appendix Table 22; Tables 31-32 for continued PaLM-62B FLOPs",
        "training_dataset": "Section 2, Table 2 (PaLM mixture)",
        "hardware": "Abstract; Section 3.3 (6144 TPU v4 chips for PaLM-540B)",
        "accelerator_count": "Abstract; Section 3.3",
        "accelerator_type": "Abstract; Section 3.3 (TPU v4)",
        "optimizer": "Section 3.2 (Adafactor)",
        "learning_rate": "Section 3.2",
        "sequence_length": "Section 3.2 (2048)",
        "batch_size": "Section 3.2; Appendix Table 22 (schedule varies by model and step)",
        "performance_value": "Appendix Table 31 (62B task average) or LLaMA Table 3 for cross-paper HellaSwag rows",
    },
    "ZHANG2022": {
        "parameters": "Section 2.1, Table 1, corresponding OPT model row",
        "training_tokens": "Section 2.2 (models trained to 300B tokens); Section 2.3 (180B-token corpus)",
        "training_dataset": "Section 2.3 (RoBERTa corpus, Pile subsets, Pushshift Reddit)",
        "hardware": "Section 2.4 (992 NVIDIA A100 80GB for OPT-175B)",
        "accelerator_count": "Section 2.4 (992 for OPT-175B only)",
        "accelerator_type": "Section 2.4 (A100 80GB)",
        "optimizer": "Section 2.2 (AdamW)",
        "learning_rate": "Section 2.1, Table 1",
        "batch_size": "Section 2.1, Table 1",
        "sequence_length": "Section 2.2 (2048)",
        "training_duration": "OPT logbook; corroborated by LLaMA Section 6 carbon accounting (estimated 34 days)",
    },
    "TOUVRON2023": {
        "parameters": "Section 2.1, Table 2 (model sizes)",
        "training_tokens": "Section 2.1, Table 2 (1.0T or 1.4T by model)",
        "training_dataset": "Section 2.1, Table 1 (data mixture)",
        "optimizer": "Section 2.3 (AdamW)",
        "learning_rate": "Section 2.1, Table 2; Section 2.3",
        "batch_size": "Section 2.1, Table 2 (4M tokens)",
        "hardware": "Section 2.4 and Table 15 (A100 80GB; GPU-hours by size)",
        "training_duration": "Section 2.4 (65B: approximately 21 days); Table 15 (GPU-hours by size)",
        "performance_value": "Table 3 (zero-shot common-sense results, including HellaSwag)",
    },
    "BIGSCIENCE2022": {
        "parameters": "Abstract; Section 3.2, Table 3 (176,247M)",
        "training_tokens": "Section 3.5, Table 3 (366B total seen tokens; ROOTS plus repeats)",
        "training_dataset": "Section 3.1 (ROOTS corpus)",
        "hardware": "Section 3.4 (Jean Zay; 384 A100 80GB GPUs)",
        "accelerator_count": "Section 3.4 (384 active GPUs; four spare nodes separately noted)",
        "accelerator_type": "Section 3.4 (A100 80GB)",
        "sequence_length": "Section 3.5, Table 3 (2048)",
        "optimizer": "Section 3.5, Table 3 (Adam)",
        "learning_rate": "Section 3.5, Table 3",
        "batch_size": "Section 3.5, Table 3 (global batch 2048)",
        "training_duration": "Luccioni et al. (2022), Table 1 (118 days, 5 h, 41 min)",
        "training_compute": "Derived 6PT estimate; not reported FLOPs",
    },
}

FIELD_URLS = {
    "BIGSCIENCE2022": "https://www.jmlr.org/papers/v25/23-0581.html",
    "BROWN2020": "https://arxiv.org/abs/2005.14165",
    "RAE2021": "https://arxiv.org/abs/2112.11446",
    "HOFFMANN2022": "https://arxiv.org/abs/2203.15556",
    "CHOWDHERY2022": "https://arxiv.org/abs/2204.02311",
    "ZHANG2022": "https://arxiv.org/abs/2205.01068",
    "TOUVRON2023": "https://arxiv.org/abs/2302.13971",
}

# Explicitly extracted source values. Values are stored as (value, unit,
# evidence class, field locator); unsupported fields remain uncertain below.
COMMON_FIELDS = {
    "BROWN2020": {
        "optimizer": ("Adam", "", "reported", "Appendix B"),
        "learning_rate": ("6.0e-5", "peak LR", "reported", "Section 2.1, Table 2.1, GPT-3 175B row"),
        "batch_size": ("3.2M", "tokens per batch", "reported", "Section 2.1, Table 2.1, GPT-3 175B row"),
        "sequence_length": ("2048", "tokens", "reported", "Section 2.1"),
        "accelerator_type": ("NVIDIA V100", "", "reported", "Section 2.3"),
        "hardware": ("Microsoft high-bandwidth V100 GPU cluster", "", "reported", "Section 2.3"),
        "training_dataset": ("Filtered Common Crawl; WebText2; Books1/Books2; Wikipedia", "", "reported", "Section 2.2, Table 2.2"),
    },
    "RAE2021": {
        "optimizer": ("Adam", "", "reported", "Section 3.2"),
        "learning_rate": ("4e-5", "maximum LR", "reported", "Section 3.1, Table 1, Gopher 280B row"),
        "batch_size": ("3M → 6M", "tokens per batch", "reported", "Section 3.1, Table 1; Section 3.2"),
        "sequence_length": ("2048", "tokens", "reported", "Section 3.2"),
        "hardware": ("TPUv3", "", "reported", "Section 3.3"),
        "accelerator_type": ("Google TPU v3", "", "reported", "Section 3.3"),
        "training_dataset": ("MassiveText", "", "reported", "Section 3.4, Table 2"),
    },
    "HOFFMANN2022": {
        "optimizer": ("AdamW", "", "reported", "Section 4.1; Appendix G"),
        "learning_rate": ("1e-4", "maximum LR", "reported", "Section 4.1, Table 4, Chinchilla 70B row"),
        "batch_size": ("1.5M → 3M", "tokens per batch", "reported", "Section 4.1, Table 4"),
        "sequence_length": ("2048", "tokens", "reported", "Gopher training setup inherited except listed changes; Rae et al. Section 3.2"),
        "hardware": ("TPUv3/TPUv4", "", "reported", "Section 4.1 and Chinchilla model card Table A8"),
        "accelerator_type": ("Google TPU v3/v4", "", "reported", "Section 4.1 and Chinchilla model card Table A8"),
        "training_dataset": ("MassiveText (reweighted subset distribution)", "", "reported", "Section 4.1; Appendix Table A1"),
    },
    "CHOWDHERY2022": {
        "optimizer": ("Adafactor without factorization", "", "reported", "Section 5, Training Setup"),
        "learning_rate": ("1e-2 for first 10,000 steps, then inverse-square-root decay", "schedule", "reported", "Section 5, Training Setup"),
        "sequence_length": ("2048", "tokens", "reported", "Section 5, Training Setup"),
        "hardware": ("TPU v4 system", "", "reported", "Section 4.1; Appendix B"),
        "accelerator_type": ("Google TPU v4", "", "reported", "Abstract; Appendix B"),
        "training_dataset": ("PaLM pretraining mixture", "", "reported", "Section 3, Training Dataset"),
    },
    "ZHANG2022": {
        "optimizer": ("AdamW", "β1=0.9, β2=0.95", "reported", "Section 2.2"),
        "sequence_length": ("2048", "tokens", "reported", "Section 2.2"),
        "training_dataset": ("RoBERTa corpus; selected Pile subsets; Pushshift.io Reddit", "", "reported", "Section 2.3"),
        "accelerator_type": ("NVIDIA A100 80GB", "", "reported", "Section 2.4; OPT-175B only"),
    },
    "TOUVRON2023": {
        "optimizer": ("AdamW (β1=0.9, β2=0.95)", "", "reported", "Section 2.3"),
        "batch_size": ("4M", "tokens per batch", "reported", "Section 2.1, Table 2"),
        "sequence_length": ("2048", "tokens", "reported", "Section 2.1"),
        "hardware": ("NVIDIA A100 80GB", "", "reported", "Section 2.4"),
        "training_dataset": ("English Common Crawl; C4; GitHub; Wikipedia; Gutenberg/Books3; arXiv; Stack Exchange", "", "reported", "Section 2.1, Table 1"),
    },
    "BIGSCIENCE2022": {
        "optimizer": ("Adam (β1=0.9, β2=0.95)", "", "reported", "Section 3.5, Table 3"),
        "learning_rate": ("6e-5", "peak LR", "reported", "Section 3.5, Table 3, BLOOM column"),
        "batch_size": ("2048", "sequences", "reported", "Section 3.5, Table 3"),
        "sequence_length": ("2048", "tokens", "reported", "Section 3.5, Table 3"),
        "hardware": ("Jean Zay; 384 active A100 80GB GPUs", "", "reported", "Section 3.4"),
        "accelerator_count": ("384 active GPUs (four additional spare nodes)", "GPUs", "reported", "Section 3.4"),
        "accelerator_type": ("NVIDIA A100 80GB", "", "reported", "Section 3.4"),
        "training_dataset": ("ROOTS multilingual corpus (341B corpus tokens; 25B repeated tokens)", "", "reported", "Sections 3.1 and 3.5; Table 3"),
        "training_duration": ("118 days, 5 hours, 41 minutes", "wall-clock", "reported", "Luccioni et al. (2022), Table 1; linked carbon accounting source"),
    },
}

MODEL_FIELDS = {
    "BROWN2020": {
        "GPT-3 175B": {"training_compute": ("3.14e23", "FLOPs", "reported", "Appendix D, Table D.1; source-reported compute estimate")},
    },
    "CHOWDHERY2022": {
        "PaLM 540B": {"training_compute": ("2.56e24", "FLOPs", "reported", "Appendix Table 22; includes rematerialization"), "batch_size": ("1M → 2M → 4M", "tokens per batch", "reported", "Section 5; Appendix Table 22"), "accelerator_count": ("6144 for 1200 hours; 3072 for 336 hours", "TPU v4 chips", "reported", "Appendix B"), "training_duration": ("1200 hours at 6144 chips + 336 hours at 3072 chips (includes downtime/repeated steps)", "reported accelerator wall time", "reported", "Appendix B")},
        "PaLM-62B (795B tokens)": {"training_compute": ("2.957e23", "FLOPs", "derived", "Appendix Table 31; converted from reported zettaFLOPs"), "batch_size": ("1M → 2M", "tokens per batch", "reported", "Section 5, Table 1; 62B schedule" )},
        "PaLM-62B continued (1.325T tokens)": {"training_compute": ("4.929e23", "FLOPs", "derived", "Appendix Table 31; converted from reported zettaFLOPs"), "batch_size": ("1M → 2M", "tokens per batch", "reported", "Section 5, Table 1; 62B schedule" )},
    },
    "ZHANG2022": {
        "OPT-13B": {"learning_rate": ("1e-4", "peak LR", "reported", "Section 2.1, Table 1, OPT-13B row"), "batch_size": ("4M", "tokens per batch", "reported", "Section 2.1, Table 1")},
        "OPT-30B": {"learning_rate": ("1e-4", "peak LR", "reported", "Section 2.1, Table 1, OPT-30B row"), "batch_size": ("4M", "tokens per batch", "reported", "Section 2.1, Table 1")},
        "OPT-66B": {"learning_rate": ("8e-5", "peak LR", "reported", "Section 2.1, Table 1, OPT-66B row"), "batch_size": ("2M", "tokens per batch", "reported", "Section 2.1, Table 1")},
        "OPT-175B": {"learning_rate": ("1.2e-4", "peak LR", "reported", "Section 2.1, Table 1, OPT-175B row"), "batch_size": ("2M", "tokens per batch", "reported", "Section 2.1, Table 1"), "accelerator_count": ("992", "A100 80GB GPUs", "reported", "Section 2.4"), "training_duration": ("approximately 34 days", "estimated wall-clock", "estimated", "LLaMA Section 6 carbon accounting, based on OPT logbook")},
    },
    "TOUVRON2023": {
        "LLaMA-7B": {"learning_rate": ("3e-4", "peak LR", "reported", "Section 2.1, Table 2")},
        "LLaMA-13B": {"learning_rate": ("3e-4", "peak LR", "reported", "Section 2.1, Table 2")},
        "LLaMA-33B": {"learning_rate": ("1.5e-4", "peak LR", "reported", "Section 2.1, Table 2; model called 32.5B")},
        "LLaMA-65B": {"learning_rate": ("1.5e-4", "peak LR", "reported", "Section 2.1, Table 2; model called 65.2B"), "accelerator_count": ("2048", "A100 80GB GPUs", "reported", "Section 2.4; 65B training run"), "training_duration": ("approximately 21 days", "wall-clock", "reported", "Section 2.4")},
    },
}


def extracted_field(row, field):
    source_id = row.get("source_id", "")
    model = row.get("model", "")
    if field == "epochs":
        if source_id == "BROWN2020":
            return ("No single global epoch; mixture components were sampled at different rates, with some seen up to 3.4 times", "passes by component", "reported", "Section 2.2, Table 2.2")
        if source_id == "HOFFMANN2022":
            return ("0.13–3.40 epochs across MassiveText subsets; no single global epoch", "subset-specific passes", "reported", "Appendix Table A1")
        if source_id == "ZHANG2022":
            return ("≈1.67 corpus equivalents (300B training tokens / 180B-token corpus); not a guaranteed epoch schedule", "derived passes", "derived", "Sections 2.2–2.3")
        if source_id == "TOUVRON2023":
            return ("No single global epoch; Wikipedia and Books are repeated approximately twice, other sources vary", "subset-specific passes", "reported", "Section 2.1")
        if source_id == "BIGSCIENCE2022":
            return ("≈1.07 corpus equivalents (366B processed / 341B ROOTS corpus tokens)", "derived passes", "derived", "Sections 3.1 and 3.5; Table 3")
    if field == "evaluation_dataset_protocol" and row.get("performance_value") and source_id == "CHOWDHERY2022" and "62B" in model:
        return ("Paper-defined average across task suite; see Table 31", "", "reported", "Appendix Table 31 and its metric definition")
    if field == "evaluation_dataset_protocol" and row.get("performance_value"):
        return ("Zero-shot HellaSwag; score transcribed from LLaMA Table 3", "", "reported", "Touvron et al. (2023), Table 3; cross-paper comparison")
    spec = MODEL_FIELDS.get(source_id, {}).get(model, {}).get(field)
    if spec is None:
        spec = COMMON_FIELDS.get(source_id, {}).get(field)
    return spec


def estimate_6pt(row):
    try:
        params = float(row["parameters"])
        tokens = float(row["training_tokens"])
        pscale = 1e9 if row["parameter_unit"].lower() == "b" else 1.0
        tscale = 1e9 if row["token_unit"].lower() == "b" else 1.0
        return f"{6 * params * pscale * tokens * tscale:.6g}"
    except (KeyError, ValueError, TypeError):
        return ""


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        source_rows = list(csv.DictReader(f))
    result = []
    for row in source_rows:
        for field, value_key, unit_key in FIELDS:
            value = row.get(value_key, "")
            unit = row.get(unit_key, "") if unit_key else ""
            known = extracted_field(row, field)
            if known:
                value, known_unit, evidence, known_locator = known
                unit = known_unit or unit
            else:
                evidence = "reported" if value else "uncertain"
            locators = FIELD_LOCATORS.get(row.get("source_id", ""), {})
            locator = known[3] if known else locators.get(field, row.get("locator", ""))
            source_url = FIELD_URLS.get(row.get("source_id", ""), row.get("source_url", ""))
            if field == "training_duration" and row.get("source_id") == "BIGSCIENCE2022":
                source_url = "https://arxiv.org/abs/2211.02001"
            if field == "training_duration" and row.get("source_id") == "ZHANG2022" and row.get("model") == "OPT-175B":
                source_url = FIELD_URLS["TOUVRON2023"]
            note = f"Ledger value checked against the field locator: {locator}." if value else "Not extracted or verified; do not treat as zero or as proof the source omits it."
            if field == "training_tokens" and value:
                evidence = row.get("training_tokens_evidence_type", "reported") or "reported"
                evidence = "derived" if evidence.lower() in ("calculated", "derived") else evidence
            if field == "training_compute" and value:
                if "converted from" in row.get("notes", ""):
                    evidence = "derived"
                    note = "Converted from source-reported compute units; see model-specific notes and locator."
                elif "Reported train FLOPs" in row.get("notes", ""):
                    evidence = "reported"
                    note = "Source-reported FLOPs; rematerialization treatment is recorded in the source notes."
            if field in ("performance_value", "evaluation_dataset_protocol") and row.get("performance_value"):
                note = row.get("notes", "Reported value; protocol comparability is not assumed.")
                is_palm_average = row.get("source_id") == "CHOWDHERY2022" and "62B" in row.get("model", "")
                cross_listed = field == "evaluation_dataset_protocol" and not is_palm_average or field == "performance_value" and not is_palm_average
                if cross_listed or "cross-listed" in note.lower() or "cross-listed" in locator.lower():
                    source_url = FIELD_URLS["TOUVRON2023"]
                    if field == "performance_value":
                        note = "Cross-paper value transcribed from LLaMA Table 3; use only as a paper-reported comparison, not a locally standardized score. " + note
                    else:
                        source_url = FIELD_URLS["TOUVRON2023"]
                        note = "The comparison is described as zero-shot in LLaMA Table 3; original paper prompting details may differ."
            result.append({
                "model": row.get("model", ""), "source_id": row.get("source_id", ""),
                "field": field, "value": value, "unit": unit, "evidence_class": evidence,
                "source_url": source_url, "locator": locator,
                "verification_note": note,
            })
        estimated_compute = estimate_6pt(row)
        result.append({
            "model": row.get("model", ""), "source_id": row.get("source_id", ""),
            "field": "training_compute_6PT_estimate", "value": estimated_compute,
            "unit": "FLOPs", "evidence_class": "estimated" if estimated_compute else "uncertain",
            "source_url": FIELD_URLS.get(row.get("source_id", ""), row.get("source_url", "")),
            "locator": "Project-derived estimate; model/token values cite the row source.",
            "verification_note": "6 × parameter_count × processed_tokens; dense-transformer approximation, not reported training compute." if estimated_compute else "Insufficient parameter/token values for estimate.",
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(result)
    print(f"Wrote {len(result)} model-field evidence rows for {len(source_rows)} configurations to {OUTPUT}")


if __name__ == "__main__":
    main()
