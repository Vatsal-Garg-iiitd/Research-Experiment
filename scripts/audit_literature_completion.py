"""Verify literature-only objectives from the primary-source evidence ledger."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/literature/large_model_literature.csv"
EVIDENCE = ROOT / "results/literature/literature_field_evidence.csv"
OUTPUT = ROOT / "results/processed/literature_completion_audit.csv"
EXPECTED_FIELDS = {
    "parameters", "training_tokens", "training_steps", "training_compute",
    "training_duration", "hardware", "accelerator_count", "accelerator_type",
    "batch_size", "sequence_length", "optimizer", "learning_rate",
    "training_dataset", "training_loss", "validation_loss", "performance_metric",
    "performance_value", "evaluation_dataset_protocol", "epochs",
    "training_compute_6PT_estimate",
}
CLASSES = {"reported", "derived", "estimated", "uncertain", "unavailable"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    source_rows = read(SOURCE)
    evidence_rows = read(EVIDENCE)
    source_by_model = {row["model"]: row for row in source_rows}
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in evidence_rows:
        by_model[row["model"]].append(row)

    if len(source_rows) != 15 or len(source_by_model) != len(source_rows):
        raise SystemExit("Literature source ledger must contain 15 unique configurations.")
    if set(by_model) != set(source_by_model):
        raise SystemExit("Source and field-evidence model sets differ.")

    out = []
    for model, source in source_by_model.items():
        fields = by_model[model]
        names = [row["field"] for row in fields]
        classes = Counter(row["evidence_class"] for row in fields)
        errors = []
        if len(fields) != len(EXPECTED_FIELDS) or set(names) != EXPECTED_FIELDS or len(names) != len(set(names)):
            errors.append("field coverage does not equal the 20-field schema")
        if any(row.get("evidence_class") not in CLASSES for row in fields):
            errors.append("unknown evidence class")
        if any(not row.get("source_url") or not row.get("locator") for row in fields):
            errors.append("missing source URL or field locator")

        field_map = {row["field"]: row for row in fields}
        mismatched_sources = [
            row["field"] for row in fields
            if row.get("source_id") != source.get("source_id")
        ]
        if mismatched_sources:
            errors.append("field rows reference a different source ID: " + ", ".join(mismatched_sources))
        parameters = field_map.get("parameters", {})
        tokens = field_map.get("training_tokens", {})
        if not parameters.get("value") or parameters.get("evidence_class") != "reported":
            errors.append("parameter count is not source-reported")
        if not tokens.get("value") or tokens.get("evidence_class") not in {"reported", "derived"}:
            errors.append("training-token quantity is not reported or transparently derived")
        if not source.get("source_url") or not source.get("source_id") or not source.get("locator"):
            errors.append("source ledger lacks source identity, URL, or locator")

        # A missing quantity is acceptable only when it is explicitly classified
        # as uncertain/unavailable; it is never converted to zero or hidden.
        unclassified_blanks = [
            row["field"] for row in fields
            if not row.get("value") and row.get("evidence_class") not in {"uncertain", "unavailable"}
        ]
        if unclassified_blanks:
            errors.append("blank fields lack an uncertainty/unavailability class: " + ", ".join(unclassified_blanks))

        # Scores are required only where a numerical score can be attributed to
        # the cited protocol. Otherwise the missing score remains explicit.
        score = field_map.get("performance_value", {})
        metric = field_map.get("performance_metric", {})
        score_status = "reported_score" if score.get("value") and metric.get("value") else "score_gap_explicitly_classified"
        status = "completed_literature_only" if not errors else "needs_review"
        out.append({
            "model": model,
            "source_id": source.get("source_id", ""),
            "primary_source_url": source.get("source_url", ""),
            "source_fields": len(fields),
            "expected_fields": len(EXPECTED_FIELDS),
            "reported_fields": classes["reported"],
            "derived_fields": classes["derived"],
            "estimated_fields": classes["estimated"],
            "uncertain_fields": classes["uncertain"],
            "unavailable_fields": classes["unavailable"],
            "parameter_evidence": parameters.get("evidence_class", ""),
            "token_evidence": tokens.get("evidence_class", ""),
            "benchmark_evidence": score_status,
            "status": status,
            "validation_errors": "; ".join(errors),
            "completion_basis": "Literature review objective: source identity verified, core model scale/token evidence present, all fields classified and source-located; unresolved source data remain disclosed. No local training or reproduction is implied.",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    columns = list(out[0])
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(out)
    failed = [row for row in out if row["status"] != "completed_literature_only"]
    print(f"Verified {len(out) - len(failed)}/{len(out)} literature-only objectives; wrote {OUTPUT}")
    if failed:
        for row in failed:
            print(row["model"], row["validation_errors"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
