"""Compute adjacent-budget marginal gains from measured experiment CSV data."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


GROUP_FIELDS = ("run_id", "paradigm", "model", "dataset", "seed", "metric_name")


def number(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def analyze(rows: list[dict[str, str]], lower_is_better: bool) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status", "").startswith("superseded"):
            continue
        key = tuple(row.get(name, "") for name in GROUP_FIELDS)
        groups[key].append(row)

    output: list[dict[str, str]] = []
    for key, series in groups.items():
        series.sort(key=lambda row: (number(row.get("budget_value")) is None,
                                     number(row.get("budget_value")) or 0.0))
        previous = None
        small_gain_counts = {0.005: 0, 0.01: 0, 0.02: 0}
        for row in series:
            score = number(row.get("metric_value"))
            cost = number(row.get("cumulative_train_seconds"))
            gain = efficiency = marginal_cost = None
            relative_gain = None
            if previous is not None and score is not None:
                prev_score = number(previous.get("metric_value"))
                prev_cost = number(previous.get("cumulative_train_seconds"))
                if prev_score is not None:
                    gain = prev_score - score if lower_is_better else score - prev_score
                    if abs(prev_score) > 1e-15:
                        relative_gain = gain / abs(prev_score)
                if cost is not None and prev_cost is not None:
                    marginal_cost = cost - prev_cost
                    if gain is not None and marginal_cost > 0:
                        efficiency = gain / marginal_cost
            result = dict(row)
            result.update({
                "marginal_gain": "" if gain is None else f"{gain:.12g}",
                "marginal_cost_seconds": "" if marginal_cost is None else f"{marginal_cost:.12g}",
                "marginal_efficiency_per_second": "" if efficiency is None else f"{efficiency:.12g}",
                "relative_improvement": "" if relative_gain is None else f"{relative_gain:.12g}",
            })
            for threshold in small_gain_counts:
                if relative_gain is not None and 0 <= relative_gain < threshold:
                    small_gain_counts[threshold] += 1
                else:
                    small_gain_counts[threshold] = 0
                suffix = str(threshold * 100).replace(".", "_")
                result[f"saturation_{suffix}pct"] = str(small_gain_counts[threshold] >= 2).lower()
            output.append(result)
            if score is not None:
                previous = row
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lower-is-better", action="store_true",
                        help="Treat the metric as loss/perplexity and calculate reductions.")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("Input CSV has no result rows.")
    required = set(GROUP_FIELDS) | {"budget_value", "metric_value", "cumulative_train_seconds"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")
    result = analyze(rows, args.lower_is_better)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"Wrote {len(result)} analyzed observations to {args.output}")


if __name__ == "__main__":
    main()
