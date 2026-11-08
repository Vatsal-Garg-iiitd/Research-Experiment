"""Evaluate the complete WikiText-2 raw-v1 test text with controlled context."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend import backend_name, get_device

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/llm/macos_full_wikitext2_evaluations.csv"
PROCESS = psutil.Process()
MODELS = [
    ("Pythia-70M", "EleutherAI/pythia-70m", "step143000", 70_426_624),
    ("Pythia-160M", "EleutherAI/pythia-160m", "step143000", 162_322_944),
    ("Pythia-410M", "EleutherAI/pythia-410m", "step143000", 405_334_016),
    ("Pythia-1B", "EleutherAI/pythia-1b", "step143000", 1_011_382_272),
    ("Pythia-1.4B", "EleutherAI/pythia-1.4b", "step143000", 1_414_647_808),
    ("Pythia-2.8B", "EleutherAI/pythia-2.8b", "step143000", 2_805_067_776),
    ("Pythia-6.9B", "EleutherAI/pythia-6.9b", "step143000", 6_915_155_968),
    ("Pythia-12B", "EleutherAI/pythia-12b", "step143000", 12_002_822_144),
    ("GPT-2 Small", "openai-community/gpt2", "main", 124_439_808),
    ("GPT-2 Medium", "openai-community/gpt2-medium", "main", 354_823_168),
    ("GPT-2 Large", "openai-community/gpt2-large", "main", 774_030_080),
    ("GPT-2 XL", "openai-community/gpt2-xl", "main", 1_557_611_200),
    ("OPT-125M", "facebook/opt-125m", "main", 125_239_808),
    ("OPT-350M", "facebook/opt-350m", "main", 331_000_000),
    ("OPT-1.3B", "facebook/opt-1.3b", "main", 1_315_000_000),
]
DEFAULT_MODELS = ["Pythia-70M", "Pythia-160M", "Pythia-410M", "GPT-2 Small", "OPT-125M", "OPT-350M"]
FIELDS = [
    "experiment_id", "model_name", "model_repo", "model_revision", "parameter_count",
    "dataset", "dataset_revision", "text_sha256", "text_characters", "text_utf8_bytes",
    "tokenizer", "token_count", "context_length", "stride", "scored_tokens", "scored_utf8_bytes",
    "negative_log_likelihood", "perplexity", "bits_per_byte", "precision", "device",
    "backend", "model_load_seconds", "evaluation_seconds", "tokens_per_second",
    "batch_size", "weight_memory_estimate_bytes", "ram_bytes_after_eval", "peak_ram_bytes", "status", "notes", "timestamp_utc",
]


def upsert(row: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if OUT.exists():
        with OUT.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    for i, old in enumerate(rows):
        if old.get("experiment_id") == row["experiment_id"]:
            rows[i] = row
            break
    else:
        rows.append(row)
    temp = OUT.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(OUT)


def char_to_byte_offsets(text: str) -> list[int]:
    result = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        result.append(total)
    return result


def evaluate_model(name, repo, revision, parameter_count, text, text_hash, max_context, stride, device):
    experiment_id = f"{name.lower().replace(' ', '_')}_{revision}_wikitext2_full_ctx{max_context}_stride{stride}_mac_v1"
    existing = []
    if OUT.exists():
        with OUT.open(encoding="utf-8-sig", newline="") as f:
            existing = [r for r in csv.DictReader(f) if r.get("experiment_id") == experiment_id]
    if existing and existing[-1].get("status") == "completed":
        print(name, "full-test evaluation already completed; skipped", flush=True)
        return

    row = dict.fromkeys(FIELDS, "")
    row.update({
        "experiment_id": experiment_id, "model_name": name, "model_repo": repo,
        "model_revision": revision, "parameter_count": parameter_count,
        "dataset": "WikiText-2 raw-v1 test", "dataset_revision": "Salesforce/wikitext@wikitext-2-raw-v1",
        "text_sha256": text_hash, "text_characters": len(text), "text_utf8_bytes": len(text.encode("utf-8")),
        "tokenizer": repo, "context_length": max_context, "stride": stride,
        "precision": "float32", "device": str(device), "backend": backend_name(device),
        "batch_size": 1, "weight_memory_estimate_bytes": parameter_count * 4,
        "ram_bytes_after_eval": "", "status": "running", "notes": "Full official test split; FP32; batch size 1; non-overlapping score ownership with rolling context windows. Weight-memory estimate excludes activations and runtime overhead.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    upsert(row)
    tokenizer = model = None
    try:
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, torch_dtype=torch.float32).to(device)
        model.eval()
        load_seconds = time.perf_counter() - started
        max_positions = int(getattr(model.config, "max_position_embeddings", max_context) or max_context)
        context = min(max_context, max_positions)
        if stride > context:
            raise ValueError(f"stride {stride} exceeds effective context {context}")
        # We tokenize the entire corpus to preserve offsets, then explicitly
        # slice bounded windows below. Raise the tokenizer warning threshold so
        # it does not mislead users into thinking a 285K-token tensor reaches the model.
        tokenizer.model_max_length = max(len(text) + 1, context)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        token_ids = encoded["input_ids"][0]
        offsets = encoded["offset_mapping"][0].tolist()
        byte_offsets = char_to_byte_offsets(text)
        total_nll = 0.0
        scored_tokens = 0
        byte_intervals = []
        previous_end = 0
        started = time.perf_counter()
        with torch.inference_mode():
            begin = 0
            while begin < len(token_ids):
                end = min(begin + context, len(token_ids))
                target_start = max(begin + 1, previous_end)
                if target_start < end:
                    ids = token_ids[begin:end].unsqueeze(0).to(device)
                    labels = ids.clone()
                    labels[:, : target_start - begin] = -100
                    output = model(input_ids=ids, labels=labels)
                    count = int((labels[:, 1:] != -100).sum().cpu())
                    total_nll += float(output.loss.detach().cpu()) * count
                    scored_tokens += count
                    for char_start, char_end in offsets[target_start:end]:
                        if char_end > char_start:
                            byte_intervals.append((byte_offsets[char_start], byte_offsets[char_end]))
                previous_end = end
                if end == len(token_ids):
                    break
                begin += stride
        evaluation_seconds = time.perf_counter() - started
        if not scored_tokens or not byte_intervals:
            raise RuntimeError("No scorable tokens or tokenizer byte offsets were returned")
        byte_intervals.sort()
        scored_bytes = 0
        covered_start, covered_end = byte_intervals[0]
        for start_byte, end_byte in byte_intervals[1:]:
            if start_byte <= covered_end:
                covered_end = max(covered_end, end_byte)
            else:
                scored_bytes += covered_end - covered_start
                covered_start, covered_end = start_byte, end_byte
        scored_bytes += covered_end - covered_start
        nll = total_nll / scored_tokens
        row.update({
            "token_count": len(token_ids), "context_length": context, "scored_tokens": scored_tokens,
            "scored_utf8_bytes": scored_bytes, "negative_log_likelihood": nll,
            "perplexity": math.exp(nll), "bits_per_byte": total_nll / (math.log(2) * scored_bytes),
            "model_load_seconds": load_seconds, "evaluation_seconds": evaluation_seconds,
            "tokens_per_second": scored_tokens / evaluation_seconds if evaluation_seconds else "",
            "ram_bytes_after_eval": PROCESS.memory_info().rss, "status": "completed",
            "notes": f"Full test text; FP32; context={context}; stride={stride}; each token scored once after at least one context token; bits/byte use the union of UTF-8 byte spans for scored tokens to avoid overlap double-counting.",
        })
    except Exception as exc:
        row.update({"status": "failed", "notes": f"{type(exc).__name__}: {exc}"[:1800]})
    finally:
        upsert(row)
        del model, tokenizer
        if device.type == "mps":
            torch.mps.empty_cache()
    print(name, row["status"], "tokens", row.get("scored_tokens"), "ppl", row.get("perplexity"), row.get("notes", "")[:180], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()
    device = get_device()
    torch.set_num_threads(8)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(line for line in dataset["text"] if line.strip())
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    for name, repo, revision, count in MODELS:
        if name in args.models:
            evaluate_model(name, repo, revision, count, text, text_hash, args.context, args.stride, device)


if __name__ == "__main__":
    main()
