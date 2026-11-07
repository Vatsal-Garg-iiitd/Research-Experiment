"""Train a canonical-dimension ViT-Tiny on Fashion-MNIST, resumably."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.models import VisionTransformer
from torchvision.transforms import ToTensor

from backend import backend_name, get_device

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw/vit_tiny_scaling.csv"
TEST = ROOT / "results/raw/vit_tiny_test_results.csv"
CONFIGS = ROOT / "results/configurations"
CHECKPOINTS = ROOT / "results/checkpoints/vit_tiny"
DEVICE = get_device()
BUDGETS = (1, 3, 5, 10, 20, 30)


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["experiment_id"] for row in csv.DictReader(f)}


def build_model() -> VisionTransformer:
    # Canonical ViT-Tiny scale: 192 hidden width, 12 blocks, 3 heads,
    # 768 MLP width. Fashion-MNIST is 28x28, so 4x4 patches are used.
    model = VisionTransformer(
        image_size=28,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=10,
    )
    model.conv_proj = nn.Conv2d(1, 192, kernel_size=4, stride=4)
    return model


def load_data(train_size: int):
    train = FashionMNIST(str(ROOT / "results/cache"), train=True, download=False, transform=ToTensor())
    test = FashionMNIST(str(ROOT / "results/cache"), train=False, download=False, transform=ToTensor())
    indices = np.random.default_rng(20261106).permutation(len(train))[: train_size + 2000]
    images = torch.stack([train[int(i)][0] for i in indices])
    labels = torch.tensor([train[int(i)][1] for i in indices], dtype=torch.long)
    test_images = torch.stack([test[i][0] for i in range(len(test))])
    test_labels = torch.tensor([test[i][1] for i in range(len(test))], dtype=torch.long)
    split = hashlib.sha256(indices.tobytes()).hexdigest()[:16]
    return images[:train_size], labels[:train_size], images[train_size:], labels[train_size:], test_images, test_labels, split


def evaluate(model, images, labels, batch_size=128):
    model.eval()
    loss_sum = correct = count = 0
    predictions = []
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            x = images[start : start + batch_size].to(DEVICE)
            y = labels[start : start + batch_size].to(DEVICE)
            logits = model(x)
            loss_sum += float(nn.functional.cross_entropy(logits, y, reduction="sum").cpu())
            pred = logits.argmax(1)
            correct += int((pred == y).sum().cpu())
            count += len(y)
            predictions.extend(pred.cpu().tolist())
    return loss_sum / count, correct / count, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(8)
    x, y, xv, yv, xt, yt, split = load_data(args.train_size)
    budgets = tuple(b for b in BUDGETS if b <= args.max_epochs)
    if not budgets or budgets[-1] != args.max_epochs:
        budgets = tuple(sorted(set((*budgets, args.max_epochs))))
    parameter_count = sum(p.numel() for p in build_model().parameters())

    for seed in args.seeds:
        run = f"vit_tiny_seed{seed}_n{args.train_size}_mac_v1"
        checkpoint = CHECKPOINTS / f"{run}.pt"
        config_path = CONFIGS / f"{run}.json"
        config = {
            "experiment_id": run,
            "model": "ViT-Tiny",
            "architecture": "ViT-Tiny dimensions adapted to one-channel 28x28 input",
            "image_resolution": [28, 28],
            "patch_size": 4,
            "embedding_dimension": 192,
            "attention_heads": 3,
            "transformer_blocks": 12,
            "mlp_dimension": 768,
            "parameter_count": parameter_count,
            "classifier_head": "linear 192-to-10",
            "dataset": "Fashion-MNIST",
            "split_id": split,
            "seed": seed,
            "train_size": args.train_size,
            "validation_size": len(yv),
            "test_size": len(yt),
            "budget_epochs": list(budgets),
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "device": str(DEVICE),
            "backend": backend_name(DEVICE),
            "test_usage": "official test evaluated only at final requested epoch",
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        torch.manual_seed(seed)
        model = build_model().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        start_epoch = 0
        cumulative_seconds = 0.0
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"])
            cumulative_seconds = float(state["cumulative_seconds"])

        recorded = ids(RAW)
        for epoch in range(start_epoch + 1, args.max_epochs + 1):
            torch.manual_seed(seed + epoch)
            order = torch.randperm(len(x))
            model.train()
            loss_sum = correct = seen = 0
            started = time.perf_counter()
            for batch_ids in order.split(128):
                xb, yb = x[batch_ids].to(DEVICE), y[batch_ids].to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(yb)
                correct += int((logits.detach().argmax(1) == yb).sum().cpu())
                seen += len(yb)
            train_seconds = time.perf_counter() - started
            cumulative_seconds += train_seconds
            val_loss, val_accuracy, _ = evaluate(model, xv, yv)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temp = checkpoint.with_suffix(".tmp")
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "cumulative_seconds": cumulative_seconds}, temp)
            temp.replace(checkpoint)
            experiment_id = f"{run}_epoch{epoch}"
            if experiment_id not in recorded:
                append(RAW, {
                    "experiment_id": experiment_id,
                    "run_id": run,
                    "model_name": "ViT-Tiny",
                    "paradigm": "supervised",
                    "dataset": "Fashion-MNIST",
                    "split_id": split,
                    "seed": seed,
                    "budget_type": "epochs",
                    "budget_value": epoch,
                    "optimizer_steps": epoch * ((args.train_size + 127) // 128),
                    "training_time_seconds": train_seconds,
                    "cumulative_training_time_seconds": cumulative_seconds,
                    "evaluation_time_seconds": "",
                    "training_loss": loss_sum / seen,
                    "training_accuracy": correct / seen,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy,
                    "parameter_count": parameter_count,
                    "backend": backend_name(DEVICE),
                    "peak_memory_bytes": psutil.Process().memory_info().rss,
                    "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                    "configuration_path": str(config_path.relative_to(ROOT)),
                    "status": "completed",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
                recorded.add(experiment_id)
            print(f"ViT-Tiny seed={seed} epoch={epoch} val_acc={val_accuracy:.4f} val_loss={val_loss:.4f} train_s={train_seconds:.1f}", flush=True)

        test_id = f"{run}_test_epoch{args.max_epochs}"
        if test_id not in ids(TEST):
            test_loss, test_accuracy, predictions = evaluate(model, xt, yt)
            append(TEST, {
                "experiment_id": test_id,
                "model_name": "ViT-Tiny",
                "seed": seed,
                "split_id": split,
                "epochs": args.max_epochs,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_size": len(yt),
                "parameter_count": parameter_count,
                "macro_precision": float(__import__("sklearn.metrics", fromlist=["precision_score"]).precision_score(yt.numpy(), predictions, average="macro", zero_division=0)),
                "macro_recall": float(__import__("sklearn.metrics", fromlist=["recall_score"]).recall_score(yt.numpy(), predictions, average="macro", zero_division=0)),
                "macro_f1": float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(yt.numpy(), predictions, average="macro", zero_division=0)),
                "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                "backend": backend_name(DEVICE),
                "status": "completed",
            })
        del model, optimizer


if __name__ == "__main__":
    main()
