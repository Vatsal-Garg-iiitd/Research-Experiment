"""Run small, measured pilots for supervised, unsupervised, SSL, RL and Pythia-70M."""
from __future__ import annotations

import argparse
import csv
import json
import os
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/raw/experiment_results.csv"
SEED = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def memory_bytes() -> int:
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def log_rows(rows: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    exists = OUT.exists() and OUT.stat().st_size > 0
    with OUT.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def result_row(run, paradigm, model, dataset, budget_unit, budget, train_s,
               cumulative_s, eval_s, metric, score, steps, params, seed=SEED, **extra):
    return {
        "run_id": run, "paradigm": paradigm, "model": model, "dataset": dataset,
        "split_hash": "fashion_mnist_train_random_12000_seed11" if dataset == "Fashion-MNIST" else f"seed_{seed}", "seed": seed,
        "budget_unit": budget_unit, "budget_value": budget, "train_seconds": train_s,
        "cumulative_train_seconds": cumulative_s, "eval_seconds": eval_s,
        "metric_name": metric, "metric_value": score, "train_steps": steps,
        "parameter_count": params, "peak_ram_bytes": memory_bytes(),
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else "",
        "hardware": f"{DEVICE}; {torch.__version__}; CPU-only" if DEVICE.type == "cpu" else str(DEVICE),
        "status": "completed", "details": json.dumps(extra, sort_keys=True),
    }


def load_data(cache: Path):
    train = FashionMNIST(str(cache), train=True, download=True, transform=ToTensor())
    test = FashionMNIST(str(cache), train=False, download=True, transform=ToTensor())
    order = np.random.default_rng(SEED).permutation(len(train))[:12000]
    train = Subset(train, order.tolist())
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train, batch_size=128, shuffle=True, generator=generator, num_workers=0)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
    return train, loader, test_loader


def evaluate_classifier(model, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            correct += int((logits.argmax(1).cpu() == y).sum())
            total += len(y)
    return correct / total


def supervised(train_loader, test_loader):
    seed_everything(SEED)
    model = SmallMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rows, elapsed, steps = [], 0.0, 0
    for epoch in range(1, 4):
        tic = time.perf_counter()
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            optimizer.step()
            steps += 1
        duration = time.perf_counter() - tic
        elapsed += duration
        tic = time.perf_counter()
        score = evaluate_classifier(model, test_loader)
        eval_s = time.perf_counter() - tic
        rows.append(result_row("pilot_fashion_mlp", "supervised", "MLP", "Fashion-MNIST",
                               "epochs", epoch, duration, elapsed, eval_s, "test_accuracy",
                               score, steps, sum(p.numel() for p in model.parameters())))
    return rows


def semi_supervised(dataset, test_loader, confidence_threshold=0.95):
    # Pseudo-labeling with a fixed 5% labeled subset, high-confidence additions each epoch.
    seed_everything(SEED)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    rng = np.random.default_rng(SEED)
    labeled_idx = rng.choice(len(dataset), size=max(1, int(0.05 * len(dataset))), replace=False)
    is_labeled = torch.zeros(len(dataset), dtype=torch.bool)
    is_labeled[labeled_idx] = True
    y_train = torch.full_like(labels, -1)
    y_train[labeled_idx] = labels[labeled_idx]
    model = SmallMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rows, elapsed, steps = [], 0.0, 0
    for epoch in range(1, 4):
        x = images[is_labeled]
        y = y_train[is_labeled]
        loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)
        tic = time.perf_counter()
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()
            steps += 1
        model.eval()
        unlabeled_idx = (~is_labeled).nonzero(as_tuple=True)[0]
        unlabeled_loader = DataLoader(TensorDataset(unlabeled_idx), batch_size=256,
                                      shuffle=False, num_workers=0)
        pseudo_y = []
        pseudo_indices = []
        with torch.inference_mode():
            for (batch_idx,) in unlabeled_loader:
                batch_x = images[batch_idx]
                probs = model(batch_x.to(DEVICE)).softmax(1)
                confidence, prediction = probs.max(1)
                keep = confidence >= confidence_threshold
                if keep.any():
                    pseudo_indices.append(batch_idx[keep].cpu())
                    pseudo_y.append(prediction[keep].cpu())
        pseudo_added = 0
        if pseudo_indices:
            ix, py = torch.cat(pseudo_indices), torch.cat(pseudo_y)
            is_labeled[ix] = True
            y_train[ix] = py
            pseudo_added = len(ix)
        duration = time.perf_counter() - tic
        elapsed += duration
        tic = time.perf_counter()
        score = evaluate_classifier(model, test_loader)
        eval_s = time.perf_counter() - tic
        rows.append(result_row(f"pilot_fashion_pseudolabel_seeded_t{int(confidence_threshold*100):02d}", "semi_supervised", "Pseudo-Labeling",
                               "Fashion-MNIST", "epochs", epoch, duration, elapsed, eval_s,
                               "test_accuracy", score, steps,
                               sum(p.numel() for p in model.parameters()),
                               confidence_threshold=confidence_threshold,
                               labeled_count=int(is_labeled.sum()), pseudo_added=pseudo_added))
    return rows


def unsupervised(dataset):
    count = 2500
    x = np.stack([dataset[i][0].numpy().reshape(-1) for i in range(count)]).astype(np.float32)
    y = np.array([dataset[i][1] for i in range(count)])
    tic = time.perf_counter()
    km = KMeans(n_clusters=10, n_init=1, max_iter=20, random_state=SEED)
    clusters = km.fit_predict(x)
    duration = time.perf_counter() - tic
    tic = time.perf_counter()
    ari = adjusted_rand_score(y, clusters)
    eval_s = time.perf_counter() - tic
    return result_row("pilot_fashion_kmeans_v2", "unsupervised", "K-Means", "Fashion-MNIST",
                      "iterations", int(km.n_iter_), duration, duration, eval_s,
                      "adjusted_rand_index", ari,
                      int(km.n_iter_), int(km.cluster_centers_.size),
                      sample_count=count, initialization_count=1)


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.layers(x)


def dqn():
    import gymnasium as gym
    seed_everything(SEED)
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    env.action_space.seed(SEED)
    model, target = QNet().to(DEVICE), QNet().to(DEVICE)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay = deque(maxlen=10000)
    rng = random.Random(SEED)
    state, _ = env.reset(seed=SEED)
    returns, episode_return, elapsed, updates, rows = [], 0.0, 0.0, 0, []
    tic = time.perf_counter()
    for step in range(1, 5001):
        epsilon = max(0.05, 1.0 - 0.95 * step / 5000)
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.inference_mode():
                action = int(model(torch.tensor(state, dtype=torch.float32, device=DEVICE)[None]).argmax(1))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.append((state, action, reward, next_state, float(terminated)))
        episode_return += reward
        state = next_state
        if done:
            returns.append(episode_return)
            state, _ = env.reset()
            episode_return = 0.0
        if len(replay) >= 256:
            batch = rng.sample(replay, 64)
            s, a, r, ns, d = zip(*batch)
            s = torch.tensor(np.asarray(s), dtype=torch.float32, device=DEVICE)
            ns = torch.tensor(np.asarray(ns), dtype=torch.float32, device=DEVICE)
            a = torch.tensor(a, dtype=torch.int64, device=DEVICE)
            r = torch.tensor(r, dtype=torch.float32, device=DEVICE)
            d = torch.tensor(d, dtype=torch.float32, device=DEVICE)
            q = model(s).gather(1, a[:, None]).squeeze(1)
            with torch.no_grad():
                y = r + 0.99 * (1-d) * target(ns).max(1).values
            loss = nn.functional.smooth_l1_loss(q, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
        if step % 250 == 0:
            target.load_state_dict(model.state_dict())
        if step in (1000, 2500, 5000):
            duration = time.perf_counter() - tic
            elapsed += duration
            tic = time.perf_counter()
            eval_tic = time.perf_counter()
            eval_returns = []
            for ep in range(5):
                s, _ = eval_env.reset(seed=SEED + ep + step)
                total = 0.0
                for _ in range(500):
                    with torch.inference_mode():
                        act = int(model(torch.tensor(s, dtype=torch.float32, device=DEVICE)[None]).argmax(1))
                    s, rew, term, trunc, _ = eval_env.step(act)
                    total += rew
                    if term or trunc:
                        break
                eval_returns.append(total)
            eval_s = time.perf_counter() - eval_tic
            rows.append(result_row("pilot_cartpole_dqn_v3", "reinforcement_learning", "DQN",
                                   "CartPole-v1", "environment_steps", step, duration, elapsed,
                                   eval_s, "mean_eval_return_5_episodes", float(np.mean(eval_returns)),
                                   updates, sum(p.numel() for p in model.parameters()),
                                   seed=SEED, training_episodes=len(returns), separate_evaluation_environment=True))
    env.close()
    eval_env.close()
    return rows


def pythia():
    from datasets import Dataset, config as datasets_config, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "EleutherAI/pythia-70m"
    revision = "step143000"
    load_start = time.perf_counter()
    local_only = bool(os.environ.get("HF_HUB_OFFLINE"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                  local_files_only=local_only).to(DEVICE).eval()
    load_s = time.perf_counter() - load_start
    cached_arrow = list((Path(datasets_config.HF_DATASETS_CACHE) /
                         "Salesforce___wikitext" / "wikitext-2-raw-v1" / "0.0.0").glob(
                             "*/wikitext-test.arrow"))
    data = (Dataset.from_file(str(cached_arrow[-1])) if cached_arrow else
            load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test"))
    text = "\n".join(s for s in data["text"] if s.strip())[:200000]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    tic = time.perf_counter()
    with torch.inference_mode():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    eval_s = time.perf_counter() - tic
    ppl = math.exp(float(loss))
    (ROOT / "results/llm").mkdir(parents=True, exist_ok=True)
    with (ROOT / "results/llm/llm_checkpoint_results.csv").open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "family", "parameters", "checkpoint_id",
            "training_step", "tokens_processed", "train_loss", "test_loss",
            "test_perplexity", "benchmark", "benchmark_protocol", "benchmark_score",
            "model_load_seconds", "evaluation_seconds", "peak_ram_bytes", "peak_vram_bytes",
            "source_type", "source_url", "status", "notes"])
        w.writerow({"model": "Pythia-70M", "family": "Pythia", "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_id": revision, "training_step": 143000, "tokens_processed": 299892736000,
            "train_loss": "", "test_loss": float(loss), "test_perplexity": ppl,
            "benchmark": "WikiText-2 raw test subset", "benchmark_protocol": "single truncated 2048-token sequence; pilot only",
            "benchmark_score": "", "model_load_seconds": load_s, "evaluation_seconds": eval_s,
            "peak_ram_bytes": memory_bytes(), "peak_vram_bytes": "",
            "source_type": "local evaluation of pretrained checkpoint", "source_url": "https://huggingface.co/EleutherAI/pythia-70m",
            "status": "completed", "notes": "Checkpoint download/evaluation cost is not original pretraining cost."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-only", action="store_true")
    parser.add_argument("--semi-only", action="store_true")
    parser.add_argument("--kmeans-only", action="store_true")
    parser.add_argument("--rl-only", action="store_true")
    parser.add_argument("--pseudo-threshold", type=float, default=0.95)
    args = parser.parse_args()
    seed_everything()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cache = ROOT / "results/cache"
    if args.llm_only:
        pythia()
        return
    if args.rl_only:
        log_rows(dqn())
        print(f"Pilots saved. Device={DEVICE}; cuda_available={torch.cuda.is_available()}")
        return
    train, train_loader, test_loader = load_data(cache)
    if args.kmeans_only:
        log_rows([unsupervised(train)])
        print(f"Pilots saved. Device={DEVICE}; cuda_available={torch.cuda.is_available()}")
        return
    if args.semi_only:
        rows = semi_supervised(train, test_loader, args.pseudo_threshold)
        log_rows(rows)
    else:
        log_rows(supervised(train_loader, test_loader))
        log_rows([unsupervised(train)])
        log_rows(semi_supervised(train, test_loader))
        log_rows(dqn())
        if not args.skip_llm:
            pythia()
    print(f"Pilots saved. Device={DEVICE}; cuda_available={torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
