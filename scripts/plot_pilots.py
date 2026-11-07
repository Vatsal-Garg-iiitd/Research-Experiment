"""Generate separate figures from measured pilot runs and compatible literature rows."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


rows = read(ROOT / "results/raw/experiment_results.csv")

def matching(run_id):
    return [r for r in rows if r["run_id"] == run_id]


def plot_run(run_id, x_key, title, xlabel, ylabel, filename):
    data = matching(run_id)
    if not data:
        return
    data.sort(key=lambda r: float(r[x_key]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([float(r[x_key]) for r in data], [float(r["metric_value"]) for r in data], "o-")
    if run_id.startswith("pilot_cartpole") and x_key == "budget_value":
        ax.set_xticks([1000, 2000, 3000, 4000, 5000])
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


plot_run("pilot_fashion_mlp", "cumulative_train_seconds",
         "Fashion-MNIST MLP pilot (single seed)", "Cumulative training time (s)",
         "Test accuracy", "pilot_mlp_accuracy_time.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
for run_id, label in [("pilot_fashion_pseudolabel_seeded_t95", "threshold 0.95"),
                      ("pilot_fashion_pseudolabel_seeded_t80", "threshold 0.80")]:
    data = matching(run_id)
    data.sort(key=lambda r: float(r["budget_value"]))
    if data:
        ax.plot([float(r["budget_value"]) for r in data],
                [float(r["metric_value"]) for r in data], "o-", label=label)
ax.set(title="Pseudo-label pilot: threshold sensitivity (single seed)",
       xlabel="Epochs", ylabel="Test accuracy")
ax.grid(True, alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "pilot_pseudolabel_thresholds.png", dpi=180)
plt.close(fig)

plot_run("pilot_cartpole_dqn_v3", "budget_value",
         "CartPole DQN pilot (single seed)", "Environment steps",
         "Mean evaluation return (5 episodes)", "pilot_dqn_return_steps.png")
plot_run("pilot_cartpole_dqn_v3", "cumulative_train_seconds",
         "CartPole DQN pilot (single seed)", "Cumulative training time (s)",
         "Mean evaluation return (5 episodes)", "pilot_dqn_return_time.png")

km = matching("pilot_fashion_kmeans_v2")
if km:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([float(km[0]["cumulative_train_seconds"])], [float(km[0]["metric_value"])])
    ax.set(title="Fashion-MNIST K-Means pilot (one run)", xlabel="Fit time (s)",
           ylabel="Adjusted Rand index")
    ax.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "pilot_kmeans_ari_time.png", dpi=180)
    plt.close(fig)

lit = read(ROOT / "results/literature/large_model_literature.csv")
lit = [r for r in lit if r["performance_metric"] == "HellaSwag zero-shot" and r["performance_value"]]
if lit:
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in lit:
        scale = 1e9 if r["parameter_unit"] == "B" else 1e6
        x = float(r["parameters"]) * scale / 1e9
        y = float(r["performance_value"])
        ax.scatter(x, y)
        ax.annotate(r["model"], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set(title="Published zero-shot HellaSwag scores, as collated by LLaMA Table 3",
           xlabel="Parameters (billions, log scale)", ylabel="HellaSwag accuracy (%)")
    ax.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "literature_hellaswag_vs_parameters.png", dpi=180)
    plt.close(fig)

palm = [r for r in read(ROOT / "results/literature/large_model_literature.csv")
        if r["model"].startswith("PaLM-62B") and r["training_compute"]]
palm.sort(key=lambda r: float(r["training_compute"]))
if len(palm) >= 2:
    compute_zf = [float(r["training_compute"]) / 1e21 for r in palm]
    scores = [float(r["performance_value"]) for r in palm]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(compute_zf, scores, "o-")
    for index, (x, y, r) in enumerate(zip(compute_zf, scores, palm)):
        offset = (-78, -16) if index == len(palm) - 1 else (5, 5)
        ax.annotate(f"{float(r['training_tokens'])/1000:g}T tokens", (x, y),
                    xytext=offset, textcoords="offset points", fontsize=8)
    ax.set(title="PaLM-62B continuation: published Table 31", xlabel="Reported training compute (zettaFLOPs)",
           ylabel="Paper-defined average English-task score")
    ax.grid(True, alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "literature_palm_continuation.png", dpi=180)
    plt.close(fig)

print(f"Generated available pilot/literature figures in {OUT}")
