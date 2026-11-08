"""Verify the 20 conventional model objectives against configured schedules."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
SEEDS = ("11", "23", "37")
EPOCHS = tuple(str(x) for x in range(1, 31))
FRACTIONS = ("0.01", "0.05", "0.1", "0.2", "0.4")
RL_BUDGETS = ("5000", "10000", "25000", "50000", "100000", "250000")


def read(name: str) -> list[dict[str, str]]:
    with (RAW / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def keys(rows, columns):
    return {tuple(r.get(c, "") for c in columns) for r in rows}


def record(model, source, n, expected, missing, detail):
    return {
        "model": model,
        "source_files": source,
        "valid_record_count": n,
        "expected_coverage": expected,
        "missing_coverage": "; ".join(missing),
        "status": "completed" if not missing else "incomplete",
        "verification_note": detail,
    }


def run() -> list[dict[str, str]]:
    out = []
    curve = read("replicated_scaling_results.csv")
    tests = read("replicated_scaling_test_results.csv")
    for name, raw_name in (("MLP", "MLP"), ("LeNet-style CNN", "CNN / LeNet"), ("ResNet-18", "ResNet-18")):
        rs = [r for r in curve if r.get("model_name") == raw_name and r.get("experiment_status") == "completed"]
        by_config = defaultdict(set)
        for r in rs:
            cfg = r.get("configuration_path", "")
            by_config[(r.get("seed", ""), cfg)].add(r.get("budget_value", ""))
        missing = []
        for seed in SEEDS:
            base = {e for (s, cfg), es in by_config.items() if s == seed and "n10000_" in cfg for e in es}
            if not set(EPOCHS) <= base:
                missing.append(f"10K seed {seed} epochs {','.join(sorted(set(EPOCHS)-base, key=int))}")
            if name in ("MLP", "LeNet-style CNN"):
                for size in ("n30000_", "n60000_"):
                    vals = {e for (s, cfg), es in by_config.items() if s == seed and size in cfg for e in es}
                    required = set(str(e) for e in range(1, 6))
                    if not required <= vals:
                        missing.append(f"{size[:-1]} seed {seed} epochs {','.join(sorted(required-vals, key=int))}")
        tr = [r for r in tests if r.get("model_name") == raw_name and r.get("status") == "completed"]
        for seed in SEEDS:
            endpoints = {r.get("examples_processed", "") for r in tr if r.get("seed") == seed}
            required = {"50000", "300000"} if name == "ResNet-18" else {"50000", "150000", "290000", "300000"}
            if not required <= endpoints:
                missing.append(f"held-out endpoints seed {seed} examples {','.join(sorted(required-endpoints))}")
        expected = "3 seeds; 10K epochs 1-30" + ("; 30K/58K epochs 1-5" if name != "ResNet-18" else "") + "; final official-test endpoints"
        out.append(record(name, "replicated_scaling_results.csv; replicated_scaling_test_results.csv", len(rs)+len(tr), expected, missing, "Coverage checked by source row, seed, configuration path, budget, and held-out-test endpoint."))

    rows = read("vit_tiny_scaling.csv")
    tests = read("vit_tiny_test_results.csv")
    valid = [r for r in rows if r.get("status") == "completed"]
    missing = []
    for seed in SEEDS:
        budgets = {r.get("budget_value") for r in valid if r.get("seed") == seed}
        if not set(EPOCHS) <= budgets: missing.append(f"seed {seed} missing validation epochs")
        if not any(r.get("seed") == seed and r.get("epochs") == "30" and r.get("status") == "completed" for r in tests):
            missing.append(f"seed {seed} missing epoch-30 test endpoint")
    out.append(record("ViT-Tiny", "vit_tiny_scaling.csv; vit_tiny_test_results.csv", len(valid)+len(tests), "12-block ViT-Tiny-dimension model; 3 seeds; 30 epochs; final test endpoint", missing, "Separate from the compact transformer proxy."))

    rows = read("random_forest_scaling.csv"); tests = read("random_forest_test_results.csv")
    valid = [r for r in rows if r.get("experiment_status") == "completed"]
    required = {"10", "25", "50", "100", "200", "400", "800", "1000"}
    missing=[]
    for seed in SEEDS:
        b={r.get("budget_value") for r in valid if r.get("seed")==seed}
        if not required<=b:missing.append(f"seed {seed} missing tree counts {','.join(sorted(required-b,key=int))}")
        if not any(r.get("seed")==seed and r.get("status")=="completed" for r in tests):missing.append(f"seed {seed} missing held-out-test endpoint")
    out.append(record("Random Forest","random_forest_scaling.csv; random_forest_test_results.csv",len(valid)+len(tests),"3 seeds; 10-1,000 warm-start trees; test endpoints",missing,"Native budget is tree count; no epoch conversion."))

    rows=read("unsupervised_scaling.csv")
    for model,raw_name,budgets in (("K-Means","MiniBatch K-Means",("1","5","10","20","50")),("Gaussian Mixture Model","Gaussian Mixture Model",("1","3","5","10","20","50","100"))):
        valid=[r for r in rows if r.get("model_name")==raw_name and r.get("experiment_status")=="completed"]
        unique={r.get("experiment_id"):r for r in valid}
        missing=[]
        for seed in SEEDS:
            b={r.get("budget_value") for r in unique.values() if r.get("seed")==seed}
            if not set(budgets)<=b:missing.append(f"seed {seed} missing native budgets {','.join(sorted(set(budgets)-b,key=int))}")
        out.append(record(model,"unsupervised_scaling.csv",len(valid),f"3 seeds; native budgets {','.join(budgets)}",missing,"Repeated K-Means source IDs are deduplicated for coverage; raw records are preserved."))

    rows=read("dbscan_size_scaling.csv");params=read("dbscan_parameter_sweep.csv")
    valid=[r for r in rows if r.get("status")=="completed"]
    missing=[]
    for seed in SEEDS:
        sizes={r.get("budget_value") for r in valid if r.get("seed")==seed}
        if not {"2500","5000","10000"}<=sizes:missing.append(f"seed {seed} missing sizes {','.join(sorted({'2500','5000','10000'}-sizes,key=int))}")
    if sum(r.get("status")=="completed" for r in params)<8:missing.append("parameter grid has fewer than 8 recorded settings")
    out.append(record("DBSCAN","dbscan_size_scaling.csv; dbscan_parameter_sweep.csv",len(valid)+len(params),"3 seeds at 2.5K/5K/10K samples plus 8-setting parameter sensitivity",missing,"DBSCAN is a parameter/size study, not an iterative epoch curve."))

    rows=read("autoencoder_scaling.csv");tests=read("autoencoder_test_results.csv")
    for model in ("Autoencoder","Variational Autoencoder"):
        valid=[r for r in rows if r.get("model_name")==model and r.get("experiment_status")=="completed"]
        tr=[r for r in tests if r.get("model_name")==model and r.get("status")=="completed"]
        missing=[]
        for seed in SEEDS:
            b={r.get("budget_value") for r in valid if r.get("seed")==seed}
            if not set(EPOCHS)<=b:missing.append(f"seed {seed} missing epochs")
            if not any(r.get("seed")==seed and r.get("epochs")=="30" for r in tr):missing.append(f"seed {seed} missing epoch-30 test endpoint")
        out.append(record(model,"autoencoder_scaling.csv; autoencoder_test_results.csv",len(valid)+len(tr),"3 seeds; epochs 1-30; reconstruction and final test metrics",missing,"Autoencoder reconstruction and downstream probe metrics remain distinct."))

    rows=read("pseudolabel_baseline_sweep.csv");tests=read("pseudolabel_test_results.csv")
    pseudo_missing=[]; baseline_missing=[]
    for method,label,missing in (("pseudo_label","Pseudo-Labeling",pseudo_missing),("labeled_only","Labeled-only matched baseline",baseline_missing)):
        valid=[r for r in rows if r.get("method")==method and r.get("experiment_status")=="completed"]
        tr=[r for r in tests if r.get("method")==method and r.get("status")=="completed"]
        for seed in SEEDS:
            for frac in FRACTIONS:
                b={r.get("budget_value") for r in valid if r.get("seed")==seed and r.get("labeled_fraction")==frac}
                if not set(EPOCHS)<=b:missing.append(f"seed {seed}, fraction {frac}: missing epochs")
                if not any(r.get("seed")==seed and r.get("labeled_fraction")==frac and r.get("epochs")=="30" for r in tr):missing.append(f"seed {seed}, fraction {frac}: missing epoch-30 test endpoint")
        if method=="pseudo_label":pseudo_records=len(valid)+len(tr)
        else:baseline_records=len(valid)+len(tr)
    pseudo_missing.extend(f"matched baseline {item}" for item in baseline_missing)
    out.append(record("Pseudo-Labeling","pseudolabel_baseline_sweep.csv; pseudolabel_test_results.csv",pseudo_records+baseline_records,"3 seeds x 5 label fractions x epochs 1-30; matched labeled-only baseline and final official-test endpoints",pseudo_missing,"Epochs 6-30 resumed from saved epoch-5 states; original checkpoints retained."))

    rows=read("ssl_algorithms.csv");tests=read("ssl_algorithms_test.csv")
    for raw_name,label in (("MeanTeacher","Mean Teacher"),("FixMatch","FixMatch"),("MixMatch","MixMatch")):
        valid=[r for r in rows if r.get("model_name")==raw_name and r.get("status")=="completed"]
        tr=[r for r in tests if r.get("model_name")==raw_name and r.get("status")=="completed"]
        missing=[]
        for seed in SEEDS:
            for frac in FRACTIONS:
                b={r.get("budget_value") for r in valid if r.get("seed")==seed and r.get("labeled_fraction")==frac}
                if not set(EPOCHS)<=b:missing.append(f"seed {seed}, fraction {frac}: missing epochs")
                if not any(r.get("seed")==seed and r.get("labeled_fraction")==frac and r.get("epochs")=="30" for r in tr):missing.append(f"seed {seed}, fraction {frac}: missing epoch-30 test endpoint")
        out.append(record(label,"ssl_algorithms.csv; ssl_algorithms_test.csv",len(valid)+len(tr),"3 seeds x 5 label fractions x epochs 1-30; final official-test endpoints",missing,"Fashion-MNIST algorithm adaptations with separate held-out test endpoints."))

    rows=read("label_propagation_scaling.csv")
    valid=[r for r in rows if r.get("status")=="completed"]
    required={"1","5","10","30","100","250","500","1000","2000"};missing=[]
    for seed in SEEDS:
        b={r.get("budget_value") for r in valid if r.get("seed")==seed}
        if not required<=b:missing.append(f"seed {seed} missing iteration ceilings {','.join(sorted(required-b,key=int))}")
    out.append(record("Label Propagation","label_propagation_scaling.csv",len(valid),"3 seeds; iteration ceilings 1-2,000; actual LabelPropagation",missing,"Only current LabelPropagation records count; superseded legacy estimator rows are excluded."))

    rows=read("rl_scaling_results.csv")
    for model,required_seeds in (("DQN",SEEDS),("Double DQN",("11","23","37","41","53")),("A2C",SEEDS),("PPO",SEEDS),("SAC",("11","23","37","41","53"))):
        valid=[r for r in rows if r.get("model_name")==model]
        missing=[]
        for seed in required_seeds:
            for budget in RL_BUDGETS:
                found=[]
                for r in valid:
                    if r.get("seed")!=seed or (r.get("requested_budget") or r.get("budget_value"))!=budget:continue
                    # The original Double DQN writer omitted one field. In its
                    # raw DictReader mapping, the source status is under backend.
                    status=(r.get("backend") if model=="Double DQN" and r.get("details") in (None,"") else r.get("experiment_status"))
                    if status=="completed":found.append(r)
                if len(found)!=1:missing.append(f"seed {seed}, {budget} steps: expected one completed row, found {len(found)}")
        out.append(record(model,"rl_scaling_results.csv",len(valid),f"{len(required_seeds)} seeds x six budgets through 250K; checkpoint evaluation",missing,"Double DQN field shift is handled analytically; its raw CSV remains unchanged."))
    return out


def main() -> int:
    rows=run()
    out=ROOT/"results"/"processed"/"conventional_completion_audit.csv"
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    done=sum(r["status"]=="completed" for r in rows)
    for r in rows:
        print(f"{r['status']}: {r['model']} ({r['valid_record_count']} records)")
        if r['missing_coverage']:print("  missing:",r['missing_coverage'])
    print(f"Verified complete conventional objectives: {done}/{len(rows)}")
    return 0 if done==len(rows) else 1


if __name__=="__main__":
    raise SystemExit(main())
