"""Summarize locally evaluated language-model checkpoints and make figures."""
import csv, math, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/llm/full_wikitext2_evaluations.csv'
OUT=ROOT/'results/processed'
FIG=ROOT/'results/figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
with SRC.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
rows=[r for r in rows if r.get('status')=='completed']
def num(r,k):
    try:
        x=float(r[k]); return x if math.isfinite(x) else None
    except (KeyError,ValueError,TypeError): return None
with (OUT/'llm_full_eval_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['model_name','family','parameters','checkpoint_id','training_step','evaluation_tokens_scored','test_cross_entropy','test_perplexity','model_load_seconds','evaluation_seconds','tokens_per_second','peak_ram_bytes','hardware','backend','status']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:r.get(k,'') for k in fields} for r in rows])
for x,y,xlab,ylab,name in [('parameters','test_perplexity','Parameters','WikiText-2 test perplexity','llm_model_size_vs_perplexity.png'),('parameters','test_cross_entropy','Parameters','WikiText-2 test cross entropy (nats/token)','llm_model_size_vs_loss.png'),('parameters','tokens_per_second','Parameters','Evaluation throughput (tokens/s)','llm_model_size_vs_throughput.png')]:
    fig,ax=plt.subplots(figsize=(8,5))
    for r in rows:
        xv,yv=num(r,x),num(r,y)
        if xv and yv: ax.scatter(xv,yv); ax.annotate(r['model_name'],(xv,yv),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xscale('log'); ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.grid(True,alpha=.25); fig.tight_layout(); fig.savefig(FIG/name,dpi=160); plt.close(fig)
models=', '.join(f"{r['model_name']} ({r['checkpoint_id']})" for r in rows)
lines=['# LLM Evaluation Report','',f"Updated from `{SRC.relative_to(ROOT)}`. All listed scores are local checkpoint inference measurements; they are not pretraining runs.",'',f'## Completed local evaluations','',f'{len(rows)} configurations completed a matched WikiText-2 raw-v1 test evaluation: {models}.','', 'The requested input budget was 50,000 tokenizer tokens for each run. Scored-token counts are 49,951–49,975 because causal next-token loss excludes each non-overlapping chunk’s first token. Pythia and OPT used 2,048-token context windows; GPT-2 Small used 1,024. That context difference is a comparability limitation. Runs used CPU FP32. The table reports model loading time, evaluation time, observed throughput, process peak RAM and the recorded checkpoint metadata. Local evaluation time is not a proxy for original pretraining compute.','', '## Results','', '| Model | Checkpoint | Parameters | Scored tokens | Cross entropy | Perplexity | Eval seconds | Tokens/s |', '|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['model_name']} | {r['checkpoint_id']} | {int(float(r['parameters'])):,} | {r['evaluation_tokens_scored']} | {float(r['test_cross_entropy']):.4f} | {float(r['test_perplexity']):.3f} | {float(r['evaluation_seconds']):.2f} | {float(r['tokens_per_second']):.1f} |")
lines += ['', '## Downstream harness evaluation', '', 'lm-evaluation-harness 0.4.13 ran 0-shot CPU FP32 evaluations on 100 examples per task for Pythia-70M step143000, GPT-2 Small, and OPT-125M. These are smoke subsets and must not be presented as full benchmark scores. The result JSONs and per-sample logs are under `results/llm/harness/`; the extracted table is `results/llm/harness_summary.csv`.','', '| Model | HellaSwag acc / normalized | PIQA acc / normalized | ARC-Easy acc / normalized | Total eval seconds |','|---|---:|---:|---:|---:|']
try:
    import csv
    with (ROOT/'results/llm/harness_summary.csv').open(encoding='utf-8-sig',newline='') as f: harness=list(csv.DictReader(f))
    grouped={}
    for r in harness: grouped.setdefault(r['model'],{})[r['task']]=r
    for model,t in grouped.items():
        h=t.get('HellaSwag',{});p=t.get('PIQA',{});a=t.get('ARC-Easy',{})
        lines.append(f"| {model} | {h.get('accuracy','')} / {h.get('normalized_accuracy','')} | {p.get('accuracy','')} / {p.get('normalized_accuracy','')} | {a.get('accuracy','')} / {a.get('normalized_accuracy','')} | {h.get('eval_seconds','')} |")
except FileNotFoundError: lines.append('| Harness subset run pending | | | | |')
lines += ['', '## Figures','', '- `results/figures/llm_model_size_vs_perplexity.png`', '- `results/figures/llm_model_size_vs_loss.png`', '- `results/figures/llm_model_size_vs_throughput.png`', '', 'No confidence intervals are reported for the WikiText comparisons: these are deterministic evaluations of distinct fixed checkpoints, not repeated seed runs. Harness standard errors are dataset-subset sampling estimates for n=100, not seed variability.']
(ROOT/'LLM_EVALUATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'Wrote summary and report for {len(rows)} runs')
