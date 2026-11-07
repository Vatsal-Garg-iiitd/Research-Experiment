"""Summarize full-corpus local LLM evaluations and regenerate their figures."""
import csv, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/llm/macos_full_wikitext2_evaluations.csv'
OUT=ROOT/'results/processed';FIG=ROOT/'results/figures'
OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
with SRC.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
rows=[r for r in rows if r.get('status')=='completed']
fields=['model_name','model_revision','parameter_count','dataset','scored_tokens','context_length','stride','negative_log_likelihood','perplexity','bits_per_byte','model_load_seconds','evaluation_seconds','tokens_per_second','precision','device','backend','ram_bytes_after_eval','weight_memory_estimate_bytes','text_sha256','status']
with (OUT/'llm_full_eval_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k,'') for k in fields} for r in rows])
def num(r,k):
    try:
        x=float(r[k]);return x if math.isfinite(x) else None
    except (KeyError,ValueError,TypeError):return None
plots=[('perplexity','Perplexity on WikiText-2 raw-v1 test','llm_model_size_vs_perplexity.png'),('negative_log_likelihood','Cross entropy (nats/scored token)','llm_model_size_vs_loss.png'),('tokens_per_second','Evaluation throughput (scored tokens/s)','llm_model_size_vs_throughput.png')]
for metric,label,name in plots:
    fig,ax=plt.subplots(figsize=(8,5))
    for r in rows:
        x,y=num(r,'parameter_count'),num(r,metric)
        if x and y:
            ax.scatter(x,y);ax.annotate(r['model_name'],(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xscale('log');ax.set_xlabel('Published checkpoint parameters');ax.set_ylabel(label)
    ax.set_title('Full WikiText-2 raw-v1 test evaluation (FP32, MPS, context 1024, stride 512)')
    ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(FIG/name,dpi=160);plt.close(fig)
print(f'Wrote full-corpus summary and regenerated figures for {len(rows)} checkpoints; report content remains maintained in LLM_EVALUATION_REPORT.md')
