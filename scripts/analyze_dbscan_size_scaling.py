import csv,statistics
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'results/raw/dbscan_size_scaling.csv';rows=list(csv.DictReader(p.open(encoding='utf-8-sig')));groups={}
for r in rows:groups.setdefault(int(r['budget_value']),[]).append(r)
summary=[]
for n,rs in sorted(groups.items()):
 for metric in ['training_time_seconds','evaluation_value','NMI','silhouette_non_noise','noise_fraction']:
  vals=[float(r[metric]) for r in rs if r.get(metric,'')!='']
  if vals:summary.append({'samples':n,'metric':metric,'n_seeds':len(vals),'mean':statistics.mean(vals),'sd':statistics.stdev(vals) if len(vals)>1 else 0})
out=ROOT/'results/processed/dbscan_size_scaling_summary.csv'
with out.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for metric,ax,title in [('training_time_seconds',axs[0],'Fit time vs sample count'),('evaluation_value',axs[1],'ARI vs sample count')]:
 q=[r for r in summary if r['metric']==metric];ax.errorbar([r['samples'] for r in q],[r['mean'] for r in q],yerr=[r['sd'] for r in q],marker='o',capsize=3);ax.set(xlabel='Samples fitted',ylabel='seconds' if metric.startswith('training') else 'ARI',title=title);ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(ROOT/'results/figures/dbscan_size_scaling.png',dpi=160);plt.close(fig)
print('Wrote DBSCAN scaling summaries')
