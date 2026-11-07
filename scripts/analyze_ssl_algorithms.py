import csv,math,statistics
from pathlib import Path
from scipy.stats import t
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'results/raw/ssl_algorithms.csv';rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
groups={}
for r in rows:
 if r.get('status')!='completed':continue
 k=(r['model_name'],r['labeled_fraction'],r['budget_value']);groups.setdefault(k,[]).append(r)
summary=[]
for (m,f,b),rs in groups.items():
 v=[float(r['validation_accuracy']) for r in rs];n=len(v);mean=statistics.mean(v);sd=statistics.stdev(v) if n>1 else 0;half=t.ppf(.975,n-1)*sd/math.sqrt(n) if n>1 else ''
 summary.append({'model_name':m,'labeled_fraction':f,'epochs':b,'n_seeds':n,'mean_validation_accuracy':mean,'sd':sd,'ci95_low':mean-half if n>1 else '', 'ci95_high':mean+half if n>1 else ''})
out=ROOT/'results/processed/ssl_algorithm_summary.csv'
with out.open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(sorted(summary,key=lambda r:(r['model_name'],float(r['labeled_fraction']),int(r['epochs']))))
fig,axs=plt.subplots(1,3,figsize=(15,4.8))
for ax,m in zip(axs,['MeanTeacher','FixMatch','MixMatch']):
 for frac in [.01,.05,.1,.2,.4]:
  q=[r for r in summary if r['model_name']==m and float(r['labeled_fraction'])==frac];q.sort(key=lambda r:int(r['epochs']))
  if q:ax.errorbar([int(r['epochs']) for r in q],[r['mean_validation_accuracy'] for r in q],yerr=[r['sd'] for r in q],marker='o',label=f'{frac:.0%}')
 ax.set_title(m);ax.set_xlabel('Epochs');ax.set_ylabel('Validation accuracy');ax.grid(alpha=.25);ax.legend(title='Labeled fraction')
fig.tight_layout();fig.savefig(ROOT/'results/figures/ssl_algorithm_budget_curves.png',dpi=160);plt.close(fig)
print(f'{len(rows)} seed-budget records; {len(summary)} grouped summaries')
