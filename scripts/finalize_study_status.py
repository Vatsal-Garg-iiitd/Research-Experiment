"""Refresh analysis artifacts, figures, completion matrix, and remaining-work ledger."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; FIG=RES/'figures'; FIG.mkdir(parents=True,exist_ok=True)
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def normalize():
 # Extended one-budget records are feasibility pilots, not complete full-budget studies.
 p=RES/'raw/full_suite_results.csv';rows=read(p)
 for r in rows:
  if r.get('status')=='completed':r['status']='pilot'
 duplicate_runs=defaultdict(list)
 for i,r in enumerate(rows):duplicate_runs[r.get('run_id','')].append(i)
 for ids in duplicate_runs.values():
  if len(ids)>1:
   for i in ids[:-1]:rows[i]['status']='superseded'
 write(p,rows)
 # Preserve all LLM retry rows, flag redundant or failed initial records transparently.
 p=RES/'llm/expanded_llm_evaluations.csv';rows=read(p); groups=defaultdict(list)
 for i,r in enumerate(rows):groups[(r.get('model'),r.get('checkpoint_id'))].append(i)
 for key,idxs in groups.items():
  completed=[i for i in idxs if rows[i].get('status')=='completed']
  if len(completed)>1:
   for i in completed[:-1]:rows[i]['status']='superseded'
  if key==('GPT-2 Small','main'):
   for i in idxs:
    if rows[i].get('status')=='failed':rows[i]['status']='superseded'
 write(p,rows)
 # Mirror the latest Pythia checkpoint row status in the older ledger.
 p=RES/'llm/llm_checkpoint_results.csv';rows=read(p);groups=defaultdict(list)
 for i,r in enumerate(rows):groups[(r.get('model'),r.get('checkpoint_id'))].append(i)
 for key,idxs in groups.items():
  completed=[i for i in idxs if rows[i].get('status')=='completed']
  if len(completed)>1:
   for i in completed[:-1]:rows[i]['status']='superseded'
 write(p,rows)

def plots():
 rows=read(RES/'raw/full_suite_results.csv')
 # Results by paradigm and native evaluation metric: every mark is a single-seed pilot.
 groups=[('supervised','heldout_accuracy','Supervised neural held-out accuracy','full_suite_supervised_accuracy.png'),('semi_supervised','heldout_accuracy','Semi-supervised held-out accuracy','full_suite_semisupervised_accuracy.png'),('unsupervised','adjusted_rand_index','Unsupervised ARI (external labels)','full_suite_unsupervised_ari.png'),('unsupervised','heldout_reconstruction_mse','Autoencoder reconstruction error','full_suite_reconstruction_mse.png'),('reinforcement_learning','mean_return_5_episodes','RL mean return (5 evaluation episodes)','full_suite_rl_return.png')]
 for par,metric,title,file in groups:
  data=[r for r in rows if r.get('paradigm')==par and r.get('metric_name')==metric and r.get('metric_value') not in ('','None') and not r.get('status','').startswith('superseded')]
  if not data:continue
  data.sort(key=lambda r:r['model']);fig,ax=plt.subplots(figsize=(9,max(4,len(data)*.34))); y=[float(r['metric_value']) for r in data]
  ax.barh([r['model'] for r in data],y);ax.set(title=title+' (one seed; feasibility budget)',xlabel=metric);ax.grid(axis='x',alpha=.25);fig.tight_layout();fig.savefig(FIG/file,dpi=180);plt.close(fig)
 # Different evaluation protocols/metrics receive their own figures.
 for name,metric,file,title in [('Random Forest','test_accuracy','full_suite_random_forest_accuracy.png','Random Forest test accuracy'),('Label Propagation','transductive_test_accuracy','full_suite_labelprop_accuracy.png','Label Propagation transductive test accuracy')]:
  d=[r for r in rows if r.get('model')==name and r.get('metric_name')==metric and not r.get('status','').startswith('superseded')]
  if d:
   fig,ax=plt.subplots(figsize=(6,4));ax.bar([name],[float(d[-1]['metric_value'])]);ax.set(title=title+' (single seed)',ylabel=metric);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(FIG/file,dpi=180);plt.close(fig)
 # Per-paradigm performance vs the task-native measured cost; no pooled cross-paradigm axis.
 fig,axs=plt.subplots(2,2,figsize=(12,8))
 costsets=[('supervised','heldout_accuracy','train_seconds','Supervised: accuracy vs train seconds'),('semi_supervised','heldout_accuracy','train_seconds','Semi-supervised: accuracy vs train seconds'),('unsupervised','adjusted_rand_index','train_seconds','Unsupervised: ARI vs fit seconds'),('reinforcement_learning','mean_return_5_episodes','budget_value','RL: return vs environment steps')]
 for ax,(par,metric,xk,title) in zip(axs.flat,costsets):
  d=[r for r in rows if r.get('paradigm')==par and r.get('metric_name')==metric and r.get('metric_value') not in ('','None') and not r.get('status','').startswith('superseded')]
  d=[r for r in d if r.get(xk) not in ('','None')]
  if d:
   for r in d:ax.scatter(float(r[xk]),float(r['metric_value']),label=r['model']);ax.annotate(r['model'],(float(r[xk]),float(r['metric_value'])),fontsize=7,xytext=(3,3),textcoords='offset points')
  ax.set(title=title,xlabel=xk,ylabel=metric);ax.grid(alpha=.2)
 fig.suptitle('Single-seed pilot results; each panel uses its native metric and cost');fig.tight_layout();fig.savefig(FIG/'full_suite_performance_native_cost.png',dpi=180);plt.close(fig)
 # Existing budget trajectories retain their real local measurements.
 llm=read(RES/'llm/expanded_llm_evaluations.csv'); trajectory=read(RES/'llm/llm_checkpoint_results.csv');p=[r for r in trajectory if r.get('model')=='Pythia-70M' and r.get('status')=='completed' and r.get('training_step') and r.get('test_loss')]
 # One record per checkpoint, excluding superseded retries.
 uniq={r['training_step']:r for r in p};p=sorted(uniq.values(),key=lambda r:int(r['training_step']))
 if p:
  x=[int(r['tokens_processed']) for r in p];loss=[float(r['test_loss']) for r in p];ppl=[float(r['test_perplexity']) for r in p]
  for ys,title,ylabel,file in [(loss,'Pythia-70M WikiText-2 slice loss vs training tokens','Cross-entropy loss','llm_pythia70m_loss_tokens.png'),(ppl,'Pythia-70M WikiText-2 slice perplexity vs training tokens','Perplexity','llm_pythia70m_perplexity_tokens.png')]:
   fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(x,ys,'o-');ax.set_xscale('log');ax.set(title=title+' (one sequence; one seed)',xlabel='Published training tokens processed (step-derived)',ylabel=ylabel);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/file,dpi=180);plt.close(fig)
 local=[r for r in llm if r.get('status')=='completed' and r.get('test_loss') and r.get('parameters')]
 if local:
  fig,ax=plt.subplots(figsize=(8,5));
  for r in local:
   if r.get('checkpoint_id') not in ('main','step143000','step100000'):continue
   ax.scatter(int(r['parameters'])/1e6,float(r['test_loss']));ax.annotate(r['model'],(int(r['parameters'])/1e6,float(r['test_loss'])),fontsize=7,xytext=(3,3),textcoords='offset points')
  ax.set(title='Local pretrained-model loss on WikiText-2 prefix (pilot)',xlabel='Parameters (millions)',ylabel='Cross-entropy loss');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/'llm_local_model_size_loss.png',dpi=180);plt.close(fig)
 litcompute=[r for r in read(RES/'literature/large_model_literature.csv') if r.get('training_compute') and r.get('parameters')]
 if litcompute:
  fig,ax=plt.subplots(figsize=(7,4.5))
  for r in litcompute:
   scale=1e9 if r.get('parameter_unit')=='B' else 1e6;x=float(r['parameters'])*scale/1e9;y=float(r['training_compute']);ax.scatter(x,y);ax.annotate(r['model'],(x,y),fontsize=7,xytext=(3,3),textcoords='offset points')
  ax.set_xscale('log');ax.set_yscale('log');ax.set(title='Published model size vs reported training compute',xlabel='Parameters (billions, log scale)',ylabel='Training FLOPs (log scale)');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/'literature_model_size_compute.png',dpi=180);plt.close(fig)
 hw=read(RES/'hardware/backend_benchmark.csv')
 if hw:
  for key,title,ylabel,file in [('inference_samples_per_second','Same-input MLP inference throughput','samples/second','hardware_inference_cpu_directml.png'),('training_steps_per_second','Same-input MLP training throughput','steps/second','hardware_training_cpu_directml.png')]:
   fig,ax=plt.subplots(figsize=(7,4));ax.bar([r['backend'].replace('AMD Radeon 860M DirectML (privateuseone:0)','Radeon 860M DirectML') for r in hw],[float(r[key]) for r in hw]);ax.set(title=title+' (microbenchmark)',ylabel=ylabel);ax.tick_params(axis='x',rotation=10);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(FIG/file,dpi=180);plt.close(fig)
 # Existing marginal-analysis data are visualized as separate model-specific pilot records.
 marg=read(RES/'processed/pilot_marginal_results.csv')
 d=[r for r in marg if r.get('marginal_gain') and r.get('marginal_cost_seconds')]
 if d:
  fig,axs=plt.subplots(1,2,figsize=(11,4.5));
  for model in sorted(set(r['model'] for r in d)):
   x=[float(r['budget_value']) for r in d if r['model']==model];g=[float(r['marginal_gain']) for r in d if r['model']==model];e=[float(r['marginal_efficiency_per_second']) for r in d if r['model']==model and r.get('marginal_efficiency_per_second')]
   if len(g)>0:axs[0].plot(x,g,'o-',label=model)
   if len(e)>0:axs[1].plot(x[:len(e)],e,'o-',label=model)
  axs[0].set(title='Observed pilot marginal gains',xlabel='Native budget value',ylabel='Metric change (native units)');axs[1].set(title='Observed pilot marginal efficiency',xlabel='Native budget value',ylabel='Metric change / second')
  for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=7)
  fig.tight_layout();fig.savefig(FIG/'pilot_marginal_gain_efficiency.png',dpi=180);plt.close(fig)

CONVENTIONAL=[
 ('MLP','supervised'),('CNN / LeNet','supervised'),('ResNet-18','supervised'),('Vision Transformer (ViT-Tiny)','supervised'),('Random Forest','supervised'),('K-Means','unsupervised'),('Gaussian Mixture Model','unsupervised'),('DBSCAN','unsupervised'),('Autoencoder','unsupervised'),('Variational Autoencoder','unsupervised'),('Pseudo-Labeling','semi_supervised'),('Mean Teacher','semi_supervised'),('FixMatch','semi_supervised'),('MixMatch','semi_supervised'),('Label Propagation','semi_supervised'),('DQN','reinforcement_learning'),('Double DQN','reinforcement_learning'),('A2C','reinforcement_learning'),('PPO','reinforcement_learning'),('SAC','reinforcement_learning')]
LLM=[('Pythia-70M','Pythia'),('Pythia-160M','Pythia'),('Pythia-410M','Pythia'),('Pythia-1B','Pythia'),('Pythia-1.4B','Pythia'),('Pythia-2.8B','Pythia'),('Pythia-6.9B','Pythia'),('Pythia-12B','Pythia'),('GPT-2 Small','GPT-2'),('GPT-2 Medium','GPT-2'),('GPT-2 Large','GPT-2'),('GPT-2 XL','GPT-2'),('OPT-125M','OPT'),('OPT-350M','OPT'),('OPT-1.3B','OPT')]
def completion():
 suite=read(RES/'raw/full_suite_results.csv');raw=read(RES/'raw/experiment_results.csv');llm=read(RES/'llm/expanded_llm_evaluations.csv');legacy_llm=read(RES/'llm/llm_checkpoint_results.csv');lit=read(RES/'literature/large_model_literature.csv')
 output=['# Final Completion Matrix','', 'Updated 2026-11-05. “Pilot” means a real run and saved evaluation exists, but the requested full budget/seed schedule is incomplete. No status is promoted from code presence alone.','', '| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |','|---|---:|---:|---|---:|---|---|---|']
 for name,par in CONVENTIONAL:
  r=[x for x in suite if x.get('model')==name and not x.get('status','').startswith('superseded')];old=[x for x in raw if x.get('model')==name and not x.get('status','').startswith('superseded')]
  combined=r+old;budget_values=set()
  for x in combined:
   if not x.get('budget_value'):continue
   unit=x.get('budget_unit','');value=x.get('budget_value')
   try:detail=json.loads(x.get('details') or '{}')
   except Exception:detail={}
   if unit=='optimizer_steps' and detail.get('epochs'):
    budget_values.add(f"{detail['epochs']} epoch ({value} optimizer steps)")
   elif unit=='optimizer_steps' and detail.get('warmup_epochs'):
    budget_values.add(f"{detail['warmup_epochs']} warm-start + {detail.get('ssl_epochs',0)} SSL epoch ({value} optimizer steps)")
   else:budget_values.add(f'{value} {unit}')
  budget='; '.join(sorted(budget_values)) or 'none'
  seeds=len({x.get('seed') for x in combined if x.get('seed')})
  if r:
   hw=', '.join(sorted({x.get('hardware','') for x in r}));status='pilot'
  elif old:
   hw=', '.join(sorted({x.get('hardware','') for x in old}));status='pilot'
  else:hw='none';status='failed'
  result='[full_suite_results.csv](results/raw/full_suite_results.csv)' if r else ('[experiment_results.csv](results/raw/experiment_results.csv)' if old else 'none')
  display=name
  if name=='Vision Transformer (ViT-Tiny)':display='Vision Transformer (compact ViT-Tiny-style; custom 3-layer/96-wide model)'
  if name=='Label Propagation' and r:status='pilot; LabelSpreading proxy, requested Label Propagation not implemented'
  output.append(f'| {display} | yes | {"yes" if combined else "no"} | {budget} | {seeds} | {hw} | {result} | {status} |')
 output+=['','## LLM configurations','', '| Model | Implemented | Evaluated | Budgets/checkpoints completed | Seeds completed | Hardware | Result location | Status |','|---|---:|---:|---|---:|---|---|---|']
 for name,fam in LLM:
  rr=[x for x in llm if x.get('model')==name]
  if name=='Pythia-70M':rr+= [x for x in legacy_llm if x.get('model')==name and x.get('status')=='completed' and x.get('checkpoint_id')=='step143000']
  success=[x for x in rr if x.get('status')=='completed'];status=('pilot' if success else (rr[-1].get('status','resource_limited') if rr else 'resource_limited'))
  cps=', '.join(x.get('checkpoint_id','') for x in success) or 'none'
  if name in ('Pythia-160M','Pythia-410M') and 'main' in cps:cps=cps.replace('main, ','')+' (main is duplicate alias of step143000)'
  hw='CPU' if success else ('CPU weight-fetch attempt' if status=='failed' else 'not loaded')
  result='[expanded_llm_evaluations.csv](results/llm/expanded_llm_evaluations.csv)'
  if name=='Pythia-70M':result+='; [llm_checkpoint_results.csv](results/llm/llm_checkpoint_results.csv)'
  output.append(f"| {name} | yes | {'yes' if success else 'no'} | {cps} | {1 if success else 0} | {hw} | {result} | {status} |")
 output+=['','## Published literature configurations','', '| Model | Implemented | Evaluated | Budgets/checkpoints completed | Seeds completed | Hardware | Result location | Status |','|---|---:|---:|---|---:|---|---|---|']
 for r in lit:
  output.append(f"| {r.get('model','')} | n/a | yes (published report) | published configuration; not locally reproduced | n/a | published hardware not fully extracted | [large_model_literature.csv](results/literature/large_model_literature.csv) | literature_only |")
 (ROOT/'FINAL_COMPLETION_MATRIX.md').write_text('\n'.join(output)+'\n',encoding='utf-8')

def remaining():
 text='''# Remaining Work\n\nThese items remain incomplete after the runs recorded in this project.\n\n1. **Conventional-model budget and seed schedules:** All 20 requested models have implementation/evaluation records, but the new runs use a single seed and one feasibility budget on fixed subsets. The full 1/3/5/10/20/30 epoch schedules, progressive unsupervised sweeps, multiple labeled fractions, and RL 5K/10K/25K/50K/100K interaction schedules were not completed. The one-seed results cannot yield reliable means, standard deviations, confidence intervals, or saturation points.\n2. **Training/validation histories:** The new neural runs saved endpoint metrics, not per-epoch train and validation loss histories. Requested loss-versus-epoch figures and confidence intervals are therefore unavailable.\n3. **LLM model coverage:** Six of the 15 model configurations have local WikiText-2 prefix evaluations; GPT-2 Medium weight retrieval stalled with zero-byte partial downloads and no score. Eight models above the conservative 450M parameter guard were identified from public model configs but their weights were not loaded because only about 4.6 GiB RAM was free at initial inspection. The local evaluation is one sequence and one seed, not lm-eval downstream benchmarks. Pythia-70M has eight checkpoints; Pythia-160M and Pythia-410M each have three checkpoints (step 0, 10K, and 143K), not the requested 8–12 checkpoints per feasible model.\n4. **NPU execution:** The NPU device and compatible driver are present, but no Ryzen AI / VitisAI or Windows ML execution provider was installed in the existing environment. No NPU inference benchmark was run. NPU provider installation/model compilation and verified inference remain outstanding.\n5. **Hardware utilization and memory telemetry:** CPU vs DirectML microbenchmark results were captured for one MLP. The system exposes global GPU performance counters, but per-run utilization and DirectML VRAM were not isolated; the benchmark is a small smoke comparison, not a robust hardware study.\n6. **Literature verification:** The existing 15-row ledger contains attributed selected values, but exact page/table locators and all requested per-configuration fields (compute, hardware, duration, epochs, train/validation loss, and compatible benchmark scores) are not verified/extracted for every row. Missing values remain blank; no cross-paper scaling fit is justified yet.\n7. **Complete computation analysis and visualizations:** The new plots show observed one-budget pilots and the small Pythia trajectory. Marginal gain/efficiency curves, full training-time/compute curves, model-size comparisons under standardized benchmarks, and error bars require the missing repeated-budget/multi-seed measurements.\n8. **GPU backend qualification:** The Radeon 860M ran the DirectML feasibility batch, but Adam emitted CPU fallback warnings for some optimizer operations. Operator-level placement and per-run utilization should be profiled before treating those runs as fully GPU-resident.\n9. **ROCm on the current Windows install:** ROCm 7.14.1 lists Ryzen AI 7 350/Radeon 860M support, but requires Adrenalin 26.6.4 or OEM driver 26.10.28. The installed 2025 display driver is older. I left the OEM driver unchanged and used verified DirectML; ROCm requires a supported driver update and backend qualification.\n10. **Method fidelity for two conventional entries:** The requested Label Propagation algorithm was represented by scikit-learn LabelSpreading, which is related but distinct. The ViT-Tiny result uses a custom compact 3-layer, 96-wide transformer rather than a standard ViT-Tiny configuration. Treat both as feasibility proxies, not exact reproductions of the named architectures/algorithms.\n'''
 (ROOT/'REMAINING_WORK.md').write_text(text,encoding='utf-8')

def main():
 normalize();
 # Marginal analysis for one-budget records is retained but does not imply a trajectory.
 suite=read(RES/'raw/full_suite_results.csv')
 out=[]
 for r in suite:
  d=dict(r);d.update(marginal_gain='',marginal_cost_seconds='',marginal_efficiency_per_second='',relative_improvement='',saturation_0_5pct='false',saturation_1_0pct='false',saturation_2_0pct='false');out.append(d)
 write(RES/'processed/full_suite_marginal_results.csv',out)
 plots();completion();remaining();print('Refreshed status matrix, remaining work, analysis and figures.')
if __name__=='__main__':main()
