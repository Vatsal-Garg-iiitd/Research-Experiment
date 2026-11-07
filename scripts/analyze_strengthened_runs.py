"""Aggregate replicated sweeps, marginal gains, uncertainty, and source figures."""
from __future__ import annotations
import csv, math, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'results/raw';PROC=ROOT/'results/processed';FIG=ROOT/'results/figures';PROC.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
SOURCES=[('replicated_scaling_results.csv','supervised_epochs'),('random_forest_scaling.csv','rf_trees'),('unsupervised_scaling.csv','unsupervised'),('pseudolabel_baseline_sweep.csv','ssl'),('rl_scaling_results.csv','rl'),('dbscan_parameter_sweep.csv','dbscan_grid'),('autoencoder_scaling.csv','autoenc')]
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def num(v):
 try:x=float(v);return x if math.isfinite(x) else None
 except (TypeError,ValueError):return None
def mean_ci(vals):
 vals=[v for v in vals if v is not None]
 if not vals:return '', '', '', '',0
 n=len(vals);m=statistics.mean(vals);sd=statistics.stdev(vals) if n>1 else 0.;half=student_t.ppf(.975,n-1)*sd/math.sqrt(n) if n>1 else ''
 return m,sd,m-half if n>1 else '',m+half if n>1 else '',n
def main():
 allrows=[]
 for filename,kind in SOURCES:
  p=RAW/filename
  if not p.exists():continue
  for r in read(p):
   status=r.get('experiment_status',r.get('status',''))
   if status.startswith(('superseded','failed','resource_limited','running')):continue
   r['_kind']=kind;r['_source_file']=filename;allrows.append(r)
 # Long-form seed-level measurements to group for honest uncertainty summaries.
 metrics=[]
 for r in allrows:
  if r['_kind']=='supervised_epochs':
   fields={'validation_accuracy':r.get('validation_accuracy'),'training_accuracy':r.get('training_accuracy'),'training_loss':r.get('training_loss'),'validation_loss':r.get('validation_loss')}
  elif r['_kind']=='rf_trees':fields={'validation_accuracy':r.get('evaluation_value'),'validation_log_loss':r.get('validation_log_loss')}
  elif r['_kind']=='unsupervised':fields={'validation_ARI':r.get('evaluation_value'),'validation_NMI':r.get('validation_NMI'),'validation_silhouette':r.get('validation_silhouette'),'validation_log_likelihood':r.get('training_log_likelihood')}
  elif r['_kind']=='ssl':fields={'validation_accuracy':r.get('evaluation_value'),'validation_loss':r.get('validation_loss'),'training_loss':r.get('mean_supervised_training_loss'),'unsupervised_loss':r.get('mean_unsupervised_loss'),'pseudo_label_acceptance_fraction':r.get('pseudo_label_acceptance_fraction')}
  elif r['_kind']=='rl':fields={'mean_return':r.get('mean_return'),'episode_return_sd':r.get('return_sd')}
  elif r['_kind']=='dbscan_grid':fields={'validation_ARI':r.get('validation_ARI'),'validation_NMI':r.get('validation_NMI'),'validation_silhouette_non_noise':r.get('validation_silhouette_non_noise')}
  else:fields={'validation_reconstruction_mse':r.get('validation_reconstruction_mse'),'downstream_validation_accuracy':r.get('downstream_validation_accuracy'),'training_objective_loss':r.get('train_objective_loss')}
  for metric,value in fields.items():
   val=num(value)
   if val is not None:metrics.append({'model_name':r.get('model_name'),'method':r.get('method',''),'labeled_fraction':r.get('labeled_fraction',''),'budget_type':'eps' if r['_kind']=='dbscan_grid' else r.get('budget_type'),'budget_value':r.get('eps') if r['_kind']=='dbscan_grid' else (r.get('requested_budget') if r['_kind']=='rl' else r.get('budget_value')),'seed':r.get('seed'),'metric':metric,'value':val,'experiment_id':r.get('experiment_id'),'source_file':r['_source_file']})
 grouped=defaultdict(list)
 for r in metrics:grouped[(r['model_name'],r['method'],r['labeled_fraction'],r['budget_type'],r['budget_value'],r['metric'])].append(r['value'])
 sums=[]
 for key,vals in sorted(grouped.items(),key=lambda kv:str(kv[0])):
  m,sd,lo,hi,n=mean_ci(vals);sums.append(dict(zip(['model_name','method','labeled_fraction','budget_type','budget_value','metric'],key))|{'n_seeds':n,'mean':m,'standard_deviation':sd,'ci95_low':lo,'ci95_high':hi})
 write(PROC/'strengthened_statistical_summaries.csv',sums)
 # Adjacent-budget improvements are paired within model, seed and SSL setting.
 series=defaultdict(list)
 for r in allrows:
  if r.get('experiment_status')!='completed':continue
  if r['_kind']=='supervised_epochs':
   model=r['model_name'];metric='validation_accuracy';score=num(r.get('validation_accuracy'));budget=num(r.get('budget_value'));cum=num(r.get('cumulative_training_time_seconds'));unit='epochs';setting=''
  elif r['_kind']=='rf_trees':
   model=r['model_name'];metric='validation_accuracy';score=num(r.get('evaluation_value'));budget=num(r.get('budget_value'));cum=num(r.get('cumulative_training_time_seconds'));unit='trees';setting=''
  elif r['_kind']=='unsupervised':
   model=r['model_name'];metric='validation_ARI';score=num(r.get('evaluation_value'));budget=num(r.get('budget_value'));cum=num(r.get('cumulative_training_time_seconds'));unit=r.get('budget_type');setting=''
  elif r['_kind']=='ssl':
   model='Pseudo-Labeling' if r.get('method')=='pseudo_label' else 'Labeled-only baseline';metric='validation_accuracy';score=num(r.get('evaluation_value'));budget=num(r.get('budget_value'));cum=num(r.get('cumulative_training_time_seconds'));unit='epochs';setting=f"frac={r.get('labeled_fraction')}"
  elif r['_kind']=='rl':
   model=r.get('model_name');metric='mean_return';score=num(r.get('mean_return'));budget=num(r.get('environment_steps'));cum=num(r.get('cumulative_training_time_seconds'));unit='environment_steps';setting=r.get('dataset','')
  elif r['_kind']=='autoenc':
   model=r.get('model_name');metric='validation_reconstruction_mse';score=num(r.get('validation_reconstruction_mse'));budget=num(r.get('budget_value'));cum=num(r.get('cumulative_training_time_seconds'));unit='epochs';setting=''
   # Add the classification probe as its own higher-is-better series.
   probe=num(r.get('downstream_validation_accuracy'))
   if probe is not None:series[(model,'','downstream linear probe',r.get('seed',''),unit,'downstream_validation_accuracy')].append((budget,probe,cum,r.get('experiment_id')))
  else:continue # DBSCAN parameter candidates are not a nested compute-budget trajectory.
  if score is None or budget is None:continue
  series[(model,r.get('method',''),setting,r.get('seed',''),unit,metric)].append((budget,score,cum,r.get('experiment_id')))
 marginal=[]
 for key,pts in series.items():
  pts=sorted(pts);prev=None
  for b,score,cum,eid in pts:
   if prev:
    pb,ps,pc,pid=prev;gain=(ps-score) if key[5] in ('validation_reconstruction_mse','training_objective_loss','validation_loss') else score-ps;dc=(b-pb);dt=(cum-pc) if cum is not None and pc is not None else None
    marginal.append({'model_name':key[0],'method':key[1],'setting':key[2],'seed':key[3],'metric':key[5],'from_budget':pb,'to_budget':b,'budget_unit':key[4],'budget_delta':dc,'performance_improvement':gain,'marginal_gain_per_budget_unit':gain/dc if dc>0 else '', 'additional_training_seconds':dt if dt is not None else '', 'marginal_improvement_per_second':gain/dt if dt is not None and dt>0 else '', 'relative_improvement':gain/abs(ps) if abs(ps)>1e-12 else '', 'from_experiment_id':pid,'to_experiment_id':eid})
   prev=(b,score,cum,eid)
 write(PROC/'strengthened_marginal_gains.csv',marginal)
 # Test-set endpoints aggregate independent seed runs, separately from validation curves.
 tests=[]
 for filename,groupkeys,metriccols in [('replicated_scaling_test_results.csv',['model_name'],['test_accuracy','test_loss','macro_precision','macro_recall','macro_f1']),('random_forest_test_results.csv',['model_name'],['test_accuracy','test_log_loss','macro_precision','macro_recall','macro_f1']),('pseudolabel_test_results.csv',['model_name','method','labeled_fraction'],['test_accuracy','test_loss','macro_precision','macro_recall','macro_f1']),('autoencoder_test_results.csv',['model_name'],['test_reconstruction_mse','test_linear_probe_accuracy'])]:
  p=RAW/filename
  if p.exists():
   for r in read(p):
    if r.get('status','').startswith(('superseded','failed','resource_limited','running')):continue
    for metriccol in metriccols:
     if num(r.get(metriccol)) is not None:tests.append((filename,r,groupkeys,metriccol))
 p=RAW/'rl_scaling_results.csv'
 if p.exists():
  for r in read(p):
   if int(r.get('requested_budget') or 0)==25000:tests.append(('rl_scaling_results.csv',r,['model_name'],'mean_return'))
 tg=defaultdict(list)
 for fn,r,keys,col in tests:
  key=(fn,*[r.get(k,'') for k in keys],col);tg[key].append(num(r.get(col)))
 tsum=[]
 for key,vals in tg.items():
  m,sd,lo,hi,n=mean_ci(vals);tsum.append({'source_file':key[0],'group':' / '.join(key[1:-1]),'metric':key[-1],'n_seeds':n,'mean':m,'standard_deviation':sd,'ci95_low':lo,'ci95_high':hi})
 write(PROC/'strengthened_test_statistics.csv',tsum)
 plots(sums,marginal)
 write_report(sums,marginal,tsum)
 print(f'Wrote {len(sums)} grouped summaries, {len(marginal)} paired marginal increments, {len(tsum)} test endpoints')

def plots(sums,marginal):
 by={}
 for r in sums:by.setdefault((r['model_name'],r['method'],r['labeled_fraction'],r['metric']),[]).append(r)
 # supervised training/validation curves with t intervals.
 for metric,file,title in [('validation_accuracy','strengthened_supervised_validation_accuracy.png','Validation accuracy vs epochs'),('validation_loss','strengthened_supervised_validation_loss.png','Validation loss vs epochs')]:
  fig,ax=plt.subplots(figsize=(7.5,4.8))
  for model in ('MLP','CNN / LeNet','ResNet-18','Vision Transformer (compact ViT-Tiny-style)'):
   rr=sorted(by.get((model,'','',''+metric),[]),key=lambda x:int(x['budget_value']))
   if rr:ax.errorbar([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],yerr=[float(x['mean'])-float(x['ci95_low']) for x in rr],marker='o',capsize=3,label=model)
  ax.set(xlabel='Epoch',ylabel=metric.replace('_',' ').title(),title=title+' (10k train subset; 3 seeds; 95% t intervals)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/file,dpi=200);plt.close(fig)
 # Representation learning reconstruction and downstream quality remain separate metrics.
 for metric,file,title in [('validation_reconstruction_mse','strengthened_autoencoder_reconstruction.png','Validation reconstruction MSE vs epochs'),('downstream_validation_accuracy','strengthened_autoencoder_linear_probe.png','Downstream linear-probe accuracy vs epochs')]:
  fig,ax=plt.subplots(figsize=(7.5,4.8))
  for model in ('Autoencoder','Variational Autoencoder'):
   rr=sorted(by.get((model,'','',metric),[]),key=lambda x:int(x['budget_value']))
   if rr:ax.errorbar([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],yerr=[float(x['mean'])-float(x['ci95_low']) for x in rr],marker='o',capsize=3,label=model)
  ax.set(xlabel='Epoch',ylabel=metric.replace('_',' ').title(),title=title+' (3 seeds; 95% t intervals)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/file,dpi=200);plt.close(fig)
 # Random forest complexity curve.
 rr=sorted(by.get(('Random Forest','','','validation_accuracy'),[]),key=lambda x:int(x['budget_value']))
 if rr:
  fig,ax=plt.subplots(figsize=(7.5,4.8));ax.errorbar([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],yerr=[float(x['mean'])-float(x['ci95_low']) for x in rr],marker='o',capsize=3);ax.set(xlabel='Trees',ylabel='Validation accuracy',title='Random Forest validation accuracy vs tree count (3 seeds; 95% t intervals)');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/'strengthened_random_forest_tree_curve.png',dpi=200);plt.close(fig)
 # Unsupervised fits use separate update units/panels.
 fig,axs=plt.subplots(1,2,figsize=(11,4.5))
 for ax,model,xlab in [(axs[0],'MiniBatch K-Means','partial_fit updates'),(axs[1],'Gaussian Mixture Model','warm-start EM iterations')]:
  rr=sorted(by.get((model,'','','validation_ARI'),[]),key=lambda x:int(x['budget_value']))
  if rr:ax.errorbar([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],yerr=[float(x['mean'])-float(x['ci95_low']) for x in rr],marker='o',capsize=3)
  ax.set(xlabel=xlab,ylabel='Validation ARI',title=model+' (3 seeds)');ax.grid(alpha=.25)
 fig.tight_layout();fig.savefig(FIG/'strengthened_unsupervised_ari_curves.png',dpi=200);plt.close(fig)
 # DBSCAN is a hyperparameter grid, not a training-budget trajectory.
 dbpath=RAW/'dbscan_parameter_sweep.csv'
 if dbpath.exists():
  dr=read(dbpath);fig,ax=plt.subplots(figsize=(7.5,4.8))
  for ms in sorted({int(r['min_samples']) for r in dr}):
   part=sorted([r for r in dr if int(r['min_samples'])==ms],key=lambda r:float(r['eps']));ax.plot([float(r['eps']) for r in part],[float(r['validation_ARI']) for r in part],marker='o',label=f'min_samples={ms}')
  ax.set(xlabel='eps (PCA32 Euclidean distance)',ylabel='Transductive validation ARI',title='DBSCAN density-parameter sensitivity');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/'strengthened_dbscan_parameter_sensitivity.png',dpi=200);plt.close(fig)
 # SSL epoch and label-fraction curves; separate chart at each method.
 for method,label in [('pseudo_label','Pseudo-labeling'),('labeled_only','Labeled-only baseline')]:
  fig,ax=plt.subplots(figsize=(7.5,4.8))
  for frac in ('0.01','0.05','0.1','0.2','0.4'):
   rr=sorted(by.get(('MLP',method,frac,'validation_accuracy'),[]),key=lambda x:int(x['budget_value']))
   if rr:ax.plot([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],marker='o',label=f'{float(frac):.0%} labels')
  ax.set(xlabel='Epoch',ylabel='Validation accuracy',title=f'{label} learning curves by labeled fraction (3 seeds)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/f"strengthened_{method}_learning_curves.png",dpi=200);plt.close(fig)
 # RL plots separate the CartPole discrete-action group from Pendulum continuous control.
 for env_id,models,slug in [('CartPole-v1',['DQN','Double DQN','A2C','PPO'],'cartpole'),('Pendulum-v1',['SAC'],'pendulum')]:
  fig,axs=plt.subplots(1,2,figsize=(11,4.6))
  for model in models:
   rr=sorted(by.get((model,'','','mean_return'),[]),key=lambda x:int(x['budget_value']))
   if rr:
    xx=[int(x['budget_value']) for x in rr];yy=[float(x['mean']) for x in rr];ee=[float(x['mean'])-float(x['ci95_low']) for x in rr]
    axs[0].errorbar(xx,yy,yerr=ee,marker='o',capsize=3,label=model)
    relevant=[r for r in marginal if r['model_name']==model and r['setting']==env_id and r['marginal_improvement_per_second'] not in ('',None)]
    eff=defaultdict(list)
    for r in relevant:eff[int(float(r['to_budget']))].append(float(r['marginal_improvement_per_second']))
    if eff:axs[1].plot(sorted(map(int,eff)),[statistics.mean(eff[x]) for x in sorted(map(int,eff))],marker='o',label=model)
  axs[0].set(xlabel='Environment steps',ylabel='Mean return over 20 eval episodes',title=f'{env_id}: return vs interactions (mean; n=3)');axs[1].set(xlabel='Checkpoint environment steps',ylabel='Return improvement / training second',title='Marginal reward efficiency (seed means)')
  for ax in axs:ax.grid(alpha=.25);ax.legend()
  fig.tight_layout();fig.savefig(FIG/f'strengthened_rl_{slug}_curves.png',dpi=200);plt.close(fig)
 # Within-run marginal efficiency is plotted separately by computational paradigm.
 groups={'mlp':['MLP'],'lenet':['CNN / LeNet'],'random_forest':['Random Forest'],'kmeans':['MiniBatch K-Means'],'gmm':['Gaussian Mixture Model']}
 for slug,models in groups.items():
  fig,ax=plt.subplots(figsize=(7.5,4.8));any_data=False
  for model in models:
   mr=[r for r in marginal if r['model_name']==model and r['marginal_improvement_per_second'] not in ('',None)]
   if not mr:continue
   agg=defaultdict(list)
   for r in mr:agg[(r['from_budget'],r['to_budget'],r['budget_unit'])].append(float(r['marginal_improvement_per_second']))
   keys=sorted(agg,key=lambda v:float(v[1]));xs=list(range(len(keys)));ys=[statistics.mean(agg[k]) for k in keys]
   ax.plot(xs,ys,marker='o',label=model);any_data=True
  if any_data:
   ax.set_xticks(range(len(keys)),[f'{a}→{b}\n{u}' for a,b,u in keys]);ax.set(ylabel='Metric improvement / training second',title=f'{slug.replace("_"," ").title()} marginal efficiency (seed means)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/f'strengthened_{slug}_marginal_efficiency.png',dpi=200)
  plt.close(fig)
 # Autoencoder cost/performance curves use separate lower/higher-is-better metrics.
 for metric,slug in [('validation_reconstruction_mse','reconstruction'),('downstream_validation_accuracy','linear_probe')]:
  fig,ax=plt.subplots(figsize=(7.5,4.8));found=False
  for model in ('Autoencoder','Variational Autoencoder'):
   mr=[r for r in marginal if r['model_name']==model and r['metric']==metric and r['marginal_improvement_per_second'] not in ('',None)];agg=defaultdict(list)
   for r in mr:agg[(int(float(r['to_budget'])),r['budget_unit'])].append(float(r['marginal_improvement_per_second']))
   if agg:
    ks=sorted(agg);ax.plot([k[0] for k in ks],[statistics.mean(agg[k]) for k in ks],marker='o',label=model);found=True
  if found:
   ax.set(xlabel='Epoch checkpoint',ylabel='Favorable metric change / model training second',title=f'Autoencoder {slug} marginal efficiency (seed means)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/f'strengthened_autoencoder_{slug}_marginal_efficiency.png',dpi=200)
  plt.close(fig)

def write_report(sums,marginal,tests):
 p=ROOT/'STATISTICAL_ANALYSIS_REPORT.md';lines=['# Statistical analysis of strengthened runs','','Analyses use saved local measurements from MLP/LeNet/ResNet/custom-ViT epoch sweeps, a Random Forest tree curve, MiniBatch K-Means/GMM optimization curves, AE/VAE reconstruction and linear-probe sweeps, pseudo-label/labeled-only ablation, RL budget curves, and a separate DBSCAN density-parameter grid. Most learning curves use a fixed 10k-train/2k-validation Fashion-MNIST subset; RL uses CartPole/Pendulum and DBSCAN has transductive validation. These are exploratory partial results, not a completed 20-model confirmatory study.','',f'- Validation summary rows: {len(sums)}.','- Paired within-seed adjacent-budget marginal increments: '+str(len(marginal))+'.','- Held-out test endpoint summary rows: '+str(len(tests))+'.','- Confidence intervals: two-sided 95% Student t intervals across seeds; n is three for the new sweeps. Intervals are wide with n=3 and do not establish generalization.','- Budget observations within a seed are repeated measures. Marginal increments are calculated within seed first; no checkpoint is treated as an independent seed.','- Test results are aggregated only at final checkpoints; validation data drives curves. No p-values or cross-model significance claims are reported.','- Initial AE/VAE v1 probe rows are retained but superseded after convergence warnings; analyses use the standardized v2 probe.','','## Final held-out test endpoints','']
 lines+=['| Run group | Metric | Seeds | Mean | SD | 95% CI |','|---|---|---:|---:|---:|---|']
 for r in tests:lines.append(f"| {r['group']} | {r['metric']} | {r['n_seeds']} | {float(r['mean']):.4f} | {float(r['standard_deviation']):.4f} | [{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}] |")
 lines+=['','## Interpretation limits','','The MLP, LeNet, ResNet-18, compact ViT-style, Random Forest, AE/VAE and RL curves use three training seeds on fixed splits/subsets; uncertainty is conditional on those splits and does not quantify dataset-sampling uncertainty. K-Means/GMM ARI varies non-monotonically across updates/iterations and seeds; no saturation model is fit. DBSCAN is a deterministic parameter grid, not a seed-level budget curve. Pseudo-label vs labeled-only comparisons are paired by seed and labeled subset; current results vary with fraction and seed, so no universal SSL benefit is claimed. RL uses 20 deterministic evaluation episodes per checkpoint and three training seeds; within-run episode dispersion is separately retained. Historical pilot curves in `experiment_results.csv` include repeated test-set looks and remain exploratory.','']
 p.write_text('\n'.join(lines),encoding='utf-8')
 if marginal:
  lines=['# Marginal-gain analysis','','All marginal values below are computed only for the new saved multi-budget runs. Improvement is positive in the favorable direction (higher accuracy/ARI is better). `gain / budget` uses the native budget; time efficiency uses the measured increment in cumulative training time. These ratios are descriptive and task-specific.','','- MLP/LeNet: epoch budgets; validation accuracy.','- Random Forest: cumulative warm-start tree count; validation accuracy.','- MiniBatch K-Means: actual `partial_fit` updates; validation ARI.','- Gaussian Mixture: repeated warm-start fit calls, one EM iteration each; validation ARI.','- Pseudo-labeling/labeled-only: epochs, per seed and label fraction; validation accuracy.','- Autoencoder/VAE: epoch increments for reconstruction MSE (lower is better) and downstream linear-probe accuracy (higher is better).','- RL: environment interactions; CartPole and Pendulum remain separate tasks.','- No cross-paradigm pooling. Saturation is not declared from these short curves.','','## Seed-level marginal data','', '| Model | Setting | Seed | Budget interval | Improvement | Gain / budget | Additional seconds | Gain / second | Relative improvement |','|---|---|---:|---|---:|---:|---:|---:|---:|']
  for r in marginal:lines.append(f"| {r['model_name']} | {r['setting'] or r['method']} | {r['seed']} | {r['from_budget']}–{r['to_budget']} {r['budget_unit']} | {float(r['performance_improvement']):.5f} | {float(r['marginal_gain_per_budget_unit']):.5f} | {float(r['additional_training_seconds']):.4f} | {float(r['marginal_improvement_per_second']):.5f} | {float(r['relative_improvement']):.3%} |")
  (ROOT/'MARGINAL_GAIN_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
