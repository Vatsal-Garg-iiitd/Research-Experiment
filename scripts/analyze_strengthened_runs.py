"""Aggregate replicated sweeps, marginal gains, uncertainty, and source figures."""
from __future__ import annotations
import csv, json, math, re, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'results/raw';PROC=ROOT/'results/processed';FIG=ROOT/'results/figures';PROC.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
SOURCES=[('replicated_scaling_results.csv','supervised_epochs'),('vit_tiny_scaling.csv','supervised_epochs'),('random_forest_scaling.csv','rf_trees'),('unsupervised_scaling.csv','unsupervised'),('pseudolabel_baseline_sweep.csv','ssl'),('ssl_algorithms.csv','ssl_methods'),('rl_scaling_results.csv','rl'),('dbscan_parameter_sweep.csv','dbscan_grid'),('autoencoder_scaling.csv','autoenc')]
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
 if not vals:return '', '', '', '', '',0
 n=len(vals);m=statistics.mean(vals);sd=statistics.stdev(vals) if n>1 else 0.;med=statistics.median(vals);half=student_t.ppf(.975,n-1)*sd/math.sqrt(n) if n>1 else ''
 return m,sd,med,m-half if n>1 else '',m+half if n>1 else '',n
def lower_error(row):
 """Return a one-sided lower CI error, or zero when n=1 has no interval."""
 try:return max(0.0,float(row['mean'])-float(row['ci95_low']))
 except (TypeError,ValueError,KeyError):return 0.0
def fmt(v, spec):
 try:return format(float(v),spec)
 except (TypeError,ValueError):return 'n/a'
def main():
 allrows=[]; seen_unsupervised=set(); excluded=[]
 for filename,kind in SOURCES:
  p=RAW/filename
  if not p.exists():continue
  for source_row,r in enumerate(read(p),start=2):
   if filename=='rl_scaling_results.csv' and r.get('model_name')=='Double DQN' and r.get('details') in (None,''):
    r['details']=r.get('timestamp_utc','');r['timestamp_utc']=r.get('checkpoint_path','');r['checkpoint_path']=r.get('configuration_path','');r['configuration_path']=r.get('source_type','');r['source_type']='local_measurement';r['experiment_status']=r.get('backend','');r['backend']=r.get('hardware','');r['hardware']=r.get('peak_memory_bytes','');r['peak_memory_bytes']=r.get('success_rate','');r['success_rate']=''
   # The later K-Means append repeats the same experiment IDs and stamps each
   # budget with the final 50-update score. Keep the first raw occurrence only.
   if filename=='unsupervised_scaling.csv' and r.get('model_name')=='MiniBatch K-Means':
    eid=r.get('experiment_id','')
    if eid in seen_unsupervised:
     excluded.append({'source_file':f'results/raw/{filename}','source_row':source_row,'experiment_id':eid,'disposition':'excluded_duplicate_checkpoint_reuse','reason':'Repeated experiment ID; later appended row carries the 50-update score at an earlier budget.'})
     continue
    seen_unsupervised.add(eid)
   status=r.get('experiment_status',r.get('status',''))
   if status.startswith(('superseded','failed','resource_limited','running')):continue
   r['_kind']=kind;r['_source_file']=filename
   raw_cum=num(r.get('cumulative_training_time_seconds'))
   if filename=='replicated_scaling_results.csv' and r.get('model_name')=='ResNet-18' and str(r.get('seed'))=='11' and int(float(r.get('epochs') or 0))>=3 and raw_cum is not None:
    r['analysis_cumulative_training_time_seconds']=str(max(0,raw_cum-1122.1625285))
   else:r['analysis_cumulative_training_time_seconds']=r.get('cumulative_training_time_seconds','')
   allrows.append(r)
 write(PROC/'data_quality_exclusions.csv',excluded)
 quality=[{'source_file':x['source_file'],'source_row':x.get('source_row',''),'issue':'Duplicate K-Means checkpoint reuse','treatment':'Excluded later duplicate from derived curves; raw record preserved.','evidence':x['reason']} for x in excluded]
 quality += [
  {'source_file':'results/raw/replicated_scaling_results.csv','source_row':'','issue':'ResNet-18 seed 11 epoch-2-to-3 interval is 1,122.1625285 seconds','treatment':'Retained in raw-time analysis; removed only from adjusted efficiency diagnostic after epoch 2.','evidence':'Adjacent timestamps show an 18m44.634s gap and subsequent epoch takes about 9.7s; logs do not identify compute vs pause.'},
  {'source_file':'results/raw/replicated_learning_curves.csv; results/raw/replicated_scaling_results.csv','source_row':'','issue':'Parallel exports share all 420 experiment IDs','treatment':'Only replicated_scaling_results.csv used in canonical curve summaries.','evidence':'The learning-curves export is a mirror with reduced schema, not additional seeds.'},
  {'source_file':'results/raw/pseudolabel_learning_curves.csv; results/raw/pseudolabel_baseline_sweep.csv','source_row':'','issue':'Parallel exports share all 150 experiment IDs','treatment':'Only pseudolabel_baseline_sweep.csv used for analysis.','evidence':'The learning-curves export is a mirror with reduced schema, not additional runs.'},
  {'source_file':'results/raw/rl_scaling_results.csv','source_row':'38-67','issue':'Double DQN CSV writer omitted success_rate cell, shifting trailing fields left under headers','treatment':'Raw rows preserved; analysis and canonical builders realign columns by value signatures. success_rate remains unavailable; peak memory, hardware/backend, status, source/config/checkpoint paths, timestamp, and details are restored to their intended fields.','evidence':'Affected records have 28 values for 29 headers; type/path/timestamp/JSON sequence establishes the offset. Recovered status is completed; local_measurement is source_type.'},
  {'source_file':'results/llm/expanded_llm_evaluations.csv','source_row':'','issue':'Pythia-160M/410M main aliases overlap step143000','treatment':'Not counted as extra trajectory checkpoints.','evidence':'Repository audit documents the aliases; full-corpus endpoint remains one row per model.'},
  {'source_file':'results/processed/data_size_scaling_summary.csv','source_row':'','issue':'Prior summary mixed 10K epoch-5 and epoch-30 endpoints','treatment':'Rebuilt comparison with epoch-5 endpoints only.','evidence':'The 10K model-seed records now align with 30K/58K five-epoch size comparisons.'},
 ]
 with (PROC/'data_quality_register.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(quality[0]));w.writeheader();w.writerows(quality)
 qmd=['# Data-quality and anomaly register','','Raw observations are preserved. Treatments below apply only to derived analyses.','', '| Source | Issue | Analytical treatment | Evidence / qualification |','|---|---|---|---|']
 for x in quality:qmd.append(f"| `{x['source_file']}`{(' row '+str(x['source_row'])) if x['source_row'] else ''} | {x['issue']} | {x['treatment']} | {x['evidence']} |")
 (ROOT/'docs/DATA_QUALITY_REGISTER.md').write_text('\n'.join(qmd)+'\n',encoding='utf-8')
 # Long-form seed-level measurements to group for honest uncertainty summaries.
 metrics=[]
 for r in allrows:
  if r['_kind']=='supervised_epochs':
   fields={'validation_accuracy':r.get('validation_accuracy'),'training_accuracy':r.get('training_accuracy'),'training_loss':r.get('training_loss'),'validation_loss':r.get('validation_loss')}
  elif r['_kind']=='rf_trees':fields={'validation_accuracy':r.get('evaluation_value'),'validation_log_loss':r.get('validation_log_loss')}
  elif r['_kind']=='unsupervised':fields={'validation_ARI':r.get('evaluation_value'),'validation_NMI':r.get('validation_NMI'),'validation_silhouette':r.get('validation_silhouette'),'validation_log_likelihood':r.get('training_log_likelihood')}
  elif r['_kind']=='ssl':fields={'validation_accuracy':r.get('evaluation_value'),'validation_loss':r.get('validation_loss'),'training_loss':r.get('mean_supervised_training_loss'),'unsupervised_loss':r.get('mean_unsupervised_loss'),'pseudo_label_acceptance_fraction':r.get('pseudo_label_acceptance_fraction')}
  elif r['_kind']=='ssl_methods':fields={'validation_accuracy':r.get('validation_accuracy'),'supervised_loss':r.get('supervised_loss'),'unsupervised_loss':r.get('unsupervised_loss'),'pseudo_label_mean_confidence':r.get('pseudo_label_mean_confidence'),'fraction_unlabeled_used':r.get('fraction_unlabeled_used')}
  elif r['_kind']=='rl':fields={'mean_return':r.get('mean_return'),'episode_return_sd':r.get('return_sd')}
  elif r['_kind']=='dbscan_grid':fields={'validation_ARI':r.get('validation_ARI'),'validation_NMI':r.get('validation_NMI'),'validation_silhouette_non_noise':r.get('validation_silhouette_non_noise')}
  else:fields={'validation_reconstruction_mse':r.get('validation_reconstruction_mse'),'downstream_validation_accuracy':r.get('downstream_validation_accuracy'),'training_objective_loss':r.get('train_objective_loss')}
  for metric,value in fields.items():
   val=num(value)
   if val is not None:metrics.append({'model_name':r.get('model_name'),'method':r.get('method',''),'labeled_fraction':r.get('labeled_fraction',''),'split_id':r.get('split_id',''),'source_file':r['_source_file'],'budget_type':'eps' if r['_kind']=='dbscan_grid' else r.get('budget_type'),'budget_value':r.get('eps') if r['_kind']=='dbscan_grid' else (r.get('requested_budget') if r['_kind']=='rl' else r.get('budget_value')),'seed':r.get('seed'),'metric':metric,'value':val,'experiment_id':r.get('experiment_id')})
 grouped=defaultdict(list)
 for r in metrics:grouped[(r['model_name'],r['method'],r['labeled_fraction'],r['split_id'],r['source_file'],r['budget_type'],r['budget_value'],r['metric'])].append(r['value'])
 sums=[]
 for key,vals in sorted(grouped.items(),key=lambda kv:str(kv[0])):
  m,sd,med,lo,hi,n=mean_ci(vals);sums.append(dict(zip(['model_name','method','labeled_fraction','split_id','source_file','budget_type','budget_value','metric'],key))|{'n_seeds':n,'mean':m,'median':med,'minimum':min(vals),'maximum':max(vals),'standard_deviation':sd,'ci95_low':lo,'ci95_high':hi})
 write(PROC/'strengthened_statistical_summaries.csv',sums)
 # Adjacent-budget improvements are paired within model, seed and SSL setting.
 series=defaultdict(list)
 for r in allrows:
  if r.get('experiment_status',r.get('status','')) not in ('completed','local_measurement'):continue
  if r['_kind']=='supervised_epochs':
   model=r['model_name'];metric='validation_accuracy';score=num(r.get('validation_accuracy'));budget=num(r.get('budget_value'));cum=num(r.get('analysis_cumulative_training_time_seconds',r.get('cumulative_training_time_seconds')));unit='epochs';setting=''
  elif r['_kind']=='rf_trees':
   model=r['model_name'];metric='validation_accuracy';score=num(r.get('evaluation_value'));budget=num(r.get('budget_value'));cum=num(r.get('cumulative_training_time_seconds'));unit='trees';setting=''
  elif r['_kind']=='unsupervised':
   model=r['model_name'];metric='validation_ARI';score=num(r.get('evaluation_value'));budget=num(r.get('budget_value'));cum=num(r.get('cumulative_training_time_seconds'));unit=r.get('budget_type');setting=''
  elif r['_kind']=='ssl':
   model='Pseudo-Labeling' if r.get('method')=='pseudo_label' else 'Labeled-only baseline';metric='validation_accuracy';score=num(r.get('evaluation_value'));budget=num(r.get('budget_value'));cum=num(r.get('cumulative_training_time_seconds'));unit='epochs';setting=f"frac={r.get('labeled_fraction')}"
  elif r['_kind']=='ssl_methods':
   model=r['model_name'];metric='validation_accuracy';score=num(r.get('validation_accuracy'));budget=num(r.get('budget_value'));cum=num(r.get('cumulative_training_time_seconds'));unit='epochs';setting=f"frac={r.get('labeled_fraction')}"
  elif r['_kind']=='rl':
   model=r.get('model_name');metric='mean_return';score=num(r.get('mean_return'));budget=num(r.get('environment_steps'));cum=num(r.get('cumulative_training_time_seconds'));unit='environment_steps';setting=r.get('dataset','')
  elif r['_kind']=='autoenc':
   model=r.get('model_name');metric='validation_reconstruction_mse';score=num(r.get('validation_reconstruction_mse'));budget=num(r.get('budget_value'));cum=num(r.get('cumulative_training_time_seconds'));unit='epochs';setting=''
   # Add the classification probe as its own higher-is-better series.
   probe=num(r.get('downstream_validation_accuracy'))
   if probe is not None:series[(model,'','downstream linear probe',r.get('seed',''),unit,'downstream_validation_accuracy',r.get('split_id',''),r['_source_file'])].append((budget,probe,cum,r.get('experiment_id')))
  else:continue # DBSCAN parameter candidates are not a nested compute-budget trajectory.
  if score is None or budget is None:continue
  series[(model,r.get('method',''),setting,r.get('seed',''),unit,metric,r.get('split_id',''),r['_source_file'])].append((budget,score,cum,r.get('experiment_id')))
 marginal=[]
 for key,pts in series.items():
  pts=sorted(pts);prev=None;small_gain_counts={0.005:0,0.01:0,0.02:0}
  for b,score,cum,eid in pts:
   if prev:
    pb,ps,pc,pid=prev;gain=(ps-score) if key[5] in ('validation_reconstruction_mse','training_objective_loss','validation_loss') else score-ps;dc=(b-pb);dt=(cum-pc) if cum is not None and pc is not None else None
    relative=gain/abs(ps) if abs(ps)>1e-12 else None
    flags={}
    for threshold in small_gain_counts:
     small_gain_counts[threshold]=small_gain_counts[threshold]+1 if relative is not None and 0<=relative<threshold else 0
     flags[f'saturation_{threshold*100:g}pct']=small_gain_counts[threshold]>=2
    marginal.append({'model_name':key[0],'method':key[1],'setting':key[2],'seed':key[3],'metric':key[5],'split_id':key[6],'source_file':key[7],'from_budget':pb,'to_budget':b,'budget_unit':key[4],'budget_delta':dc,'performance_improvement':gain,'marginal_gain_per_budget_unit':gain/dc if dc>0 else '', 'additional_training_seconds':dt if dt is not None else '', 'marginal_improvement_per_second':gain/dt if dt is not None and dt>0 else '', 'seconds_per_unit_improvement':dt/gain if dt is not None and dt>0 and gain>0 else '', 'relative_improvement':relative if relative is not None else '', **flags,'from_experiment_id':pid,'to_experiment_id':eid})
   prev=(b,score,cum,eid)
 write(PROC/'strengthened_marginal_gains.csv',marginal)
 # Test-set endpoints aggregate independent seed runs, separately from validation curves.
 tests=[]
 for filename,groupkeys,metriccols in [('replicated_scaling_test_results.csv',['model_name','split_id','epochs'],['test_accuracy','test_loss','macro_precision','macro_recall','macro_f1']),('vit_tiny_test_results.csv',['model_name','split_id','epochs'],['test_accuracy','test_loss','macro_precision','macro_recall','macro_f1']),('random_forest_test_results.csv',['model_name','split_id','trees'],['test_accuracy','test_log_loss','macro_precision','macro_recall','macro_f1']),('pseudolabel_test_results.csv',['model_name','method','labeled_fraction','split_id','epochs'],['test_accuracy','test_loss','macro_precision','macro_recall','macro_f1']),('ssl_algorithms_test.csv',['model_name','labeled_fraction','split_id','epochs'],['test_accuracy']),('autoencoder_test_results.csv',['model_name','split_id','epochs'],['test_reconstruction_mse','test_linear_probe_accuracy'])]:
  p=RAW/filename
  if p.exists():
   for r in read(p):
    if r.get('status','').startswith(('superseded','failed','resource_limited','running')):continue
    for metriccol in metriccols:
     if num(r.get(metriccol)) is not None:tests.append((filename,r,groupkeys,metriccol))
 p=RAW/'rl_scaling_results.csv'
 if p.exists():
  for r in read(p):
   if int(r.get('requested_budget') or 0)==25000:tests.append(('rl_scaling_results.csv',r,['model_name','dataset'],'mean_return'))
 tg=defaultdict(list)
 for fn,r,keys,col in tests:
  key=(fn,*[r.get(k,'') for k in keys],col);tg[key].append(num(r.get(col)))
 tsum=[]
 for key,vals in tg.items():
  m,sd,med,lo,hi,n=mean_ci(vals);tsum.append({'source_file':key[0],'group':' / '.join(key[1:-1]),'metric':key[-1],'n_seeds':n,'mean':m,'median':med,'minimum':min(vals),'maximum':max(vals),'standard_deviation':sd,'ci95_low':lo,'ci95_high':hi})
 write(PROC/'strengthened_test_statistics.csv',tsum)
 plots(sums,marginal)
 write_report(sums,marginal,tsum)
 plot_data_size_endpoints()
 print(f'Wrote {len(sums)} grouped summaries, {len(marginal)} paired marginal increments, {len(tsum)} test endpoints')

def plot_data_size_endpoints():
 """Compare held-out test endpoints at matched five-epoch budgets by data size."""
 source=RAW/'replicated_scaling_test_results.csv'
 if not source.exists():return
 grouped=defaultdict(list);details=[]
 for row in read(source):
  match=re.search(r'_n(\d+)_',row.get('experiment_id',''))
  if not match or row.get('status')!='completed' or int(float(row.get('epochs') or 0))!=5 or num(row.get('test_accuracy')) is None:continue
  requested=int(match.group(1));run=row['experiment_id'].split('_test_epoch')[0]
  cfg=ROOT/'results/configurations'/f'{run}.json';actual=requested
  if cfg.exists():
   try:actual=int(json.loads(cfg.read_text(encoding='utf-8')).get('train_size',requested))
   except (ValueError,TypeError):pass
  grouped[(row.get('model_name',''),requested,actual)].append(float(row['test_accuracy']))
  details.append({'model_name':row.get('model_name',''),'requested_train_size':requested,'actual_train_size':actual,'seed':row.get('seed',''),'test_accuracy':row['test_accuracy'],'experiment_id':row['experiment_id'],'status':row['status']})
 summary=[]
 for (model,requested,actual),values in sorted(grouped.items()):
  mean,sd,median,lo,hi,n=mean_ci(values)
  summary.append({'model_name':model,'requested_train_size':requested,'actual_train_size':actual,'budget_type':'epochs','budget_value':5,'metric':'test_accuracy','n_seeds':n,'mean':mean,'median':median,'standard_deviation':sd,'ci95_low':lo,'ci95_high':hi})
 write(PROC/'data_size_scaling_seed_results.csv',details)
 write(PROC/'data_size_scaling_summary.csv',summary)
 fig,ax=plt.subplots(figsize=(7.5,4.8));found=False
 for model in ('MLP','CNN / LeNet'):
  rows=[r for r in summary if r['model_name']==model]
  if rows:
   rows.sort(key=lambda r:r['actual_train_size']);x=[r['actual_train_size'] for r in rows]
   y=[r['mean'] for r in rows];err=[lower_error(r) for r in rows]
   ax.errorbar(x,y,yerr=err,marker='o',capsize=3,label=model);found=True
 if found:
  ax.set(xlabel='Training examples (58k means requested 60k minus 2k validation)',ylabel='Official test accuracy at epoch 5',title='Fashion-MNIST performance by training-set size (3 seeds; 95% t intervals)')
  ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/'strengthened_data_size_scaling.png',dpi=200)
 plt.close(fig)

def plots(sums,marginal):
 by={}
 # The original fixed 10k split is the comparable base curve. Larger-data
 # splits are analyzed separately rather than merged as extra seeds.
 base_split='24574ee00a20ab31'
 for r in sums:
  if r.get('split_id')==base_split:by.setdefault((r['model_name'],r['method'],r['labeled_fraction'],r['metric'],r['source_file']),[]).append(r)
 # supervised training/validation curves with t intervals.
 for metric,file,title in [('validation_accuracy','strengthened_supervised_validation_accuracy.png','Validation accuracy vs epochs'),('validation_loss','strengthened_supervised_validation_loss.png','Validation loss vs epochs')]:
  fig,ax=plt.subplots(figsize=(7.5,4.8))
  for model in ('MLP','CNN / LeNet','ResNet-18','Vision Transformer (compact ViT-Tiny-style)','ViT-Tiny'):
   source='vit_tiny_scaling.csv' if model=='ViT-Tiny' else 'replicated_scaling_results.csv'
   rr=sorted(by.get((model,'','',metric,source),[]),key=lambda x:int(x['budget_value']))
   if rr:ax.errorbar([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],yerr=[lower_error(x) for x in rr],marker='o',capsize=3,label=model)
  ax.set(xlabel='Epoch',ylabel=metric.replace('_',' ').title(),title=title+' (10k train subset; 3 seeds; 95% t intervals)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/file,dpi=200);plt.close(fig)
 # SSL algorithms are plotted at each labeled fraction to preserve the matched setting.
 for metric in ('validation_accuracy','unsupervised_loss'):
  fig,axs=plt.subplots(1,5,figsize=(18,4),sharey=False)
  for ax,frac in zip(axs,('0.01','0.05','0.1','0.2','0.4')):
   for model in ('MeanTeacher','FixMatch','MixMatch'):
    rr=sorted(by.get((model,'',frac,metric,'ssl_algorithms.csv'),[]),key=lambda x:int(x['budget_value']))
    if rr:ax.plot([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],marker='o',label=model)
   ax.set(title=f'{float(frac):.0%} labels',xlabel='Epoch');ax.grid(alpha=.25)
  axs[0].set_ylabel(metric.replace('_',' ').title());axs[-1].legend()
  fig.suptitle('Mean Teacher / FixMatch / MixMatch task-adaptation curves')
  fig.tight_layout();fig.savefig(FIG/f"strengthened_ssl_methods_{metric}.png",dpi=200);plt.close(fig)
 # Representation learning reconstruction and downstream quality remain separate metrics.
 for metric,file,title in [('validation_reconstruction_mse','strengthened_autoencoder_reconstruction.png','Validation reconstruction MSE vs epochs'),('downstream_validation_accuracy','strengthened_autoencoder_linear_probe.png','Downstream linear-probe accuracy vs epochs')]:
  fig,ax=plt.subplots(figsize=(7.5,4.8))
  for model in ('Autoencoder','Variational Autoencoder'):
   rr=sorted(by.get((model,'','',metric,'autoencoder_scaling.csv'),[]),key=lambda x:int(x['budget_value']))
   if rr:ax.errorbar([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],yerr=[lower_error(x) for x in rr],marker='o',capsize=3,label=model)
  ax.set(xlabel='Epoch',ylabel=metric.replace('_',' ').title(),title=title+' (3 seeds; 95% t intervals)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/file,dpi=200);plt.close(fig)
 # Random forest complexity curve.
 rr=sorted(by.get(('Random Forest','','','validation_accuracy','random_forest_scaling.csv'),[]),key=lambda x:int(x['budget_value']))
 if rr:
  fig,ax=plt.subplots(figsize=(7.5,4.8));ax.errorbar([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],yerr=[lower_error(x) for x in rr],marker='o',capsize=3);ax.set(xlabel='Trees',ylabel='Validation accuracy',title='Random Forest validation accuracy vs tree count (3 seeds; 95% t intervals)');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/'strengthened_random_forest_tree_curve.png',dpi=200);plt.close(fig)
 # Unsupervised fits use separate update units/panels.
 fig,axs=plt.subplots(1,2,figsize=(11,4.5))
 for ax,model,xlab in [(axs[0],'MiniBatch K-Means','partial_fit updates'),(axs[1],'Gaussian Mixture Model','warm-start EM iterations')]:
  rr=sorted(by.get((model,'','','validation_ARI','unsupervised_scaling.csv'),[]),key=lambda x:int(x['budget_value']))
  if rr:ax.errorbar([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],yerr=[lower_error(x) for x in rr],marker='o',capsize=3)
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
   rr=sorted(by.get(('MLP',method,frac,'validation_accuracy','pseudolabel_baseline_sweep.csv'),[]),key=lambda x:int(x['budget_value']))
   if rr:ax.plot([int(x['budget_value']) for x in rr],[float(x['mean']) for x in rr],marker='o',label=f'{float(frac):.0%} labels')
  ax.set(xlabel='Epoch',ylabel='Validation accuracy',title=f'{label} learning curves by labeled fraction (3 seeds)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/f"strengthened_{method}_learning_curves.png",dpi=200);plt.close(fig)
 # RL plots separate the CartPole discrete-action group from Pendulum continuous control.
 for env_id,models,slug in [('CartPole-v1',['DQN','Double DQN','A2C','PPO'],'cartpole'),('Pendulum-v1',['SAC','PPO-Pendulum'],'pendulum')]:
  fig,axs=plt.subplots(1,2,figsize=(11,4.6))
  for model in models:
   rr=sorted([r for r in sums if r['model_name']==model and r['metric']=='mean_return' and r.get('split_id','').startswith(env_id)],key=lambda x:int(x['budget_value']))
   if rr:
    xx=[int(x['budget_value']) for x in rr];yy=[float(x['mean']) for x in rr];ee=[lower_error(x) for x in rr]
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
   mr=[r for r in marginal if r['model_name']==model and r.get('split_id')==base_split and r['marginal_improvement_per_second'] not in ('',None)]
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
   mr=[r for r in marginal if r['model_name']==model and r.get('split_id')==base_split and r['metric']==metric and r['marginal_improvement_per_second'] not in ('',None)];agg=defaultdict(list)
   for r in mr:agg[(int(float(r['to_budget'])),r['budget_unit'])].append(float(r['marginal_improvement_per_second']))
   if agg:
    ks=sorted(agg);ax.plot([k[0] for k in ks],[statistics.mean(agg[k]) for k in ks],marker='o',label=model);found=True
  if found:
   ax.set(xlabel='Epoch checkpoint',ylabel='Favorable metric change / model training second',title=f'Autoencoder {slug} marginal efficiency (seed means)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIG/f'strengthened_autoencoder_{slug}_marginal_efficiency.png',dpi=200)
  plt.close(fig)

def write_report(sums,marginal,tests):
 saturation={}
 for threshold in ('0.5','1','2'):
  saturation[threshold]=len({(r['model_name'],r['setting'],r['seed'],r['source_file']) for r in marginal if r.get(f'saturation_{threshold}pct')})
 saturation_text='; '.join(f'{t}% threshold: {n} model/setting/seed series met the two-increment screen' for t,n in saturation.items())
 p=ROOT/'reports/STATISTICAL_ANALYSIS_REPORT.md';lines=['# Statistical analysis of strengthened runs','','Analyses use verified local measurements from MLP/LeNet/ResNet/transformer epoch sweeps, Random Forest, K-Means/GMM, AE/VAE, pseudo-labeling and its labeled-only comparator, other SSL methods, RL, and DBSCAN. Most learning curves use the fixed Fashion-MNIST 10K-train/2K-validation split; RL uses CartPole/Pendulum. All 20 registered conventional schedules pass the coverage audit. This is a descriptive multi-paradigm study, not a confirmatory cross-paradigm comparison. Pseudo-labeling and its matched baseline resumed from saved epoch-5 checkpoints to the original 30-epoch target; all other results were pre-existing records.','',f'- Validation metric/budget summary rows: {len(sums)}.','- Paired within-seed adjacent-budget marginal transitions: '+str(len(marginal))+'.','- Held-out endpoint metric groups: '+str(len(tests))+'.','- Grouped summaries report mean, median, minimum, maximum, standard deviation, valid seed count, and two-sided 95% Student t intervals where n > 1. Intervals are imprecise with n=3 and describe seed variation conditional on the fixed split.','- Budget observations within a seed are repeated measures. Marginal increments pair only within model, seed, split, and setting; no checkpoint is treated as an independent seed.','- The dataset-size endpoint summary uses epoch 5 only. This prevents the 10K epoch-30 endpoints from being pooled into the matched 10K/30K/58K comparison.','- Fifteen later K-Means rows duplicate experiment IDs and reuse the 50-update state at earlier budgets. They remain in raw data but are excluded from the derived K-Means curves.','- Double DQN rows are restored to completed status and local_measurement source type in normalized outputs; success_rate is unavailable because the raw field was omitted.','- Two consecutive favorable relative gains below the stated screen: '+saturation_text+'. These are descriptive flags; short curves and small seed counts do not establish general saturation.','- Final test endpoints are grouped by model, metric, split, and final budget; validation curves are analyzed separately. No p-values or cross-model significance claims are reported.','- Initial AE/VAE v1 probe rows are retained but superseded after convergence warnings; analyses use the standardized v2 probe.','','## Final held-out test endpoints','']
 lines+=['| Run group | Metric | Seeds | Mean | Median | Min | Max | SD | 95% CI |','|---|---|---:|---:|---:|---:|---:|---:|---|']
 for r in tests:
  ci=f"[{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}]" if r['ci95_low']!='' else 'n/a (n=1)'
  lines.append(f"| {r['group']} | {r['metric']} | {r['n_seeds']} | {float(r['mean']):.4f} | {float(r['median']):.4f} | {float(r['minimum']):.4f} | {float(r['maximum']):.4f} | {float(r['standard_deviation']):.4f} | {ci} |")
 lines+=['','## Interpretation limits','','The MLP, LeNet, ResNet-18, compact ViT-style, Random Forest, AE/VAE and RL curves use three training seeds on fixed splits/subsets; uncertainty is conditional on those splits and does not quantify dataset-sampling uncertainty. K-Means/GMM ARI varies non-monotonically across updates/iterations and seeds; no saturation model is fit. DBSCAN is a deterministic parameter grid, not a seed-level budget curve. Pseudo-label vs labeled-only comparisons are paired by seed and labeled subset; current results vary with fraction and seed, so no universal SSL benefit is claimed. RL uses 20 deterministic evaluation episodes per checkpoint and three training seeds; within-run episode dispersion is separately retained. Historical pilot curves in `experiment_results.csv` include repeated test-set looks and remain exploratory.','']
 p.write_text('\n'.join(lines),encoding='utf-8')
 if marginal:
  lines=['# Marginal-gain analysis','','All marginal values below are computed only for saved multi-budget runs. Improvement is positive in the favorable direction; lower-is-better losses are sign-adjusted. Native-budget and elapsed-time ratios are task-specific. Rows pair adjacent budgets only within model, seed, split, setting, and source series. Cost per positive improvement is undefined for nonpositive gains. Fifteen later duplicate K-Means records are preserved in raw data but excluded because they reuse the final checkpoint value. ResNet seed-11 time-based marginal calculations subtract the unexplained 1,122.1625-second interval after epoch 2 for the adjusted diagnostic; raw time remains in source records and raw-time figures. The Double DQN CSV omits one success_rate field; derived analysis realigns values, leaves success_rate unavailable, and preserves raw records.','','- MLP/LeNet and neural models: epochs; validation accuracy/loss.','- Random Forest: warm-start tree count; validation accuracy/log loss.','- MiniBatch K-Means: partial-fit updates; validation ARI.','- Gaussian Mixture: warm-start EM calls; validation ARI.','- Pseudo-labeling/labeled-only and SSL methods: epochs, paired by seed and label fraction.','- Autoencoder/VAE: reconstruction and linear-probe metrics remain separate.','- RL: environment steps; CartPole and Pendulum remain separate.','- No cross-paradigm pooling. Small-gain screens are descriptive and do not establish saturation.','','## Seed-level marginal data','', '| Model | Setting | Seed | Budget interval | Improvement | Gain / budget | Additional seconds | Gain / second | Seconds / positive unit gain | Relative improvement |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
  for r in marginal:lines.append(f"| {r['model_name']} | {r['setting'] or r['method']} | {r['seed']} | {r['from_budget']}–{r['to_budget']} {r['budget_unit']} | {fmt(r['performance_improvement'],'.5f')} | {fmt(r['marginal_gain_per_budget_unit'],'.5f')} | {fmt(r['additional_training_seconds'],'.4f')} | {fmt(r['marginal_improvement_per_second'],'.5f')} | {fmt(r['seconds_per_unit_improvement'],'.4f')} | {fmt(float(r['relative_improvement'])*100 if r['relative_improvement'] not in ('',None) else '', '.3f')}% |")
  (ROOT/'reports/MARGINAL_GAIN_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
