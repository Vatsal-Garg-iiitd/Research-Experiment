"""Build a lossless-enough common index over heterogeneous experiment CSVs."""
import csv, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'results/raw'; OUT=ROOT/'results/processed/canonical_results.csv'
SOURCES=['experiment_results.csv','full_suite_results.csv','replicated_scaling_results.csv','random_forest_scaling.csv','unsupervised_scaling.csv','pseudolabel_baseline_sweep.csv','rl_scaling_results.csv','dbscan_parameter_sweep.csv','dbscan_size_scaling.csv','autoencoder_scaling.csv','label_propagation_scaling.csv','ssl_algorithms.csv']
FIELDS=['canonical_id','experiment_id','source_file','source_row','model_name','paradigm','dataset','split_id','seed','budget_type','budget_value','budget_unit','metric_name','metric_value','metric_direction','training_time_seconds','cumulative_training_time_seconds','evaluation_time_seconds','training_compute','evaluation_compute','hardware','backend','status','run_type','source_type','model_checkpoint_revision','raw_record_json']
def get(r,*keys,default=''):
    for k in keys:
        v=r.get(k)
        if v not in (None,''): return v
    return default
rows=[]
for name in SOURCES:
    p=RAW/name
    if not p.exists(): continue
    with p.open(encoding='utf-8-sig',newline='') as f:
        for i,r in enumerate(csv.DictReader(f),start=2):
            model=get(r,'model_name','model')
            if name=='pseudolabel_baseline_sweep.csv': model=f"{model} / {get(r,'method')} / label={get(r,'labeled_fraction')}"
            metric=get(r,'evaluation_metric','metric_name',default='validation_accuracy' if 'validation_accuracy' in r else '')
            value=get(r,'evaluation_value','metric_value','validation_ARI','validation_reconstruction_mse','mean_return','validation_accuracy')
            status=get(r,'experiment_status','status',default='unknown')
            identity=get(r,'experiment_id','run_id',default=f'{name}:{i}')
            import json
            direction='lower_is_better' if any(s in metric.lower() for s in ('loss','mse','perplexity','cross_entropy')) else ('higher_is_better' if metric else '')
            rows.append(dict(canonical_id=f'{name}:{identity}:{i}',experiment_id=identity,source_file=f'results/raw/{name}',source_row=i,model_name=model,paradigm=get(r,'paradigm'),dataset=get(r,'dataset'),split_id=get(r,'split_id'),seed=get(r,'seed'),budget_type=get(r,'budget_type','budget_unit'),budget_value=get(r,'budget_value'),budget_unit=get(r,'budget_type','budget_unit'),training_time_seconds=get(r,'training_time_seconds','train_seconds'),cumulative_training_time_seconds=get(r,'cumulative_training_time_seconds','cumulative_train_seconds'),evaluation_time_seconds=get(r,'evaluation_time_seconds','eval_seconds'),training_compute='',evaluation_compute='',metric_name=metric,metric_value=value,metric_direction=direction,hardware=get(r,'hardware'),backend=get(r,'backend'),status=status,run_type='pilot_or_partial',source_type=get(r,'source_type',default='local_experiment'),model_checkpoint_revision=get(r,'checkpoint_id','revision'),raw_record_json=json.dumps(r,ensure_ascii=False,separators=(',',':'))))
for name in ['full_wikitext2_evaluations.csv','harness_summary.csv']:
    p=ROOT/'results/llm'/name
    if not p.exists():continue
    with p.open(encoding='utf-8-sig',newline='') as f:
        for i,r in enumerate(csv.DictReader(f),start=2):
            if name.startswith('full_'):
                model=r.get('model_name','');metric='test_cross_entropy';value=r.get('test_cross_entropy');budget='evaluation_tokens';bv=r.get('evaluation_tokens_scored');revision=r.get('checkpoint_id','');stype='local_checkpoint_evaluation'
            else:
                model=r.get('model','');metric='accuracy';value=r.get('accuracy');budget='smoke_examples_per_task';bv=r.get('sample_len');revision='';stype='local_harness_smoke'
            import json
            direction='lower_is_better' if metric=='test_cross_entropy' else 'higher_is_better'
            rows.append(dict(canonical_id=f'llm:{name}:{i}:{model}:{metric}',experiment_id='',source_file=f'results/llm/{name}',source_row=i,model_name=model,paradigm='language_modeling',dataset=r.get('dataset','WikiText-2 raw-v1 test') if name.startswith('full_') else r.get('task',''),split_id='',seed='',budget_type=budget,budget_value=bv,budget_unit=budget,training_time_seconds='',cumulative_training_time_seconds='',evaluation_time_seconds=r.get('evaluation_seconds',''),training_compute='',evaluation_compute='',metric_name=metric,metric_value=value,metric_direction=direction,hardware=r.get('hardware','CPU'),backend=r.get('backend',r.get('device','CPU')),status=r.get('status','completed'),run_type='smoke' if 'smoke' in stype else 'checkpoint_eval',source_type=stype,model_checkpoint_revision=revision,raw_record_json=json.dumps(r,ensure_ascii=False,separators=(',',':'))))
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
print(f'Wrote {len(rows)} rows to {OUT}')
