"""Build a lossless-enough common index over heterogeneous experiment CSVs."""
import csv, math
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'results/raw'; OUT=ROOT/'results/processed/canonical_results.csv'
SOURCES=['experiment_results.csv','full_suite_results.csv','replicated_scaling_results.csv','vit_tiny_scaling.csv','random_forest_scaling.csv','unsupervised_scaling.csv','pseudolabel_baseline_sweep.csv','rl_scaling_results.csv','dbscan_parameter_sweep.csv','dbscan_size_scaling.csv','autoencoder_scaling.csv','label_propagation_scaling.csv','ssl_algorithms.csv']
FIELDS=['canonical_id','evaluation_id','experiment_id','run_id','source_file','source_row','model_id','model_name','paradigm','dataset_id','dataset','split_id','seed','configuration_id','checkpoint_id','checkpoint_path','timestamp_utc','budget_type','budget_value','budget_unit','metric_name','metric_value','metric_direction','training_time_seconds','cumulative_training_time_seconds','evaluation_time_seconds','training_compute','evaluation_compute','hardware','backend','status','run_type','source_type','model_checkpoint_revision','raw_record_json']
def slug(v):
    return re.sub(r'[^a-z0-9]+','-',str(v or '').lower()).strip('-')
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
            source_record=dict(r)
            # A legacy Double DQN writer omitted the success_rate cell. The
            # original values shifted one field left in CSV parsing. Normalize
            # the common index from the evidence sequence without editing raw.
            if name=='rl_scaling_results.csv' and r.get('model_name')=='Double DQN' and r.get('details') in (None,''):
                r=dict(r)
                r['details']=r.get('timestamp_utc','')
                r['timestamp_utc']=r.get('checkpoint_path','')
                r['checkpoint_path']=r.get('configuration_path','')
                r['configuration_path']=r.get('source_type','')
                r['source_type']='local_measurement'
                r['experiment_status']=r.get('backend','')
                r['backend']=r.get('hardware','')
                r['hardware']=r.get('peak_memory_bytes','')
                r['peak_memory_bytes']=r.get('success_rate','')
                r['success_rate']=''
            model=get(r,'model_name','model')
            if name=='pseudolabel_baseline_sweep.csv': model=f"{model} / {get(r,'method')} / label={get(r,'labeled_fraction')}"
            metric=get(r,'evaluation_metric','metric_name',default='validation_accuracy' if 'validation_accuracy' in r else '')
            value=get(r,'evaluation_value','metric_value','validation_ARI','validation_reconstruction_mse','mean_return','validation_accuracy')
            status=get(r,'experiment_status','status',default='unknown')
            identity=get(r,'experiment_id')
            run_identity=get(r,'run_id')
            source_type=get(r,'source_type',default='local_experiment')
            if source_type.replace('\\','/').startswith('results/configurations/'):
                # Some legacy RL CSV rows shifted this configuration path into
                # the source_type column. Preserve the row verbatim in JSON,
                # but give the normalized index a truthful provenance label.
                source_type='local_measurement (legacy provenance field was a configuration path)'
            import json
            direction='lower_is_better' if any(s in metric.lower() for s in ('loss','mse','perplexity','cross_entropy')) else ('higher_is_better' if metric else '')
            cid=f'results/raw/{name}:{i}'
            rows.append(dict(canonical_id=cid,evaluation_id=f'eval:{name}:{i}',experiment_id=identity,run_id=run_identity,source_file=f'results/raw/{name}',source_row=i,model_id=f'model:{slug(model)}',model_name=model,paradigm=get(r,'paradigm'),dataset_id=f'dataset:{slug(get(r,"dataset"))}' if get(r,'dataset') else '',dataset=get(r,'dataset'),split_id=get(r,'split_id'),seed=get(r,'seed'),configuration_id=get(r,'configuration_path','config_path'),checkpoint_id=get(r,'checkpoint_id','revision'),checkpoint_path=get(r,'checkpoint_path'),timestamp_utc=get(r,'timestamp_utc'),budget_type=get(r,'budget_type','budget_unit'),budget_value=get(r,'budget_value'),budget_unit=get(r,'budget_type','budget_unit'),training_time_seconds=get(r,'training_time_seconds','train_seconds'),cumulative_training_time_seconds=get(r,'cumulative_training_time_seconds','cumulative_train_seconds'),evaluation_time_seconds=get(r,'evaluation_time_seconds','eval_seconds'),training_compute='',evaluation_compute='',metric_name=metric,metric_value=value,metric_direction=direction,hardware=get(r,'hardware'),backend=get(r,'backend'),status=status,run_type='pilot_or_partial',source_type=source_type,model_checkpoint_revision=get(r,'checkpoint_id','revision'),raw_record_json=json.dumps(source_record,ensure_ascii=False,separators=(',',':'))))
for name in ['macos_full_wikitext2_evaluations.csv','full_wikitext2_evaluations.csv','expanded_llm_evaluations.csv','harness_summary.csv','harness_full_summary.csv']:
    p=ROOT/'results/llm'/name
    if not p.exists():continue
    with p.open(encoding='utf-8-sig',newline='') as f:
        for i,r in enumerate(csv.DictReader(f),start=2):
            if name=='macos_full_wikitext2_evaluations.csv':
                model=r.get('model_name','');metric='negative_log_likelihood';value=r.get('negative_log_likelihood');budget='full_test_scored_tokens';bv=r.get('scored_tokens');revision=r.get('model_revision','');stype='local_full_corpus_checkpoint_evaluation'
            elif name=='expanded_llm_evaluations.csv':
                model=r.get('model','');metric='test_loss';value=r.get('test_loss');budget='prefix_evaluation_tokens';bv=r.get('tokens_processed');revision=r.get('checkpoint_id','');stype='local_checkpoint_prefix_evaluation'
            elif name=='harness_full_summary.csv':
                model=r.get('model','');metric='accuracy';value=r.get('metric_accuracy');budget='full_validation_examples';bv=r.get('evaluated_examples');revision=r.get('checkpoint_revision','');stype='local_full_zero_shot_benchmark'
            elif name.startswith('full_'):
                model=r.get('model_name','');metric='test_cross_entropy';value=r.get('test_cross_entropy');budget='evaluation_tokens';bv=r.get('evaluation_tokens_scored');revision=r.get('checkpoint_id','');stype='local_checkpoint_evaluation'
            else:
                model=r.get('model','');metric='accuracy';value=r.get('accuracy');budget='smoke_examples_per_task';bv=r.get('sample_len');revision='';stype='local_harness_smoke'
            import json
            direction='lower_is_better' if metric=='test_cross_entropy' else 'higher_is_better'
            dataset=r.get('dataset','WikiText-2 raw-v1 test') if 'wikitext' in name.lower() or name=='expanded_llm_evaluations.csv' else r.get('task','')
            cid=f'llm:{name}:{i}:{model}:{metric}'
            rows.append(dict(canonical_id=cid,evaluation_id=f'eval:{name}:{i}',experiment_id=r.get('experiment_id',''),source_file=f'results/llm/{name}',source_row=i,model_id=f'model:{slug(model)}',model_name=model,paradigm='language_modeling',dataset_id=f'dataset:{slug(dataset)}' if dataset else '',dataset=dataset,split_id='',seed='',configuration_id='',checkpoint_id=revision,checkpoint_path='',timestamp_utc=r.get('timestamp_utc',''),budget_type=budget,budget_value=bv,budget_unit=budget,training_time_seconds='',cumulative_training_time_seconds='',evaluation_time_seconds=r.get('evaluation_seconds',''),training_compute='',evaluation_compute='',metric_name=metric,metric_value=value,metric_direction=direction,hardware=r.get('hardware','Apple MPS' if 'mps' in r.get('device','').lower() else 'CPU'),backend=r.get('backend',r.get('device','CPU')),status=r.get('status','completed'),run_type='smoke' if 'smoke' in stype else ('benchmark_eval' if 'benchmark' in stype else 'checkpoint_eval'),source_type=stype,model_checkpoint_revision=revision,raw_record_json=json.dumps(r,ensure_ascii=False,separators=(',',':'))))
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
print(f'Wrote {len(rows)} rows to {OUT}')
