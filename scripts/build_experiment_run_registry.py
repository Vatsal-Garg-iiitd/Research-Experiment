"""Create a row-level provenance index over raw local and literature results."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'EXPERIMENT_RUN_REGISTRY.csv'
FIELDS=['registry_id','source_file','source_row','record_id','model','paradigm','dataset','split_id','seed','budget_type','budget_value','requested_budget','actual_budget','metric','metric_value','status','hardware','backend','configuration_path','checkpoint_path','source_type','timestamp']

def rows_from(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        for i,row in enumerate(csv.DictReader(f),2):
            yield i,row

def first(row,*keys):
    for key in keys:
        v=row.get(key)
        if v not in (None,''):return v
    return ''

records=[]
sources=[]
sources.extend(sorted((ROOT/'results/raw').glob('*.csv')))
sources.extend(sorted((ROOT/'results/llm').glob('*.csv')))
sources.extend([ROOT/'results/literature/large_model_literature.csv',ROOT/'results/literature/literature_field_evidence.csv'])
for path in sources:
    if not path.is_file():continue
    for i,r in rows_from(path):
        name=path.relative_to(ROOT).as_posix();rid=first(r,'experiment_id','run_id','canonical_id') or f'{name}:{i}'
        budget=first(r,'budget_value','requested_budget','environment_steps','training_step','scored_tokens','evaluation_tokens_scored','tokens_processed','evaluated_examples')
        actual=first(r,'environment_steps','optimizer_steps','examples_processed','scored_tokens','evaluation_tokens_scored','tokens_processed')
        records.append({
            'registry_id':f'{name}:{i}:{rid}','source_file':name,'source_row':i,'record_id':rid,
            'model':first(r,'model_name','model'),'paradigm':first(r,'paradigm','family'),'dataset':first(r,'dataset','benchmark'),
            'split_id':first(r,'split_id','evaluation_split'),'seed':r.get('seed',''),
            'budget_type':first(r,'budget_type','budget_unit','benchmark_protocol'),'budget_value':budget,
            'requested_budget':first(r,'requested_budget'),'actual_budget':actual,
            'metric':first(r,'evaluation_metric','metric_name','test_metric','metric','performance_metric'),
            'metric_value':first(r,'evaluation_value','metric_value','test_accuracy','test_loss','test_perplexity','perplexity','negative_log_likelihood','accuracy','performance_value'),
            'status':first(r,'experiment_status','status','evidence_class'),'hardware':first(r,'hardware','device'),
            'backend':first(r,'backend','device','precision'),'configuration_path':first(r,'configuration_path','config_path'),
            'checkpoint_path':first(r,'checkpoint_path','checkpoint'),'source_type':first(r,'source_type','record_type','evidence_class'),
            'timestamp':first(r,'timestamp_utc','timestamp')})

with OUT.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(records)
print(f'Wrote {len(records)} registry rows across {len(sources)} source paths to {OUT}')
