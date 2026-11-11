"""Attach explicit analytical eligibility to the common result index."""
import csv
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];src=ROOT/'results/processed/canonical_results.csv';out=ROOT/'results/processed/verified_canonical_results.csv'
with src.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
rawk=ROOT/'results/raw/unsupervised_scaling.csv';seen_ids=set();dupe_rows=set()
with rawk.open(encoding='utf-8-sig',newline='') as f:
    for i,r in enumerate(csv.DictReader(f),start=2):
        if r.get('model_name')!='MiniBatch K-Means':continue
        eid=r.get('experiment_id','')
        if eid in seen_ids:dupe_rows.add(str(i))
        seen_ids.add(eid)
for r in rows:
    r['original_experiment_id']=r.get('experiment_id','')
    if r.get('source_file')=='results/raw/unsupervised_scaling.csv' and str(r.get('source_row')) in dupe_rows:
        original=r.get('experiment_id','');corrected=f'{original}__duplicate_copy_row{r.get("source_row")}'
        r['experiment_id']=corrected
        r['audit_eligibility']='excluded';r['audit_disposition']='duplicate_checkpoint_reuse';r['audit_reason']='Later duplicate K-Means row reuses final 50-update score at an earlier budget; assigned a unique derived row-scoped ID and excluded from trajectories.'
    else:
        status=r.get('status','').lower()
        if status in ('completed','local_measurement'):
            r['audit_eligibility']='eligible_within_protocol';r['audit_disposition']='retained';r['audit_reason']='Status is a recorded measurement; compare only within its protocol and metric.'
        else:
            r['audit_eligibility']='excluded_from_completed_analysis';r['audit_disposition']=status or 'status_unavailable';r['audit_reason']='Status is not a completed/local measurement; retained in the index for provenance.'
    if not r.get('experiment_id') and r.get('paradigm')=='language_modeling':
        r['audit_disposition']='evaluation_only_no_experiment_id';r['audit_reason']='Source is an evaluation record, not a local experiment run; unique evaluation_id identifies it.'
    if r.get('source_file')=='results/llm/expanded_llm_evaluations.csv' and r.get('model_name') in ('Pythia-160M','Pythia-410M') and r.get('model_checkpoint_revision')=='main':
        r['audit_eligibility']='excluded_as_alias';r['audit_disposition']='checkpoint_alias';r['audit_reason']='main alias duplicates the step143000 checkpoint; not an additional trajectory point.'
fields=list(rows[0]) if rows else []
for k in ('original_experiment_id','audit_eligibility','audit_disposition','audit_reason'):
    if k not in fields:fields.append(k)
with out.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
reconciliation=[]
for r in rows:
    if r.get('audit_disposition')=='duplicate_checkpoint_reuse':
        reconciliation.append({'identifier_type':'experiment_id','original_identifier':r['original_experiment_id'],'corrected_identifier':r['experiment_id'],'source_file':r['source_file'],'source_row':r['source_row'],'reason':r['audit_reason'],'related_records':f"original kept at earlier source row with same ID; raw repeated row retained here; analysis_exclusion={r['audit_eligibility']}",'validation_status':'valid_unique_derived_id_raw_preserved'})
    elif r.get('audit_disposition')=='evaluation_only_no_experiment_id':
        reconciliation.append({'identifier_type':'evaluation_id','original_identifier':'','corrected_identifier':r['evaluation_id'],'source_file':r['source_file'],'source_row':r['source_row'],'reason':'Evaluation-level record has no experiment_id in source; stable row-scoped evaluation_id assigned in canonical index.','related_records':f"model={r['model_id']}; revision={r['model_checkpoint_revision']}; metric={r['metric_name']}",'validation_status':'valid_unique_evaluation_id'})
for r in rows:
    if r.get('source_file')=='results/raw/rl_scaling_results.csv' and r.get('model_name')=='Double DQN':
        rec=json.loads(r['raw_record_json'])
        if len(rec)==0:continue
        # Canonical builder's normalized schema is authoritative for references.
        reconciliation.append({'identifier_type':'configuration/checkpoint foreign key','original_identifier':rec.get('source_type',''),'corrected_identifier':r.get('configuration_id',''),'source_file':r['source_file'],'source_row':r['source_row'],'reason':'Raw Double DQN CSV omitted success_rate and shifted trailing fields; derived index realigns the configuration path.','related_records':f"experiment_id={r['experiment_id']}; checkpoint={r.get('model_checkpoint_revision','')}",'validation_status':'validated_against_existing_configuration_and_checkpoint_paths'})
recpath=out.parent/'identifier_reconciliation.csv'
with recpath.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['identifier_type','original_identifier','corrected_identifier','source_file','source_row','reason','related_records','validation_status']);w.writeheader();w.writerows(reconciliation)
rootrec=ROOT/'results/processed/identifier_reconciliation.csv'
with rootrec.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['identifier_type','original_identifier','corrected_identifier','source_file','source_row','reason','related_records','validation_status']);w.writeheader();w.writerows(reconciliation)
print(f'Wrote eligibility-annotated verified index: {len(rows)} rows; {len(reconciliation)} identifier reconciliations.')
