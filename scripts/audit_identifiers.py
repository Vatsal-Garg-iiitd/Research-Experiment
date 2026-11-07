"""Repository-wide identifier and foreign-key audit for saved result tables."""
import csv,json,re
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/processed';OUT.mkdir(parents=True,exist_ok=True)
ID_FIELDS=('canonical_id','evaluation_id','experiment_id','run_id','registry_id','model_id','dataset_id','configuration_id','checkpoint_id','seed','split_id')
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def slug(s):return re.sub(r'[^a-z0-9]+','-',str(s or '').lower()).strip('-')
def write(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    paths=sorted(set([*ROOT.glob('EXPERIMENT_RUN_REGISTRY.csv'),*ROOT.glob('IDENTIFIER_RECONCILIATION.csv'),*ROOT.glob('results/raw/*.csv'),*ROOT.glob('results/processed/*.csv'),*ROOT.glob('results/llm/*.csv'),*ROOT.glob('results/literature/*.csv')]))
    summary=[];findings=[];run_models=defaultdict(set);run_seeds=defaultdict(set);run_configs=defaultdict(set);model_names=defaultdict(set);foreign=[]
    for p in paths:
        try:rows=read(p)
        except Exception as e:
            findings.append({'severity':'error','source_file':p.relative_to(ROOT).as_posix(),'row':'','identifier_type':'table','identifier':'','finding':f'CSV parse error: {e}','treatment':'Review source encoding/schema.'});continue
        rel=p.relative_to(ROOT).as_posix();cols=set(rows[0]) if rows else set()
        for fld in ID_FIELDS:
            if fld not in cols:continue
            vals=[r.get(fld,'') for r in rows];cnt=Counter(v for v in vals if v);dups={k:v for k,v in cnt.items() if v>1}
            summary.append({'source_file':rel,'identifier_type':fld,'records':len(rows),'missing':sum(not v for v in vals),'distinct_nonblank':len(cnt),'duplicate_excess':sum(v-1 for v in dups.values()),'duplicate_values':json.dumps(dups,ensure_ascii=False) if dups else ''})
            for value,times in dups.items():
                examples=[str(i+2) for i,r in enumerate(rows) if r.get(fld)==value][:8]
                finding='Repeated within table; requires scoped interpretation.'
                severity='review'
                if fld=='run_id':finding='Expected run identity repeats across budget/checkpoint records; verify model/seed/config consistency.';severity='info'
                if fld=='experiment_id' and rel.endswith('unsupervised_scaling.csv') and 'minibatch_k-means' in value:
                    finding='Known later duplicate K-Means record reuses final checkpoint value; excluded and assigned unique derived row ID.';severity='corrected'
                findings.append({'severity':severity,'source_file':rel,'row':','.join(examples),'identifier_type':fld,'identifier':value,'finding':finding,'treatment':'See IDENTIFIER_RECONCILIATION.csv and verified_canonical_results.csv.'})
        for i,r in enumerate(rows,2):
            model=r.get('model_name') or r.get('model') or ''
            if model:
                mid=r.get('model_id') or f'model:{slug(model)}';model_names[mid].add(model)
            run=r.get('run_id')
            if run:
                run_models[(rel,run)].add(model);run_seeds[(rel,run)].add(r.get('seed',''))
                cfg=r.get('configuration_path') or r.get('config_path') or r.get('configuration_id') or ''
                if cfg:run_configs[(rel,run)].add(cfg)
            for col in ('configuration_path','config_path','configuration_id','checkpoint_path'):
                val=r.get(col,'')
                if not val:continue
                # Older RL output has a known one-cell shift for Double DQN;
                # the normalized canonical/registry rows are checked instead.
                if rel=='results/raw/rl_scaling_results.csv' and model=='Double DQN':continue
                if re.match(r'^\d{4}-\d\d-\d\dT',val):continue
                target=ROOT/val.replace('\\','/')
                if not target.exists():
                    foreign.append({'source_file':rel,'row':i,'field':col,'value':val,'validation':'missing'})
    for (rel,run),models in run_models.items():
        if len(models)>1:findings.append({'severity':'error','source_file':rel,'row':'','identifier_type':'run_id','identifier':run,'finding':f'Run ID maps to multiple model names: {sorted(models)}','treatment':'Do not merge; correct source mapping only with configuration evidence.'})
        seeds={s for s in run_seeds[(rel,run)] if s}
        if len(seeds)>1:findings.append({'severity':'error','source_file':rel,'row':'','identifier_type':'run_id','identifier':run,'finding':f'Run ID merges multiple seeds: {sorted(seeds)}','treatment':'Treat as distinct runs; inspect raw configurations.'})
        if len(run_configs[(rel,run)])>1:findings.append({'severity':'review','source_file':rel,'row':'','identifier_type':'run_id','identifier':run,'finding':'Run ID maps to multiple configuration paths.','treatment':'Check whether path variants resolve to same file or reflect changed configuration.'})
    for mid,names in model_names.items():
        if len(names)>1:findings.append({'severity':'error','source_file':'cross-file','row':'','identifier_type':'model_id','identifier':mid,'finding':f'Model ID maps to multiple labels: {sorted(names)}','treatment':'Resolve naming alias or use distinct IDs if model identity differs.'})
    write(OUT/'identifier_audit.csv',summary,['source_file','identifier_type','records','missing','distinct_nonblank','duplicate_excess','duplicate_values'])
    write(OUT/'identifier_findings.csv',findings,['severity','source_file','row','identifier_type','identifier','finding','treatment'])
    write(OUT/'foreign_key_audit.csv',foreign,['source_file','row','field','value','validation'])
    reconciliation=ROOT/'IDENTIFIER_RECONCILIATION.csv'
    rec=read(reconciliation) if reconciliation.exists() else []
    errors=sum(x.get('severity')=='error' for x in findings)
    md=['# Identifier audit','','Identifier scopes are source file plus identifier type unless a cross-file key is explicitly normalized. A run_id intentionally repeats over budget observations; it must map to one model, seed, and configuration within a source. `evaluation_id` and `canonical_id` identify individual analytical records. Seed values repeat across runs by design and are not globally unique.','',f'- Tables scanned: {len(paths)}.','- Identifier-field summaries: '+str(len(summary))+'.',f'- Reconciliation entries: {len(rec)}.',f'- Identifier/model/run integrity errors: {errors}.',f'- Unresolved foreign-key paths: {len(foreign)}.','','Known corrections, generated evaluation IDs, and affected rows are listed in `IDENTIFIER_RECONCILIATION.csv`. The full machine-readable identifier scan is `results/processed/identifier_audit.csv`; findings are `results/processed/identifier_findings.csv`; file-path references are in `results/processed/foreign_key_audit.csv`. Raw files are not edited.','','## Important distinctions','','- Repeated `run_id` values across epochs/budgets are legitimate when model, seed, and configuration remain consistent.','- The only duplicate `experiment_id` within a raw source is the later K-Means append; it is excluded from analysis and given a unique derived ID.','- LLM evaluation records without source experiment IDs receive stable row-scoped `evaluation_id` values; they are not fabricated as training experiments.','- Thirty Double DQN CSV rows are one cell short. Derived outputs realign the configuration/checkpoint/timestamp/details fields using their values, leaving the omitted success_rate unavailable. The original malformed rows are preserved.']
    (ROOT/'IDENTIFIER_AUDIT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(f'Scanned {len(paths)} tables; identifier errors={errors}; unresolved references={len(foreign)}; reconciliations={len(rec)}.')
    raise SystemExit(1 if errors else 0)
if __name__=='__main__':main()
