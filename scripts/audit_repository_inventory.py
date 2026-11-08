"""Inventory research artifacts and tabular records without changing source data."""
import csv, json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/processed'; OUT.mkdir(parents=True,exist_ok=True)
SKIP={'.git','__pycache__','snapshots'}
def eligible(p):
    rel=p.relative_to(ROOT)
    return not any(part in SKIP or part.startswith('.venv') for part in rel.parts) and p.suffix not in {'.pyc','.pyo'}
def rows_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); data=list(reader); return reader.fieldnames or [],data
def main():
    files=[p for p in ROOT.rglob('*') if p.is_file() and eligible(p)]
    kinds=Counter(p.suffix.lower() or '[no extension]' for p in files)
    (OUT/'repository_file_inventory.json').write_text(json.dumps({'root':str(ROOT),'excluded':['.git','all .venv* directories','__pycache__','*.pyc','*.pyo'],'file_count':len(files),'extension_counts':dict(sorted(kinds.items())),'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'suffix':p.suffix.lower()} for p in sorted(files)]},indent=2),encoding='utf-8')
    detail=[]
    for p in sorted((ROOT/'results').rglob('*.csv')):
        if not eligible(p):continue
        try: cols,data=rows_csv(p)
        except Exception as e:
            detail.append({'path':p.relative_to(ROOT).as_posix(),'parse_error':str(e)});continue
        missing={c:sum(1 for r in data if not (r.get(c) or '').strip()) for c in cols}
        candidate=next((k for k in ('canonical_id','registry_id','experiment_id','run_id') if k in cols),'')
        counts=Counter((r.get(candidate) or '') for r in data) if candidate else Counter()
        duplicate_records=sum(n-1 for k,n in counts.items() if k and n>1)
        status_key='experiment_status' if 'experiment_status' in cols else ('status' if 'status' in cols else '')
        detail.append({'path':p.relative_to(ROOT).as_posix(),'rows':len(data),'columns':len(cols),'column_names':cols,'candidate_identifier':candidate or None,'candidate_identifier_duplicate_excess':duplicate_records,'missing_cells':sum(missing.values()),'missing_by_column':{k:v for k,v in missing.items() if v},'status_column':status_key or None,'status_counts':dict(Counter(r.get(status_key,'') for r in data)) if status_key else {},'provenance_columns':[c for c in cols if any(s in c.lower() for s in ('source','path','revision','checkpoint','hardware','backend','seed','split','run_id','experiment_id'))]})
    (OUT/'tabular_data_inventory.json').write_text(json.dumps({'scope':'current results CSV files excluding preserved local snapshots, virtual environments, and caches','files':detail},indent=2,ensure_ascii=False),encoding='utf-8')
    json_detail=[]
    for p in sorted((ROOT/'results').rglob('*')):
        if not p.is_file() or not eligible(p) or p.suffix.lower() not in ('.json','.jsonl'):continue
        try:
            if p.suffix.lower()=='.jsonl':
                count=0;keys=set();ids=Counter();
                with p.open(encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():continue
                        obj=json.loads(line);count+=1
                        if isinstance(obj,dict):
                            keys.update(obj)
                            for k in ('experiment_id','run_id','model','model_name','task','checkpoint_id','id'):
                                if obj.get(k) not in (None,''):ids[k]+=1
                json_detail.append({'path':p.relative_to(ROOT).as_posix(),'format':'jsonl','records':count,'top_level_keys':sorted(keys),'identifier_field_counts':dict(ids),'parse_error':None})
            else:
                obj=json.loads(p.read_text(encoding='utf-8'))
                if isinstance(obj,dict):keys=sorted(obj);records=1;kind='object'
                elif isinstance(obj,list):keys=sorted(set(k for x in obj if isinstance(x,dict) for k in x));records=len(obj);kind='array'
                else:keys=[];records=1;kind=type(obj).__name__
                json_detail.append({'path':p.relative_to(ROOT).as_posix(),'format':'json','top_level_type':kind,'records_or_items':records,'top_level_keys':keys,'identifier_fields':[k for k in ('experiment_id','run_id','model','model_name','checkpoint_id','revision','dataset','seed','status') if k in keys],'parse_error':None})
        except Exception as e:json_detail.append({'path':p.relative_to(ROOT).as_posix(),'format':p.suffix.lower(),'parse_error':str(e)})
    (OUT/'json_data_inventory.json').write_text(json.dumps({'scope':'current JSON and JSONL files under results, excluding preserved local snapshots, virtual environments, and caches','files':json_detail},indent=2,ensure_ascii=False),encoding='utf-8')
    md=['# Repository and data inventory','',f'Inventory generated from `{ROOT}`; {len(files)} files after excluding Git internals, virtual environments, bytecode, and cache folders. Raw records are read-only inputs.','', '## File types','', '| Type | Files |','|---|---:|']
    md += [f'| `{k}` | {v} |' for k,v in sorted(kinds.items())]
    md += ['', '## Result tables','', 'CSV row counts below exclude headers. Candidate identifier repetition is a screening signal, not an automatic duplicate verdict: repeated identifiers may represent legitimate checkpoint budgets. Full columns, missingness, status counts, and provenance fields are recorded in `results/processed/tabular_data_inventory.json`. JSON/JSONL record counts and key/identifier coverage are in `results/processed/json_data_inventory.json`.','', '| File | Rows | Columns | Candidate ID | Repeated ID excess | Missing cells |','|---|---:|---:|---|---:|---:|']
    for x in detail:
        if 'rows' in x:md.append(f"| `{x['path']}` | {x['rows']} | {x['columns']} | {x['candidate_identifier'] or '—'} | {x['candidate_identifier_duplicate_excess']} | {x['missing_cells']} |")
    (ROOT/'DATA_INVENTORY.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(f'Inventoried {len(files)} files and {len(detail)} result CSVs.')
if __name__=='__main__':main()
