import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TASKS={'arc_easy':'ARC-Easy','hellaswag':'HellaSwag','piqa':'PIQA'}
rows=[]
files=sorted((ROOT/'results/llm/harness').rglob('results_*.json'))
latest={}
for p in files: latest[p.parent]=p
for p in sorted(latest.values()):
    try: d=json.loads(p.read_text(encoding='utf-8'))
    except Exception: continue
    model=d.get('model_name','')
    for task,data in d.get('results',{}).items():
        if task not in TASKS: continue
        rows.append({'model':model,'task':TASKS[task],'sample_len':data.get('sample_len'),'fewshot':d.get('configs',{}).get(task,{}).get('num_fewshot'),'accuracy':data.get('acc,none'),'accuracy_stderr':data.get('acc_stderr,none'),'normalized_accuracy':data.get('acc_norm,none'),'normalized_accuracy_stderr':data.get('acc_norm_stderr,none'),'eval_seconds':d.get('total_evaluation_time_seconds'),'device':'CPU','dtype':'float32','lm_eval_version':d.get('lm_eval_version'),'output_json':str(p.relative_to(ROOT))})
out=ROOT/'results/llm/harness_summary.csv'
with out.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['model','task']); w.writeheader();w.writerows(rows)
print(f'Extracted {len(rows)} task-model records from harness outputs')
