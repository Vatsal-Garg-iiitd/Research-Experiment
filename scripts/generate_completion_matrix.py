import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
reg=read(ROOT/'configs/model_registry.csv')
def count(file,model_key,model):
 p=ROOT/'results/raw'/file
 if not p.exists():return []
 return [r for r in read(p) if r.get(model_key)==model and r.get('experiment_status',r.get('status','')) not in ('superseded','failed','resource_limited')]
out=['# Final Completion Matrix','', 'Updated 2026-11-07 from result files. `pilot` means actual runs exist but the requested full study for that entry is incomplete. Local checkpoint evaluation of an LLM does not mean local pretraining.', '', '**Strict objective completion: 0/50 configurations (0%).** The denominator is 20 conventional models + 15 specified LLM configurations + 15 literature configurations. A configuration counts complete only when its requested study objective is met; pilot/smoke/subset evaluations and partially verified literature rows do not count. This does not mean no work was completed; it distinguishes partial evidence from full requested scope.','', '| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |','|---|---|---|---|---:|---|---|---|']
for x in reg:
 m=x['model']; p=x['paradigm']; rows=[]; loc=[]; hardware=''
 if m in ('MLP','LeNet-style CNN','ResNet-18','ViT-Tiny'):
  key={'MLP':'MLP','LeNet-style CNN':'CNN / LeNet','ResNet-18':'ResNet-18','ViT-Tiny':'Vision Transformer (compact ViT-Tiny-style)'}[m]
  rows=count('replicated_scaling_results.csv','model_name',key);loc=['replicated_scaling_results.csv','replicated_scaling_test_results.csv']
  budget='epochs 1–5'; hardware='AMD Radeon 860M / DirectML (with operator CPU fallback)'
  if m=='ViT-Tiny': m='ViT-Tiny (custom compact 3-layer, 96-wide proxy)'
 elif m=='Random Forest':
  rows=count('random_forest_scaling.csv','model_name',m);loc=['random_forest_scaling.csv','random_forest_test_results.csv'];budget='10,25,50,100,200,400 trees';hardware='CPU'
 elif p=='unsupervised' and m in ('K-Means','Gaussian Mixture Model'):
  key='MiniBatch K-Means' if m=='K-Means' else m
  rows=count('unsupervised_scaling.csv','model_name',key);loc=['unsupervised_scaling.csv'];budget='algorithm-specific iterative budgets';hardware='CPU'
 elif m in ('Autoencoder','Variational Autoencoder'):
  key=m;rows=count('autoencoder_scaling.csv','model_name',key);loc=['autoencoder_scaling.csv','autoencoder_test_results.csv'];budget='epochs 1–5; 3 seeds';hardware='AMD Radeon 860M / DirectML'
 elif m=='Pseudo-Labeling':
  rows=count('pseudolabel_baseline_sweep.csv','method','pseudo_label');loc=['pseudolabel_baseline_sweep.csv','pseudolabel_test_results.csv'];budget='epochs 1–5; label fractions 1–40%; 3 seeds';hardware='AMD Radeon 860M / DirectML'
 elif m=='DBSCAN':
  rows=count('dbscan_size_scaling.csv','model_name',m);loc=['dbscan_parameter_sweep.csv','dbscan_size_scaling.csv'];budget='2.5K,5K,10K samples x 3 seeds (data-size cost curve); separate 8-setting parameter sensitivity';hardware='CPU'
 elif m=='Label Propagation':
  rows=count('label_propagation_scaling.csv','model_name',m);loc=['label_propagation_scaling.csv'];budget='max iterations 1–2,000; tol 1e-3; 3 seeds; convergence observed';hardware='CPU sklearn; actual LabelPropagation'
 elif m in ('Mean Teacher','FixMatch','MixMatch'):
  key={'Mean Teacher':'MeanTeacher','FixMatch':'FixMatch','MixMatch':'MixMatch'}[m]
  rows=count('ssl_algorithms.csv','model_name',key);loc=['ssl_algorithms.csv','ssl_algorithms_test.csv'];budget='epochs 1–3; labeled fractions 1,5,10,20,40%; 3 seeds';hardware='DirectML for MeanTeacher/FixMatch; CPU MixMatch'
 elif p=='reinforcement_learning':
  key=m.lower().replace(' ','_');rows=count('rl_scaling_results.csv','model_name',m);loc=['rl_scaling_results.csv'];budget='5K,10K,25K environment steps; 3 seeds';hardware='CPU'
 else:
  # Retain original genuine feasibility pilot evidence.
  key={'Label Propagation':'Label Propagation','Mean Teacher':'Mean Teacher','FixMatch':'FixMatch','MixMatch':'MixMatch'}.get(m,m)
  raw=read(ROOT/'results/raw/full_suite_results.csv');rows=[r for r in raw if r.get('model')==key and not r.get('status','').startswith('superseded')];loc=['full_suite_results.csv'];budget='single feasibility point / legacy pilot'
  if m=='Label Propagation': m='Label Propagation (legacy LabelSpreading pilot)'
  hardware='CPU or DirectML; see raw record'
 if rows and not hardware: hardware=rows[0].get('hardware') or rows[0].get('backend') or 'see raw records'
 seeds=len(set(r.get('seed','') for r in rows if r.get('seed','')!=''))
 status='pilot' if rows else 'implemented'
 path=', '.join(f'`results/raw/{p}`' for p in loc) if loc else 'none'
 out.append(f"| {m} | {'yes' if x.get('implementation_target') else 'unverified'} | {'yes' if rows else 'no verified run'} | {budget} | {seeds} | {hardware} | {path} | {status} |")
llm=[('Pythia-70M','step143000; 50K WikiText-2 tokens + 100/task harness subset',True),('Pythia-160M','step143000; 50K tokens',True),('Pythia-410M','step143000; 50K tokens',True),('Pythia-1B','no long local evaluation',False),('Pythia-1.4B','no long local evaluation',False),('Pythia-2.8B','no long local evaluation',False),('Pythia-6.9B','no long local evaluation',False),('Pythia-12B','no long local evaluation',False),('GPT-2 Small','main; 50K WikiText-2 tokens + 100/task harness subset',True),('GPT-2 Medium','weight retrieval/evaluation incomplete',False),('GPT-2 Large','no long local evaluation',False),('GPT-2 XL','no long local evaluation',False),('OPT-125M','main; 50K WikiText-2 tokens + 100/task harness subset',True),('OPT-350M','main; 50K WikiText-2 tokens',True),('OPT-1.3B','no long local evaluation',False)]
out += ['', '## LLM configurations (15)', '', '| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |','|---|---|---|---|---:|---|---|---|']
for m,b,done in llm:
 out.append(f"| {m} | checkpoint/config available | {'yes, local inference' if done else 'attempt/status in prior ledger only'} | {b} | 1 or n/a | CPU | `results/llm/full_wikitext2_evaluations.csv` / existing LLM ledgers | {'pilot' if done else 'resource_limited'} |")
lit=read(ROOT/'results/literature/large_model_literature.csv')
out += ['',f'## Literature configurations ({len(lit)})','', '| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |','|---|---|---|---|---:|---|---|---|']
for r in lit:
 out.append(f"| {r['model']} | published configuration | yes, literature report only | published observation; not locally reproduced | n/a | {r.get('hardware') or 'not fully extracted'} | `results/literature/large_model_literature.csv` | literature_only |")
(ROOT/'FINAL_COMPLETION_MATRIX.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
(ROOT/'EXPERIMENT_COMPLETION_MATRIX.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
print('Wrote matrix:',len(reg),'conventional,',len(llm),'LLM,',len(lit),'literature rows')
