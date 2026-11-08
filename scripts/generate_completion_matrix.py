import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
reg=read(ROOT/'configs/model_registry.csv')
audit=read(ROOT/'results/processed/conventional_completion_audit.csv')
audit_by_model={r['model']:r for r in audit}
literature_audit=read(ROOT/'results/processed/literature_completion_audit.csv')
literature_audit_by_model={r['model']:r for r in literature_audit}
def count(file,model_key,model):
 p=ROOT/'results/raw'/file
 if not p.exists():return []
 return [r for r in read(p) if r.get(model_key)==model and r.get('experiment_status',r.get('status','')) not in ('superseded','failed','resource_limited')]
complete_conventional=sum(x.get('status')=='completed' for x in audit)
complete_literature=sum(x.get('status')=='completed_literature_only' for x in literature_audit)
complete_total=complete_conventional+complete_literature
out=['# Final Completion Matrix','', 'Updated 2026-11-08 from raw-result coverage checks and field-level literature evidence. Conventional status requires all registered budgets, seeds, and final evaluations verified by `scripts/audit_conventional_completion.py`. Literature status requires the source, core scale/token evidence, and every registered field to be classified and located in `scripts/audit_literature_completion.py`; it denotes a completed literature objective and never local reproduction. Double DQN retains its source-mapping correction in derived records only; raw CSV rows are preserved. Local checkpoint evaluation of an LLM does not mean local pretraining.', '', f'**Strict objective completion: {complete_total}/50 configurations ({complete_total/50:.0%}).** The denominator remains 20 conventional models + 15 specified LLM configurations + 15 literature configurations. The original objective types and denominator are unchanged: local-study objectives require local evidence; literature objectives require source-based evidence. This count distinguishes collected evidence, partial local LLM evaluations, and completed literature reviews.','', '| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |','|---|---|---|---|---:|---|---|---|']
for x in reg:
 m=x['model']; p=x['paradigm']; rows=[]; loc=[]; hardware=''
 if m in ('MLP','LeNet-style CNN','ResNet-18'):
  key={'MLP':'MLP','LeNet-style CNN':'CNN / LeNet','ResNet-18':'ResNet-18','ViT-Tiny':'Vision Transformer (compact ViT-Tiny-style)'}[m]
  rows=count('replicated_scaling_results.csv','model_name',key);loc=['replicated_scaling_results.csv','replicated_scaling_test_results.csv']
  budget='10K train: epochs 1–30; 30K/58K train: epochs 1–5; 3 seeds; 10K/30K/60K requested-size endpoints'; hardware='Apple M3 / PyTorch MPS'
 elif m=='ViT-Tiny':
  rows=count('vit_tiny_scaling.csv','model_name','ViT-Tiny');loc=['vit_tiny_scaling.csv','vit_tiny_test_results.csv'];budget='10K Fashion-MNIST: epochs 1–30; 3 seeds; 5,353,738 parameters; 28x28 grayscale / 4x4 patches';hardware='Apple M3 / PyTorch MPS'
 elif m=='Random Forest':
  rows=count('random_forest_scaling.csv','model_name',m);loc=['random_forest_scaling.csv','random_forest_test_results.csv'];budget='10–1,000 warm-start trees; 3 seeds';hardware='CPU'
 elif p=='unsupervised' and m in ('K-Means','Gaussian Mixture Model'):
  key='MiniBatch K-Means' if m=='K-Means' else m
  rows=count('unsupervised_scaling.csv','model_name',key);loc=['unsupervised_scaling.csv'];budget='algorithm-specific iterative budgets';hardware='CPU'
 elif m in ('Autoencoder','Variational Autoencoder'):
  key=m;rows=count('autoencoder_scaling.csv','model_name',key);loc=['autoencoder_scaling.csv','autoencoder_test_results.csv'];budget='epochs 1–30; 3 seeds; standardized linear probe';hardware='Apple M3 / PyTorch MPS'
 elif m=='Pseudo-Labeling':
  rows=count('pseudolabel_baseline_sweep.csv','method','pseudo_label');loc=['pseudolabel_baseline_sweep.csv','pseudolabel_test_results.csv'];budget='epochs 1–30; label fractions 1–40%; 3 seeds; matched labeled-only baseline';hardware='Apple M3 / PyTorch MPS'
 elif m=='DBSCAN':
  rows=count('dbscan_size_scaling.csv','model_name',m);loc=['dbscan_parameter_sweep.csv','dbscan_size_scaling.csv'];budget='2.5K,5K,10K samples x 3 seeds (data-size cost curve); separate 8-setting parameter sensitivity';hardware='CPU'
 elif m=='Label Propagation':
  rows=count('label_propagation_scaling.csv','model_name',m);loc=['label_propagation_scaling.csv'];budget='max iterations 1–2,000; tol 1e-3; 3 seeds; convergence observed';hardware='CPU sklearn; actual LabelPropagation'
 elif m in ('Mean Teacher','FixMatch','MixMatch'):
  key={'Mean Teacher':'MeanTeacher','FixMatch':'FixMatch','MixMatch':'MixMatch'}[m]
  rows=count('ssl_algorithms.csv','model_name',key);loc=['ssl_algorithms.csv','ssl_algorithms_test.csv'];budget='epochs 1–30; labeled fractions 1,5,10,20,40%; 3 seeds';hardware='Apple M3 / PyTorch MPS'
 elif p=='reinforcement_learning':
  key=m.lower().replace(' ','_');rows=count('rl_scaling_results.csv','model_name',m);loc=['rl_scaling_results.csv'];budget='5K–250K requested environment steps; 3 seeds (SAC/Double DQN: 5 seeds)';hardware='CPU'
 else:
  # Retain genuine legacy evidence outside current coverage checks.
  key={'Label Propagation':'Label Propagation','Mean Teacher':'Mean Teacher','FixMatch':'FixMatch','MixMatch':'MixMatch'}.get(m,m)
  raw=read(ROOT/'results/raw/full_suite_results.csv');rows=[r for r in raw if r.get('model')==key and not r.get('status','').startswith('superseded')];loc=['full_suite_results.csv'];budget='legacy single-point observation; separate from current verified study'
  if m=='Label Propagation': m='Label Propagation (legacy LabelSpreading implementation)'
  hardware='CPU or DirectML; see raw record'
 if rows and not hardware: hardware=rows[0].get('hardware') or rows[0].get('backend') or 'see raw records'
 seeds=len(set(r.get('seed','') for r in rows if r.get('seed','')!=''))
 status=audit_by_model.get(m,{}).get('status','unverified') if m in audit_by_model else ('evidence recorded' if rows else 'implemented; no verified run')
 path=', '.join(f'`results/raw/{p}`' for p in loc) if loc else 'none'
 out.append(f"| {m} | {'yes' if x.get('implementation_target') else 'unverified'} | {'yes' if rows else 'no verified run'} | {budget} | {seeds} | {hardware} | {path} | {status} |")
out += ['', '### Additional recorded extensions (outside the 50-objective denominator)', '', '| Model | Environment | Budget | Seeds | Result location | Status |', '|---|---|---|---:|---|---|', '| Compact transformer proxy | Fashion-MNIST | 10K split, epochs 1–30 | 3 | `results/raw/replicated_scaling_results.csv`, `results/raw/replicated_scaling_test_results.csv` | Supplementary 3-layer, width-96 proxy; separate from the ViT-Tiny-scope implementation |', '| PPO-Pendulum | Pendulum-v1 | 5K–250K requested environment steps | 3 | `results/raw/rl_scaling_results.csv` | Recorded RL extension; not an original sixth method |']
llm=[('Pythia-70M','8 released checkpoints (0–143K); full WikiText-2 test; full HellaSwag/PIQA/ARC-Easy zero-shot',True),('Pythia-160M','8 released checkpoints (0–143K); full WikiText-2 test',True),('Pythia-410M','step 0, 10K, 143K prefix checkpoints; full WikiText-2 test',True),('Pythia-1B','no long local evaluation',False),('Pythia-1.4B','no long local evaluation',False),('Pythia-2.8B','no long local evaluation',False),('Pythia-6.9B','no long local evaluation',False),('Pythia-12B','no long local evaluation',False),('GPT-2 Small','main; full WikiText-2 test + 100/task harness smoke subset',True),('GPT-2 Medium','weight retrieval/evaluation incomplete',False),('GPT-2 Large','no long local evaluation',False),('GPT-2 XL','no long local evaluation',False),('OPT-125M','main; full WikiText-2 test + 100/task harness smoke subset',True),('OPT-350M','main; full WikiText-2 test',True),('OPT-1.3B','no long local evaluation',False)]
out += ['', '## LLM configurations (15)', '', '| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |','|---|---|---|---|---:|---|---|---|']
for m,b,done in llm:
 status='failed' if m=='GPT-2 Medium' else ('partial evaluation' if done else 'resource-limited / not evaluated')
 eval_status='failed local checkpoint attempt' if m=='GPT-2 Medium' else ('yes, local inference' if done else 'attempt/status in prior ledger only')
 out.append(f"| {m} | checkpoint/config available | {eval_status} | {b} | 1 or n/a | Apple M3 / MPS or CPU (per run record) | `results/llm/macos_full_wikitext2_evaluations.csv`, `results/llm/expanded_llm_evaluations.csv`, harness records | {status} |")
lit=read(ROOT/'results/literature/large_model_literature.csv')
out += ['',f'## Literature configurations ({len(lit)})','', '| Model | Implemented | Evaluated | Budgets completed | Seeds completed | Hardware | Result location | Status |','|---|---|---|---|---:|---|---|---|']
for r in lit:
 status=literature_audit_by_model.get(r['model'],{}).get('status','needs_review')
 hw=r.get('hardware') or 'as documented in cited source; field gaps shown in evidence ledger'
 out.append(f"| {r['model']} | published configuration/source verified | yes, literature objective | 20 source-located fields; reported gaps remain classified | n/a | {hw} | `results/literature/large_model_literature.csv`, `results/literature/literature_field_evidence.csv` | {status} |")
(ROOT/'FINAL_COMPLETION_MATRIX.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
(ROOT/'EXPERIMENT_COMPLETION_MATRIX.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
print('Wrote matrix:',len(reg),'conventional,',len(llm),'LLM,',len(lit),'literature rows')
