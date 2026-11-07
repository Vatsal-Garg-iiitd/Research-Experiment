"""Sequential CPU evaluation of feasible small causal LMs on one shared WikiText-2 slice.

Each checkpoint is appended as soon as evaluated. Larger models are entered as
resource_limited after metadata inspection, never assigned fabricated scores.
"""
from __future__ import annotations
import argparse,csv,gc,json,time,math
from pathlib import Path
import torch,psutil
from datasets import load_dataset
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/llm/expanded_llm_evaluations.csv'
LEGACY=ROOT/'results/llm/llm_checkpoint_results.csv'
PROC=psutil.Process(); MAX_PARAMS=450_000_000
CONFIGS=[
 ('Pythia-70M','EleutherAI/pythia-70m',70_426_624,'Pythia'),('Pythia-160M','EleutherAI/pythia-160m',162_322_944,'Pythia'),('Pythia-410M','EleutherAI/pythia-410m',405_334_016,'Pythia'),('Pythia-1B','EleutherAI/pythia-1b',1_011_382_272,'Pythia'),('Pythia-1.4B','EleutherAI/pythia-1.4b',1_414_647_808,'Pythia'),('Pythia-2.8B','EleutherAI/pythia-2.8b',2_805_067_776,'Pythia'),('Pythia-6.9B','EleutherAI/pythia-6.9b',6_915_155_968,'Pythia'),('Pythia-12B','EleutherAI/pythia-12b',12_002_822_144,'Pythia'),
 ('GPT-2 Small','openai-community/gpt2',124_439_808,'GPT-2'),('GPT-2 Medium','openai-community/gpt2-medium',354_823_168,'GPT-2'),('GPT-2 Large','openai-community/gpt2-large',774_030_080,'GPT-2'),('GPT-2 XL','openai-community/gpt2-xl',1_557_611_200,'GPT-2'),
 ('OPT-125M','facebook/opt-125m',125_239_808,'OPT'),('OPT-350M','facebook/opt-350m',331_000_000,'OPT'),('OPT-1.3B','facebook/opt-1.3b',1_315_000_000,'OPT')]

def append(path,row):
 exists=path.exists() and path.stat().st_size>0
 with path.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if not exists:w.writeheader()
  w.writerow(row);f.flush()

def tokens():
 ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test')
 return '\n'.join(x for x in ds['text'] if x.strip())[:30000]

def measure(name,repo,params,family,text,revision='main',step=None,local=True):
 effective_step=143000 if family=='Pythia' and revision=='main' and step is None else step
 row={'model':name,'family':family,'parameters':params,'checkpoint_id':revision,'training_step':effective_step if effective_step is not None else '', 'tokens_processed':'','test_loss':'','test_perplexity':'','benchmark':'WikiText-2 raw test prefix','benchmark_protocol':'first 30000 raw characters; up to 2048 model tokens (capped by model context); one sequence; CPU fp32','benchmark_score':'','model_load_seconds':'','evaluation_seconds':'','peak_ram_bytes':'','peak_vram_bytes':'','source_type':'local pretrained checkpoint','source_url':f'https://huggingface.co/{repo}','status':'running','notes':'single-seed inference; evaluation is not original pretraining compute'}
 if family=='Pythia' and effective_step is not None:
  if effective_step==143000:
   row['tokens_processed']=299_892_736_000;row['notes']+='; final training token count source-reported in model card'
  else:
   row['tokens_processed']=effective_step*2_097_152;row['notes']+='; token count calculated from model-card final tokens/step (2,097,152)'
 tok=model=None
 try:
  before=time.perf_counter(); tok=AutoTokenizer.from_pretrained(repo,revision=revision); model=AutoModelForCausalLM.from_pretrained(repo,revision=revision,torch_dtype=torch.float32,low_cpu_mem_usage=False); model_load=time.perf_counter()-before
  model.eval(); limit=min(2048,int(getattr(model.config,'max_position_embeddings',2048))); row['benchmark_protocol']=f"first 30000 raw characters; first {limit} model tokens (model context cap); one sequence; CPU fp32"; ids=tok(text,return_tensors='pt',truncation=True,max_length=limit)['input_ids']
  t=time.perf_counter()
  with torch.inference_mode(): loss=model(input_ids=ids,labels=ids).loss
  elapsed=time.perf_counter()-t; lv=float(loss); row.update(test_loss=lv,test_perplexity=math.exp(lv) if lv<20 else '',model_load_seconds=model_load,evaluation_seconds=elapsed,peak_ram_bytes=getattr(PROC.memory_info(),'peak_wset',PROC.memory_info().rss),status='completed')
 except Exception as e:
  row.update(status='failed',notes=(row['notes']+'; '+type(e).__name__+': '+str(e))[:1800])
 append(OUT,row)
 if family=='Pythia': append(LEGACY,{'model':name,'family':family,'parameters':params,'checkpoint_id':revision,'training_step':step or '', 'tokens_processed':row['tokens_processed'],'train_loss':'','test_loss':row['test_loss'],'test_perplexity':row['test_perplexity'],'benchmark':row['benchmark'],'benchmark_protocol':row['benchmark_protocol'],'benchmark_score':'','model_load_seconds':row['model_load_seconds'],'evaluation_seconds':row['evaluation_seconds'],'peak_ram_bytes':row['peak_ram_bytes'],'peak_vram_bytes':'','source_type':'local evaluation of pretrained checkpoint','source_url':row['source_url'],'status':row['status'],'notes':row['notes']})
 print(name,revision,row['status'],row['test_loss'],row['notes'][-180:],flush=True)
 if model is not None: del model
 if tok is not None: del tok
 gc.collect()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--retry-failed',action='store_true',help='Retry model configurations with a previous failed attempt; this may repeat large network downloads.');args=parser.parse_args()
 torch.set_num_threads(8); text=tokens()
 # Eight points on the Pythia-70M trajectory; published branches, same data slice/protocol.
 steps=[0,1000,5000,10000,20000,50000,100000,143000]
 for s in steps:
  already=False
  if OUT.exists():
   with OUT.open(encoding='utf-8') as f: already=any(r.get('model')=='Pythia-70M' and r.get('checkpoint_id')==f'step{s}' and r.get('status')=='completed' for r in csv.DictReader(f))
  if s==143000 and LEGACY.exists():
   with LEGACY.open(encoding='utf-8') as f: already=already or any(r.get('model')=='Pythia-70M' and r.get('checkpoint_id')=='step143000' and r.get('status')=='completed' for r in csv.DictReader(f))
  if already: continue
  measure('Pythia-70M','EleutherAI/pythia-70m',70_426_624,'Pythia',text,revision=f'step{s}',step=s)
 # Add three directly named checkpoints for the next two resource-feasible Pythia sizes.
 for name,repo,params,step in [('Pythia-160M','EleutherAI/pythia-160m',162_322_944,0),('Pythia-160M','EleutherAI/pythia-160m',162_322_944,10000),('Pythia-160M','EleutherAI/pythia-160m',162_322_944,143000),('Pythia-410M','EleutherAI/pythia-410m',405_334_016,0),('Pythia-410M','EleutherAI/pythia-410m',405_334_016,10000),('Pythia-410M','EleutherAI/pythia-410m',405_334_016,143000)]:
  rev=f'step{step}'; exists=[]
  if OUT.exists():
   with OUT.open(encoding='utf-8') as f:exists=[r for r in csv.DictReader(f) if r.get('model')==name and r.get('checkpoint_id')==rev and r.get('status')=='completed']
  if not exists:measure(name,repo,params,'Pythia',text,revision=rev,step=step)
 for name,repo,params,family in CONFIGS[1:]:
  existing=[]
  if OUT.exists():
   with OUT.open(encoding='utf-8') as f: existing=[r for r in csv.DictReader(f) if r.get('model')==name and r.get('checkpoint_id')=='main']
  if existing and existing[-1].get('status') in ('completed','resource_limited'): continue
  if existing and existing[-1].get('status')=='failed' and not args.retry_failed:continue
  if params>MAX_PARAMS:
   # Pull only public config metadata so model presence/parameter scale is verified before restricting weights.
   row={'model':name,'family':family,'parameters':params,'checkpoint_id':'main','training_step':'','tokens_processed':'','test_loss':'','test_perplexity':'','benchmark':'WikiText-2 raw test prefix','benchmark_protocol':'not run: CPU resident-memory guard at 450M parameters under observed free RAM','benchmark_score':'','model_load_seconds':'','evaluation_seconds':'','peak_ram_bytes':getattr(PROC.memory_info(),'peak_wset',PROC.memory_info().rss),'peak_vram_bytes':'','source_type':'published checkpoint identified; local weights not loaded','source_url':f'https://huggingface.co/{repo}','status':'resource_limited','notes':'No local score. Parameter count exceeds conservative 450M resident-memory guard; total visible RAM ~24.4GB, only ~4.6GB free at hardware inspection. Larger local load may destabilize the active Windows session.'}
   try:
    cfg=AutoConfig.from_pretrained(repo); row['notes']+=f"; HF config loaded ({cfg.model_type})"
   except Exception as e: row['notes']+=f'; config lookup failed: {type(e).__name__}: {e}'
   append(OUT,row); print(name,'resource_limited',flush=True); continue
  measure(name,repo,params,family,text,revision='main')
 print('saved',OUT)
if __name__=='__main__':main()
