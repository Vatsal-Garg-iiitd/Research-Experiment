"""Evaluate locally feasible causal LMs on a fixed 50k-token WikiText-2 test slice."""
from __future__ import annotations
import argparse,csv,gc,json,math,time
from datetime import datetime,timezone
from pathlib import Path
import psutil,torch
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/llm/full_wikitext2_evaluations.csv';PROC=psutil.Process()
MODELS=[('Pythia-70M','EleutherAI/pythia-70m','step143000',70426624),('Pythia-160M','EleutherAI/pythia-160m','step143000',162322944),('Pythia-410M','EleutherAI/pythia-410m','step143000',405334016),('GPT-2 Small','openai-community/gpt2','main',124439808),('OPT-125M','facebook/opt-125m','main',125239808),('OPT-350M','facebook/opt-350m','main',331000000)]
def append(row):
 OUT.parent.mkdir(parents=True,exist_ok=True);exists=OUT.exists() and OUT.stat().st_size>0
 with OUT.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if not exists:w.writeheader()
  w.writerow(row);f.flush()
def completed_ids():
 if not OUT.exists():return set()
 with OUT.open(encoding='utf-8-sig',newline='') as f:return {r['experiment_id'] for r in csv.DictReader(f) if r.get('status')=='completed'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--token-budget',type=int,default=50000);ap.add_argument('--models',nargs='+',default=[m[0] for m in MODELS]);a=ap.parse_args();torch.set_num_threads(8)
 ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test');text='\n'.join(t for t in ds['text'] if t.strip())
 done=completed_ids()
 for name,repo,revision,params in MODELS:
  if name not in a.models:continue
  eid=f'{name.lower().replace(" ","_")}_{revision}_wikitext2_{a.token_budget}_v1'
  if eid in done:print(name,'already completed; skipped',flush=True);continue
  row={'experiment_id':eid,'model_name':name,'family':repo.split('/')[0],'parameters':params,'checkpoint_id':revision,'training_step':143000 if name.startswith('Pythia') else '','tokens_processed_during_pretraining':'299892736000' if name.startswith('Pythia') else '', 'dataset':'WikiText-2 raw-v1 test','dataset_revision':'Salesforce/wikitext/wikitext-2-raw-v1','evaluation_token_budget_requested':a.token_budget,'evaluation_tokens_input':'','evaluation_tokens_scored':'','context_length':'','chunks':'','test_cross_entropy':'','test_perplexity':'','model_load_seconds':'','evaluation_seconds':'','tokens_per_second':'','peak_ram_bytes':'','peak_vram_bytes':'','hardware':'AMD Ryzen AI 7 350 CPU','backend':f'CPU; PyTorch {torch.__version__}; Transformers','status':'running','source_type':'local_checkpoint_evaluation','source_url':f'https://huggingface.co/{repo}','timestamp_utc':datetime.now(timezone.utc).isoformat(),'notes':"Checkpoint evaluation compute/time is not original pretraining compute. Non-overlapping context windows; each window's first token is excluded by causal loss shift."}
  model=tok=None
  try:
   t=time.perf_counter();tok=AutoTokenizer.from_pretrained(repo,revision=revision);model=AutoModelForCausalLM.from_pretrained(repo,revision=revision,torch_dtype=torch.float32);load_s=time.perf_counter()-t
   cap=int(getattr(model.config,'max_position_embeddings',2048) or 2048);cap=min(cap,2048);ids=tok(text,return_tensors='pt',add_special_tokens=False)['input_ids'][0,:a.token_budget]
   win=cap;loss_sum=0.;scored=0;chunks=0;t=time.perf_counter();model.eval()
   with torch.inference_mode():
    for i in range(0,len(ids),win):
     block=ids[i:i+win]
     if len(block)<2:continue
     out=model(input_ids=block.unsqueeze(0),labels=block.unsqueeze(0));count=len(block)-1;loss_sum+=float(out.loss)*count;scored+=count;chunks+=1
   eval_s=time.perf_counter()-t;loss=loss_sum/scored;row.update(evaluation_tokens_input=len(ids),evaluation_tokens_scored=scored,context_length=win,chunks=chunks,test_cross_entropy=loss,test_perplexity=math.exp(loss),model_load_seconds=load_s,evaluation_seconds=eval_s,tokens_per_second=scored/eval_s if eval_s else '',peak_ram_bytes=getattr(PROC.memory_info(),'peak_wset',PROC.memory_info().rss),status='completed')
  except Exception as e:row.update(status='failed',notes=(row['notes']+'; '+type(e).__name__+': '+str(e))[:2000])
  append(row);print(name,row['status'],row.get('evaluation_tokens_input'),row.get('test_cross_entropy'),row.get('evaluation_seconds'),row.get('notes','')[-150:],flush=True)
  if model is not None:del model
  if tok is not None:del tok
  gc.collect()
 print('saved',OUT)
if __name__=='__main__':main()
