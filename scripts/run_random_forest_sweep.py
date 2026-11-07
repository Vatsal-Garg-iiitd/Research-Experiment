"""Resumable tree-count and seed sweep for a fixed Fashion-MNIST split."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,psutil
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,log_loss

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/raw/random_forest_scaling.csv';TEST=ROOT/'results/raw/random_forest_test_results.csv';CK=ROOT/'results/checkpoints/random_forest';CFG=ROOT/'results/configurations'
BUDGETS=[10,25,50,100,200,400]
def append(path,row):
 path.parent.mkdir(parents=True,exist_ok=True);exists=path.exists() and path.stat().st_size>0
 with path.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if not exists:w.writeheader()
  w.writerow(row);f.flush()
def rows(path):
 if not path.exists():return []
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def data(n):
 ds=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor());te=FashionMNIST(str(ROOT/'results/cache'),train=False,download=True,transform=ToTensor())
 idx=np.random.default_rng(20261106).permutation(len(ds))[:n+2000]
 x=np.stack([ds[int(i)][0].numpy().reshape(-1) for i in idx]);y=np.array([ds[int(i)][1] for i in idx]);xt=np.stack([te[i][0].numpy().reshape(-1) for i in range(len(te))]);yt=np.array([te[i][1] for i in range(len(te))])
 return x[:n],y[:n],x[n:],y[n:],xt,yt,hashlib.sha256(idx.tobytes()).hexdigest()[:16]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);ap.add_argument('--train-size',type=int,default=10000);a=ap.parse_args();x,y,xv,yv,xt,yt,split=data(a.train_size)
 for seed in a.seeds:
  rid=f'rf_trees_seed{seed}_n{a.train_size}_v1';path=CK/f'{rid}.joblib';cfg=CFG/f'{rid}.json'
  cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(json.dumps({'experiment_id':rid,'model':'Random Forest','dataset':'Fashion-MNIST','split_id':split,'seed':seed,'train_size':len(y),'validation_size':len(yv),'test_size':len(yt),'tree_budgets':BUDGETS,'parameters':{'criterion':'gini','max_features':'sqrt','warm_start':True,'n_jobs':-1},'evaluation':'validation at each tree budget; official test once at 400 trees'},indent=2),encoding='utf-8')
  forest=joblib.load(path) if path.exists() else RandomForestClassifier(n_estimators=0,warm_start=True,random_state=seed,n_jobs=-1,max_features='sqrt',criterion='gini',min_samples_leaf=1)
  done={(r['seed'],int(r['budget_value'])) for r in rows(OUT) if r.get('experiment_status')=='completed'}
  cumulative=sum(float(r['training_time_seconds']) for r in rows(OUT) if r.get('experiment_id','').startswith(rid))
  for trees in BUDGETS:
   if forest.n_estimators<trees:
    forest.set_params(n_estimators=trees);t=time.perf_counter();forest.fit(x,y);dt=time.perf_counter()-t;cumulative+=dt
    CK.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');joblib.dump(forest,tmp);tmp.replace(path)
   else:dt=0.
   if (str(seed),trees) in done:continue
   t=time.perf_counter();prob=forest.predict_proba(xv);pred=prob.argmax(1);ev=time.perf_counter()-t
   append(OUT,{'experiment_id':f'{rid}_trees{trees}','model_name':'Random Forest','paradigm':'supervised','dataset':'Fashion-MNIST','dataset_revision':'torchvision FashionMNIST','split_id':split,'seed':seed,'budget_type':'trees','budget_value':trees,'optimizer_steps':'','training_tokens':'','epochs':'','environment_steps':'','training_time_seconds':dt,'cumulative_training_time_seconds':cumulative,'evaluation_time_seconds':ev,'inference_throughput_samples_per_second':len(yv)/ev if ev else '', 'peak_memory_bytes':psutil.Process().memory_info().rss,'hardware':'AMD Ryzen AI 7 350 CPU','backend':'scikit-learn CPU','evaluation_metric':'validation_accuracy','evaluation_value':accuracy_score(yv,pred),'validation_log_loss':log_loss(yv,prob,labels=np.arange(10)),'experiment_status':'completed','source_type':'local_measurement','configuration_path':str(cfg.relative_to(ROOT)),'timestamp_utc':datetime.now(timezone.utc).isoformat(),'details':json.dumps({'warm_start_incremental_trees':True,'parameter_count':'not applicable','model_complexity':trees})})
   print(f'RF seed={seed} trees={trees} val_acc={accuracy_score(yv,pred):.4f} cumulative_s={cumulative:.1f}',flush=True)
  tid=f'{rid}_test400';existing={r['experiment_id'] for r in rows(TEST)}
  if tid not in existing:
   prob=forest.predict_proba(xt);pred=prob.argmax(1);p,r,f,_=precision_recall_fscore_support(yt,pred,average='macro',zero_division=0)
   append(TEST,{'experiment_id':tid,'model_name':'Random Forest','seed':seed,'trees':forest.n_estimators,'test_accuracy':accuracy_score(yt,pred),'macro_precision':p,'macro_recall':r,'macro_f1':f,'test_log_loss':log_loss(yt,prob,labels=np.arange(10)),'confusion_matrix':json.dumps(confusion_matrix(yt,pred).tolist()),'test_size':len(yt),'status':'completed'})
 print('saved',OUT,TEST)
if __name__=='__main__':main()
