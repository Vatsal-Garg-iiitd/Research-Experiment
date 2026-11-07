"""Multi-seed optimization-budget curves for K-Means and diagonal GMM."""
from __future__ import annotations
import argparse,csv,hashlib,json,time,warnings
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,psutil
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/raw/unsupervised_scaling.csv';CK=ROOT/'results/checkpoints/unsupervised';CFG=ROOT/'results/configurations'
K_BUDGETS=[1,5,10,20,50];G_BUDGETS=[1,3,5,10,20]
def append(path,row):
 path.parent.mkdir(parents=True,exist_ok=True);exists=path.exists() and path.stat().st_size>0
 with path.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if not exists:w.writeheader()
  w.writerow(row);f.flush()
def rows():
 if not OUT.exists():return []
 with OUT.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def arrays(n):
 ds=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor());idx=np.random.default_rng(20261106).permutation(len(ds))[:n+2000]
 x=np.stack([ds[int(i)][0].numpy().reshape(-1) for i in idx]);y=np.array([ds[int(i)][1] for i in idx]);return x[:n],y[:n],x[n:],y[n:],hashlib.sha256(idx.tobytes()).hexdigest()[:16]
def addrow(name,seed,budget,dt,cum,eval_s,score,split,pre_s,config,extra):
 append(OUT,{'experiment_id':f'{name.lower().replace(" ","_")}_seed{seed}_budget{budget}_v1','model_name':name,'paradigm':'unsupervised','dataset':'Fashion-MNIST','dataset_revision':'torchvision FashionMNIST','split_id':split,'seed':seed,'budget_type':'partial_fit_updates' if name=='MiniBatch K-Means' else 'warm_start_em_iterations','budget_value':budget,'training_time_seconds':dt,'cumulative_training_time_seconds':cum,'preprocessing_time_seconds':pre_s,'evaluation_time_seconds':eval_s,'peak_memory_bytes':psutil.Process().memory_info().rss,'hardware':'AMD Ryzen AI 7 350 CPU','backend':'scikit-learn CPU','evaluation_metric':'validation_ARI','evaluation_value':score[0],'validation_NMI':score[1],'validation_silhouette':score[2],'training_log_likelihood':score[3],'experiment_status':'completed','source_type':'local_measurement','configuration_path':str(config.relative_to(ROOT)),'timestamp_utc':datetime.now(timezone.utc).isoformat(),'details':json.dumps(extra)})
def evaluate(est,zv,yv,seed,gmm=False):
 pred=est.predict(zv) if gmm else est.predict(zv);unique=np.unique(pred)
 ari=adjusted_rand_score(yv,pred);nmi=normalized_mutual_info_score(yv,pred)
 sil=silhouette_score(zv,pred,sample_size=min(1000,len(zv)),random_state=seed) if 1<len(unique)<len(pred) else ''
 ll=float(est.score(zv)) if gmm else ''
 return ari,nmi,sil,ll
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);ap.add_argument('--train-size',type=int,default=10000);a=ap.parse_args()
 x,y,xv,yv,split=arrays(a.train_size);t=time.perf_counter();pca=PCA(n_components=32,random_state=20261106);z=pca.fit_transform(x);zv=pca.transform(xv);pre=time.perf_counter()-t
 for seed in a.seeds:
  kname='MiniBatch K-Means';krun=f'kmeans_seed{seed}_n{a.train_size}_v1';kcfg=CFG/f'{krun}.json';kcfg.parent.mkdir(parents=True,exist_ok=True);kcfg.write_text(json.dumps({'experiment_id':krun,'model':'MiniBatchKMeans','seed':seed,'split_id':split,'train_size':len(y),'pca_components':32,'pca_fit':'train only','partial_fit_budgets':K_BUDGETS,'batch_size':512,'random_state':seed},indent=2),encoding='utf-8')
  kp=CK/f'{krun}.joblib';kstate=joblib.load(kp) if kp.exists() else {'model':MiniBatchKMeans(n_clusters=10,random_state=seed,batch_size=512,n_init=3),'updates':0,'cum':0.};k=kstate['model'];updates=int(kstate['updates']);kcum=float(kstate['cum']);rng=np.random.default_rng(seed)
  if 'rng_state' in kstate:rng.bit_generator.state=kstate['rng_state'];perm=kstate['perm'];cursor=int(kstate['cursor'])
  else:perm=rng.permutation(len(z));cursor=0
  done={(r['experiment_id']) for r in rows()}
  for target in K_BUDGETS:
   dt=0.
   while updates<target:
    if cursor+512>len(perm):perm=rng.permutation(len(z));cursor=0
    batch=perm[cursor:cursor+512];cursor+=512;t=time.perf_counter();k.partial_fit(z[batch]);dt=time.perf_counter()-t;kcum+=dt;updates+=1
    CK.mkdir(parents=True,exist_ok=True);tmp=kp.with_suffix('.tmp');joblib.dump({'model':k,'updates':updates,'cum':kcum,'rng_state':rng.bit_generator.state,'perm':perm,'cursor':cursor},tmp);tmp.replace(kp)
   eid=f'mini-batch_k-means_seed{seed}_budget{target}_v1'
   if eid not in done:
    t=time.perf_counter();sc=evaluate(k,zv,yv,seed);ev=time.perf_counter()-t;addrow(kname,seed,target,dt,kcum,ev,sc,split,pre,kcfg,{'feature_transform':'PCA32 fit on training subset only','explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'algorithm':'incremental MiniBatchKMeans partial_fit','unlabeled_training':True})
    print(f'KMeans seed={seed} updates={target} val_ARI={sc[0]:.4f} NMI={sc[1]:.4f}',flush=True)
  gname='Gaussian Mixture Model';grun=f'gmm_seed{seed}_n{a.train_size}_v1';gcfg=CFG/f'{grun}.json';gcfg.write_text(json.dumps({'experiment_id':grun,'model':'GaussianMixture','seed':seed,'split_id':split,'train_size':len(y),'pca_components':32,'pca_fit':'train only','warm_start_em_iterations':G_BUDGETS,'components':10,'covariance_type':'diag','n_init':1},indent=2),encoding='utf-8')
  gp=CK/f'{grun}.joblib';gstate=joblib.load(gp) if gp.exists() else {'model':GaussianMixture(n_components=10,covariance_type='diag',n_init=1,max_iter=1,warm_start=True,random_state=seed,init_params='kmeans'),'updates':0,'cum':0.};g=gstate['model'];updates=int(gstate['updates']);gcum=float(gstate['cum']);
  for target in G_BUDGETS:
   dt=0.
   while updates<target:
    t=time.perf_counter()
    with warnings.catch_warnings():warnings.simplefilter('ignore',ConvergenceWarning);g.fit(z)
    dt=time.perf_counter()-t;gcum+=dt;updates+=1
    CK.mkdir(parents=True,exist_ok=True);tmp=gp.with_suffix('.tmp');joblib.dump({'model':g,'updates':updates,'cum':gcum},tmp);tmp.replace(gp)
   eid=f'gaussian_mixture_model_seed{seed}_budget{target}_v1'
   if eid not in done:
    t=time.perf_counter();sc=evaluate(g,zv,yv,seed,True);ev=time.perf_counter()-t;addrow(gname,seed,target,dt,gcum,ev,sc,split,pre,gcfg,{'feature_transform':'PCA32 fit on training subset only','explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'algorithm':'GaussianMixture warm_start, one EM iteration per fit call','converged_last_iteration':bool(g.converged_)})
    print(f'GMM seed={seed} EM={target} val_ARI={sc[0]:.4f} NMI={sc[1]:.4f}',flush=True)
 print('saved',OUT)
if __name__=='__main__':main()
