"""Execute the previously unrun classical learning models on a fixed Fashion-MNIST subset.

This is an incremental, one-seed, one-budget resource-feasibility batch. It appends
records; it does not overwrite the earlier pilot CSV.
"""
from __future__ import annotations
import csv, json, time, hashlib
from pathlib import Path
import numpy as np
import psutil
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/raw/full_suite_results.csv'
SEED=11

def save(row):
    OUT.parent.mkdir(parents=True,exist_ok=True)
    exists=OUT.exists() and OUT.stat().st_size>0
    with OUT.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if not exists: w.writeheader()
        w.writerow(row); f.flush()

def main():
    ds=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor())
    rng=np.random.default_rng(SEED); ids=rng.permutation(len(ds))[:2500]
    X=np.stack([ds[i][0].numpy().reshape(-1) for i in ids]).astype('float32')
    y=np.array([ds[i][1] for i in ids]); split=int(.8*len(X))
    Xtr,Xte,ytr,yte=X[:split],X[split:],y[:split],y[split:]
    hashid=hashlib.sha256(ids.tobytes()).hexdigest()[:16]
    # RF is supervised; fit once and measure held-out accuracy.
    t=time.perf_counter(); rf=RandomForestClassifier(n_estimators=100,max_depth=None,n_jobs=-1,random_state=SEED)
    rf.fit(Xtr,ytr); fit=time.perf_counter()-t; t=time.perf_counter(); pred=rf.predict(Xte); ev=time.perf_counter()-t
    save(dict(run_id='full_rf_seed11_v1',paradigm='supervised',model='Random Forest',dataset='Fashion-MNIST-subset-2500',split_hash=hashid,seed=SEED,budget_unit='trees',budget_value=100,train_seconds=fit,cumulative_train_seconds=fit,eval_seconds=ev,metric_name='test_accuracy',metric_value=accuracy_score(yte,pred),train_steps=100,parameter_count='',peak_ram_bytes=psutil.Process().memory_info().rss,peak_vram_bytes='',hardware='CPU; scikit-learn',status='completed',details=json.dumps({'train_n':split,'test_n':len(Xte),'purpose':'new full-suite feasibility run; 1 seed, single forest size'})))
    # Label propagation uses only a fixed 10% labelled subset in the training split.
    unl=-np.ones(split,dtype=int); lab_ids=rng.choice(split,size=max(1,split//10),replace=False); unl[lab_ids]=ytr[lab_ids]
    # PCA and scaling keep the graph tractable while preserving a fixed transductive protocol.
    t=time.perf_counter(); allz=PCA(n_components=32,random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
    # Propagation is evaluated transductively over train and test images; test labels are hidden.
    ally=np.r_[unl,np.full(len(Xte),-1)]
    model=LabelSpreading(kernel='knn',n_neighbors=10,max_iter=30); model.fit(allz,ally); fit=time.perf_counter()-t
    t=time.perf_counter(); pred=model.transduction_[split:]; ev=time.perf_counter()-t
    save(dict(run_id='full_labelprop_seed11_v1',paradigm='semi_supervised',model='Label Propagation',dataset='Fashion-MNIST-transductive-2500',split_hash=hashid,seed=SEED,budget_unit='graph_iterations',budget_value=int(model.n_iter_),train_seconds=fit,cumulative_train_seconds=fit,eval_seconds=ev,metric_name='transductive_test_accuracy',metric_value=accuracy_score(yte,pred),train_steps=int(model.n_iter_),parameter_count='',peak_ram_bytes=psutil.Process().memory_info().rss,peak_vram_bytes='',hardware='CPU; scikit-learn PCA+LabelSpreading',status='completed',details=json.dumps({'label_fraction':.1,'train_n':split,'unlabeled_test_n':len(Xte),'pca_dimensions':32,'transductive':True,'converged':bool(model.n_iter_<30)})))
    # Clustering evaluations use the same images and labels only for external ARI scoring.
    Zall=PCA(n_components=32,random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
    methods=[('K-Means',KMeans(n_clusters=10,n_init=5,max_iter=100,random_state=SEED)),('Gaussian Mixture Model',GaussianMixture(n_components=10,covariance_type='diag',max_iter=100,random_state=SEED)),('DBSCAN',DBSCAN(eps=2.5,min_samples=8,n_jobs=-1))]
    for name,est in methods:
        t=time.perf_counter(); est.fit(Zall); fit=time.perf_counter()-t; labels=getattr(est,'labels_',None)
        if labels is None: labels=est.predict(Zall)
        it=int(getattr(est,'n_iter_',getattr(est,'n_iter',1)))
        save(dict(run_id='full_'+name.lower().replace(' ','_')+'_seed11_v1',paradigm='unsupervised',model=name,dataset='Fashion-MNIST-subset-2500',split_hash=hashid,seed=SEED,budget_unit='algorithm_iterations',budget_value=it,train_seconds=fit,cumulative_train_seconds=fit,eval_seconds=0.0,metric_name='adjusted_rand_index',metric_value=adjusted_rand_score(y,labels),train_steps=it,parameter_count='',peak_ram_bytes=psutil.Process().memory_info().rss,peak_vram_bytes='',hardware='CPU; scikit-learn',status='completed',details=json.dumps({'n_samples':len(X),'pca_dimensions':32,'parameters':est.get_params()})))
    print('saved',OUT)
if __name__=='__main__': main()
