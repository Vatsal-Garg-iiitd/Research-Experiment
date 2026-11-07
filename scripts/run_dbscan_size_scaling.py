"""DBSCAN data-size cost curve (not an iterative training-budget curve)."""
import csv,json,time,hashlib
from pathlib import Path
import numpy as np,psutil
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/raw/dbscan_size_scaling.csv';
def append(r):
 e=OUT.exists() and OUT.stat().st_size>0
 with OUT.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(r));
  if not e:w.writeheader()
  w.writerow(r);f.flush()
ds=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor())
for seed in [11,23,37]:
 idx=np.random.default_rng(seed).permutation(len(ds))[:10000];X=np.stack([ds[int(i)][0].numpy().reshape(-1) for i in idx]);y=np.array([ds[int(i)][1] for i in idx]);sc=StandardScaler().fit_transform(X);z=PCA(n_components=32,random_state=seed).fit_transform(sc);split=hashlib.sha256(idx.tobytes()).hexdigest()[:16]
 for n in [2500,5000,10000]:
  t=time.perf_counter();labels=DBSCAN(eps=2.8803592324,min_samples=5,metric='euclidean',n_jobs=-1).fit_predict(z[:n]);sec=time.perf_counter()-t;mask=labels!=-1;sil=silhouette_score(z[:n],labels,sample_size=min(1000,n),random_state=seed) if len(set(labels[mask]))>1 and mask.sum()>2 else ''
  append({'experiment_id':f'dbscan_n{n}_seed{seed}_eps288_v1','model_name':'DBSCAN','dataset':'Fashion-MNIST unsupervised data-size scaling','split_id':split,'seed':seed,'budget_type':'samples_fit','budget_value':n,'training_time_seconds':sec,'evaluation_metric':'adjusted_rand_index','evaluation_value':adjusted_rand_score(y[:n],labels),'NMI':normalized_mutual_info_score(y[:n],labels),'silhouette_non_noise':sil,'cluster_count':len(set(labels))-int(-1 in labels),'noise_fraction':float(np.mean(labels==-1)),'eps':2.8803592324,'min_samples':5,'pca_dimensions':32,'backend':'CPU sklearn','status':'completed','details':json.dumps({'fixed_parameters_across_sizes':True,'PCA_fit_per_seed_on_full_10000_pool':True,'external_labels_used_only_for_scoring':True,'data_size_is_compute_budget':True})})
  print(seed,n,round(sec,2),flush=True)
