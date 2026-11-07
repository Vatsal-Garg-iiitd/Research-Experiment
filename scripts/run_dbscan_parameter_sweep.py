"""Tune DBSCAN neighborhoods by training-only k-distance quantiles."""
from __future__ import annotations
import csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,psutil
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/raw/dbscan_parameter_sweep.csv'
def main():
 ds=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor());idx=np.random.default_rng(20261106).permutation(len(ds))[:12000]
 x=np.stack([ds[int(i)][0].numpy().reshape(-1) for i in idx]);y=np.array([ds[int(i)][1] for i in idx]);split=hashlib.sha256(idx.tobytes()).hexdigest()[:16]
 pca=PCA(n_components=32,random_state=20261106);z=pca.fit_transform(x[:10000]);zv=pca.transform(x[10000:]);t=time.perf_counter();rows=[]
 for ms in [5,10]:
  nn=NearestNeighbors(n_neighbors=ms,n_jobs=-1).fit(z);dist,_=nn.kneighbors(z);kd=dist[:,-1]
  for q in [.5,.75,.9,.95]:
   eps=float(np.quantile(kd,q));model=DBSCAN(eps=eps,min_samples=ms,n_jobs=-1);start=time.perf_counter();labels=model.fit_predict(np.vstack([z,zv]));fit=time.perf_counter()-start;pred=labels[len(z):];clusters=set(pred)-{-1};noise=float(np.mean(pred==-1));sil=''
   kept=pred!=-1
   if len(set(pred[kept]))>1 and kept.sum()>2:
    sil=float(silhouette_score(zv[kept],pred[kept],sample_size=min(1000,int(kept.sum())),random_state=20261106))
   rows.append({'experiment_id':f'dbscan_ms{ms}_kdistq{int(q*100)}','model_name':'DBSCAN','dataset':'Fashion-MNIST','split_id':split,'training_size':len(z),'transductive_validation_size':len(zv),'feature_transform':'PCA32 fit on training subset only','pca_explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'distance_metric':'euclidean','eps':eps,'min_samples':ms,'k_distance_quantile':q,'training_time_seconds':fit,'evaluation_time_seconds':'','validation_ARI':adjusted_rand_score(y[10000:],pred),'validation_NMI':normalized_mutual_info_score(y[10000:],pred),'validation_silhouette_non_noise':sil,'validation_cluster_count':len(clusters),'validation_noise_fraction':noise,'peak_memory_bytes':psutil.Process().memory_info().rss,'backend':'scikit-learn CPU','status':'completed','timestamp_utc':datetime.now(timezone.utc).isoformat(),'details':json.dumps({'procedure':'DBSCAN fitted jointly on unlabeled train+validation embeddings; only validation labels used for external scores','eps_source':'training-only min_samples-th-neighbor distance quantile'})})
   print(f'DBSCAN min_samples={ms} eps={eps:.4f} q={q:.2f} clusters={len(clusters)} noise={noise:.3f} ARI={rows[-1]["validation_ARI"]:.4f}',flush=True)
 OUT.parent.mkdir(parents=True,exist_ok=True)
 with OUT.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('saved',OUT,'PCA_seconds',time.perf_counter()-t)
if __name__=='__main__':main()
