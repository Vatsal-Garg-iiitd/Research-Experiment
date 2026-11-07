"""Actual sklearn LabelPropagation sweep on fixed transductive Fashion-MNIST splits."""
import csv,time,hashlib,json
from pathlib import Path
import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score,adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/raw/label_propagation_scaling.csv'; N=2500
def append(row):
 exists=OUT.exists() and OUT.stat().st_size>0
 with OUT.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if not exists:w.writeheader()
  w.writerow(row);f.flush()
ds=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor())
for seed in [11,23,37]:
 ids=np.random.default_rng(seed).permutation(len(ds))[:N];X=np.stack([ds[int(i)][0].numpy().reshape(-1) for i in ids]).astype('float32');y=np.array([ds[int(i)][1] for i in ids]);n=2000
 z=PCA(n_components=32,random_state=seed).fit_transform(StandardScaler().fit_transform(X));rng=np.random.default_rng(seed+500);lab=rng.choice(n,n//10,replace=False);yl=np.full(N,-1,dtype=int);yl[lab]=y[lab];split=hashlib.sha256(ids.tobytes()).hexdigest()[:16]
 # Keep graph/split/seed fixed and vary only the iteration budget. Test labels are never supplied.
 graph=kneighbors_graph(z,n_neighbors=10,mode='connectivity',include_self=True);graph=graph.maximum(graph.T);components,labels=connected_components(graph,directed=False)
 for budget in [100,250,500,1000,2000]:
  model=LabelPropagation(kernel='knn',n_neighbors=10,max_iter=budget,tol=1e-3);t=time.perf_counter();model.fit(z,yl);sec=time.perf_counter()-t;pred=model.transduction_[n:];mask=pred>=0;acc=accuracy_score(y[n:][mask],pred[mask]) if mask.any() else ''
  append({'experiment_id':f'labelprop_seed{seed}_iter{budget}_tol1e-3_v2','model_name':'Label Propagation','dataset':'Fashion-MNIST transductive 2500','split_id':split,'seed':seed,'labeled_fraction':.1,'budget_type':'max_propagation_iterations','budget_value':budget,'actual_iterations':model.n_iter_,'training_time_seconds':sec,'evaluation_metric':'transductive_test_accuracy','evaluation_value':acc,'validation_ARI':adjusted_rand_score(y[n:],pred),'n_neighbors':10,'pca_dimensions':32,'backend':'CPU sklearn','status':'completed','details':json.dumps({'labeled_points':len(lab),'unlabeled_training_points':n-len(lab),'test_labels_hidden_during_fit':True,'converged':model.n_iter_<budget,'tolerance':1e-3,'graph_components':components,'isolated_nodes':int(np.sum(np.asarray(graph.sum(axis=1)).ravel()==0)),'graph_symmetric_after_union':True,'graph_normalization':'sklearn LabelPropagation internal normalization'})})
  print(seed,budget,model.n_iter_,acc,flush=True)
