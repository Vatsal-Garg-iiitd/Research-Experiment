"""Matched compute semi-supervised pseudo-label vs labeled-only baseline curves."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,psutil,torch
from torch import nn
import torch_directml
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score,precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/raw/pseudolabel_baseline_sweep.csv';TEST=ROOT/'results/raw/pseudolabel_test_results.csv';CURVE=ROOT/'results/raw/pseudolabel_learning_curves.csv';CFG=ROOT/'results/configurations';CK=ROOT/'results/checkpoints/pseudolabel'
FRACTIONS=[.01,.05,.10,.20,.40];MAX_EPOCH=5;DEVICE=torch_directml.device()
class MLP(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))
 def forward(self,x):return self.net(x)
def append(path,row):
 path.parent.mkdir(parents=True,exist_ok=True);ex=path.exists() and path.stat().st_size>0
 with path.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if not ex:w.writeheader()
  w.writerow(row);f.flush()
def readids(p):
 if not p.exists():return set()
 with p.open(encoding='utf-8-sig',newline='') as f:return {r['experiment_id'] for r in csv.DictReader(f)}
def load(n):
 ds=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor());te=FashionMNIST(str(ROOT/'results/cache'),train=False,download=True,transform=ToTensor())
 idx=np.random.default_rng(20261106).permutation(len(ds))[:n+2000];x=torch.stack([ds[int(i)][0] for i in idx]);y=torch.tensor([ds[int(i)][1] for i in idx]);xt=torch.stack([te[i][0] for i in range(len(te))]);yt=torch.tensor([te[i][1] for i in range(len(te))]);return x[:n],y[:n],x[n:],y[n:],xt,yt,hashlib.sha256(idx.tobytes()).hexdigest()[:16]
def evaluate(m,x,y):
 m.eval();s=ok=0
 with torch.no_grad():
  for i in range(0,len(x),256):
   xb=x[i:i+256].to(DEVICE);yb=y[i:i+256].to(DEVICE);z=m(xb);s+=float(nn.functional.cross_entropy(z,yb,reduction='sum').cpu());ok+=int((z.argmax(1)==yb).sum().cpu())
 return s/len(y),ok/len(y)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);ap.add_argument('--train-size',type=int,default=10000);args=ap.parse_args();torch.set_num_threads(8);x,y,xv,yv,xt,yt,split=load(args.train_size)
 for seed in args.seeds:
  for frac in FRACTIONS:
   rng=np.random.default_rng(seed+int(frac*10000));perm=rng.permutation(len(x));nl=max(10,int(len(x)*frac));li=perm[:nl];ui=perm[nl:];lx,ly=x[li],y[li];ux=x[ui];steps=int(np.ceil(len(ui)/128))
   for method in ('labeled_only','pseudo_label'):
    run=f'{method}_frac{int(frac*100):02d}_seed{seed}_v1';cfg=CFG/f'{run}.json';cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(json.dumps({'experiment_id':run,'model':'MLP','method':method,'dataset':'Fashion-MNIST','split_id':split,'seed':seed,'labeled_fraction':frac,'labeled_n':nl,'unlabeled_n':len(ui),'epochs':list(range(1,MAX_EPOCH+1)),'optimizer':'Adam','learning_rate':.001,'batch_size':128,'matched_optimizer_steps_per_epoch':steps,'pseudo_threshold':.8 if method=='pseudo_label' else None,'unsupervised_loss_weight':1.0 if method=='pseudo_label' else 0,'test_evaluation':'final epoch only'},indent=2),encoding='utf-8')
    ck=CK/f'{run}.pt';m=MLP().to(DEVICE);opt=torch.optim.Adam(m.parameters(),lr=.001);start=0;cum=0.
    if ck.exists():
     state=torch.load(ck,map_location='cpu',weights_only=False);m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['epoch'];cum=state['cumulative_seconds']
    exist=readids(OUT)
    for epoch in range(start+1,MAX_EPOCH+1):
     rs=np.random.default_rng(seed+epoch+int(frac*1e6));lp=rs.permutation(nl);up=rs.permutation(len(ui));lcur=ucur=0;loss_total=0.;sup_total=unsup_total=0.;accepted=seen=0;actual_steps=0;m.train();t0=time.perf_counter()
     for _ in range(steps):
      if lcur+128>nl:lp=rs.permutation(nl);lcur=0
      ids=lp[lcur:min(lcur+128,nl)];lcur+=len(ids)
      if len(ids)<128:ids=rs.choice(nl,size=128,replace=True)
      bx=lx[ids].to(DEVICE);by=ly[ids].to(DEVICE);opt.zero_grad(set_to_none=True);logits=m(bx);sup=nn.functional.cross_entropy(logits,by);unloss=logits.sum()*0
      if method=='pseudo_label':
       uids=up[ucur:min(ucur+128,len(ui))];ucur+=len(uids)
       if len(uids)==0:up=rs.permutation(len(ui));ucur=0;uids=up[ucur:min(ucur+128,len(ui))];ucur+=len(uids)
       ub=ux[uids].to(DEVICE)
       with torch.no_grad():probs=m(ub).softmax(1);conf,pseudo=probs.max(1);mask=conf>=.8;accepted+=int(mask.sum().cpu());seen+=len(mask)
       if mask.any():unlogits=m(ub);unloss=nn.functional.cross_entropy(unlogits[mask],pseudo[mask])
      loss=sup+unloss;loss.backward();opt.step();actual_steps+=1;loss_total+=float(loss.detach().cpu());sup_total+=float(sup.detach().cpu());unsup_total+=float(unloss.detach().cpu())
     dt=time.perf_counter()-t0;cum+=dt;vl,va=evaluate(m,xv,yv);eid=f'{run}_epoch{epoch}';row={'experiment_id':eid,'run_id':run,'model_name':'MLP','paradigm':'semi_supervised' if method=='pseudo_label' else 'supervised_labeled_baseline','method':method,'dataset':'Fashion-MNIST','split_id':split,'seed':seed,'labeled_fraction':frac,'labeled_n':nl,'unlabeled_n':len(ui),'budget_type':'epochs','budget_value':epoch,'optimizer_steps':actual_steps,'training_time_seconds':dt,'cumulative_training_time_seconds':cum,'evaluation_time_seconds':'','peak_memory_bytes':psutil.Process().memory_info().rss,'backend':'DirectML; AMD Radeon(TM) 860M; PyTorch 2.4.1','evaluation_metric':'validation_accuracy','evaluation_value':va,'validation_loss':vl,'mean_supervised_training_loss':sup_total/actual_steps,'mean_unsupervised_loss':unsup_total/actual_steps,'pseudo_labels_accepted':accepted,'pseudo_label_acceptance_fraction':accepted/seen if seen else 0,'experiment_status':'completed','source_type':'local_measurement','configuration_path':str(cfg.relative_to(ROOT)),'timestamp_utc':datetime.now(timezone.utc).isoformat(),'details':json.dumps({'matched_optimizer_steps':True,'labeled_index_seed':seed+int(frac*10000),'validation_split_fixed':'yes'})}
     CK.mkdir(parents=True,exist_ok=True);tmp=ck.with_suffix('.tmp');torch.save({'epoch':epoch,'model':m.state_dict(),'optimizer':opt.state_dict(),'cumulative_seconds':cum},tmp);tmp.replace(ck)
     if eid not in exist:append(OUT,row);exist.add(eid)
     append(CURVE,{'experiment_id':eid,'run_id':run,'method':method,'seed':seed,'labeled_fraction':frac,'epoch':epoch,'validation_accuracy':va,'validation_loss':vl,'mean_supervised_loss':sup_total/actual_steps,'mean_unsupervised_loss':unsup_total/actual_steps,'accepted_pseudo_labels':accepted,'pseudo_label_acceptance_fraction':accepted/seen if seen else 0,'cumulative_training_time_seconds':cum})
     print(f'{method} seed={seed} frac={frac:.2f} epoch={epoch} val={va:.4f} pseudo_accept={accepted}/{seen}',flush=True)
    tid=f'{run}_test_epoch{MAX_EPOCH}';teids=readids(TEST)
    if tid not in teids:
     loss,acc=evaluate(m,xt,yt);pred=[];m.eval()
     with torch.no_grad():
      for i in range(0,len(xt),256):pred.extend(m(xt[i:i+256].to(DEVICE)).argmax(1).cpu().tolist())
     p,r,f,_=precision_recall_fscore_support(yt.numpy(),np.asarray(pred),average='macro',zero_division=0)
     append(TEST,{'experiment_id':tid,'model_name':'MLP','method':method,'seed':seed,'labeled_fraction':frac,'labeled_n':nl,'epochs':MAX_EPOCH,'test_loss':loss,'test_accuracy':acc,'macro_precision':p,'macro_recall':r,'macro_f1':f,'test_size':len(yt),'status':'completed'})
 print('saved',OUT,TEST,CURVE)
if __name__=='__main__':main()
