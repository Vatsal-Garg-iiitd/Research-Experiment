"""Three-seed AE/VAE reconstruction and linear-probe budget curves."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,psutil,torch
from torch import nn
import torch_directml
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/raw/autoencoder_scaling.csv';TEST=ROOT/'results/raw/autoencoder_test_results.csv';CFG=ROOT/'results/configurations';CK=ROOT/'results/checkpoints/autoencoders';DEV=torch_directml.device();MAXE=5
class AE(nn.Module):
 def __init__(self,variational=False):
  super().__init__();self.v=variational;self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU());self.mu=nn.Linear(256,32);self.lv=nn.Linear(256,32) if variational else None;self.decoder=nn.Sequential(nn.Linear(32,256),nn.ReLU(),nn.Linear(256,784),nn.Sigmoid())
 def encode(self,x):
  h=self.encoder(x);return self.mu(h)
 def forward(self,x):
  h=self.encoder(x);mu=self.mu(h);lv=self.lv(h).clamp(-10,10) if self.v else None;z=mu+torch.randn_like(mu)*torch.exp(.5*lv) if self.v and self.training else mu;return self.decoder(z),mu,lv
def append(path,row):
 path.parent.mkdir(parents=True,exist_ok=True);ex=path.exists() and path.stat().st_size>0
 with path.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if not ex:w.writeheader()
  w.writerow(row);f.flush()
def readids(path):
 if not path.exists():return set()
 with path.open(encoding='utf-8-sig',newline='') as f:return {r['experiment_id'] for r in csv.DictReader(f)}
def arrays(n):
 tr=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor());te=FashionMNIST(str(ROOT/'results/cache'),train=False,download=True,transform=ToTensor());idx=np.random.default_rng(20261106).permutation(len(tr))[:n+2000];x=torch.stack([tr[int(i)][0] for i in idx]);y=torch.tensor([tr[int(i)][1] for i in idx]);xt=torch.stack([te[i][0] for i in range(len(te))]);yt=torch.tensor([te[i][1] for i in range(len(te))]);return x[:n],y[:n],x[n:],y[n:],xt,yt,hashlib.sha256(idx.tobytes()).hexdigest()[:16]
def encode(m,x):
 m.eval();out=[]
 with torch.no_grad():
  for i in range(0,len(x),256):out.append(m.encode(x[i:i+256].to(DEV)).cpu().numpy())
 return np.concatenate(out)
def recon(m,x):
 m.eval();ss=0;n=0
 with torch.no_grad():
  for i in range(0,len(x),256):
   xb=x[i:i+256].to(DEV);out,_,_=m(xb);ss+=float(((out-xb.flatten(1))**2).sum().cpu());n+=xb.shape[0]*784
 return ss/n
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);ap.add_argument('--train-size',type=int,default=10000);a=ap.parse_args();torch.set_num_threads(8);x,y,xv,yv,xt,yt,split=arrays(a.train_size)
 for name,var in [('Autoencoder',False),('Variational Autoencoder',True)]:
  for seed in a.seeds:
   run=f'{"vae" if var else "ae"}_seed{seed}_n{a.train_size}_v2';cfg=CFG/f'{run}.json';cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(json.dumps({'experiment_id':run,'model':name,'dataset':'Fashion-MNIST','split_id':split,'seed':seed,'train_size':len(y),'validation_size':len(yv),'budget_epochs':list(range(1,MAXE+1)),'latent_dim':32,'optimizer':'Adam','learning_rate':.001,'batch_size':128,'objective':'reconstruction MSE + KL/784' if var else 'reconstruction MSE','backend':'DirectML','downstream':'per-checkpoint standardized logistic-regression linear probe fit on train latents, scored on validation; test probe at final checkpoint','probe_max_iter':500,'probe_tolerance':.001},indent=2),encoding='utf-8')
   ck=CK/f'{run}.pt';m=AE(var).to(DEV);opt=torch.optim.Adam(m.parameters(),lr=.001);start=0;cum=0.
   if ck.exists():
    s=torch.load(ck,map_location='cpu',weights_only=False);m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);start=int(s['epoch']);cum=float(s['cumulative_seconds'])
   done=readids(OUT)
   for ep in range(start+1,MAXE+1):
    torch.manual_seed(seed+ep);order=torch.randperm(len(x));m.train();tot=0.;samples=0;t=time.perf_counter()
    for ids in order.split(128):
     xb=x[ids].to(DEV);opt.zero_grad(set_to_none=True);out,mu,lv=m(xb);re=((out-xb.flatten(1))**2).mean();kl=.5*(-1-lv+mu.pow(2)+lv.exp()).sum(1).mean()/784 if var else torch.zeros((),device=DEV);loss=re+kl;loss.backward();opt.step();tot+=float(loss.detach().cpu())*len(ids);samples+=len(ids)
    dt=time.perf_counter()-t;cum+=dt;val_recon=recon(m,xv);et=time.perf_counter();zt=encode(m,x);zv=encode(m,xv);probe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,tol=1e-3,random_state=seed)).fit(zt,y.numpy());val_acc=accuracy_score(yv.numpy(),probe.predict(zv));probe_s=time.perf_counter()-et;eid=f'{run}_epoch{ep}'
    CK.mkdir(parents=True,exist_ok=True);tmp=ck.with_suffix('.tmp');torch.save({'epoch':ep,'model':m.state_dict(),'optimizer':opt.state_dict(),'cumulative_seconds':cum},tmp);tmp.replace(ck)
    if eid not in done:append(OUT,{'experiment_id':eid,'run_id':run,'model_name':name,'paradigm':'unsupervised_representation','dataset':'Fashion-MNIST','split_id':split,'seed':seed,'budget_type':'epochs','budget_value':ep,'training_time_seconds':dt,'cumulative_training_time_seconds':cum,'evaluation_time_seconds':probe_s,'preprocessing_time_seconds':'','peak_memory_bytes':psutil.Process().memory_info().rss,'backend':'DirectML; AMD Radeon(TM) 860M; PyTorch 2.4.1','train_objective_loss':tot/samples,'validation_reconstruction_mse':val_recon,'downstream_validation_accuracy':val_acc,'experiment_status':'completed','source_type':'local_measurement','configuration_path':str(cfg.relative_to(ROOT)),'timestamp_utc':datetime.now(timezone.utc).isoformat(),'details':json.dumps({'linear_probe':'StandardScaler + sklearn LogisticRegression(max_iter=500,tol=0.001)','probe_n_iter':int(probe[-1].n_iter_.max()),'probe_train_n':len(y),'probe_val_n':len(yv),'latent_dimension':32,'VAE_KL_scaled_by_784':var})})
    print(f'{name} seed={seed} epoch={ep} val_mse={val_recon:.5f} probe_val_acc={val_acc:.4f}',flush=True)
   tid=f'{run}_test_epoch{MAXE}'
   if tid not in readids(TEST):
    te_recon=recon(m,xt);t=time.perf_counter();zt=encode(m,x);ztest=encode(m,xt);probe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,tol=1e-3,random_state=seed)).fit(zt,y.numpy());test_acc=accuracy_score(yt.numpy(),probe.predict(ztest));eval_s=time.perf_counter()-t
    append(TEST,{'experiment_id':tid,'model_name':name,'seed':seed,'split_id':split,'epochs':MAXE,'test_reconstruction_mse':te_recon,'test_linear_probe_accuracy':test_acc,'test_n':len(yt),'probe_train_n':len(y),'probe_n_iter':int(probe[-1].n_iter_.max()),'probe_evaluation_seconds':eval_s,'status':'completed'})
 print('saved',OUT,TEST)
if __name__=='__main__':main()
