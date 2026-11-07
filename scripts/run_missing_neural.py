"""One-budget DirectML feasibility runs for neural models not covered by pilots.

Uses a fixed Fashion-MNIST 2,500 image subset and appends immutable run records.
This does not claim the requested multi-seed/full-budget study is complete.
"""
from __future__ import annotations
import argparse, csv, json, time, hashlib, copy
from pathlib import Path
import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch_directml
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/raw/full_suite_results.csv'; SEED=11
D=torch_directml.device()

class MLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))
    def forward(self,x): return self.net(x)
class LeNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,16,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*7*7,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.net(x)
class TinyViT(nn.Module):
    def __init__(self):
        super().__init__(); self.patch=nn.Conv2d(1,96,7,7); self.cls=nn.Parameter(torch.zeros(1,1,96)); self.pos=nn.Parameter(torch.zeros(1,17,96)); layer=nn.TransformerEncoderLayer(96,3,192,batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,3); self.head=nn.Linear(96,10)
    def forward(self,x):
        z=self.patch(x).flatten(2).transpose(1,2); z=torch.cat([self.cls.expand(len(x),-1,-1),z],1)+self.pos; return self.head(self.enc(z)[:,0])
class AE(nn.Module):
    def __init__(self,variational=False):
        super().__init__(); self.v=variational; self.enc=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU()); self.mu=nn.Linear(256,24); self.lv=nn.Linear(256,24) if variational else None; self.dec=nn.Sequential(nn.Linear(24,256),nn.ReLU(),nn.Linear(256,784),nn.Sigmoid())
    def forward(self,x):
        h=self.enc(x); mu=self.mu(h)
        if self.v:
            lv=self.lv(h).clamp(-10,10); z=mu+torch.randn_like(mu)*torch.exp(.5*lv) if self.training else mu; return self.dec(z),mu,lv
        return self.dec(mu),mu,None
class ResNet18(nn.Module):
    def __new__(cls):
        from torchvision.models import resnet18
        m=resnet18(weights=None); m.conv1=nn.Conv2d(1,64,7,stride=2,padding=3,bias=False); m.fc=nn.Linear(m.fc.in_features,10); return m

def save(row):
    OUT.parent.mkdir(parents=True,exist_ok=True); exists=OUT.exists() and OUT.stat().st_size>0
    with OUT.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if not exists: w.writeheader()
        w.writerow(row); f.flush()

def eval_acc(m,x,y):
    m.eval(); good=0
    with torch.no_grad():
        for i in range(0,len(x),128): good+=int((m(x[i:i+128].to(D)).argmax(1).cpu()==y[i:i+128]).sum())
    return good/len(x)
def aug(x,strong=False):
    noise=.25 if strong else .08
    return (x+torch.randn_like(x)*noise).clamp(0,1)
def train_ce(m,x,y,epochs=1):
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss(); m.train(); steps=0
    for ep in range(epochs):
        order=torch.randperm(len(x))
        for ids in order.split(128):
            bx=x[ids].to(D); by=y[ids].to(D); opt.zero_grad(); loss=lossfn(m(bx),by); loss.backward(); opt.step(); steps+=1
    return steps
def record(name,para,metric,value,train_s,eval_s,steps,extra):
    save(dict(run_id='full_'+name.lower().replace(' ','_').replace('/','_')+'_seed11_v1',paradigm=para,model=name,dataset='Fashion-MNIST-subset-2500',split_hash=SPLIT,seed=SEED,budget_unit='optimizer_steps',budget_value=steps,train_seconds=train_s,cumulative_train_seconds=train_s,eval_seconds=eval_s,metric_name=metric,metric_value=value,train_steps=steps,parameter_count='',peak_ram_bytes=psutil.Process().memory_info().rss,peak_vram_bytes='',hardware=f'DirectML; AMD Radeon(TM) 860M Graphics; torch-directml 0.2.5.dev240914; PyTorch {torch.__version__}',status='completed',details=json.dumps(extra)))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--only-mixmatch',action='store_true'); args=parser.parse_args()
    global SPLIT
    torch.manual_seed(SEED); np.random.seed(SEED)
    ds=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor()); ids=np.random.default_rng(SEED).permutation(len(ds))[:2500]
    X=torch.stack([ds[i][0] for i in ids]); Y=torch.tensor([ds[i][1] for i in ids],dtype=torch.long); SPLIT=hashlib.sha256(ids.tobytes()).hexdigest()[:16]
    n=int(.8*len(X)); xt,yt=X[:n],Y[:n]; xv,yv=X[n:],Y[n:]
    # Supervised neural models; one actual epoch, one seed.
    for name,ctor in [('MLP',MLP),('CNN / LeNet',LeNet),('ResNet-18',ResNet18),('Vision Transformer (ViT-Tiny)',TinyViT)]:
        torch.manual_seed(SEED); m=ctor().to(D); t=time.perf_counter(); steps=train_ce(m,xt,yt,1); ts=time.perf_counter()-t; t=time.perf_counter(); acc=eval_acc(m,xv,yv); es=time.perf_counter()-t
        record(name,'supervised','heldout_accuracy',acc,ts,es,steps,{'train_n':n,'test_n':len(xv),'epochs':1,'directml_device':str(D),'single_seed':True})
        del m
    # Reconstruction models: one epoch on the same images; report held-out reconstruction MSE.
    for name,var in [('Autoencoder',False),('Variational Autoencoder',True)]:
        torch.manual_seed(SEED); m=AE(var).to(D); opt=torch.optim.Adam(m.parameters(),lr=1e-3); t=time.perf_counter(); m.train(); steps=0
        for ids0 in torch.randperm(n).split(128):
            bx=xt[ids0].to(D); opt.zero_grad(); out,mu,lv=m(bx); mse=((out-bx.flatten(1))**2).mean()
            loss=mse + (.5*(-1-lv+mu.pow(2)+lv.exp()).sum(1).mean()/784 if var else 0); loss.backward(); opt.step(); steps+=1
        ts=time.perf_counter()-t; t=time.perf_counter(); m.eval()
        with torch.no_grad():
            vals=[]
            for i in range(0,len(xv),128):
                bx=xv[i:i+128].to(D); out,_,_=m(bx); vals.append(float(((out-bx.flatten(1))**2).mean().cpu()))
            score=float(np.mean(vals))
        es=time.perf_counter()-t; record(name,'unsupervised','heldout_reconstruction_mse',score,ts,es,steps,{'train_n':n,'test_n':len(xv),'epochs':1,'objective':'MSE + KL' if var else 'MSE','single_seed':True}); del m
    # Semi-supervised methods: fixed 10% labels; warm-start one epoch, then one SSL epoch.
    labeled=np.random.default_rng(SEED).choice(n,size=n//10,replace=False); islab=torch.zeros(n,dtype=torch.bool); islab[labeled]=True; lx,ly=xt[islab],yt[islab]; ux=xt[~islab]
    for name in (['MixMatch'] if args.only_mixmatch else ['Pseudo-Labeling','Mean Teacher','FixMatch','MixMatch']):
        torch.manual_seed(SEED); m=MLP().to(D); t=time.perf_counter(); steps=train_ce(m,lx,ly,1); opt=torch.optim.Adam(m.parameters(),lr=5e-4); m.train(); teacher=copy.deepcopy(m).eval() if name=='Mean Teacher' else None
        # One additional pass over unlabeled data with method-specific targets/loss.
        for ids0 in torch.randperm(len(ux)).split(128):
            bx=ux[ids0].to(D); opt.zero_grad()
            if name=='Mean Teacher':
                with torch.no_grad(): target=teacher(aug(bx)).softmax(1)
                logits=m(aug(bx,True)); loss=nn.functional.mse_loss(logits.softmax(1),target)
                # EMA update is the defining teacher operation.
                for tp,sp in zip(teacher.parameters(),m.parameters()): tp.data.mul_(.99).add_(sp.data,alpha=.01)
            elif name=='Pseudo-Labeling':
                with torch.no_grad(): probs=m(bx).softmax(1); conf,pseudo=probs.max(1); mask=conf>=.80
                logits=m(bx); loss=nn.functional.cross_entropy(logits[mask],pseudo[mask]) if mask.any() else logits.sum()*0
            elif name=='FixMatch':
                with torch.no_grad(): probs=m(aug(bx)).softmax(1); conf,pseudo=probs.max(1); mask=conf>=.8
                logits=m(aug(bx,True)); loss=nn.functional.cross_entropy(logits[mask],pseudo[mask]) if mask.any() else logits.sum()*0
            else:
                # MixMatch core: guessed/sharpened labels for U, concatenate L and U,
                # then mix inputs and soft targets with the same sampled permutation.
                li=torch.randint(0,len(lx),(len(bx),)); bl=lx[li].to(D); yl=ly[li].to(D)
                with torch.no_grad():
                    probs=(m(aug(bx)).softmax(1)+m(aug(bx)).softmax(1))/2
                    target=probs.pow(2); target=target/target.sum(1,keepdim=True)
                    onehot=nn.functional.one_hot(yl.cpu(),10).float().to(D); tx=torch.cat([onehot,target],0)
                xx=torch.cat([bl,bx],0); perm=torch.randperm(len(xx),device=D); lam=float(np.random.beta(.75,.75)); mx=lam*xx+(1-lam)*xx[perm]; mt=lam*tx+(1-lam)*tx[perm]
                logits=m(aug(mx,True)); loss=(-(mt*logits.log_softmax(1)).sum(1)).mean()
            loss.backward(); opt.step(); steps+=1
        ts=time.perf_counter()-t; t=time.perf_counter(); score=eval_acc(m,xv,yv); es=time.perf_counter()-t
        record(name,'semi_supervised','heldout_accuracy',score,ts,es,steps,{'run_revision':'mixmatch_core_v2' if args.only_mixmatch else 'v1','labeled_fraction':.10,'labeled_n':len(lx),'unlabeled_n':len(ux),'warmup_epochs':1,'ssl_epochs':1,'confidence_threshold':.8 if name in ['Pseudo-Labeling','FixMatch'] else None,'note':'compact method-specific feasibility implementation; one seed; not full benchmark recipe'})
        del m,teacher
    print('saved',OUT,'device',D)
if __name__=='__main__': main()
