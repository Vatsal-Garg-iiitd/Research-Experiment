"""Matched multi-seed Fashion-MNIST MeanTeacher/FixMatch/MixMatch experiments.

Compact MLP and lightweight image perturbations are explicitly a controlled small-study
recipe. Checkpoints and epoch records are appended incrementally for safe resumption.
"""
import argparse,csv,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch
from backend import backend_name, get_device
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/raw/ssl_algorithms.csv'; TEST=ROOT/'results/raw/ssl_algorithms_test.csv'; CFG=ROOT/'results/configurations'; CK=ROOT/'results/checkpoints/ssl_algorithms'; DEV=get_device()
class Net(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))
 def forward(self,x):return self.net(x)
def append(p,row):
 ex=p.exists() and p.stat().st_size>0
 fields=list(row)
 if ex:
  with p.open(encoding='utf-8-sig',newline='') as f:
   reader=csv.DictReader(f); old_fields=list(reader.fieldnames or []); old_rows=list(reader)
  if not set(fields).issubset(old_fields):
   fields=old_fields+[name for name in fields if name not in old_fields]
   tmp=p.with_suffix(p.suffix+'.tmp')
   with tmp.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(old_rows)
   tmp.replace(p)
 with p.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
  if not ex:w.writeheader()
  w.writerow(row);f.flush()
def ids(p):
 if not p.exists():return set()
 with p.open(encoding='utf-8-sig',newline='') as f:return {r['experiment_id'] for r in csv.DictReader(f)}
def aug(x,level):
 # Paired weak/strong photometric perturbations used by this task adaptation.
 noise=.04 if level==0 else .18;contrast=.08 if level==0 else .35
 return ((x-.5)*(1+torch.randn((len(x),1,1,1),device=x.device)*contrast)+.5+torch.randn_like(x)*noise).clamp(0,1)
def evaluate(m,x,y):
 m.eval();pred=[]
 with torch.no_grad():
  for i in range(0,len(y),256):pred.extend(m(x[i:i+256].to(DEV)).argmax(1).cpu().tolist())
 return accuracy_score(y.numpy(),pred)
def main():
 global DEV
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);ap.add_argument('--fractions',nargs='+',type=float,default=[.01,.05,.1,.2,.4]);ap.add_argument('--max-epochs',type=int,default=30);a=ap.parse_args();torch.set_num_threads(8)
 tr=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor());te=FashionMNIST(str(ROOT/'results/cache'),train=False,download=True,transform=ToTensor())
 allidx=np.random.default_rng(20261106).permutation(len(tr))[:12000];X=torch.stack([tr[int(i)][0] for i in allidx]);Y=torch.tensor([tr[int(i)][1] for i in allidx]);x,y=X[:10000],Y[:10000];xv,yv=X[10000:],Y[10000:];xt=torch.stack([te[i][0] for i in range(len(te))]);yt=torch.tensor([te[i][1] for i in range(len(te))]);split=hashlib.sha256(allidx.tobytes()).hexdigest()[:16]
 for seed in a.seeds:
  for frac in a.fractions:
   rng=np.random.default_rng(seed+int(frac*10000));perm=rng.permutation(len(x));nl=max(20,int(frac*len(x)));li=perm[:nl];ui=perm[nl:];lx,ly=x[li],y[li];ux=x[ui];steps=int(np.ceil(len(x)/256))
   for method in ['MeanTeacher','FixMatch','MixMatch']:
    run=f'{method.lower()}_seed{seed}_frac{int(frac*100):02d}_v1';cfg=CFG/f'{run}.json';cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(json.dumps({'experiment_id':run,'method':method,'seed':seed,'fraction':frac,'labeled_n':nl,'unlabeled_n':len(ui),'split_id':split,'budgets_epochs':list(range(1,a.max_epochs+1)),'steps_per_epoch':steps,'architecture':'MLP 784-256-10','confidence_threshold':.95 if method=='FixMatch' else None,'ema_decay':.99 if method=='MeanTeacher' else None,'mixup_alpha':.75 if method=='MixMatch' else None,'temperature':.5 if method=='MixMatch' else None},indent=2),encoding='utf-8')
    ck=CK/f'{run}.pt';m=Net().to(DEV);teacher=Net().to(DEV) if method=='MeanTeacher' else None
    if teacher:teacher.load_state_dict(m.state_dict());teacher.eval()
    opt=torch.optim.Adam(m.parameters(),lr=.001);start=0;cum=0
    if ck.exists():
     s=torch.load(ck,map_location='cpu',weights_only=False);m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);start=s['epoch'];cum=s['seconds'];
     if teacher and s.get('teacher'):teacher.load_state_dict(s['teacher'])
    done=ids(OUT)
    for ep in range(start+1,a.max_epochs+1):
     np.random.seed(seed+ep);torch.manual_seed(seed+ep);order=torch.randperm(len(ui));m.train();sup_sum=unsup_sum=confidence_sum=used_sum=unlabeled_seen=0.;t=time.perf_counter()
     for it in range(steps):
      lii=torch.randint(len(lx),(128,));uii=order[(it*128)%len(ui):((it+1)*128)%len(ui)]
      if len(uii)<128:uii=order[torch.arange(128)%len(order)]
      bl,yl=lx[lii].to(DEV),ly[lii].to(DEV);bu=ux[uii].to(DEV);opt.zero_grad(set_to_none=True);sup=nn.functional.cross_entropy(m(bl),yl)
      if method=='MeanTeacher':
       with torch.no_grad():target=teacher(aug(bu,0)).softmax(-1);conf=target.max(-1).values;mask=torch.ones_like(conf,dtype=torch.bool)
       uns=nn.functional.mse_loss(m(aug(bu,1)).softmax(-1),target);loss=sup+uns
      elif method=='FixMatch':
       with torch.no_grad():p=m(aug(bu,0)).softmax(-1);conf,pseudo=p.max(-1);mask=conf.ge(.95)
       logits=m(aug(bu,1));uns=(nn.functional.cross_entropy(logits,pseudo,reduction='none')*mask.to(dtype=logits.dtype)).mean();loss=sup+uns
      else:
       with torch.no_grad():p=(m(aug(bu,0)).softmax(-1)+m(aug(bu,0)).softmax(-1))/2;p=p.pow(2);p=p/p.sum(-1,keepdim=True);conf=p.max(-1).values;mask=torch.ones_like(conf,dtype=torch.bool)
       lam=float(np.random.beta(.75,.75));u=torch.cat([bl,bu]);targets=torch.cat([nn.functional.one_hot(yl,10).float(),p]);perm2=torch.randperm(len(u)).to(DEV);mix=lam*u+(1-lam)*u[perm2];mt=lam*targets+(1-lam)*targets[perm2];logits=m(aug(mix,1));per=-(mt*logits.log_softmax(-1)).sum(-1);uns=per[128:].mean();loss=sup+per[:128].mean()+uns
      confidence_sum+=float(conf.detach().sum().cpu());used_sum+=float(mask.detach().sum().cpu());unlabeled_seen+=len(mask)
      loss.backward();opt.step();sup_sum+=float(sup.detach().cpu());unsup_sum+=float(uns.detach().cpu())
      if teacher:
       with torch.no_grad():
        for tp,sp in zip(teacher.parameters(),m.parameters()):tp.mul_(.99).add_(sp,alpha=.01)
     dt=time.perf_counter()-t;cum+=dt;va=evaluate(m,xv,yv);eid=f'{run}_epoch{ep}';CK.mkdir(parents=True,exist_ok=True);tmp=ck.with_suffix('.tmp');torch.save({'epoch':ep,'model':m.state_dict(),'teacher':teacher.state_dict() if teacher else None,'optimizer':opt.state_dict(),'seconds':cum},tmp);tmp.replace(ck)
     teacher_delta=''
     if teacher:
      with torch.no_grad():teacher_delta=float(torch.sqrt(sum((tp-sp).pow(2).sum() for tp,sp in zip(teacher.parameters(),m.parameters()))).cpu())
     row={'experiment_id':eid,'run_id':run,'model_name':method,'paradigm':'semi_supervised','dataset':'Fashion-MNIST','split_id':split,'seed':seed,'labeled_fraction':frac,'labeled_n':nl,'budget_type':'epochs','budget_value':ep,'optimizer_steps':steps*ep,'training_time_seconds':dt,'cumulative_training_time_seconds':cum,'validation_accuracy':va,'supervised_loss':sup_sum/steps,'unsupervised_loss':unsup_sum/steps,'pseudo_label_mean_confidence':confidence_sum/unlabeled_seen if unlabeled_seen else '', 'fraction_unlabeled_used':used_sum/unlabeled_seen if unlabeled_seen else '', 'ema_teacher_student_l2':teacher_delta,'backend':backend_name(DEV),'configuration_path':str(cfg.relative_to(ROOT)),'status':'completed','timestamp_utc':datetime.now(timezone.utc).isoformat(),'details':json.dumps({'semi_supervised_algorithm':method,'test_labels_hidden':True,'test_split':'official FashionMNIST test; final epoch only','task_adaptation':True,'unlabeled_examples_seen':int(unlabeled_seen),'unlabeled_examples_used':int(used_sum)})}
     if eid not in done:append(OUT,row);done.add(eid)
     print(method,seed,frac,ep,round(va,4),flush=True)
    tid=f'{run}_test_epoch{a.max_epochs}'
    if tid not in ids(TEST):append(TEST,{'experiment_id':tid,'model_name':method,'seed':seed,'split_id':split,'labeled_fraction':frac,'epochs':a.max_epochs,'test_accuracy':evaluate(m,xt,yt),'test_size':len(yt),'status':'completed','checkpoint':str(ck.relative_to(ROOT))})
if __name__=='__main__':main()
