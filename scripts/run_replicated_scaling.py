"""Run resumable multi-seed Fashion-MNIST MLP/LeNet budget curves on DirectML."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
import torch_directml
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/raw/replicated_scaling_results.csv'
TEST_OUT=ROOT/'results/raw/replicated_scaling_test_results.csv'
CURVES=ROOT/'results/raw/replicated_learning_curves.csv'
CFGDIR=ROOT/'results/configurations'
CKPTDIR=ROOT/'results/checkpoints/replicated_scaling'
DEVICE=torch_directml.device()
EPOCHS=(1,3,5)

class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))
    def forward(self,x): return self.net(x)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,16,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*7*7,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.net(x)

class ResNet18(nn.Module):
    def __new__(cls):
        from torchvision.models import resnet18
        model=resnet18(weights=None);model.conv1=nn.Conv2d(1,64,7,stride=2,padding=3,bias=False);model.fc=nn.Linear(model.fc.in_features,10);return model

class CompactViT(nn.Module):
    """Custom compact ViT-style feasibility architecture; not standard ViT-Tiny."""
    def __init__(self):
        super().__init__();self.patch=nn.Conv2d(1,96,7,7);self.cls=nn.Parameter(torch.zeros(1,1,96));self.pos=nn.Parameter(torch.zeros(1,17,96));layer=nn.TransformerEncoderLayer(96,3,192,batch_first=True,norm_first=True);self.encoder=nn.TransformerEncoder(layer,3);self.head=nn.Linear(96,10)
    def forward(self,x):
        z=self.patch(x).flatten(2).transpose(1,2);z=torch.cat([self.cls.expand(len(x),-1,-1),z],1)+self.pos;return self.head(self.encoder(z)[:,0])

def append(path:Path,row:dict):
    path.parent.mkdir(parents=True,exist_ok=True); existed=path.exists() and path.stat().st_size>0
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if not existed:w.writeheader()
        w.writerow(row);f.flush()

def existing_ids(path:Path):
    if not path.exists():return set()
    with path.open(encoding='utf-8-sig',newline='') as f:return {r.get('experiment_id','') for r in csv.DictReader(f)}

def score(model,x,y,batch=256):
    model.eval(); lossfn=nn.CrossEntropyLoss(reduction='sum'); total=correct=0; loss_sum=0.
    with torch.no_grad():
        for start in range(0,len(x),batch):
            xb=x[start:start+batch].to(DEVICE);yb=y[start:start+batch].to(DEVICE);logits=model(xb)
            loss_sum+=float(lossfn(logits,yb).cpu());correct+=int((logits.argmax(1)==yb).sum().cpu());total+=len(yb)
    return loss_sum/total,correct/total

def load_arrays(subset_n):
    train=FashionMNIST(str(ROOT/'results/cache'),train=True,download=True,transform=ToTensor())
    test=FashionMNIST(str(ROOT/'results/cache'),train=False,download=True,transform=ToTensor())
    g=np.random.default_rng(20261106);idx=g.permutation(len(train))[:subset_n+2000]
    x=torch.stack([train[int(i)][0] for i in idx]);y=torch.tensor([train[int(i)][1] for i in idx],dtype=torch.long)
    xt=torch.stack([test[i][0] for i in range(len(test))]);yt=torch.tensor([test[i][1] for i in range(len(test))],dtype=torch.long)
    split_id=hashlib.sha256(idx.tobytes()).hexdigest()[:16]
    return x[:subset_n],y[:subset_n],x[subset_n:],y[subset_n:],xt,yt,split_id

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',choices=['mlp','lenet','resnet18','vit_compact'],default=['mlp','lenet','resnet18','vit_compact']);ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);ap.add_argument('--train-size',type=int,default=10000);args=ap.parse_args()
    torch.set_num_threads(8);x,y,xv,yv,xt,yt,split=load_arrays(args.train_size)
    for key in args.models:
      for seed in args.seeds:
        name={'mlp':'MLP','lenet':'CNN / LeNet','resnet18':'ResNet-18','vit_compact':'Vision Transformer (compact ViT-Tiny-style)'}[key];cls={'mlp':MLP,'lenet':LeNet,'resnet18':ResNet18,'vit_compact':CompactViT}[key]
        run=f'rep_{key}_seed{seed}_n{args.train_size}_v1';ckpt=CKPTDIR/f'{run}.pt';cfg=CFGDIR/f'{run}.json'
        config={'experiment_id':run,'model_name':name,'architecture':'custom compact 3-layer 96-wide transformer (not standard ViT-Tiny)' if key=='vit_compact' else key,'paradigm':'supervised','dataset':'Fashion-MNIST','dataset_revision':'torchvision packaged FashionMNIST train/test; fixed indexed subset v20261106','split_id':split,'seed':seed,'budget_type':'epochs','budget_points':list(EPOCHS),'train_size':args.train_size,'validation_size':2000,'test_size':len(yt),'optimizer':'Adam','learning_rate':.001,'batch_size':128,'scheduler':None,'backend':'torch-directml privateuseone:0','device':str(DEVICE),'evaluate':'validation each epoch; test once after epoch 5'}
        cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(json.dumps(config,indent=2),encoding='utf-8')
        model=cls().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.001);start_epoch=0;cum=0.
        if ckpt.exists():
            state=torch.load(ckpt,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start_epoch=int(state['epoch']);cum=float(state['cumulative_seconds'])
        known=existing_ids(OUT);known_test=existing_ids(TEST_OUT)
        for epoch in range(start_epoch+1,max(EPOCHS)+1):
            torch.manual_seed(seed+epoch);order=torch.randperm(len(x));model.train();loss_sum=0.;n_seen=0;correct=0;st=time.perf_counter()
            for ids in order.split(128):
                xb=x[ids].to(DEVICE);yb=y[ids].to(DEVICE);opt.zero_grad(set_to_none=True);logits=model(xb);loss=nn.functional.cross_entropy(logits,yb);loss.backward();opt.step()
                loss_sum+=float(loss.detach().cpu())*len(ids);n_seen+=len(ids);correct+=int((logits.detach().argmax(1)==yb).sum().cpu())
            train_s=time.perf_counter()-st;cum+=train_s;ev=time.perf_counter();vl,va=score(model,xv,yv);eval_s=time.perf_counter()-ev
            metric_id=f'{run}_epoch{epoch}'
            row={'experiment_id':metric_id,'model_name':name,'paradigm':'supervised','dataset':'Fashion-MNIST','dataset_revision':'torchvision FashionMNIST','split_id':split,'seed':seed,'budget_type':'epoch','budget_value':epoch,'training_tokens':'','optimizer_steps':epoch*((args.train_size+127)//128),'epochs':epoch,'environment_steps':'','training_time_seconds':train_s,'cumulative_training_time_seconds':cum,'evaluation_time_seconds':eval_s,'inference_throughput_samples_per_second':len(yv)/eval_s if eval_s else '','peak_memory_bytes':psutil.Process().memory_info().rss,'hardware':'AMD Radeon(TM) 860M Graphics','backend':'DirectML / PyTorch 2.4.1','evaluation_metric':'validation_accuracy','evaluation_value':va,'training_loss':loss_sum/n_seen,'validation_loss':vl,'training_accuracy':correct/n_seen,'validation_accuracy':va,'test_accuracy':'','precision':'','recall':'','f1':'','confusion_matrix':'','mean':'','standard_deviation':'','confidence_interval_95':'','experiment_status':'completed','source_type':'local_measurement','configuration_path':str(cfg.relative_to(ROOT)),'execution_log':'EXECUTION_LOG.md','timestamp_utc':datetime.now(timezone.utc).isoformat(),'details':json.dumps({'note':'validation metric at every epoch; 10k fixed training subset, 2k validation; full official test evaluated only at final checkpoint','parameter_count':sum(p.numel() for p in model.parameters())})}
            # Save a resumable checkpoint after the measured epoch, before writing its row.
            CKPTDIR.mkdir(parents=True,exist_ok=True);tmp=ckpt.with_suffix('.tmp');torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'cumulative_seconds':cum},tmp);tmp.replace(ckpt)
            if metric_id not in known:append(OUT,row);known.add(metric_id)
            append(CURVES,{'experiment_id':metric_id,'run_id':run,'epoch':epoch,'seed':seed,'training_loss':row['training_loss'],'validation_loss':vl,'training_accuracy':correct/n_seen,'validation_accuracy':va,'training_seconds':train_s,'cumulative_training_seconds':cum,'split_id':split})
            print(f'{name} seed={seed} epoch={epoch} val_acc={va:.4f} train_s={train_s:.1f}',flush=True)
        test_id=f'{run}_test_epoch5'
        if test_id not in known_test:
            loss,acc=score(model,xt,yt);pred=[];model.eval()
            with torch.no_grad():
                for i in range(0,len(xt),256):pred.extend(model(xt[i:i+256].to(DEVICE)).argmax(1).cpu().tolist())
            precision,recall,f1,_=precision_recall_fscore_support(yt.numpy(),np.asarray(pred),average='macro',zero_division=0)
            append(TEST_OUT,{'experiment_id':test_id,'model_name':name,'seed':seed,'split_id':split,'epochs':max(EPOCHS),'test_loss':loss,'test_accuracy':acc,'macro_precision':precision,'macro_recall':recall,'macro_f1':f1,'confusion_matrix':json.dumps(confusion_matrix(yt.numpy(),pred).tolist()),'test_size':len(yt),'backend':'DirectML','status':'completed'})
            print(f'{name} seed={seed} FINAL_TEST acc={acc:.4f} loss={loss:.4f}',flush=True)
    print('results',OUT,TEST_OUT,CURVES)
if __name__=='__main__':main()
