"""Same-input CPU vs DirectML MLP inference/training smoke benchmark."""
from __future__ import annotations
import csv,time
from pathlib import Path
import numpy as np,psutil,torch,torch_directml
from torch import nn

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/hardware/backend_benchmark.csv'
class Net(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,128),nn.ReLU(),nn.Linear(128,10))
 def forward(self,x):return self.net(x)
def benchmark(label,device,state,x,y):
 p=psutil.Process(); before=p.memory_info().rss; t=time.perf_counter(); m=Net().to(device);m.load_state_dict(state);load=time.perf_counter()-t;m.eval();xx=x.to(device);yy=y.to(device)
 with torch.no_grad():
  for _ in range(10): _=m(xx)
 lat=[];t0=time.perf_counter()
 with torch.no_grad():
  for _ in range(50):
   t=time.perf_counter();z=m(xx);_=float(z[0,0].cpu());lat.append(time.perf_counter()-t)
 infer=time.perf_counter()-t0
 m.train(); opt=torch.optim.SGD(m.parameters(),lr=.01); t=time.perf_counter()
 for _ in range(30):
  opt.zero_grad();loss=nn.functional.cross_entropy(m(xx),yy);loss.backward();opt.step();_=float(loss.detach().cpu())
 train=time.perf_counter()-t
 return {'backend':label,'model':'MLP 784-128-10','batch_size':len(x),'inference_batches':50,'inference_seconds':infer,'inference_samples_per_second':50*len(x)/infer,'median_batch_latency_ms':1000*float(np.median(lat)),'model_load_seconds':load,'train_steps':30,'training_seconds':train,'training_steps_per_second':30/train,'rss_delta_bytes':p.memory_info().rss-before,'peak_rss_bytes':getattr(p.memory_info(),'peak_wset',p.memory_info().rss),'device_utilization':'GPU engine counters available globally; per-run utilization not isolated','device_memory':'not available from tested DirectML API'}
def main():
 torch.manual_seed(11);x=torch.randn(128,1,28,28);y=torch.randint(0,10,(128,));base=Net();state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
 rows=[benchmark('CPU','cpu',state,x,y)]
 dml=torch_directml.device();rows.append(benchmark(f'AMD Radeon 860M DirectML ({dml})',dml,state,x,y))
 cols=list(rows[0]);OUT.parent.mkdir(parents=True,exist_ok=True)
 with OUT.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 print(rows)
if __name__=='__main__':main()
