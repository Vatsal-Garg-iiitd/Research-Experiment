"""Compact CPU RL evaluations for the four RL algorithms absent from the old pilot."""
from __future__ import annotations
import csv,json,time,random
from collections import deque
from pathlib import Path
import numpy as np
import psutil,torch,gymnasium as gym
from torch import nn
from stable_baselines3 import DQN,A2C,PPO,SAC

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/raw/full_suite_results.csv'; SEED=11
def save(r):
    exists=OUT.exists() and OUT.stat().st_size>0
    with OUT.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(r));
        if not exists:w.writeheader()
        w.writerow(r);f.flush()
def eval_model(model,envid,episodes=5):
    env=gym.make(envid); vals=[]
    for i in range(episodes):
        obs,_=env.reset(seed=SEED+100+i); done=False; score=0
        while not done:
            action=model.predict(obs,deterministic=True)[0]; obs,r,term,trunc,_=env.step(action); score+=r;done=term or trunc
        vals.append(score)
    env.close();return vals
def row(name,envid,steps,train_s,eval_s,vals,updates,impl):
    save(dict(run_id='full_'+name.lower().replace(' ','_')+'_seed11_v1',paradigm='reinforcement_learning',model=name,dataset=envid,split_hash=f'{envid}_seed11_eval5',seed=SEED,budget_unit='environment_steps',budget_value=steps,train_seconds=train_s,cumulative_train_seconds=train_s,eval_seconds=eval_s,metric_name='mean_return_5_episodes',metric_value=float(np.mean(vals)),train_steps=updates,parameter_count='',peak_ram_bytes=psutil.Process().memory_info().rss,peak_vram_bytes='',hardware=f'CPU; PyTorch {torch.__version__}',status='completed',details=json.dumps({'environment':envid,'evaluation_returns':vals,'eval_episodes':5,'algorithm':impl,'gradient_updates':updates,'single_seed':True})))
def double_dqn(steps=5000):
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);env=gym.make('CartPole-v1');obs,_=env.reset(seed=SEED);q=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,2));target=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,2));target.load_state_dict(q.state_dict());opt=torch.optim.Adam(q.parameters(),lr=1e-3);buf=deque(maxlen=10000);updates=0;t=time.perf_counter()
    for step in range(1,steps+1):
        eps=max(.05,1-step/3000); act=env.action_space.sample() if random.random()<eps else int(q(torch.tensor(obs,dtype=torch.float32)).argmax())
        nxt,r,term,trunc,_=env.step(act); done=term or trunc;buf.append((obs,act,r,nxt,done));obs=nxt
        if done: obs,_=env.reset()
        if step>=500 and step%4==0:
            batch=random.sample(buf,32); s,a,re,ns,d=zip(*batch); s=torch.tensor(np.asarray(s),dtype=torch.float32); a=torch.tensor(a); re=torch.tensor(re,dtype=torch.float32); ns=torch.tensor(np.asarray(ns),dtype=torch.float32); d=torch.tensor(d,dtype=torch.float32)
            pred=q(s).gather(1,a[:,None]).squeeze(1)
            with torch.no_grad(): na=q(ns).argmax(1); val=re+.99*(1-d)*target(ns).gather(1,na[:,None]).squeeze(1)
            loss=nn.functional.smooth_l1_loss(pred,val);opt.zero_grad();loss.backward();opt.step();updates+=1
        if step%500==0: target.load_state_dict(q.state_dict())
    train_s=time.perf_counter()-t;env.close()
    class QPolicy:
        def predict(self,o,deterministic=True):return np.asarray(int(q(torch.tensor(o,dtype=torch.float32)).argmax())),None
    t=time.perf_counter(); vals=eval_model(QPolicy(),'CartPole-v1'); ev=time.perf_counter()-t
    row('Double DQN','CartPole-v1',steps,train_s,ev,vals,updates,'custom Double-DQN target selection (online argmax, target evaluation)')
def main():
    specs=[('DQN',DQN,'CartPole-v1',dict(learning_starts=500,train_freq=4,gradient_steps=1,buffer_size=10000,batch_size=32)),('A2C',A2C,'CartPole-v1',dict(n_steps=128)),('PPO',PPO,'CartPole-v1',dict(n_steps=256,batch_size=64,n_epochs=4)),('SAC',SAC,'Pendulum-v1',dict(learning_starts=500,buffer_size=10000,batch_size=64))]
    for name,cls,env_id,kw in specs:
        torch.set_num_threads(4);env=gym.make(env_id);env.reset(seed=SEED);env.action_space.seed(SEED);model=cls('MlpPolicy',env,seed=SEED,device='cpu',verbose=0,**kw);t=time.perf_counter();model.learn(total_timesteps=5000);train_s=time.perf_counter()-t;updates=int(getattr(model,'_n_updates',0));env.close();t=time.perf_counter();vals=eval_model(model,env_id);ev=time.perf_counter()-t;row(name,env_id,5000,train_s,ev,vals,updates,'Stable-Baselines3 '+cls.__name__)
    double_dqn()
    print('saved',OUT)
if __name__=='__main__':main()
