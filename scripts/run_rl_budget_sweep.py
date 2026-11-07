"""Three-seed RL checkpoint curves with separate 20-episode evaluation envs."""
from __future__ import annotations
import argparse,csv,json,time,random
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
import gymnasium as gym,numpy as np,psutil,torch
from stable_baselines3 import DQN,A2C,PPO,SAC

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/raw/rl_scaling_results.csv';CFG=ROOT/'results/configurations';CK=ROOT/'results/checkpoints/rl_scaling';BUDGETS=[5000,10000,25000]
SPECS={'DQN':(DQN,'CartPole-v1',{'learning_starts':500,'train_freq':4,'gradient_steps':1,'buffer_size':50000,'batch_size':32}), 'A2C':(A2C,'CartPole-v1',{'n_steps':128}), 'PPO':(PPO,'CartPole-v1',{'n_steps':256,'batch_size':64,'n_epochs':4}), 'SAC':(SAC,'Pendulum-v1',{'learning_starts':500,'buffer_size':50000,'batch_size':64})}
def append(row):
 OUT.parent.mkdir(parents=True,exist_ok=True);ex=OUT.exists() and OUT.stat().st_size>0
 with OUT.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if not ex:w.writeheader()
  w.writerow(row);f.flush()
def ids():
 if not OUT.exists():return set()
 with OUT.open(encoding='utf-8-sig',newline='') as f:return {r['experiment_id'] for r in csv.DictReader(f)}
def evaluate(model,env_id,seed,episodes=20):
 env=gym.make(env_id);vals=[]
 for i in range(episodes):
  obs,_=env.reset(seed=100000+seed*100+i);done=False;ret=0.
  while not done:
   act=model.predict(obs,deterministic=True)[0];obs,r,term,trunc,_=env.step(act);ret+=float(r);done=term or trunc
  vals.append(ret)
 env.close();return vals
def run_double_dqn(seed):
 from torch import nn
 torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(2)
 env=gym.make('CartPole-v1');env.action_space.seed(seed);obs,_=env.reset(seed=seed)
 q=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,2));target=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,2));target.load_state_dict(q.state_dict());opt=torch.optim.Adam(q.parameters(),lr=1e-3);buf=deque(maxlen=50000);updates=0;cum_train=0.;eval_total=0.;train_mark=0.;run=f'double_dqn_seed{seed}_budget_v1';cfg=CFG/f'{run}.json';cfg.write_text(json.dumps({'experiment_id':run,'model':'Double DQN','environment':'CartPole-v1','seed':seed,'budgets':BUDGETS,'batch_size':64,'replay_capacity':50000,'learning_starts':500,'target_sync_steps':500,'epsilon_schedule':'linear 1 to .05 over 10k interactions','double_dqn_target':'online argmax; target-network evaluation','evaluation_episodes':20},indent=2),encoding='utf-8')
 seen=ids();ep=0;start=time.perf_counter();
 for step in range(1,max(BUDGETS)+1):
  eps=max(.05,1-.95*step/10000)
  action=env.action_space.sample() if random.random()<eps else int(q(torch.as_tensor(obs,dtype=torch.float32)).argmax().item())
  nxt,reward,term,trunc,_=env.step(action);done=term or trunc;buf.append((obs.copy(),action,reward,nxt.copy(),float(done)));obs=nxt
  if done:ep+=1;obs,_=env.reset(seed=seed+ep)
  if step>=500 and step%4==0:
   batch=random.sample(buf,64);ss,aa,rr,next_states,dd=zip(*batch);ss=torch.tensor(np.asarray(ss),dtype=torch.float32);aa=torch.tensor(aa);rr=torch.tensor(rr,dtype=torch.float32);next_states=torch.tensor(np.asarray(next_states),dtype=torch.float32);dd=torch.tensor(dd,dtype=torch.float32)
   pred=q(ss).gather(1,aa[:,None]).squeeze(1)
   with torch.no_grad():nexta=q(next_states).argmax(1);boot=rr+.99*(1-dd)*target(next_states).gather(1,nexta[:,None]).squeeze(1)
   loss=__import__('torch').nn.functional.smooth_l1_loss(pred,boot);opt.zero_grad();loss.backward();opt.step();updates+=1
  if step%500==0:target.load_state_dict(q.state_dict())
  if step in BUDGETS:
   traincum=max(0.,time.perf_counter()-start-eval_total);dt=traincum-train_mark;train_mark=traincum;t=time.perf_counter();vals=[];ee=gym.make('CartPole-v1')
   for i in range(20):
    o,_=ee.reset(seed=100000+seed*100+i);d=False;ret=0.
    while not d:
     a=int(q(torch.as_tensor(o,dtype=torch.float32)).argmax().item());o,r,te,tr,_=ee.step(a);ret+=float(r);d=te or tr
    vals.append(ret)
   ee.close();ev=time.perf_counter()-t;eval_total+=ev;path=CK/f'{run}_steps{step}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save({'q':q.state_dict(),'target':target.state_dict(),'optimizer':opt.state_dict(),'step':step,'updates':updates,'replay_buffer':buf},path)
   eid=f'{run}_target{step}'
   if eid not in seen:
    append({'experiment_id':eid,'model_name':'Double DQN','paradigm':'reinforcement_learning','dataset':'CartPole-v1','dataset_revision':'Gymnasium CartPole-v1','split_id':f'CartPole-v1_eval_seed{seed}','seed':seed,'budget_type':'environment_steps','budget_value':step,'requested_budget':step,'environment_steps':step,'optimizer_steps':updates,'training_time_seconds':dt,'cumulative_training_time_seconds':traincum,'evaluation_time_seconds':ev,'evaluation_episodes':20,'mean_return':float(np.mean(vals)),'return_sd':float(np.std(vals,ddof=1)),'return_median':float(np.median(vals)),'peak_memory_bytes':psutil.Process().memory_info().rss,'hardware':'AMD Ryzen AI 7 350 CPU','backend':f'CPU; PyTorch {torch.__version__}; custom Double DQN','experiment_status':'completed','source_type':'local_measurement','configuration_path':str(cfg.relative_to(ROOT)),'checkpoint_path':str(path.relative_to(ROOT)),'timestamp_utc':datetime.now(timezone.utc).isoformat(),'details':json.dumps({'returns':vals,'double_dqn_target':'online network selects next action; target network evaluates','overshoot_steps':0})});seen.add(eid)
   print(f'Double DQN seed={seed} env_steps={step} eval20_mean={np.mean(vals):.2f} +/- {np.std(vals,ddof=1):.2f}',flush=True)
 env.close()
class BudgetCallback:
 def __init__(self,model_name,env_id,seed,run,cfg,start,requested,base_cum=0.):
  from stable_baselines3.common.callbacks import BaseCallback
  self.base=BaseCallback
  class _Callback(BaseCallback):
   def __init__(self,outer):super().__init__(verbose=0);self.o=outer
   def _on_step(self):return self.o.step(self.model)
  self.callback=_Callback(self);self.name=model_name;self.env_id=env_id;self.seed=seed;self.run=run;self.cfg=cfg;self.start=start;self.requested=requested;self.eval_total=0.;self.base_cum=base_cum;self.train_mark=base_cum;self.seen=ids()
 def step(self,model):
  actual=int(model.num_timesteps)
  while self.requested and actual>=self.requested[0]:
   target=self.requested.pop(0);before=time.perf_counter();cum=self.base_cum+max(0.,before-self.start-self.eval_total);dt=max(0.,cum-self.train_mark);vals=evaluate(model,self.env_id,self.seed,20);ev=time.perf_counter()-before;self.eval_total+=ev;self.train_mark=cum
   sd=float(np.std(vals,ddof=1));mean=float(np.mean(vals));checkpoint=CK/f'{self.run}_steps{actual}.zip';checkpoint.parent.mkdir(parents=True,exist_ok=True);model.save(checkpoint)
   if self.name in ('DQN','SAC'):model.save_replay_buffer(str(checkpoint)+'.replay.pkl')
   eid=f'{self.run}_target{target}'
   if eid not in self.seen:
    append({'experiment_id':eid,'model_name':self.name,'paradigm':'reinforcement_learning','dataset':self.env_id,'dataset_revision':'Gymnasium environment ID; fixed package versions in active .venv','split_id':f'{self.env_id}_eval_seeds_100000_plus_{self.seed}','seed':self.seed,'budget_type':'environment_steps','budget_value':actual,'requested_budget':target,'environment_steps':actual,'optimizer_steps':getattr(model,'_n_updates',''),'training_time_seconds':dt,'cumulative_training_time_seconds':cum,'evaluation_time_seconds':ev,'evaluation_episodes':len(vals),'mean_return':mean,'return_sd':sd,'return_median':float(np.median(vals)),'success_rate':'','peak_memory_bytes':psutil.Process().memory_info().rss,'hardware':'AMD Ryzen AI 7 350 CPU','backend':f'CPU; PyTorch {torch.__version__}; Stable-Baselines3','experiment_status':'completed','source_type':'local_measurement','configuration_path':str(self.cfg.relative_to(ROOT)),'checkpoint_path':str(checkpoint.relative_to(ROOT)),'timestamp_utc':datetime.now(timezone.utc).isoformat(),'details':json.dumps({'returns':vals,'evaluation_deterministic':True,'training_env_separate':True,'overshoot_steps':actual-target})});self.seen.add(eid)
   print(f'{self.name} seed={self.seed} env_steps={actual} eval20_mean={mean:.2f} +/- {sd:.2f}',flush=True)
  return True
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);ap.add_argument('--models',nargs='+',choices=list(SPECS)+['Double DQN'],default=list(SPECS)+['Double DQN']);a=ap.parse_args();torch.set_num_threads(4)
 for name in a.models:
  if name=='Double DQN':continue
  cls,env_id,kw=SPECS[name]
  for seed in a.seeds:
   run=f'{name.lower()}_seed{seed}_budget_v1';cfg=CFG/f'{run}.json';cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(json.dumps({'experiment_id':run,'model':name,'environment':env_id,'seed':seed,'budgets':BUDGETS,'evaluation_episodes':20,'deterministic_evaluation':True,'algorithm_parameters':kw,'device':'CPU','separate_eval_environment':True},indent=2),encoding='utf-8')
   existing_rows=[]
   if OUT.exists():
    with OUT.open(encoding='utf-8-sig',newline='') as f:existing_rows=[r for r in csv.DictReader(f) if r.get('model_name')==name and r.get('seed')==str(seed)]
   completed_targets={int(r['requested_budget']) for r in existing_rows if r.get('experiment_status')=='completed' and r.get('requested_budget')}
   pending=[b for b in BUDGETS if b not in completed_targets]
   if not pending:print(f'{name} seed={seed} already complete; skipped',flush=True);continue
   env=gym.make(env_id);env.reset(seed=seed);env.action_space.seed(seed)
   if name=='SAC':env.observation_space.seed(seed)
   latest=max(existing_rows,key=lambda r:int(r.get('requested_budget') or 0),default=None);base_cum=float(latest.get('cumulative_training_time_seconds') or 0.) if latest else 0.
   if latest and latest.get('checkpoint_path') and (ROOT/latest['checkpoint_path']).exists():
    checkpoint=ROOT/latest['checkpoint_path'];model=cls.load(checkpoint,env=env,device='cpu')
    replay=Path(str(checkpoint)+'.replay.pkl')
    if name in ('DQN','SAC') and replay.exists():model.load_replay_buffer(str(replay))
    print(f'{name} seed={seed} resuming from {model.num_timesteps} steps; environment restarts at reset',flush=True)
   else:model=cls('MlpPolicy',env,seed=seed,device='cpu',verbose=0,**kw)
   start=time.perf_counter();cb=BudgetCallback(name,env_id,seed,run,cfg,start,pending,base_cum);model.learn(total_timesteps=max(0,max(BUDGETS)-int(model.num_timesteps)),callback=cb.callback,reset_num_timesteps=False);env.close()
 if 'Double DQN' in a.models:
  for seed in a.seeds:
   if all(f'double_dqn_seed{seed}_budget_v1_target{b}' in ids() for b in BUDGETS):print(f'Double DQN seed={seed} already complete; skipped',flush=True)
   else:run_double_dqn(seed)
 print('saved',OUT)
if __name__=='__main__':main()
