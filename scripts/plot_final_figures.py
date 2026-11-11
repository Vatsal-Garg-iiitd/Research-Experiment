"""Publication-style figures from verified existing result files only."""
import csv, math
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'results/raw'; PROC=ROOT/'results/processed'; LLM=ROOT/'results/llm'; FIG=ROOT/'results/figures/final'; FIG.mkdir(parents=True,exist_ok=True)
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#6A4C93', '#4D4D4D']
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'axes.labelweight': 'normal', 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'axes.grid': True,
    'grid.color': '#D9E1E8', 'grid.alpha': .65, 'grid.linewidth': .7,
    'font.family': 'DejaVu Sans', 'axes.prop_cycle': plt.cycler(color=COLORS),
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'svg.fonttype': 'none',
})
catalog=[]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def n(r,k):
    try:x=float(r.get(k,''));return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None
def save(fig,num,title,sources,interpretation,limitations):
    stem=f'figure_{num:02d}'
    for ax in fig.axes:
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', color='#D9E1E8', alpha=.65, linewidth=.7)
        ax.grid(False, axis='x')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#8B98A5'); ax.spines['bottom'].set_color('#8B98A5')
        leg=ax.get_legend()
        if leg is not None:
            leg.set_frame_on(True)
            leg.get_frame().set_facecolor('white');leg.get_frame().set_edgecolor('#D9E1E8');leg.get_frame().set_alpha(.92)
        # Keep protocol notes legible without covering low-valued observations.
        for note in ax.texts:
            if note.get_transform() == ax.transAxes:
                x,y=note.get_position()
                if y < .08:
                    note.set_position((max(.015,x),.97));note.set_va('top')
                    note.set_bbox({'boxstyle':'round,pad=.25','facecolor':'white','edgecolor':'#D9E1E8','alpha':.9})
    fig.tight_layout(pad=1.25); fig.savefig(FIG/(stem+'.png'),bbox_inches='tight',facecolor='white');fig.savefig(FIG/(stem+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
    catalog.append({'number':num,'title':title,'file_png':f'results/figures/final/{stem}.png','file_vector':f'results/figures/final/{stem}.svg','data_source':sources,'analysis_script':'scripts/plot_final_figures.py','interpretation':interpretation,'limitations':limitations})
def errorbar(ax,x,ys,label):
    vals=[v for v in ys if v is not None]
    if not vals:return
    xx=[x[i] for i,v in enumerate(ys) if v is not None]
    ax.plot(xx,vals,'o-',label=label,linewidth=1.4,markersize=3)

def mean_ci(values, confidence=.95):
    """Return mean and symmetric Student-t CI half-width for independent runs."""
    vals=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    if not len(vals): return np.nan, 0.0, 0
    if len(vals)==1: return float(vals.mean()), 0.0, 1
    half=float(student_t.ppf((1+confidence)/2,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals)))
    return float(vals.mean()), half, len(vals)

def plot_ci(ax, xs, groups, label=None, marker='o', color=None, linewidth=1.8, capsize=2.5):
    """Plot means and 95% t intervals from a sequence of seed-level groups."""
    means=[]; errors=[]; counts=[]; valid_x=[]
    for x,values in zip(xs,groups):
        mean,half,count=mean_ci(values)
        if count:
            valid_x.append(x); means.append(mean); errors.append(half); counts.append(count)
    if not valid_x:return
    count_note=f'n={min(counts)}' if min(counts)==max(counts) else f'n={min(counts)}–{max(counts)}'
    full_label=f'{label} ({count_note})' if label else None
    ax.errorbar(valid_x,means,yerr=errors,marker=marker,markersize=4,linewidth=linewidth,
                capsize=capsize,color=color,label=full_label,zorder=3)
    return valid_x,means,errors,counts

def main():
    # 1: supervised performance by epoch, using fixed split and only completed runs.
    rows=[r for r in read(RAW/'replicated_scaling_results.csv') if r.get('experiment_status')=='completed' and r.get('split_id')=='24574ee00a20ab31' and n(r,'validation_accuracy') is not None]
    vit_rows=[r for r in read(RAW/'vit_tiny_scaling.csv') if r.get('status')=='completed' and r.get('split_id')=='24574ee00a20ab31' and n(r,'validation_accuracy') is not None]
    rows += vit_rows
    fig,ax=plt.subplots(figsize=(8.2,5));
    for model in sorted(set(r['model_name'] for r in rows)):
        groups=defaultdict(list)
        for r in rows:
            if r['model_name']==model:groups[int(float(r['budget_value']))].append(n(r,'validation_accuracy'))
        xs=sorted(groups);plot_ci(ax,xs,[groups[x] for x in xs],model)
    ax.set(xlabel='Training epoch',ylabel='Validation accuracy',title='Supervised learning on Fashion-MNIST\n10K training examples · fixed split · mean ± 95% t interval',ylim=(0,1));handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='center left',bbox_to_anchor=(1.0,.5),ncol=1,frameon=False);save(fig,1,'Supervised validation accuracy versus epoch','results/raw/replicated_scaling_results.csv; results/raw/vit_tiny_scaling.csv','Seed means and 95% t intervals show early improvement and later method-specific changes; actual ViT-Tiny-dimension run is separate from the compact proxy.','Three seeds per model and epoch; intervals are wide for the actual 12-block ViT-Tiny.')
    # 2/3: measured time and measured compute; anomalous interval retained separately.
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    time_models=sorted(set(r['model_name'] for r in rows))
    for model_index,model in enumerate(time_models):
        rr=sorted([r for r in rows if r['model_name']==model],key=lambda r:int(float(r['budget_value'])))
        color=COLORS[model_index%len(COLORS)]
        # seed-level traces support visibility of the single-seed adjustment
        if model=='ResNet-18':
            rr11=sorted([r for r in rr if str(r['seed'])=='11'],key=lambda r:int(float(r['budget_value'])))
            axs[1].plot([int(float(r['budget_value'])) for r in rr11],[max(0,n(r,'cumulative_training_time_seconds')-1122.1625285) if int(float(r['budget_value']))>=3 else n(r,'cumulative_training_time_seconds') for r in rr11],':',alpha=.8,color=color,lw=1.1,label='ResNet-18 seed 11 adjusted')
        by_raw=defaultdict(list);by_adjusted=defaultdict(list)
        for seed in sorted(set(str(r['seed']) for r in rr)):
            seed_rows=sorted([r for r in rr if str(r['seed'])==seed],key=lambda r:int(float(r['budget_value'])))
            xs=[int(float(r['budget_value'])) for r in seed_rows];raw=[n(r,'cumulative_training_time_seconds') for r in seed_rows];adjusted=[]
            for r,val in zip(seed_rows,raw):
                if model=='ResNet-18' and seed=='11' and int(float(r['budget_value']))>=3 and val is not None:val=max(0,val-1122.1625285)
                adjusted.append(val)
            axs[0].plot(xs,raw,color=color,alpha=.22,lw=.8)
            axs[1].plot(xs,adjusted,color=color,alpha=.22,lw=.8)
            for x,v1,v2 in zip(xs,raw,adjusted):
                if v1 is not None:by_raw[x].append(v1)
                if v2 is not None:by_adjusted[x].append(v2)
        xs=sorted(by_raw);axs[0].plot(xs,[np.mean(by_raw[x]) for x in xs],'-o',color=color,lw=2.2,ms=3.5,label=model)
        xs=sorted(by_adjusted);axs[1].plot(xs,[np.mean(by_adjusted[x]) for x in xs],'-o',color=color,lw=2.2,ms=3.5,label=model)
    axs[0].set(xlabel='Training epoch',ylabel='Cumulative wall time (seconds)',title='Measured cumulative time')
    axs[1].set(xlabel='Training epoch',ylabel='Adjusted cumulative time (seconds)',title='Sensitivity view with ResNet interval removed')
    h,l=axs[1].get_legend_handles_labels();fig.legend(h,l,loc='center left',bbox_to_anchor=(1.01,.5),ncol=1,frameon=False,fontsize=7)
    fig.text(.5,.005,'Faint traces are individual seeds; bold lines are seed means. Raw ResNet-18 seed 11 includes the 1,122.16 s interval.',ha='center',fontsize=7.5,color='#52616B')
    save(fig,2,'Training time versus epoch, raw and adjusted','results/raw/replicated_scaling_results.csv','The paired panels expose the 1,122.16 s ResNet-18 seed-11 interval and show its effect on mean wall time.','Adjusted panel subtracts the interval after epoch 2 as a sensitivity analysis only; its cause is unknown and the raw measure remains shown.')
    # 4 marginal accuracy / second from paired per-seed results.
    marg=read(PROC/'strengthened_marginal_gains.csv'); rr=[r for r in marg if r.get('source_file') in ('replicated_scaling_results.csv','vit_tiny_scaling.csv') and r.get('metric')=='validation_accuracy' and r.get('model_name') in ('MLP','CNN / LeNet','ResNet-18','Vision Transformer (compact ViT-Tiny-style)','ViT-Tiny') and n(r,'marginal_improvement_per_second') is not None]
    fig,ax=plt.subplots(figsize=(8,4.8))
    for model in sorted(set(r['model_name'] for r in rr)):
        g=defaultdict(list)
        for r in rr:
            if r['model_name']==model:g[int(float(r['to_budget']))].append(n(r,'marginal_improvement_per_second'))
        plot_ci(ax,sorted(g),[g[x] for x in sorted(g)],model)
    ax.axhline(0,color='#59636E',lw=1,ls='--');ax.set(xlabel='Destination epoch',ylabel='Accuracy fraction gained per added second',title='Marginal validation gain per logged second\nPaired within seed · whiskers are 95% t intervals');handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='center left',bbox_to_anchor=(1.0,.5),ncol=1,frameon=False);save(fig,3,'Marginal accuracy improvement per additional compute','results/processed/strengthened_marginal_gains.csv','Marginal returns fluctuate and can be negative; uncertainty intervals make that variability visible.','Per-second metric is sensitive to wall-clock noise; only the adjusted diagnostic handles the ResNet anomaly interval.')
    # 5/6 scaling by data and RF trees.
    ds=read(PROC/'data_size_scaling_summary.csv');fig,ax=plt.subplots(figsize=(7,4.5))
    for model in sorted(set(r['model_name'] for r in ds)):
        q=sorted([r for r in ds if r['model_name']==model],key=lambda r:int(r['actual_train_size']));x=[int(r['actual_train_size']) for r in q];y=[n(r,'mean') for r in q];lo=[y[i]-n(r,'ci95_low') for i,r in enumerate(q)];hi=[n(r,'ci95_high')-y[i] for i,r in enumerate(q)];ax.errorbar(x,y,yerr=[lo,hi],marker='o',markersize=5,capsize=3,lw=1.8,label=f'{model} (n=3)')
    ax.set(xlabel='Training examples',ylabel='Official test accuracy at epoch 5',title='Data-size endpoints at a matched 5-epoch budget');ax.legend();ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'{v/1000:g}K'));ax.text(.01,.02,'Whiskers are 95% t intervals; train/validation split held fixed.',transform=ax.transAxes,fontsize=7.5,color='#52616B');save(fig,4,'Dataset-size scaling for MLP and LeNet','results/processed/data_size_scaling_summary.csv','LeNet improves across the two reported sizes; MLP is not monotone across its available range.','Only 30K and 58K requested-size points are plotted; three-seed intervals are imprecise.')
    rf=[r for r in read(RAW/'random_forest_scaling.csv') if r.get('experiment_status')=='completed'];g=defaultdict(list)
    for r in rf:
        if n(r,'evaluation_value') is not None:g[int(float(r['budget_value']))].append(n(r,'evaluation_value'))
    fig,ax=plt.subplots(figsize=(7,4.5));xs=sorted(g);plot_ci(ax,xs,[g[x] for x in xs],'Random Forest');ax.set(xlabel='Number of trees',ylabel='Validation accuracy',title='Random Forest performance versus tree count');ax.text(.01,.02,'Points show seed means; whiskers are 95% t intervals.',transform=ax.transAxes,fontsize=7.5,color='#52616B');save(fig,5,'Random Forest performance versus number of trees','results/raw/random_forest_scaling.csv','The curve shows validation accuracy over the native tree-count budget with seed uncertainty.','Warm-start forest runs; no interpretation as neural epochs.')
    # 7-10 unsupervised separate units and metrics.
    u=[r for r in read(RAW/'unsupervised_scaling.csv') if r.get('experiment_status')=='completed'];seen=set();clean=[]
    for r in u:
        key=r.get('experiment_id')
        if r['model_name']=='MiniBatch K-Means' and key in seen:continue
        seen.add(key);clean.append(r)
    for model,numfig,budget_label in [('MiniBatch K-Means',6,'partial_fit updates'),('Gaussian Mixture Model',7,'warm-start EM fit calls')]:
        q=[r for r in clean if r['model_name']==model]
        metrics=[('evaluation_value','Adjusted Rand index'),('validation_NMI','Normalized mutual information'),('validation_silhouette','Silhouette score')]
        if model=='Gaussian Mixture Model':metrics.append(('training_log_likelihood','Training log-likelihood'))
        fig,axs=plt.subplots(1,len(metrics),figsize=(4.25*len(metrics),4.2),squeeze=False);axs=axs.ravel()
        for ax,(field,metric_label) in zip(axs,metrics):
            g=defaultdict(list)
            for r in q:
                if n(r,field) is not None:g[int(float(r['budget_value']))].append(n(r,field))
            xs=sorted(g);plot_ci(ax,xs,[g[x] for x in xs],model);ax.set(xlabel=budget_label,ylabel=metric_label,title=metric_label)
            if field=='training_log_likelihood':ax.set_ylabel('Mean log-likelihood per sample')
        fig.suptitle(f'{model}: clustering quality and fit objective',y=1.03,fontweight='bold')
        save(fig,numfig,f'{model} multi-metric performance versus computational budget','results/raw/unsupervised_scaling.csv','The panels keep external clustering metrics distinct; GMM log-likelihood is shown separately from clustering quality.','Seed means with 95% t intervals; K-Means reused-checkpoint duplicate rows are excluded; native update units are retained.')
    db=[r for r in read(RAW/'dbscan_size_scaling.csv') if r.get('status')=='completed'];fig,axs=plt.subplots(1,3,figsize=(12,4.2))
    sizes=sorted(set(int(float(r['budget_value'])) for r in db))
    for ax,field,label,title in [(axs[0],'evaluation_value','Adjusted Rand index','Clustering agreement'),(axs[1],'training_time_seconds','Fit time (seconds)','Observed fit time'),(axs[2],'noise_fraction','Noise fraction','Points labeled noise')]:
        g=defaultdict(list)
        for r in db:
            if n(r,field) is not None:g[int(float(r['budget_value']))].append(n(r,field))
        if field=='training_time_seconds':
            means=[]
            for x in sizes:
                vals=g[x];means.append(np.mean(vals));jitter=np.linspace(-45,45,len(vals))
                ax.scatter(np.asarray([x]*len(vals))+jitter,vals,s=22,alpha=.5,color=COLORS[1],edgecolor='white',linewidth=.4,zorder=2)
            ax.plot(sizes,means,'s-',color=COLORS[1],lw=1.8,ms=5,zorder=3)
        else:plot_ci(ax,sizes,[g[x] for x in sizes],None)
        ax.set(xlabel='Samples in fit',ylabel=label,title=title)
        if field=='evaluation_value':ax.axhline(0,color='#59636E',lw=.8,ls='--')
        if field=='noise_fraction':ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'{100*v:.1f}%'))
    fig.suptitle('DBSCAN fit-size sweep · fixed density settings',y=1.02,fontweight='bold');fig.text(.5,.005,'Three seeds per size · ARI/noise use 95% t intervals; faint fit-time points are individual seeds.',ha='center',fontsize=7.5,color='#52616B')
    save(fig,8,'DBSCAN quality, fit time and noise versus dataset size','results/raw/dbscan_size_scaling.csv','Three matched panels show clustering agreement, fit time and noise share; individual fit-time seed observations are visible.','Three fit sizes and three seeds; this is descriptive evidence, not a runtime scaling law.')
    ae=[r for r in read(RAW/'autoencoder_scaling.csv') if r.get('experiment_status')=='completed'];
    for metric,numfig,label in [('validation_reconstruction_mse',9,'Validation reconstruction MSE'),('downstream_validation_accuracy',10,'Downstream linear-probe accuracy')]:
        fig,ax=plt.subplots(figsize=(7.4,4.5))
        for model in sorted(set(r['model_name'] for r in ae)):
            g=defaultdict(list)
            for r in ae:
                if r['model_name']==model and n(r,metric) is not None:g[int(float(r['budget_value']))].append(n(r,metric))
            xs=sorted(g);plot_ci(ax,xs,[g[x] for x in xs],model)
        ax.set(xlabel='Training epochs',ylabel=label,title=label+' by training budget');ax.legend();ax.text(.01,.02,'Points show seed means; whiskers are 95% t intervals.',transform=ax.transAxes,fontsize=7.5,color='#52616B');save(fig,numfig,label+' versus training budget','results/raw/autoencoder_scaling.csv',f'{label} is reported separately for each representation model.','Three seeds; reconstruction and downstream metrics answer different questions.')
    # SSL label fraction (final common 30 epoch) and training budget curves.
    ssl=[r for r in read(RAW/'ssl_algorithms_test.csv') if r.get('status')=='completed' and n(r,'test_accuracy') is not None]
    g=defaultdict(list)
    for r in ssl:
        if int(float(r.get('epochs') or 0))==30:g[(r.get('model_name'),float(r.get('labeled_fraction')))].append(n(r,'test_accuracy'))
    fig,ax=plt.subplots(figsize=(8,5))
    for model in sorted(set(k[0] for k in g)):
        xs=sorted(k[1] for k in g if k[0]==model);plot_ci(ax,[100*x for x in xs],[g[(model,x)] for x in xs],model)
    ax.set(xlabel='Labeled training data (%)',ylabel='Official test accuracy',title='Semi-supervised endpoints at 30 epochs');ax.legend(ncol=2);ax.text(.01,.02,'Points show seed means; whiskers are 95% t intervals. Labeled fraction uses the same split protocol.',transform=ax.transAxes,fontsize=7.5,color='#52616B');save(fig,11,'Semi-supervised performance versus labeled fraction','results/raw/ssl_algorithms_test.csv','Algorithm and label-fraction effects are shown at the shared 30-epoch endpoint with seed uncertainty.','Three seeds per condition; supervised baseline is plotted separately in Figure 21.')
    # SSL curves: all methods separate to avoid combining data regimes.
    sr=[r for r in read(RAW/'ssl_algorithms.csv') if r.get('status')=='completed' and r.get('labeled_fraction')=='0.1'];fig,ax=plt.subplots(figsize=(8,5))
    for model in sorted(set(r['model_name'] for r in sr)):
        q=[r for r in sr if r['model_name']==model];g=defaultdict(list)
        for r in q:
            if n(r,'validation_accuracy') is not None:g[int(float(r['budget_value']))].append(n(r,'validation_accuracy'))
        xs=sorted(g);plot_ci(ax,xs,[g[x] for x in xs],model)
    ax.set(xlabel='Training epoch',ylabel='Validation accuracy',title='Semi-supervised training curves · 10% labeled data');ax.legend(ncol=2);ax.text(.01,.02,'Points show seed means; whiskers are 95% t intervals.',transform=ax.transAxes,fontsize=7.5,color='#52616B');save(fig,12,'Semi-supervised performance versus training budget','results/raw/ssl_algorithms.csv','Methods have distinct trajectories at a common label fraction; uncertainty is shown across seeds.','Three seeds; task adaptations, not canonical implementations on standard SSL benchmark splits.')
    # RL curves with valid groups by model/environment; include seed SD.
    rl=[]
    for raw_row in read(RAW/'rl_scaling_results.csv'):
        r=dict(raw_row)
        if r.get('model_name')=='Double DQN' and r.get('details') in (None,''):
            r['details']=r.get('timestamp_utc','');r['timestamp_utc']=r.get('checkpoint_path','');r['checkpoint_path']=r.get('configuration_path','');r['configuration_path']=r.get('source_type','');r['source_type']='local_measurement';r['experiment_status']=r.get('backend','');r['backend']=r.get('hardware','');r['hardware']=r.get('peak_memory_bytes','');r['peak_memory_bytes']=r.get('success_rate','');r['success_rate']=''
        if r.get('experiment_status')=='completed' and n(r,'mean_return') is not None:rl.append(r)
    for env,fig_no in [('CartPole-v1',13),('Pendulum-v1',14)]:
        q=[r for r in rl if r.get('dataset')==env];fig,axs=plt.subplots(1,2,figsize=(11,4.5))
        for model in sorted(set(r['model_name'] for r in q)):
            g=defaultdict(list);t=defaultdict(list)
            for r in q:
                if r['model_name']==model:
                    b=int(float(r['environment_steps']));g[b].append(n(r,'mean_return'))
                    if n(r,'cumulative_training_time_seconds') is not None:t[b].append((n(r,'cumulative_training_time_seconds'),n(r,'mean_return')))
            xs=sorted(g);plot_ci(axs[0],xs,[g[x] for x in xs],model)
            tt=[]
            for x in xs:
                if not t[x]:continue
                times=np.asarray([v[0] for v in t[x]],dtype=float);returns=np.asarray([v[1] for v in t[x]],dtype=float)
                mx,ex,_=mean_ci(times);my,ey,_=mean_ci(returns);tt.append((mx,my,ex,ey))
            if tt:axs[1].errorbar([v[0] for v in tt],[v[1] for v in tt],xerr=[v[2] for v in tt],yerr=[v[3] for v in tt],marker='o',ms=4,lw=1.6,capsize=2,label=f'{model} (seed means)')
        axs[0].set(xlabel='Environment interactions (steps)',ylabel='Mean return per checkpoint',title=f'{env}: return vs steps');axs[1].set(xlabel='Cumulative training time (seconds)',ylabel='Mean return per checkpoint',title=f'{env}: return vs training time')
        for ax in axs:ax.legend(fontsize=7,ncol=2)
        fig.text(.5,.01,'Whiskers on the step curve are 95% t intervals across independent training seeds; evaluation uses 20 episodes per checkpoint.',ha='center',fontsize=7.5,color='#52616B')
        save(fig,fig_no,f'{env} reward versus steps and time','results/raw/rl_scaling_results.csv',f'Algorithm-specific return trajectories are shown for {env}.','Seed counts differ by algorithm; separate environments and reward scales; logged elapsed time is hardware/protocol specific.')
    # LLM full test endpoint and prefix curves / zero-shot separate protocol plots.
    full=read(LLM/'macos_full_wikitext2_evaluations.csv');full=[r for r in full if r.get('status')=='completed' and n(r,'parameter_count') and n(r,'perplexity')]
    families=defaultdict(list)
    for r in full:families[r['model_name'].split('-')[0] if not r['model_name'].startswith('GPT') else 'GPT-2'].append(r)
    family_colors={'Pythia':COLORS[0],'GPT-2':COLORS[1],'OPT':COLORS[2]}
    ordered=sorted(full,key=lambda r:n(r,'parameter_count'))
    fig,(ax,table_ax)=plt.subplots(1,2,figsize=(11,5.4),gridspec_kw={'width_ratios':[1.8,1.05]})
    fig.suptitle('Full-test checkpoint results under one local evaluation protocol',fontweight='bold',y=.99)
    for family,items in families.items():
        ax.scatter([n(r,'parameter_count')/1e6 for r in items],[n(r,'perplexity') for r in items],s=124,label=family,edgecolor='white',linewidth=.8,zorder=3,color=family_colors[family])
    for idx,r in enumerate(ordered,start=1):
        ax.annotate(str(idx),(n(r,'parameter_count')/1e6,n(r,'perplexity')),ha='center',va='center',fontsize=7.5,color='white',fontweight='bold',zorder=4)
    ax.set_xscale('log');ax.set_xlim(55,2400);ax.set(xlabel='Parameter count (millions; log scale)',ylabel='WikiText-2 raw-v1 test perplexity');ax.legend(title='Model family',loc='upper right')
    table_ax.axis('off');table_ax.set_xlim(0,1);table_ax.set_ylim(0,1)
    table_ax.text(0,.985,'Point key',fontsize=10,fontweight='bold',va='top',color='#17365D')
    table_ax.text(0,.94,'Full test · FP32 MPS · context 1,024 · stride 512',fontsize=7.1,va='top',color='#52616B')
    table_ax.text(.01,.88,'#',fontsize=7.5,fontweight='bold',color='#52616B');table_ax.text(.13,.88,'Model',fontsize=7.5,fontweight='bold',color='#52616B');table_ax.text(.70,.88,'Params (M)',fontsize=7.5,fontweight='bold',ha='right',color='#52616B');table_ax.text(.99,.88,'PPL',fontsize=7.5,fontweight='bold',ha='right',color='#52616B')
    for idx,r in enumerate(ordered,start=1):
        y=.82-(idx-1)*.068
        family=r['model_name'].split('-')[0] if not r['model_name'].startswith('GPT') else 'GPT-2'
        color=family_colors[family]
        table_ax.text(.01,y,str(idx),fontsize=8,color='#17365D',fontweight='bold',va='center')
        table_ax.text(.13,y,r['model_name'],fontsize=7.7,color=color,va='center')
        table_ax.text(.70,y,f"{n(r,'parameter_count')/1e6:,.0f}",fontsize=7.7,ha='right',va='center',color='#263746')
        table_ax.text(.99,y,f"{n(r,'perplexity'):.2f}",fontsize=7.7,ha='right',va='center',color='#263746')
        if idx%2==0:table_ax.axhspan(y-.028,y+.028,color='#F4F7FA',zorder=-1)
    save(fig,15,'Full WikiText-2 perplexity versus parameter count','results/llm/macos_full_wikitext2_evaluations.csv','The numbered points pair with a table listing each checkpoint size and measured perplexity under the shared local test protocol.','Eleven checkpoints across families; tokenizers and training histories differ; parameter count is not training compute. Each point is one checkpoint evaluation, not a seed replicate.')
    pref=read(LLM/'expanded_llm_evaluations.csv');pref=[r for r in pref if r.get('status')=='completed' and r.get('benchmark')=='WikiText-2 raw test prefix' and n(r,'training_step') is not None and n(r,'test_loss') is not None]
    for modelname,fig_no in [('Pythia-70M',16),('Pythia-160M',17)]:
        q=[r for r in pref if r.get('model')==modelname and r.get('checkpoint_id') not in ('main',)];g=defaultdict(list)
        for r in q:g[int(float(r['training_step']))].append(n(r,'test_loss'))
        fig,ax=plt.subplots(figsize=(7.5,4.5));xs=sorted(g);means=[np.mean(g[x]) for x in xs];ax.plot(xs,means,'o-',lw=2,ms=5)
        ax.set_xscale('symlog',linthresh=1000);ax.set_xticks([0,1000,10000,100000]);ax.set_xticklabels(['0','1k','10k','100k']);ax.set(xlabel='Released checkpoint training step (symlog scale)',ylabel='Loss on restricted WikiText-2 prefix',title=f'{modelname} selected checkpoint trajectory')
        ax.text(.01,.02,'2,048-token prefix · CPU FP32 · selected released checkpoints; step-to-token values are published metadata.',transform=ax.transAxes,fontsize=7.2,color='#52616B')
        save(fig,fig_no,f'{modelname} checkpoint-prefix evaluation trajectory','results/llm/expanded_llm_evaluations.csv','Loss across selected released checkpoints is shown on a symlog step axis so step zero remains visible.','Restricted local-text prefix and single evaluations; step-to-token mapping is published metadata, not measured local training compute.')
    harness=read(LLM/'harness_full_summary.csv');fig,ax=plt.subplots(figsize=(8,4.8));harness=[r for r in harness if n(r,'metric_accuracy') is not None];harness.sort(key=lambda r:n(r,'metric_accuracy'))
    labels=[f"{r['task']}  (n={int(float(r['evaluated_examples'])):,})" for r in harness];values=[n(r,'metric_accuracy') for r in harness];errors=[1.96*(n(r,'accuracy_stderr') or 0) for r in harness]
    y=np.arange(len(harness));ax.barh(y,values,color=COLORS[0],height=.58,alpha=.9);ax.errorbar(values,y,xerr=errors,fmt='none',ecolor='#273746',capsize=3,lw=1.2,zorder=4);ax.set_yticks(y,labels);ax.set(xlim=(0,1),xlabel='Zero-shot accuracy (95% normal-approximation interval)',title='Pythia-70M on full validation splits');ax.text(.02,.97,'0-shot · FP32 · MPS · tasks shown separately (not averaged)',transform=ax.transAxes,va='top',fontsize=7.5,color='#52616B',bbox={'boxstyle':'round,pad=.3','facecolor':'white','edgecolor':'#D9E1E8','alpha':.9});save(fig,18,'Zero-shot benchmark results (full validation splits)','results/llm/harness_full_summary.csv','All available validation examples are shown by task with score uncertainty and sample counts.','One model and three tasks; task scores are not pooled; smoke-test subsets are excluded.')
    # Within-paradigm normalized cross-paradigm panel: relative change from first point only.
    fig,axs=plt.subplots(2,2,figsize=(10,7));
    panels=[('Supervised validation accuracy', [(r['model_name'],int(float(r['budget_value'])),str(r.get('seed','')),n(r,'validation_accuracy')) for r in rows]),('Semi-supervised accuracy',[(r['model_name'],int(float(r['budget_value'])),str(r.get('seed','')),n(r,'validation_accuracy')) for r in sr]),('RL return (CartPole)',[(r['model_name'],int(float(r['environment_steps'])),str(r.get('seed','')),n(r,'mean_return')) for r in rl if r.get('dataset')=='CartPole-v1']),('Representation reconstruction MSE',[(r['model_name'],int(float(r['budget_value'])),str(r.get('seed','')),n(r,'validation_reconstruction_mse')) for r in ae])]
    for ax,(title,triples) in zip(axs.flat,panels):
        series=defaultdict(dict)
        for name,b,seed,v in triples:
            if v is None:continue
            series[(name,seed)][b]=v
        normalized=defaultdict(lambda:defaultdict(list))
        for (name,seed),budget_values in series.items():
            budgets=sorted(budget_values);base=float(budget_values[budgets[0]])
            if abs(base)<1e-8:continue
            for index,budget in enumerate(budgets):
                val=float(budget_values[budget])
                normalized[name][index].append((val-base)*(100 if 'accuracy' in title.lower() else 1))
        for name,by_index in normalized.items():
            indices=sorted(by_index);plot_ci(ax,indices,[by_index[i] for i in indices],name)
        ylabel='Δ accuracy (percentage points)' if 'accuracy' in title.lower() else ('Δ return (reward units)' if 'return' in title.lower() else 'Δ reconstruction MSE')
        ax.axhline(0,color='#59636E',lw=.8,ls='--');ax.set(title=title,xlabel='Within-series budget order',ylabel=ylabel);ax.legend(fontsize=6,ncol=2)
    fig.suptitle('Change from each series baseline · separate native metrics',y=1.02,fontweight='bold');save(fig,19,'Cross-paradigm within-task changes','multiple verified raw result tables','Each panel shows seed-paired absolute change from its own first checkpoint; accuracy is in percentage points, while RL and reconstruction retain native units.','Panels do not share a metric or cost scale; budget order is ordinal and series baselines differ.')
    # Additional required views that retain matched budget/protocol boundaries.
    fig,ax=plt.subplots(figsize=(8,5))
    for model in sorted(set(r['model_name'] for r in rows)):
        q=[r for r in rows if r['model_name']==model];g=defaultdict(list)
        for r in q:
            epoch=int(float(r['budget_value']));tm=n(r,'cumulative_training_time_seconds')
            if tm is not None and model=='ResNet-18' and str(r['seed'])=='11' and epoch>=3:tm=max(0,tm-1122.1625285)
            if tm is not None and n(r,'validation_accuracy') is not None:g[epoch].append((tm,n(r,'validation_accuracy')))
        xs=sorted(g);mean_x=[];mean_y=[];xerr=[];yerr=[]
        for x in xs:
            a=np.asarray([v[0] for v in g[x]],dtype=float);b=np.asarray([v[1] for v in g[x]],dtype=float)
            mx,ex,_=mean_ci(a);my,ey,_=mean_ci(b);mean_x.append(mx);mean_y.append(my);xerr.append(ex);yerr.append(ey)
        ax.errorbar(mean_x,mean_y,xerr=xerr,yerr=yerr,marker='o',ms=4,lw=1.8,capsize=2,label=f'{model} (seed means)')
    ax.set(xlabel='Cumulative training time (seconds; adjusted ResNet diagnostic)',ylabel='Validation accuracy',title='Supervised performance versus measured training time');ax.legend(ncol=2,fontsize=7);ax.text(.01,.02,'Error bars show 95% t intervals across paired seed observations.',transform=ax.transAxes,fontsize=7.5,color='#52616B');save(fig,20,'Supervised performance versus cumulative compute time','results/raw/replicated_scaling_results.csv','Matched seed means of accuracy and cumulative time show task-specific performance-cost trajectories.','Time is elapsed runtime, not energy/FLOPs; the ResNet seed-11 interval is removed only in this adjusted diagnostic.')
    pseudo=read(RAW/'pseudolabel_test_results.csv');pseudo=[r for r in pseudo if r.get('status')=='completed' and int(float(r.get('epochs') or 0))==30 and n(r,'test_accuracy') is not None]
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for method,label in [('labeled_only','Labeled-only MLP baseline'),('pseudo_label','Pseudo-labeling')]:
        g=defaultdict(list)
        for r in pseudo:
            if r.get('method')==method:g[float(r['labeled_fraction'])].append(n(r,'test_accuracy'))
        xs=sorted(g);plot_ci(ax,[100*x for x in xs],[g[x] for x in xs],label)
    ax.set(xlabel='Labeled training data (%)',ylabel='Official test accuracy at epoch 30',title='Pseudo-labeling versus matched labeled-only MLP');ax.legend();ax.text(.02,.97,'Matched label fractions and seeds · mean ± 95% t interval',transform=ax.transAxes,va='top',fontsize=7.5,color='#52616B',bbox={'boxstyle':'round,pad=.3','facecolor':'white','edgecolor':'#D9E1E8','alpha':.9});save(fig,21,'Semi-supervised method versus supervised baseline','results/raw/pseudolabel_test_results.csv','Pseudo-labeling is compared with the labeled-only MLP under matched fractions, seeds, and the registered 30-epoch endpoint.','Three seeds per fraction; this ablation is separate from the multi-method SSL study.')
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for model in sorted(set(r['model_name'] for r in ae)):
        g=defaultdict(list)
        for r in ae:
            if r['model_name']!=model:continue
            tm=n(r,'cumulative_training_time_seconds');acc=n(r,'downstream_validation_accuracy')
            if tm is not None and acc is not None:g[int(float(r['budget_value']))].append((tm,acc))
        xs=sorted(g);mean_x=[];mean_y=[];xerr=[];yerr=[]
        for x in xs:
            a=np.asarray([v[0] for v in g[x]],dtype=float);b=np.asarray([v[1] for v in g[x]],dtype=float)
            mx,ex,_=mean_ci(a);my,ey,_=mean_ci(b);mean_x.append(mx);mean_y.append(my);xerr.append(ex);yerr.append(ey)
        ax.errorbar(mean_x,mean_y,xerr=xerr,yerr=yerr,marker='o',ms=4,lw=1.8,capsize=2,label=f'{model} (n=3)')
    ax.set(xlabel='Cumulative training time (seconds)',ylabel='Downstream linear-probe validation accuracy',title='Representation quality versus measured training cost');ax.legend();ax.text(.01,.02,'Points pair seed-specific cost and probe score; whiskers show 95% t intervals.',transform=ax.transAxes,fontsize=7.2,color='#52616B');save(fig,22,'Representation quality versus reconstruction-model training cost','results/raw/autoencoder_scaling.csv','Linear-probe performance is plotted against paired cumulative training time for each representation model.','Three seeds; the probe is a separate downstream assessment and not a universal representation metric.')
    model_colors={name:COLORS[i%len(COLORS)] for i,name in enumerate(sorted({r['model_name'] for r in full}))}
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for r in full:ax.scatter(n(r,'evaluation_seconds'),n(r,'perplexity'),s=58,zorder=3,edgecolor='white',linewidth=.7,color=model_colors[r['model_name']],label=r['model_name'])
    ax.set(xlabel='Full-test evaluation time (seconds; Apple M3 MPS)',ylabel='WikiText-2 raw-v1 test perplexity',title='Perplexity and runtime under the matched local protocol');ax.legend(title='Checkpoint',loc='center left',bbox_to_anchor=(1.01,.5),fontsize=7);ax.text(.01,.1,'Same corpus · FP32 · context 1,024 · stride 512 · one evaluation per model',transform=ax.transAxes,va='center',fontsize=7.2,color='#52616B',bbox={'boxstyle':'round,pad=.3','facecolor':'white','edgecolor':'#D9E1E8','alpha':.9});save(fig,23,'Perplexity versus comparable full-test evaluation time','results/llm/macos_full_wikitext2_evaluations.csv','The same local device, precision, context, stride, and corpus protocol permit descriptive inference-time comparison.','Tokenizer-specific scored token counts differ; evaluation time is not original training compute and each model has one run.')
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for r in full:ax.scatter(n(r,'parameter_count')/1e6,n(r,'negative_log_likelihood'),s=58,zorder=3,edgecolor='white',linewidth=.7,color=model_colors[r['model_name']],label=r['model_name'])
    ax.set_xscale('log');ax.set(xlabel='Parameter count (millions; log scale)',ylabel='Mean negative log-likelihood (nats/token)',title='Full-test loss versus model size');ax.legend(title='Checkpoint',loc='center left',bbox_to_anchor=(1.01,.5),fontsize=7);ax.text(.01,.1,'WikiText-2 raw-v1 · FP32 · MPS · context 1,024 · stride 512',transform=ax.transAxes,va='center',fontsize=7.2,color='#52616B',bbox={'boxstyle':'round,pad=.3','facecolor':'white','edgecolor':'#D9E1E8','alpha':.9});save(fig,24,'Full-test evaluation loss versus parameter count','results/llm/macos_full_wikitext2_evaluations.csv','Loss and parameter count are plotted for the same six local checkpoint evaluations.','Cross-family training data and tokenization differ; one evaluation per model; size is not training compute.')
    smoke=read(LLM/'harness_summary.csv');fig,ax=plt.subplots(figsize=(8,5))
    model_params={'EleutherAI/pythia-70m':70e6,'openai-community/gpt2':124439808,'facebook/opt-125m':125239808}
    for task in sorted(set(r['task'] for r in smoke)):
        q=[r for r in smoke if r.get('task')==task and r.get('model') in model_params and n(r,'accuracy') is not None]
        q.sort(key=lambda r:model_params[r['model']]);x=[model_params[r['model']]/1e6 for r in q];yvals=[n(r,'accuracy') for r in q];err=[1.96*(n(r,'accuracy_stderr') or 0) for r in q];ax.errorbar(x,yvals,yerr=err,marker='o',ms=5,capsize=2,lw=1.6,label=task)
    ax.set_xscale('log');ax.set(ylim=(0,1),xlabel='Parameter count (millions; log scale)',ylabel='Zero-shot accuracy on restricted 100-example subsets',title='Harness smoke checks · not full benchmark results');ax.legend(title='Task');ax.text(.01,.02,'CPU FP32 · 0-shot · 100 examples per task · whiskers are approx. 95% intervals',transform=ax.transAxes,fontsize=7.2,color='#52616B');save(fig,25,'Restricted zero-shot benchmark smoke tests','results/llm/harness_summary.csv','Three checkpoints were evaluated on 100 examples per task with score uncertainty; this is separate from the full Pythia-70M validation evaluation.','CPU FP32 smoke runs on limited examples; no claims about full-split comparative benchmark performance.')
    # Catalogue Markdown and CSV.
    with (PROC/'figure_catalogue.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(catalog[0]),lineterminator='\n');w.writeheader();w.writerows(catalog)
    lines=['# Figure catalogue','','All figures are regenerated by `scripts/plot_final_figures.py` from existing result files. Each figure is saved as PNG and vector SVG. No model execution occurs.','']
    for x in catalog:lines += [f"## Figure {x['number']}. {x['title']}",'',f"- PNG: `{x['file_png']}`",f"- Vector: `{x['file_vector']}`",f"- Data source: `{x['data_source']}`",f"- Script: `{x['analysis_script']}`",f"- Interpretation: {x['interpretation']}",f"- Limitations: {x['limitations']}",'']
    (ROOT/'reports/FIGURE_CATALOGUE.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Generated {len(catalog)} figures in PNG and SVG.')
if __name__=='__main__':main()
