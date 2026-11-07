"""Publication-style figures from verified existing result files only."""
import csv, math
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'results/raw'; PROC=ROOT/'results/processed'; LLM=ROOT/'results/llm'; FIG=ROOT/'results/figures/final'; FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':150,'savefig.dpi':240,'axes.grid':True,'grid.alpha':.25,'font.family':'DejaVu Sans'})
catalog=[]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def n(r,k):
    try:x=float(r.get(k,''));return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None
def save(fig,num,title,sources,interpretation,limitations):
    stem=f'figure_{num:02d}'
    fig.tight_layout(); fig.savefig(FIG/(stem+'.png'),bbox_inches='tight');fig.savefig(FIG/(stem+'.svg'),bbox_inches='tight');plt.close(fig)
    catalog.append({'number':num,'title':title,'file_png':f'results/figures/final/{stem}.png','file_vector':f'results/figures/final/{stem}.svg','data_source':sources,'analysis_script':'scripts/plot_final_figures.py','interpretation':interpretation,'limitations':limitations})
def errorbar(ax,x,ys,label):
    vals=[v for v in ys if v is not None]
    if not vals:return
    xx=[x[i] for i,v in enumerate(ys) if v is not None]
    ax.plot(xx,vals,'o-',label=label,linewidth=1.4,markersize=3)

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
        xs=sorted(groups);means=[np.mean(groups[x]) for x in xs];sds=[np.std(groups[x],ddof=1) if len(groups[x])>1 else 0 for x in xs]
        ax.errorbar(xs,means,yerr=sds,marker='o',capsize=2,label=f'{model} (n={len(groups[xs[0]])})')
    ax.set(xlabel='Training epoch',ylabel='Validation accuracy',title='Supervised learning curves on fixed 10,000-example Fashion-MNIST split');ax.legend();save(fig,1,'Supervised validation accuracy versus epoch','results/raw/replicated_scaling_results.csv; results/raw/vit_tiny_scaling.csv','Accuracy generally increases early, with model-specific later changes; the actual ViT-Tiny-dimension run is shown separately from the compact proxy.','Three seeds; standard deviation bars; the 12-block ViT-Tiny has notably variable terminal test results.')
    # 2/3: measured time and measured compute; anomalous interval retained separately.
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for model in sorted(set(r['model_name'] for r in rows)):
        rr=sorted([r for r in rows if r['model_name']==model],key=lambda r:int(float(r['budget_value'])))
        xs=[int(float(r['budget_value'])) for r in rr]
        yy=[]
        for r in rr:
            val=n(r,'cumulative_training_time_seconds')
            if model=='ResNet-18' and str(r['seed'])=='11' and int(float(r['budget_value']))>=3 and val is not None:val=max(0,val-1122.1625285)
            yy.append(val)
        # seed-level traces support visibility of the single-seed adjustment
        if model=='ResNet-18':
            rr11=sorted([r for r in rr if str(r['seed'])=='11'],key=lambda r:int(float(r['budget_value'])))
            axs[1].plot([int(float(r['budget_value'])) for r in rr11],[max(0,n(r,'cumulative_training_time_seconds')-1122.1625285) if int(float(r['budget_value']))>=3 else n(r,'cumulative_training_time_seconds') for r in rr11],':',alpha=.7,label='ResNet-18 seed 11 (adjusted)')
        # plot seed means (unadjusted; only seed11 affected in right panel separately)
        by=defaultdict(list)
        for r in rr:
            val=n(r,'cumulative_training_time_seconds')
            if val is not None:by[int(float(r['budget_value']))].append(val)
        axs[0].plot(sorted(by),[np.mean(by[x]) for x in sorted(by)],'o-',label=model)
        by2=defaultdict(list)
        for r in rr:
            val=n(r,'cumulative_training_time_seconds')
            if val is not None:
                if model=='ResNet-18' and str(r['seed'])=='11' and int(float(r['budget_value']))>=3:val=max(0,val-1122.1625285)
                by2[int(float(r['budget_value']))].append(val)
        axs[1].plot(sorted(by2),[np.mean(by2[x]) for x in sorted(by2)],'o-',label=model)
    axs[0].set(xlabel='Epoch',ylabel='Cumulative logged time (seconds)',title='Raw logged wall-clock time')
    axs[1].set(xlabel='Epoch',ylabel='Cumulative adjusted time (seconds)',title='Efficiency diagnostic excluding ResNet interval')
    for ax in axs:ax.legend()
    save(fig,2,'Training time versus epoch, raw and adjusted','results/raw/replicated_scaling_results.csv','The ResNet epoch-2-to-3 interval dominates raw time; adjusted diagnostic removes 1,122.16 s from seed 11 after epoch 2.','Adjustment is analytical only, not a claim about true compute duration; no process logs identify the pause.')
    # 4 marginal accuracy / second from paired per-seed results.
    marg=read(PROC/'strengthened_marginal_gains.csv'); rr=[r for r in marg if r.get('source_file') in ('replicated_scaling_results.csv','vit_tiny_scaling.csv') and r.get('metric')=='validation_accuracy' and r.get('model_name') in ('MLP','CNN / LeNet','ResNet-18','Vision Transformer (compact ViT-Tiny-style)','ViT-Tiny') and n(r,'marginal_improvement_per_second') is not None]
    fig,ax=plt.subplots(figsize=(8,4.8))
    for model in sorted(set(r['model_name'] for r in rr)):
        g=defaultdict(list)
        for r in rr:
            if r['model_name']==model:g[int(float(r['to_budget']))].append(n(r,'marginal_improvement_per_second'))
        ax.plot(sorted(g),[np.mean(g[x]) for x in sorted(g)],'o-',label=model)
    ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Destination epoch',ylabel='Accuracy-point improvement per logged second',title='Paired marginal validation gain per added time');ax.legend();save(fig,3,'Marginal accuracy improvement per additional compute','results/processed/strengthened_marginal_gains.csv','Marginal returns fluctuate and can be negative.','Per-second metric is sensitive to wall-clock noise; ResNet anomaly interval is excluded in adjusted calculation only if available in summary.')
    # 5/6 scaling by data and RF trees.
    ds=read(PROC/'data_size_scaling_summary.csv');fig,ax=plt.subplots(figsize=(7,4.5))
    for model in sorted(set(r['model_name'] for r in ds)):
        q=sorted([r for r in ds if r['model_name']==model],key=lambda r:int(r['actual_train_size']));ax.errorbar([int(r['actual_train_size']) for r in q],[n(r,'mean') for r in q],yerr=[n(r,'ci95_high')-n(r,'mean') if n(r,'ci95_high') is not None else 0 for r in q],marker='o',capsize=3,label=model)
    ax.set(xlabel='Training examples',ylabel='Test accuracy at 5 epochs',title='Dataset-size scaling (3 seeds; 95% t intervals)');ax.legend();save(fig,4,'Dataset-size scaling for MLP and LeNet','results/processed/data_size_scaling_summary.csv','LeNet increases across the two reported sizes; MLP is not monotone across its available range.','Only 30K and 58K requested-size points; intervals with three seeds are imprecise.')
    rf=[r for r in read(RAW/'random_forest_scaling.csv') if r.get('experiment_status')=='completed'];g=defaultdict(list)
    for r in rf:
        if n(r,'evaluation_value') is not None:g[int(float(r['budget_value']))].append(n(r,'evaluation_value'))
    fig,ax=plt.subplots(figsize=(7,4.5));xs=sorted(g);ax.errorbar(xs,[np.mean(g[x]) for x in xs],yerr=[np.std(g[x],ddof=1) if len(g[x])>1 else 0 for x in xs],marker='o',capsize=3);ax.set(xlabel='Number of trees',ylabel='Validation accuracy',title='Random Forest validation accuracy versus tree count (3 seeds)');save(fig,5,'Random Forest performance versus number of trees','results/raw/random_forest_scaling.csv','The curve provides a native model-size budget comparison.','Warm-start forest runs; no interpretation as neural epochs.')
    # 7-10 unsupervised separate units and metrics.
    u=[r for r in read(RAW/'unsupervised_scaling.csv') if r.get('experiment_status')=='completed'];seen=set();clean=[]
    for r in u:
        key=r.get('experiment_id')
        if r['model_name']=='MiniBatch K-Means' and key in seen:continue
        seen.add(key);clean.append(r)
    for model,numfig,label in [('MiniBatch K-Means',6,'partial_fit updates'),('Gaussian Mixture Model',7,'warm-start EM iterations')]:
        q=[r for r in clean if r['model_name']==model and n(r,'evaluation_value') is not None];g=defaultdict(list)
        for r in q:g[int(float(r['budget_value']))].append(n(r,'evaluation_value'))
        fig,ax=plt.subplots(figsize=(7,4.5));xs=sorted(g);ax.errorbar(xs,[np.mean(g[x]) for x in xs],yerr=[np.std(g[x],ddof=1) if len(g[x])>1 else 0 for x in xs],marker='o',capsize=3);ax.set(xlabel=label,ylabel='Validation adjusted Rand index',title=f'{model}: clustering quality (n=3 seeds)');save(fig,numfig,f'{model} performance versus computational budget','results/raw/unsupervised_scaling.csv','ARI changes over the algorithm-specific iterative budget.','K-Means reused-checkpoint duplicate rows are excluded; no epoch equivalence is implied.')
    db=read(RAW/'dbscan_size_scaling.csv');g=defaultdict(list)
    for r in db:
        if r.get('status')=='completed' and n(r,'validation_ARI') is not None:g[int(float(r['budget_value']))].append(n(r,'validation_ARI'))
    fig,ax=plt.subplots(figsize=(7,4.5));xs=sorted(g);ax.plot(xs,[np.mean(g[x]) for x in xs],'o-');ax.set(xlabel='Samples in DBSCAN fit',ylabel='Validation adjusted Rand index',title='DBSCAN behavior as fitted sample size changes');save(fig,8,'DBSCAN performance versus dataset size','results/raw/dbscan_size_scaling.csv','The size sweep tests data-fit cost with fixed density settings.','Nine observations are parameter and budget conditions; no replicated seed error bars; mostly near-zero ARI per existing records.')
    ae=[r for r in read(RAW/'autoencoder_scaling.csv') if r.get('experiment_status')=='completed'];
    for metric,numfig,label in [('validation_reconstruction_mse',9,'Validation reconstruction MSE'),('downstream_validation_accuracy',10,'Downstream linear-probe accuracy')]:
        fig,ax=plt.subplots(figsize=(7.4,4.5))
        for model in sorted(set(r['model_name'] for r in ae)):
            g=defaultdict(list)
            for r in ae:
                if r['model_name']==model and n(r,metric) is not None:g[int(float(r['budget_value']))].append(n(r,metric))
            xs=sorted(g);ax.errorbar(xs,[np.mean(g[x]) for x in xs],yerr=[np.std(g[x],ddof=1) if len(g[x])>1 else 0 for x in xs],marker='o',capsize=2,label=model)
        ax.set(xlabel='Training epochs',ylabel=label,title=label+' by training budget');ax.legend();save(fig,numfig,label+' versus training budget','results/raw/autoencoder_scaling.csv',f'{label} is reported separately for each representation model.','Three seeds; reconstruction and downstream metrics answer different questions.')
    # SSL label fraction (final common 30 epoch) and training budget curves.
    ssl=[r for r in read(RAW/'ssl_algorithms_test.csv') if r.get('status')=='completed' and n(r,'test_accuracy') is not None]
    g=defaultdict(list)
    for r in ssl:
        if int(float(r.get('epochs') or 0))==30:g[(r.get('model_name'),float(r.get('labeled_fraction')))].append(n(r,'test_accuracy'))
    fig,ax=plt.subplots(figsize=(8,5))
    for model in sorted(set(k[0] for k in g)):
        xs=sorted(k[1] for k in g if k[0]==model);ax.plot([100*x for x in xs],[np.mean(g[(model,x)]) for x in xs],'o-',label=model)
    ax.set(xlabel='Labeled training data (%)',ylabel='Official test accuracy',title='Semi-supervised endpoint performance by labeled fraction (30 epochs)');ax.legend();save(fig,11,'Semi-supervised performance versus labeled fraction','results/raw/ssl_algorithms_test.csv','Algorithm and label-fraction effects are visible at the shared terminal budget.','Three seeds per condition; supervised baseline only where matched and present in raw result tables.')
    # SSL curves: all methods separate to avoid combining data regimes.
    sr=[r for r in read(RAW/'ssl_algorithms.csv') if r.get('status')=='completed' and r.get('labeled_fraction')=='0.1'];fig,ax=plt.subplots(figsize=(8,5))
    for model in sorted(set(r['model_name'] for r in sr)):
        q=[r for r in sr if r['model_name']==model];g=defaultdict(list)
        for r in q:
            if n(r,'validation_accuracy') is not None:g[int(float(r['budget_value']))].append(n(r,'validation_accuracy'))
        xs=sorted(g);ax.plot(xs,[np.mean(g[x]) for x in xs],'o-',label=model)
    ax.set(xlabel='Training epochs',ylabel='Validation accuracy',title='Semi-supervised training-budget comparison (10% labels)');ax.legend();save(fig,12,'Semi-supervised performance versus training budget','results/raw/ssl_algorithms.csv','Methods have distinct trajectories at a common label fraction.','Three seeds; task adaptations, not canonical implementations on standard SSL benchmark splits.')
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
            xs=sorted(g);axs[0].errorbar(xs,[np.mean(g[x]) for x in xs],yerr=[np.std(g[x],ddof=1) if len(g[x])>1 else 0 for x in xs],marker='o',capsize=2,label=model)
            tt=[(np.mean([v[0] for v in t[x]]),np.mean([v[1] for v in t[x]])) for x in xs if t[x]]
            if tt:axs[1].plot([v[0] for v in tt],[v[1] for v in tt],'o-',label=model)
        axs[0].set(xlabel='Environment interactions (steps)',ylabel='Mean return (20 deterministic evaluation episodes)',title=f'{env}: reward vs steps');axs[1].set(xlabel='Cumulative training time (s)',ylabel='Mean return',title=f'{env}: reward vs elapsed time')
        for ax in axs:ax.legend()
        save(fig,fig_no,f'{env} reward versus steps and time','results/raw/rl_scaling_results.csv',f'Algorithm-specific return trajectories are shown for {env}.','Seed counts differ by algorithm; separate environments and reward scales; logged elapsed time is hardware/protocol specific.')
    # LLM full test endpoint and prefix curves / zero-shot separate protocol plots.
    full=read(LLM/'macos_full_wikitext2_evaluations.csv');full=[r for r in full if r.get('status')=='completed' and n(r,'parameter_count') and n(r,'perplexity')]
    fig,ax=plt.subplots(figsize=(7.5,4.7));ax.scatter([n(r,'parameter_count')/1e6 for r in full],[n(r,'perplexity') for r in full])
    for r in full:ax.annotate(r['model_name'],(n(r,'parameter_count')/1e6,n(r,'perplexity')),xytext=(4,4),textcoords='offset points',fontsize=7)
    ax.set_xscale('log');ax.set(yscale='log',xlabel='Parameters (millions; log scale)',ylabel='WikiText-2 raw-v1 test perplexity (log scale)',title='Full test-set checkpoint perplexity (MPS FP32, context 1,024, stride 512)');save(fig,15,'Full WikiText-2 perplexity versus parameter count','results/llm/macos_full_wikitext2_evaluations.csv','Larger checkpoints in the Pythia series have lower measured perplexity under the matched local evaluation protocol.','Six checkpoints across different model families; not matched pretraining compute, and tokenizer differences remain.')
    pref=read(LLM/'expanded_llm_evaluations.csv');pref=[r for r in pref if r.get('status')=='completed' and r.get('benchmark')=='WikiText-2 raw test' and n(r,'training_step') is not None and n(r,'test_loss') is not None]
    for modelname,fig_no in [('Pythia-70M',16),('Pythia-160M',17)]:
        q=[r for r in pref if r.get('model')==modelname and r.get('checkpoint_id') not in ('main',)];g=defaultdict(list)
        for r in q:g[int(float(r['training_step']))].append(n(r,'test_loss'))
        fig,ax=plt.subplots(figsize=(7.5,4.5));xs=sorted(g);ax.plot(xs,[np.mean(g[x]) for x in xs],'o-');ax.set(xlabel='Released checkpoint training step',ylabel='Loss on restricted prefix evaluation',title=f'{modelname} checkpoint-prefix trajectory (CPU FP32)');save(fig,fig_no,f'{modelname} checkpoint-prefix evaluation trajectory','results/llm/expanded_llm_evaluations.csv','Loss is evaluated at released checkpoints, including step 0, on a fixed restricted local text prefix.','Single forward evaluations of selected released checkpoints on a small raw-text prefix; step-to-token mapping is published metadata, not measured local training compute.')
    harness=read(LLM/'harness_full_summary.csv');fig,ax=plt.subplots(figsize=(7.5,4.5));
    for r in harness:
        if n(r,'metric_accuracy') is not None:ax.bar(r['task'],n(r,'metric_accuracy'),label=f"n={r.get('evaluated_examples')}")
    ax.set(ylim=(0,1),ylabel='Zero-shot accuracy',title='Pythia-70M full validation-split harness results');save(fig,18,'Zero-shot benchmark results (full validation splits)','results/llm/harness_full_summary.csv','Three full validation tasks were evaluated without few-shot examples.','One model, three tasks; task scores are not pooled; smoke-test subsets are excluded.')
    # Within-paradigm normalized cross-paradigm panel: relative change from first point only.
    fig,axs=plt.subplots(2,2,figsize=(10,7));
    panels=[('Supervised validation accuracy', [(r['model_name'],int(float(r['budget_value'])),n(r,'validation_accuracy')) for r in rows]),('Semi-supervised accuracy',[(r['model_name'],int(float(r['budget_value'])),n(r,'validation_accuracy')) for r in sr]),('RL return (CartPole)',[(r['model_name'],int(float(r['environment_steps'])),n(r,'mean_return')) for r in rl if r.get('dataset')=='CartPole-v1']),('Representation reconstruction MSE',[(r['model_name'],int(float(r['budget_value'])),n(r,'validation_reconstruction_mse')) for r in ae])]
    for ax,(title,triples) in zip(axs.flat,panels):
        gg=defaultdict(dict)
        for name,b,v in triples:
            if v is not None:gg[name][b]=v
        for name,d in gg.items():
            xs=sorted(d);base=d[xs[0]]
            if not base:continue
            vals=[(d[x]-base)/abs(base)*100 for x in xs];ax.plot(range(len(xs)),vals,'o-',label=name)
        ax.axhline(0,color='black',lw=.7);ax.set(title=title,xlabel='Ordered native budget (index)',ylabel='Change from first point (%)');ax.legend(fontsize=6)
    fig.suptitle('Within-paradigm normalized change; panels retain distinct metrics and native budgets');save(fig,19,'Cross-paradigm within-task normalized changes','multiple verified raw result tables','Each panel normalizes a metric only within its own series; no pooled score or shared compute axis.','Index axis intentionally avoids pretending epochs, steps, and rewards are equivalent; percent change can be unstable near zero or for negative returns.')
    # Additional required views that retain matched budget/protocol boundaries.
    fig,ax=plt.subplots(figsize=(8,5))
    for model in sorted(set(r['model_name'] for r in rows)):
        q=[r for r in rows if r['model_name']==model];g=defaultdict(list)
        for r in q:
            epoch=int(float(r['budget_value']));tm=n(r,'cumulative_training_time_seconds')
            if tm is not None and model=='ResNet-18' and str(r['seed'])=='11' and epoch>=3:tm=max(0,tm-1122.1625285)
            if tm is not None and n(r,'validation_accuracy') is not None:g[epoch].append((tm,n(r,'validation_accuracy')))
        xs=sorted(g);ax.plot([np.mean([v[0] for v in g[x]]) for x in xs],[np.mean([v[1] for v in g[x]]) for x in xs],'o-',label=model)
    ax.set(xlabel='Cumulative training time (seconds; ResNet adjusted)',ylabel='Validation accuracy',title='Supervised validation performance versus cumulative measured time');ax.legend();save(fig,20,'Supervised performance versus cumulative compute time','results/raw/replicated_scaling_results.csv','Mean validation accuracy is plotted against mean cumulative wall time at each epoch.','Time is elapsed runtime, not energy/FLOPs; ResNet seed-11 pause-like interval is removed only in this adjusted diagnostic.')
    pseudo=read(RAW/'pseudolabel_test_results.csv');pseudo=[r for r in pseudo if r.get('status')=='completed' and int(float(r.get('epochs') or 0))==5 and n(r,'test_accuracy') is not None]
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for method,label in [('labeled_only','Labeled-only MLP baseline'),('pseudo_label','Pseudo-labeling')]:
        g=defaultdict(list)
        for r in pseudo:
            if r.get('method')==method:g[float(r['labeled_fraction'])].append(n(r,'test_accuracy'))
        xs=sorted(g);ax.errorbar([100*x for x in xs],[np.mean(g[x]) for x in xs],yerr=[np.std(g[x],ddof=1) if len(g[x])>1 else 0 for x in xs],marker='o',capsize=3,label=label)
    ax.set(xlabel='Labeled training data (%)',ylabel='Official test accuracy at epoch 5',title='Pseudo-labeling versus matched labeled-only baseline');ax.legend();save(fig,21,'Semi-supervised method versus supervised baseline','results/raw/pseudolabel_test_results.csv','Pseudo-labeling is compared with the labeled-only MLP under matched fractions, seeds, and terminal epoch.','Three seeds per fraction; this ablation is separate from the 30-epoch Mean Teacher/FixMatch/MixMatch runs.')
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for model in sorted(set(r['model_name'] for r in ae)):
        g=defaultdict(list)
        for r in ae:
            if r['model_name']!=model:continue
            tm=n(r,'cumulative_training_time_seconds');acc=n(r,'downstream_validation_accuracy')
            if tm is not None and acc is not None:g[int(float(r['budget_value']))].append((tm,acc))
        xs=sorted(g);ax.plot([np.mean([v[0] for v in g[x]]) for x in xs],[np.mean([v[1] for v in g[x]]) for x in xs],'o-',label=model)
    ax.set(xlabel='Cumulative training time (seconds)',ylabel='Downstream linear-probe validation accuracy',title='Representation quality versus training cost');ax.legend();save(fig,22,'Representation quality versus reconstruction-model training cost','results/raw/autoencoder_scaling.csv','Linear-probe accuracy is plotted against measured cumulative training time for each representation model.','Three seeds; the probe is a separate downstream assessment and does not represent a universal representation metric.')
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for r in full:
        ax.scatter(n(r,'evaluation_seconds'),n(r,'perplexity'));ax.annotate(r['model_name'],(n(r,'evaluation_seconds'),n(r,'perplexity')),xytext=(4,4),textcoords='offset points',fontsize=7)
    ax.set(xlabel='Full-test evaluation time (seconds; Apple M3 MPS)',ylabel='WikiText-2 raw-v1 test perplexity',title='Full-test perplexity versus evaluation time');save(fig,23,'Perplexity versus comparable full-test evaluation time','results/llm/macos_full_wikitext2_evaluations.csv','The same local device, precision, context, stride, and corpus protocol permit descriptive inference-time comparison.','Tokenizer-specific scored token counts differ; evaluation time is not original training compute and each model has one run.')
    fig,ax=plt.subplots(figsize=(7.5,4.8));ax.scatter([n(r,'parameter_count')/1e6 for r in full],[n(r,'negative_log_likelihood') for r in full])
    for r in full:ax.annotate(r['model_name'],(n(r,'parameter_count')/1e6,n(r,'negative_log_likelihood')),xytext=(4,4),textcoords='offset points',fontsize=7)
    ax.set_xscale('log');ax.set(xlabel='Parameters (millions; log scale)',ylabel='Mean negative log likelihood (nats/token)',title='Full-test evaluation loss versus model size');save(fig,24,'Full-test evaluation loss versus parameter count','results/llm/macos_full_wikitext2_evaluations.csv','Loss and parameter count are plotted for the same six local checkpoint evaluations.','Cross-family training data and tokenization differ; one evaluation per model; size is not training compute.')
    smoke=read(LLM/'harness_summary.csv');fig,ax=plt.subplots(figsize=(8,5))
    model_params={'EleutherAI/pythia-70m':70e6,'openai-community/gpt2':124439808,'facebook/opt-125m':125239808}
    for task in sorted(set(r['task'] for r in smoke)):
        q=[r for r in smoke if r.get('task')==task and r.get('model') in model_params and n(r,'accuracy') is not None]
        q.sort(key=lambda r:model_params[r['model']]);ax.plot([model_params[r['model']]/1e6 for r in q],[n(r,'accuracy') for r in q],'o-',label=task)
    ax.set_xscale('log');ax.set(ylim=(0,1),xlabel='Parameters (millions; log scale)',ylabel='Zero-shot accuracy on 100-example smoke subset',title='Restricted harness smoke checks (not full benchmarks)');ax.legend();save(fig,25,'Restricted zero-shot benchmark smoke tests','results/llm/harness_summary.csv','Three checkpoints were evaluated on 100 examples per task; this view is explicitly separate from the full Pythia-70M validation evaluation.','CPU FP32 smoke runs, limited examples; no claims about full-split comparative benchmark performance.')
    # Catalogue Markdown and CSV.
    with (PROC/'figure_catalogue.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(catalog[0]));w.writeheader();w.writerows(catalog)
    lines=['# Figure catalogue','','All figures are regenerated by `scripts/plot_final_figures.py` from existing result files. Each figure is saved as PNG and vector SVG. No model execution occurs.','']
    for x in catalog:lines += [f"## Figure {x['number']}. {x['title']}",'',f"- PNG: `{x['file_png']}`",f"- Vector: `{x['file_vector']}`",f"- Data source: `{x['data_source']}`",f"- Script: `{x['analysis_script']}`",f"- Interpretation: {x['interpretation']}",f"- Limitations: {x['limitations']}",'']
    (ROOT/'FIGURE_CATALOGUE.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Generated {len(catalog)} figures in PNG and SVG.')
if __name__=='__main__':main()
