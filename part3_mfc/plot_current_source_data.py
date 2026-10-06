"""Redraw permitted HUP060 public Source Data only; never execute a model."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def draw(number,output):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise RuntimeError('Use a new empty output directory; no frozen export overwrite')
    if number in (7,9):
        name='plot_ablation_original_style.py' if number==7 else 'plot_baselines_original_style.py'
        source=ROOT/('output/part3/source_data/figure_07' if number==7 else 'output/part3/source_data/figure_09_trivial_baselines')
        flags=['--source-dir',str(source),'--output-dir',str(output)] if number==7 else ['--source',str(source),'--output',str(output)]
        return subprocess.run([sys.executable,'-B',str(ROOT/'part3_mfc'/name),*flags],check=False).returncode
    if number==10:
        raise RuntimeError('Default verifies approved raw/median loss exports; exact eight-panel redraw uses plot_loss_approved.py with separately supplied history, never an implicit data open')
    if number not in (6,8):raise ValueError('Unsupported current figure')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    labels=('Observed ictal','Free Graph–RC','Preictal reference','WGAN-GP Full')
    colors=dict(zip(labels,('#272727','#9A9A9A','#3C8D62','#2166AC')));styles=dict(zip(labels,('-',':','-.','-')))
    handles=[Line2D([0],[0],color=colors[label],ls=styles[label],lw=1.1,label=label) for label in labels]
    if number==6:
        source=ROOT/'output/part3/source_data/figure_06'
        density=pd.read_csv(source/'source_data_representative_densities.csv');paths=pd.read_csv(source/'source_data_representative_trajectories.csv');controls=pd.read_csv(source/'source_data_representative_controls.csv')
        fig,axes=plt.subplots(3,3,figsize=(170/25.4,175/25.4))
        for i,ax in enumerate(axes.flat):
            row,col=divmod(i,3);ax.set_position([.105+col*.305,(.800,.455,.207)[row]-(.195,.160,.130)[row],.235,(.195,.160,.130)[row]])
            ax.text(-.17,1.08,'abcdefghi'[i],transform=ax.transAxes,fontweight='bold',fontsize=10)
        for col,channel in enumerate(('RPFa3','RA3','RAFa4')):
            for label in labels:
                path=paths[(paths.node==channel)&(paths.series==label)];d=density[(density.node==channel)&(density.series==label)]
                axes[0,col].plot(path.time_s,path.standardized_amplitude,color=colors[label],lw=.9);axes[1,col].plot(d.standardized_amplitude,d.density,color=colors[label],ls=styles[label],lw=.9)
            command=controls[controls.node==channel];axes[2,col].plot(command.time_s,command.input_value,color=colors[labels[-1]],lw=1)
            axes[0,col].set_title(channel,loc='left',fontweight='bold');axes[0,col].set_xlabel('Time (s)');axes[1,col].set_xlabel('Standardized amplitude');axes[2,col].set_xlabel('Time (s)');axes[1,col].set_yticks([])
        axes[0,0].set_ylabel('Standardized amplitude');axes[1,0].set_ylabel('Density');axes[2,0].set_ylabel('Control input')
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.94),ncol=2,frameon=False);fig.text(.09,.984,'HUP060 run-02: reference-directed control (WGAN-GP Full)',va='top',fontweight='bold',fontsize=10);stem='hup060_actor_wgan_mfc_preview'
    else:
        source=ROOT/'output/part3/source_data/figure_08';density=pd.read_csv(source/'source_data_actor_wgan_all36_densities.csv');metrics=pd.read_csv(source/'source_data_all36_metrics.csv')
        fig,axes=plt.subplots(6,6,figsize=(170/25.4,180/25.4),sharex=True)
        for c,ax in enumerate(axes.flat):
            row,col=divmod(c,6);channel=metrics.iloc[c]['channel'];ax.set_position([.030+col*.160,(180-29.7-row*24.66-17.1)/180,.145,17.1/180]);maximum=0
            for label in labels:
                d=density[(density.channel==channel)&(density.series==label)];ax.plot(d.standardized_amplitude,d.density,color=colors[label],ls=styles[label],lw=.9);maximum=max(maximum,float(d.density.max()))
            ax.set_title(channel,fontsize=9,fontweight='bold');ax.set_ylim(0,maximum*1.5);ax.set_yticks([]);ax.tick_params(labelbottom=row==5)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.947),ncol=4,frameon=False);fig.text(.09,.984,'HUP060: all-contact control (WGAN-GP Full)',va='top',fontweight='bold',fontsize=10);fig.supxlabel('Standardized amplitude',fontsize=9,y=2.16/180);stem='hup060_actor_wgan_all36_distribution_grid'
    output.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','png','svg'):fig.savefig(output/(stem+'.'+extension),dpi=300)
    plt.close(fig)
    print(json.dumps(dict(status='public_SourceData_redrawn',output=str(output),figure=number,model_evaluated=False,training_started=False,visual_redraw_not_replacing_approved_export=True),indent=2));return 0
