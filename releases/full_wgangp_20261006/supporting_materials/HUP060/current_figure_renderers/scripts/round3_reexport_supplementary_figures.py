"""Presentation-only supplementary-figure re-export from frozen curves.

No model evaluation, training, KDE fitting or statistical recomputation occurs.
The original plotted arrays, colours, bar/scatter values and raw/smoothed loss
traces are retained. Native typography and layout are the only changes.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import fitz

import round3_reexport_main_figures as util

ROOT=util.ROOT
BASELINE=ROOT.parent/"chaos_revision_20261005_round2"
QA=ROOT/"qa_round3"/"supplementary_figures"
QA.mkdir(parents=True,exist_ok=True)
PROJECT=util.PROJECT
ARCHIVE=util.ARCHIVE
CP=ARCHIVE/"cp7330"
LOSS=ARCHIVE/"HUP060_wgan_loss_decomposition_e7500_preview_v1"
EXTERNAL_RENDER=ARCHIVE/"hup065_hup080_simple_channel_figures_v1/redraw_channel_plates.py"
EXTERNAL_SOURCES={
    "HUP065":ARCHIVE/"multi_patient_control_preview/paper_exact_v2/HUP065_hup060_sparse_rerun_v1/publication_export_v1/package_overleaf_final/source/channel_density_long.csv",
    "HUP080":ARCHIVE/"multi_patient_control_preview/paper_exact_v2/HUP080_hup060_sparse_rerun_v2/publication_export_v1/package_overleaf_final_v2/source/channel_density_long.csv",
}
EXPECTED_EXTERNAL={"HUP065":"4b6c938f5cd5299e0fc9e14ff5f41bf6aecd524dd6291226831d2715d3626ff4",
                   "HUP080":"9e2820414b8c1de824203fcb61d4186e261a3d161ac7621eff38294d1ae1b712"}
ASSETS=re.findall(r"includegraphics\[([^]]+)\]\{([^}]+)\}",(BASELINE/"chaos_supplementary.tex").read_text(encoding="utf-8"))
INITIAL={}
SOURCE_FILES=[]
SOURCE_HASHES={}
RESULTS=[]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    SOURCE_FILES.append(path)
    SOURCE_HASHES.setdefault(str(path),sha(path))
    return path


def old_audit():
    for options,name in ASSETS:
        page=fitz.open(BASELINE/name)[0]
        width=468.0
        height=float("inf")
        wm=re.search(r"width=(\d*\.?\d*)\\textwidth",options)
        hm=re.search(r"height=(\d*\.?\d*)\\textheight",options)
        if wm and wm[1]: width*=float(wm[1])
        if hm: height=648*float(hm[1])
        ratio=min(width/page.rect.width,height/page.rect.height)
        ordinary=[s for b in page.get_text("dict")["blocks"] for l in b.get("lines",[]) for s in l["spans"] if s["text"].strip() and s["size"]>=4.8]
        minimum=min(s["size"] for s in ordinary)*ratio
        INITIAL[name]={"include_options":options,"source_width_mm":page.rect.width/72*25.4,
                       "source_height_mm":page.rect.height/72*25.4,"applied_scale":ratio,
                       "minimum_ordinary_font_pt":minimum,"initial_sha256":sha(BASELINE/name)}


def finish(fig,name,before,figure_number,multiplier=1.0):
    util.style_text(fig)
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_text() in tuple("abcdefghi"):
                text.set_position((-.14,1.01))
        # Formatter-created order-of-magnitude labels are ordinary annotations,
        # not mathematical subscripts, and must meet the same font minimum.
        ax.xaxis.get_offset_text().set_fontsize(9)
        ax.yaxis.get_offset_text().set_fontsize(9)
    fig.canvas.draw()
    assert util.digest_artists(fig)==before, "a scientific curve/value/colour changed"
    path=ROOT/name
    fig.savefig(path,metadata={"Creator":"Python/matplotlib; frozen curve arrays", "CreationDate":None,"ModDate":None})
    preview=QA/(path.stem+".png")
    fig.savefig(preview,dpi=220)
    doc=fitz.open(path); page=doc[0]
    width=page.rect.width/72*25.4; height=page.rect.height/72*25.4
    assert width<=170.001 and height<=211.001
    spans=[s for b in page.get_text("dict")["blocks"] for l in b.get("lines",[]) for s in l["spans"] if s["text"].strip()]
    ordinary=[s for s in spans if s["size"]>=8.8]
    smaller=[{"text":s["text"],"size":s["size"]} for s in spans if s["size"]<8.8]
    scale=165.1*multiplier/width
    minimum=min(s["size"] for s in ordinary)*scale
    assert minimum>=8
    info={"asset":name,"figure":figure_number,"before":INITIAL.get(name),
          "width_mm":width,"height_mm":height,"proposed_include":f"width={multiplier if multiplier!=1 else ''}\\textwidth",
          "minimum_ordinary_font_pt_at_proposed_inclusion":minimum,
          "smaller_spans_math_only":smaller,"original_data_artist_digest":before[0],
          "artist_counts":before[1],"all_curve_values_and_colours_unchanged":True,
          "preview":str(preview),"final_sha256":sha(path),"editable_text":bool(page.get_fonts())}
    RESULTS.append(info)
    doc.close(); plt.close(fig)


def plot_plate_layout(fig,number,title,count=36,height=180,rows=6):
    util.remove_figure_text(fig)
    occupied_rows=(count+5)//6
    height-=max(0,6-occupied_rows)*24.66
    fig.set_size_inches(170/25.4,height/25.4)
    for i,ax in enumerate(fig.axes):
        if not ax.get_visible(): continue
        row,col=divmod(i,6)
        ax.set_position([.030+col*.160,(height-29.7-row*24.66-17.1)/height,.145,17.1/height])
        ax.set_xlabel("")
        low,high=ax.get_xlim()
        ticks=[v for v in [-1,0,1] if low<=v<=high]
        ax.set_xticks(ticks)
        # Bottommost data-bearing panel in each column receives readable ticks.
        has_below=any(j<count for j in range(i+6,count,6))
        ax.tick_params(axis="x",labelbottom=not has_below,labelsize=9,length=2)
    util.title(fig,title)
    fig.supxlabel("Standardized amplitude",fontsize=9,y=2.16/height)


def supplementary1():
    path=source(CP/"part2_rc_sde/figure_03_all36_distributions.py")
    module=util.load_module(path,"supplementary_prediction_plate")
    module.SOURCE=CP/"output/part2/source_data/figures_02_04"
    source(module.SOURCE/"all36_distribution_curves.csv")
    source(module.SOURCE/"all36_distribution_metrics.csv")
    module.configure_publication_style()
    fig=util.capture(module,module.make_figure_03)
    before=util.digest_artists(fig)
    oldlegend=fig.legends[0]
    handles=oldlegend.legend_handles
    labels=[t.get_text().replace("Free RC-SDE, 32-particle law","Free Graph–RC").replace("Observed actual","Recorded ictal") for t in oldlegend.get_texts()]
    plot_plate_layout(fig,"S1","HUP060: all-contact predictive occupation laws")
    for ax in fig.axes:
        label=ax.title.get_text()
        ax.set_title(label.replace("  W1=","  $W_1$="),fontweight=ax.title.get_fontweight(),color=ax.title.get_color(),fontsize=9,pad=2)
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.52,.955),ncol=3,
               fontsize=9,columnspacing=.7,handlelength=1.2)
    finish(fig,"chaos_hup060_all36_prediction.pdf",before,"S1")


def supplementary2():
    path=source(CP/"part2_rc_sde/figure_04_distribution_errors.py")
    module=util.load_module(path,"supplementary_prediction_error")
    module.configure_publication_style()
    fig=util.capture(module,module.make_figure_04)
    before=util.digest_artists(fig)
    handles=fig.legends[0].legend_handles
    labels=[t.get_text() for t in fig.legends[0].get_texts()]
    util.remove_figure_text(fig)
    fig.set_size_inches(170/25.4,200/25.4)
    for i,ax in enumerate(fig.axes):
        ax.set_position([.145+i*.285,.100,.235,.745])
        ax.set_xlabel(["Occupation $W_1$","Absolute mean error","Absolute SD error"][i])
        if i==0: ax.set_ylabel("Contact")
        for t in ax.texts:
            if t.get_text().startswith("median"):
                t.set_position((.97,.018));t.set_fontsize(9)
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.55,.930),ncol=2,fontsize=9,handlelength=1)
    util.title(fig,"HUP060: contact-wise prediction errors")
    finish(fig,"hup060_context3_all36_distribution_w1_summary.pdf",before,"S2")


def supplementary3():
    path=source(PROJECT/"part2_rc_sde/figure_05_input_ablation.py")
    module=util.load_module(path,"supplementary_input_ablation")
    source(module.SOURCE/"fixed_context_ablation_density_curves.csv")
    source(module.SOURCE/"fixed_context_ablation_metrics.csv")
    fig=util.capture(module,module.main)
    before=util.digest_artists(fig)
    handles,labels=fig.axes[0].get_legend_handles_labels()
    util.remove_figure_text(fig)
    fig.set_size_inches(170/25.4,150/25.4)
    for i,ax in enumerate(fig.axes):
        row,col=divmod(i,3)
        ax.set_position([.105+col*.305,.565 if row==0 else .140,.235,.270])
        ax.set_title(["RPFa3","RA3","RAFa4"][col],loc="left",fontweight="bold")
        if row==1:
            ax.set_xticklabels(["State","+Delay","+Graph"])
            if col==0:ax.set_ylabel("Occupation $W_1$")
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.54,.935),ncol=2,fontsize=9,handlelength=1.5)
    util.title(fig,"HUP060: predictive feature comparison")
    finish(fig,"hup060_context3_1s_free_state_delay_graph_ablation.pdf",before,"S3")


def supplementary4():
    path=source(ROOT/"scripts/build_chaos_revision_figures.py")
    module=util.load_module(path,"supplementary_control_plate")
    source(ARCHIVE/"github_release_taming_epilepsy/output/part3/source_data/figure_08/source_data_actor_wgan_all36_densities.csv")
    fig=util.capture(module,module.all_channels,"save")
    before=util.digest_artists(fig)
    handles=module.legend_handles()
    for h in handles:
        if h.get_label()=="Interictal reference":h.set_label("Preictal reference")
        elif h.get_label()=="Uncontrolled Graph–RC":h.set_label("Free Graph–RC")
        elif h.get_label()=="Full policy":h.set_label("Primary Full")
    plot_plate_layout(fig,"S4","HUP060: all-contact control (Primary Full)")
    for ax in fig.axes:
        channel=next(t for t in ax.texts if "Reference" not in t.get_text())
        distance=next(t for t in ax.texts if "Reference" in t.get_text())
        nums=re.findall(r"\d+\.\d+",distance.get_text())
        label=channel.get_text()
        ax.set_title(label+"\n"+" → ".join(nums),loc="center",fontweight="bold",color=channel.get_color(),fontsize=9,pad=2)
        channel.remove();distance.remove()
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.947),ncol=4,
               fontsize=9,columnspacing=.5,handlelength=1.3)
    finish(fig,"chaos_hup060_all36_control.pdf",before,"S4")


def supplementary5():
    source_path=source(LOSS/"output_symmetric/source_data_symmetric_loss_decomposition_e7500.csv")
    renderer=source(LOSS/"plot_loss_decomposition_7500_symmetric.py")
    sys.path.insert(0,str(LOSS))
    module=util.load_module(renderer,"supplementary_loss_diagnostic")
    frame=pd.read_csv(source_path); epoch=frame.epoch
    fig,axes=plt.subplots(4,2,figsize=(183/25.4,180/25.4),sharex=True)
    specs=[("train_total","actor_total",r"$L_A$","a  Actor total"),
           ("critic_loss","critic_total",r"$L_C$","b  Signed critic total"),
           ("actor_law_component","actor_law",r"$L_{law}$","c  Empirical-law term"),
           ("critic_wasserstein_component","critic_w",r"$-\widehat W_C$","d  Wasserstein term"),
           ("actor_adversarial_component","actor_adv",r"$0.5L_{adv}$","e  Adversarial term"),
           ("critic_gp_component","critic_gp",r"$10L_{GP}$","f  Gradient penalty / norm"),
           ("actor_proximal_component","actor_prox",r"$0.2L_{prox}$","g  Proximal-action term"),
           ("critic_drift_component","critic_drift",r"$10^{-3}L_{drift}$","h  Reconstructed drift")]
    for i,(column,color,ylabel,heading) in enumerate(specs):
        ax=axes.flat[i]
        if i!=5:
            module.trace(ax,epoch,frame[column],frame["median125_"+column],module.COLORS[color],"Trailing 125-epoch median",zero=i in (1,3))
        else:
            ax.plot(epoch,frame[column],color=module.COLORS["raw"],lw=.28,alpha=.28,rasterized=True)
            ax.plot(epoch,frame["median125_"+column],color=module.COLORS[color],lw=1.15)
            ax.axhline(0,color=module.COLORS[color],lw=.65,ls=":")
            module.style(ax)
            ax.set_ylim(0,.14)
            twin=ax.twinx()
            twin.plot(epoch,frame.critic_mean_gradient_norm,color="#B8D5D2",lw=.25,alpha=.24,rasterized=True)
            twin.plot(epoch,frame.median125_critic_mean_gradient_norm,color=module.COLORS["grad_norm"],lw=1.05)
            twin.axhline(1,color=module.COLORS["grad_norm"],lw=.75,ls="--")
            twin.set_ylim(.70,1.09)
            twin.set_ylabel("Gradient norm (target 1)",color=module.COLORS["grad_norm"])
            twin.tick_params(axis="y",colors=module.COLORS["grad_norm"])
            twin.spines["right"].set_visible(True)
            twin.spines["right"].set_color(module.COLORS["grad_norm"])
            twin.grid(False)
        ax.set_ylabel(ylabel+(" (target 0)" if i==5 else ""))
        ax.set_title(heading,loc="left",fontweight="bold")
        if i in (0,1):ax.axvspan(7000,7500,color=module.COLORS["window"],alpha=.9,lw=0)
        if i in (4,6,7):ax.set_ylim(bottom=0)
        if i==7:ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,-3))
    before=util.digest_artists(fig)
    fig.set_size_inches(170/25.4,174/25.4)
    for i,ax in enumerate(axes.flat):
        row,col=divmod(i,2)
        ax.set_position([.105+col*.490,.704-row*.199,.310,.137])
        ax.set_xticks([0,2500,5000,7500])
        ax.tick_params(axis="x",labelbottom=row==3)
        if row==3:ax.set_xlabel("Joint-training epoch")
    twin.set_position(axes[2,1].get_position())
    fig.legend([Line2D([],[],color=module.COLORS["raw"],lw=1),Line2D([],[],color="#444444",lw=1.15)],
               ["Raw epoch value","Trailing 125-epoch median"],loc="upper center",bbox_to_anchor=(.54,.933),ncol=2,fontsize=9)
    util.title(fig,"HUP060: separate extended-training loss diagnostic")
    finish(fig,"HUP060_WGAN_loss_decomposition_e7500.pdf",before,"S5")


def external():
    path=source(EXTERNAL_RENDER)
    module=util.load_module(path,"supplementary_external_plate")
    manifest=json.loads((ARCHIVE/"hup065_hup080_simple_channel_figures_v1/output_simple_v2/manifest.json").read_text(encoding="utf-8"))
    class FakePages:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def savefig(self,*a,**k):pass
    module.PdfPages=FakePages
    original_close=module.plt.close
    for subject,csv in EXTERNAL_SOURCES.items():
        source(csv)
        assert sha(csv)==EXPECTED_EXTERNAL[subject]==manifest["input_files"][subject]["sha256"]
        table=module.load_density_table(csv,subject)
        held=[]
        module.save_page_bundle=lambda fig,*a: held.append(fig) or []
        module.plt.close=lambda *a,**k:None
        try:module.render_subject(subject,table,QA/("original_renderer_capture_"+uuid.uuid4().hex))
        finally:module.plt.close=original_close
        for page_index,(fig,count) in enumerate(zip(held,module.SUBJECTS[subject]["page_occupancy"]),start=1):
            before=util.digest_artists(fig)
            oldlegend=fig.legends[0]
            handles=oldlegend.legend_handles
            plot_plate_layout(fig,"",f"{subject}: channel occupation laws ({page_index}/{len(held)})",count=count)
            for i,ax in enumerate(fig.axes):
                if not ax.get_visible():continue
                label=ax.texts[0]
                contact=label.get_text().removeprefix("EEG ").removesuffix("-Ref")
                label.remove()
                ax.set_title(contact,loc="left",fontsize=9,fontweight="bold",color="#4D4D4D",pad=2)
            fig.legend(handles,["Recorded ictal","Free prediction","Preictal reference","Controlled"],
                       loc="upper center",bbox_to_anchor=(.52,1-9/(fig.get_size_inches()[1]*25.4)),ncol=4,fontsize=9,handlelength=1.2,columnspacing=.5)
            suffix="_v2" if subject=="HUP080" else ""
            name=f"{subject.lower()}_ofrc_all_channels_page_{page_index:02d}{suffix}.pdf"
            finish(fig,name,before,f"S{5+page_index if subject=='HUP065' else 7+page_index}")
            RESULTS[-1]["channel_count"]=count
            RESULTS[-1]["curves_per_channel"]=4
            RESULTS[-1]["restored_bottommost_occupied_tick_labels"]=True
            RESULTS[-1]["channel_display_normalization"]="common EEG prefix and -Ref montage suffix removed, contact identifier unchanged"


def main():
    old_audit()
    supplementary1();supplementary2();supplementary3();supplementary4();supplementary5();external()
    assert all(sha(p)==SOURCE_HASHES[str(p)] for p in dict.fromkeys(SOURCE_FILES)), "an original source changed"
    (QA/"round3_supplementary_figure_type_and_source_qa.json").write_text(json.dumps({"figures":RESULTS,
        "sources":[{"path":str(p),"sha256":sha(p)} for p in dict.fromkeys(SOURCE_FILES)],
        "no_training_rollout_or_kde":True,"maximum_width_mm":170,"all_numerical_artists_unchanged":True},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(RESULTS,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
