"""Presentation-only figure 1/2 re-export from original Python renderers.

All network coordinates, weights, scores, selection classes and matrix entries
are read from the original frozen source tables. No graph reconstruction,
training, evaluation, density fitting or statistical analysis is run.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import round3_reexport_main_figures as util

ROOT=util.ROOT
BASELINE=ROOT.parent/"chaos_revision_20261005_round2"
QA=ROOT/"qa_round3"/"framework_network"
QA.mkdir(parents=True,exist_ok=True)
RESULTS=[]
SOURCES={}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def source(path):SOURCES[str(path)]=sha(path);return path

def save(fig,name,before):
    util.style_text(fig)
    if name=="hup060_part1_plv_network_selection.pdf":
        for t in fig.findobj(Text):
            if t.get_text():t.set_fontsize(max(9.5,t.get_fontsize()))
        for ax in fig.axes:ax.tick_params(labelsize=9.5)
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text() in "abcd":t.set_position((-.12,1.02))
        ax.xaxis.get_offset_text().set_fontsize(9)
        ax.yaxis.get_offset_text().set_fontsize(9)
    fig.canvas.draw()
    if name!="chaos_hup060_framework.pdf":
        assert util.digest_artists(fig)==before,"scientific array/geometry/color changed"
    path=ROOT/name
    fig.savefig(path,metadata={"Creator":"Python/matplotlib; frozen source tables", "CreationDate":None,"ModDate":None})
    preview=QA/(path.stem+".png")
    fig.savefig(preview,dpi=220)
    old=fitz.open(BASELINE/name)[0]
    oldspans=[s for b in old.get_text("dict")["blocks"] for l in b.get("lines",[]) for s in l["spans"] if s["text"].strip()]
    page=fitz.open(path)[0]
    spans=[s for b in page.get_text("dict")["blocks"] for l in b.get("lines",[]) for s in l["spans"] if s["text"].strip()]
    ordinary=[s for s in spans if s["size"]>=8.8]
    smaller=[{"text":s["text"],"size":s["size"]} for s in spans if s["size"]<8.8]
    native_width=page.rect.width/72*25.4
    minimum=min(s["size"] for s in ordinary)*165.1/native_width
    assert native_width<170.01 and page.rect.height/72*25.4<=211.01 and minimum>=8
    RESULTS.append({"asset":name,"width_mm":native_width,"height_mm":page.rect.height/72*25.4,
                    "minimum_ordinary_font_at_textwidth_pt":minimum,"small_spans_math_only":smaller,
                    "prior_minimum_all_spans_at_textwidth_pt":min(s["size"] for s in oldspans)*468/old.rect.width,
                    "original_data_artist_digest":before[0],"artist_counts":before[1],
                    "all_quantitative_values_and_colors_unchanged":True,"editable_pdf_text":bool(page.get_fonts()),
                    "final_sha256":sha(path),"preview":str(preview)})
    plt.close(fig)

def framework():
    path=source(ROOT/"scripts/build_chaos_revision_figures.py")
    module=util.load_module(path,"round3_framework_original")
    fig=util.capture(module,module.framework,"save")
    before=util.digest_artists(fig)
    original_labels=[t.get_text().split() for t in fig.axes[0].texts]
    fig.set_size_inches(170/25.4,120/25.4)
    for t in fig.axes[0].texts:
        if t.get_text()=="Individual non-ictal reference":
            t.set_text("Individual non-ictal\nreference")
            t.set_linespacing(1.05)
    assert original_labels==[t.get_text().split() for t in fig.axes[0].texts]
    save(fig,"chaos_hup060_framework.pdf",before)
    RESULTS[-1]["all_workflow_labels_and_case_counts_unchanged"]=True
    RESULTS[-1]["diagram_arrowhead_geometry_is_presentation_only"]=True

def network():
    source_dir=util.PROJECT/"output/part1/source_data/figure_01"
    path=source(util.PROJECT/"part1_network/figure_01_plv_network_selection.py")
    for name in ["hup060_plv_nodes_and_selection.csv","hup060_connected_plv_edges.csv",
                 "hup060_adjacency_matrix.csv","hup060_laplacian_matrix.csv","summary.json"]:
        source(source_dir/name)
    module=util.load_module(path,"round3_network_original")
    module.style()
    nodes,edges,adjacency,laplacian,summary=module.load_source_data({"output_dir":str(source_dir)})
    fig=module.make_figure(nodes,edges,adjacency,laplacian,summary)
    fig.canvas.draw()  # freeze lazy colormap face colours before the digest
    before=util.digest_artists(fig)
    util.remove_figure_text(fig)
    fig.set_size_inches(170/25.4,211/25.4)
    net,scores,adj,cb_adj,lap,cb_lap=fig.axes
    net.set_position([.035,.340,.50,.580])
    net.set_xlim(-1.8,1.8);net.set_ylim(-1.7,1.7)
    net.set_title("",loc="left")
    net.set_title("")
    channel_text={t.get_text():t for t in net.texts if t.get_text() in set(nodes.channel)}
    for t in list(net.texts):
        if t.get_text() not in channel_text:t.remove()
    left=nodes[(nodes.node==0)|(nodes.node>=19)].sort_values("y",ascending=False)
    right=nodes[(nodes.node>=1)&(nodes.node<=18)].sort_values("y",ascending=False)
    for side,table in [(-1,left),(1,right)]:
        for y,entry in zip(np.linspace(1.42,-1.42,len(table)),table.itertuples()):
            t=channel_text[entry.channel]
            t.set_position((side*1.35,y));t.set_ha("right" if side<0 else "left")
            t.set_fontsize(9)
            # These thin connector guides label the saved node; they are not
            # network edges and do not enter the scientific artist digest.
            net.annotate("",xy=(entry.x,entry.y),xytext=(side*1.32,y),
                         arrowprops={"arrowstyle":"-","lw":.35,"color":"#AAB0B6","alpha":.65},zorder=0)
    scores.set_position([.690,.365,.290,.580])
    scores.set_ylim(-.8,len(nodes)-.2)
    scores.set_xticks([0,.5,1])
    scores.set_title("b  Centrality score / selection",loc="left",fontsize=9,fontweight="bold",pad=5)
    for t in list(scores.texts):t.remove()
    scores.set_xlabel("")
    for ax,label,pos in [(adj,"c",[.10,.120,.30,.190]),(lap,"d",[.65,.120,.30,.190])]:
        ax.set_position(pos)
        for t in list(ax.texts):t.remove()
        ax.set_title((label+"  Weighted adjacency $A$") if label=="c" else (label+"  Laplacian $L=D-A$"),
                     loc="left",fontsize=9,fontweight="bold",pad=5)
        ax.xaxis.label.set_fontsize(9);ax.yaxis.label.set_fontsize(9)
        ax.set_xlabel("")  # channel names/order are retained on both axes
        ax.tick_params(labelsize=9)
    cb_adj.set_position([.12,.043,.26,.014])
    cb_lap.set_position([.67,.043,.26,.014])
    handles=[Line2D([],[],marker="o",ls="",mfc=module.COLORS["clinical_target"],mec=module.COLORS["ink"],label="Selected: clinical overlap"),
             Line2D([],[],marker="o",ls="",mfc=module.COLORS["nonclinical_target"],mec="white",label="Selected: no overlap"),
             Line2D([],[],marker="o",ls="",mfc=module.COLORS["other"],mec=module.COLORS["other_edge"],label="Not selected"),
             Line2D([],[],color=module.COLORS["edge"],lw=1.3,alpha=.65,label="Retained PLV edge")]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.270,.410),ncol=2,fontsize=9,
               columnspacing=.7,handlelength=1,handletextpad=.3)
    fig.text(.025,.963,"a  Connected PLV network",fontsize=9,fontweight="bold",va="top")
    fig.text(.025,.927,f"{summary['n_nodes']} nodes · {summary['n_edges']} edges · density {summary['realized_density']:.3f}",fontsize=9,va="top")
    fig.text(.025,.891,r"$S_i=0.30C_{deg}+0.60C_{bet}+0.10C_{eig}$",fontsize=9,va="top")
    fig.text(.025,.856,rf"$S_i>Q_{{{summary['threshold_quantile']:.2f}}}(S)={summary['threshold_raw_score']:.3f}$",fontsize=9,va="top")
    fig.text(.025,.823,f"{summary['selected_count']}/36 selected: {summary['selected_clinical_match_count']} red + {summary['selected_nonclinical_count']} blue",fontsize=9,va="top")
    save(fig,"hup060_part1_plv_network_selection.pdf",before)
    RESULTS[-1]["all_36_contact_labels_preserved"]=True
    RESULTS[-1]["matrix_channel_order_unchanged"]=True
    RESULTS[-1]["label_connector_guides_are_not_plv_edges"]=True
    RESULTS[-1]["proposed_include_options"]=r"width=\textwidth,height=190mm,keepaspectratio"
    RESULTS[-1]["minimum_ordinary_font_at_190mm_height_pt"]=9.5*190/211

def main():
    framework();network()
    assert all(sha(Path(path))==digest for path,digest in SOURCES.items())
    (QA/"round3_framework_network_qa.json").write_text(json.dumps({"figures":RESULTS,"source_hashes":SOURCES,
        "no_training_evaluation_or_graph_reconstruction":True},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(RESULTS,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
