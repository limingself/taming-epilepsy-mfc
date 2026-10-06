"""Re-export frozen main figures at readable journal dimensions.

The plotting artists are produced by the archived Python generators. No model
rollouts, training, density fitting, or quantitative source edits are performed.
The only changed artist properties are wording, font size, legend arrangement,
panel positions, and canvas dimensions. A before/after artist-data digest checks
that all curve, scatter, bar, image, and colour values remain unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text, Annotation
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import fitz

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "qa_round3" / "figures"
QA.mkdir(parents=True, exist_ok=True)
PROJECT = Path(r"D:\VS code\distribution control")
ARCHIVE = Path(r"C:\Users\LiMing\Documents\改论文")
TEXT_WIDTH_MM = 165.1  # article/letter, one-inch margins
STYLE_PT = 9.0

CONTRACT = {
    "backend": "Python/matplotlib only",
    "archetype": "quantitative grid",
    "maximum_width_mm": 170,
    "minimum_ordinary_font_size_at_inclusion_pt": 8,
    "export": "editable vector PDF and PNG preview",
    "source_status": "frozen CSV/NPZ, read only; no training or rollout",
    "claims": {
        "figure3": "One-second stochastic prediction has limited single-path fidelity but measurable occupation-law agreement.",
        "figure4": "The Primary Full checkpoint changes the frozen-model law toward its preictal reference.",
        "figure5": "The component-removal audit uses Audit Full; structural changes also change command RMS.",
        "figure6": "Matched-command-RMS generic inputs and changed-diffusion sensitivities answer different questions.",
    },
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class Captured(Exception):
    pass


def capture(module, function, save_name="save_bundle"):
    held = []
    def intercept(fig, *args, **kwargs):
        held.append(fig)
        raise Captured()
    setattr(module, save_name, intercept)
    try:
        function()
    except Captured:
        pass
    if len(held) != 1:
        raise RuntimeError("exactly one frozen figure must be captured")
    return held[0]


def digest_artists(fig):
    digest = hashlib.sha256()
    def add(value):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    count = {"lines": 0, "collections": 0, "patches": 0, "images": 0}
    for axis in fig.axes:
        for line in axis.lines:
            add(line.get_xdata()); add(line.get_ydata())
            digest.update(str(line.get_color()).encode())
            count["lines"] += 1
        for collection in axis.collections:
            add(collection.get_offsets())
            add(collection.get_facecolors()); add(collection.get_edgecolors())
            for path in collection.get_paths():
                add(path.vertices)
            count["collections"] += 1
        for patch in axis.patches:
            add(patch.get_path().vertices)
            add(patch.get_facecolor()); add(patch.get_edgecolor())
            if hasattr(patch, "get_x"):
                add([patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()])
            count["patches"] += 1
        for image in axis.images:
            add(image.get_array())
            count["images"] += 1
    return digest.hexdigest(), count


def remove_figure_text(fig):
    for text in list(fig.texts):
        text.remove()
    for legend in list(fig.legends):
        legend.remove()


def style_text(fig):
    for text in fig.findobj(Text):
        if text.get_text():
            text.set_fontsize(max(STYLE_PT, text.get_fontsize()))
    for axis in fig.axes:
        axis.tick_params(axis="both", labelsize=STYLE_PT)
        axis.xaxis.label.set_fontsize(STYLE_PT)
        axis.yaxis.label.set_fontsize(STYLE_PT)
        axis.title.set_fontsize(STYLE_PT)
        for text in axis.texts:
            if text.get_text() in tuple("abcdefghi"):
                text.set_position((-.17, 1.08))
                text.set_fontsize(10)


def title(fig, label):
    fig.text(.09, .984, label, va="top", ha="left", fontsize=10,
             fontweight="bold")


def grid3(fig, height_mm, top=.865, middle=.510, bottom=.155, widths=.235,
          heights=(.210,.180,.145)):
    fig.set_size_inches(170 / 25.4, height_mm / 25.4)
    for i, axis in enumerate(fig.axes):
        row, col = divmod(i, 3)
        axis.set_position([.105 + col * .305, (top, middle, bottom)[row] - heights[row],
                           widths, heights[row]])
        if row != 1:
            axis.set_xticks([0, .5, 1])
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))


def export(fig, filename, before, multiplier):
    style_text(fig)
    # Draw once so dynamically generated tick labels are included in the check.
    fig.canvas.draw()
    after = digest_artists(fig)
    assert before == after, "frozen numerical artists or colours changed"
    pdf = ROOT / filename
    png = QA / (pdf.stem + ".png")
    fig.savefig(pdf, metadata={"Creator": "Python/matplotlib; frozen source data", "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=220)
    page = fitz.open(pdf)[0]
    width_mm = page.rect.width / 72 * 25.4
    height_mm = page.rect.height / 72 * 25.4
    assert width_mm <= 170.001
    scaling = TEXT_WIDTH_MM * multiplier / width_mm
    ordinary = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip() and span["size"] >= 8.8:
                    ordinary.append(span["size"] * scaling)
    assert min(ordinary) >= 8, "ordinary type below 8 pt at manuscript size"
    assert page.get_text().strip(), "vector text must be selectable"
    fonts = page.get_fonts(full=True)
    assert fonts, "font resources absent"
    info = {"filename": filename, "preview": png.name,
            "width_mm": width_mm, "height_mm": height_mm,
            "inclusion_width_mm": TEXT_WIDTH_MM * multiplier,
            "minimum_ordinary_font_at_inclusion_pt": min(ordinary),
            "math_subscripts_superscripts_exempt": True,
            "data_artist_digest_unchanged": before[0], "artist_counts": before[1],
            "editable_text": True, "font_types": sorted({row[2] for row in fonts}),
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    plt.close(fig)
    return info


def prediction():
    module = load_module(PROJECT / "part2_rc_sde/figure_02_rc_sde_prediction.py", "frozen_prediction")
    module.configure_publication_style()
    fig = capture(module, module.make_figure_02)
    before = digest_artists(fig)
    fig.savefig(QA / "original_size_prediction_reproduction.pdf")
    remove_figure_text(fig)
    grid3(fig, 174, top=.798, middle=.472, bottom=.210, heights=(.195,.160,.135))
    for i, axis in enumerate(fig.axes):
        row, col = divmod(i,3)
        if row == 0:
            axis.set_title(("RPFa3 · selected SOZ", "RA3 · selected non-SOZ", "RAFa4 · unselected")[col], loc="left", fontweight="bold")
        else:
            axis.set_title("", loc="left")
        for text in axis.texts:
            label = text.get_text()
            if label.startswith("distribution"):
                text.set_text(label.replace("distribution ", ""))
                text.set_position((.98,.94)); text.set_va("top")
            elif label.startswith("nRMSE"):
                text.set_position((.98,.94)); text.set_va("top")
    handles = []
    labels = []
    for row in range(3):
        for handle, label in zip(*fig.axes[row*3].get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle); labels.append(label)
    replacements = {"Observed actual":"Recorded ictal", "Free RC-SDE, particle 0":"Free Graph–RC, particle 0",
                    "Free RC-SDE, 32 particles":"Free Graph–RC, 32 particles",
                    "Instantaneous absolute error":"Absolute error", "Mean absolute error":"MAE"}
    fig.legend(handles, [replacements.get(s,s) for s in labels], loc="upper center",
               bbox_to_anchor=(.54,.945), ncol=3, fontsize=9, columnspacing=.8,
               handlelength=1.8)
    title(fig, "HUP060 run-02: one-second stochastic Graph–RC prediction")
    return export(fig, "chaos_hup060_prediction.pdf", before, 1)


def control():
    module = load_module(ROOT / "scripts/build_chaos_revision_figures.py", "frozen_control")
    module.OUTPUT = QA
    fig = capture(module, module.representative, "save")
    before = digest_artists(fig)
    fig.savefig(QA / "original_size_control_reproduction.pdf")
    remove_figure_text(fig)
    grid3(fig, 175, top=.800, middle=.455, bottom=.207, heights=(.195,.160,.130))
    for i, axis in enumerate(fig.axes):
        if i//3==1:
            for text in axis.texts:
                if "Uncontrolled" in text.get_text():
                    text.set_text(text.get_text().replace("Uncontrolled → Reference", "Free → Ref.").replace("Controlled → Reference", "Full → Ref."))
                    text.set_position((.98,1.025)); text.set_va("bottom")
    handles = module.legend_handles()
    for handle in handles:
        if "Interictal" in handle.get_label(): handle.set_label("Preictal reference")
        if "Full policy" in handle.get_label(): handle.set_label("Primary Full")
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.54,.940),ncol=2,
               fontsize=9, handlelength=1.8,columnspacing=1.8)
    title(fig, "HUP060 run-02: reference-directed control (Primary Full)")
    return export(fig, "chaos_hup060_control.pdf", before, .96)


def ablation():
    path = ARCHIVE / "github_release_taming_epilepsy/part3_mfc/figure_07_mfc_ablation.py"
    module = load_module(path,"frozen_component")
    argv = sys.argv
    sys.argv = [str(path), "--source-directory", str(PROJECT / "output/part3/source_data/figure_07"),
                "--output-directory", str(QA)]
    try: fig = capture(module,module.main)
    finally: sys.argv = argv
    before = digest_artists(fig)
    fig.savefig(QA / "original_size_component_reproduction.pdf")
    remove_figure_text(fig)
    fig.set_size_inches(170/25.4, 178/25.4)
    # Six unchanged panels reflow from 2x3 to 3x2 so 9-pt labels fit.
    for i, axis in enumerate(fig.axes):
        row, col = divmod(i,2)
        axis.set_position([.120 + col*.465, .710-row*.300, .335, .205])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4)) if i in (0,1,4,5) else None
    a,b,c,d,e,f = fig.axes
    for axis in (a,b):
        axis.set_xticklabels(["Free","−WGAN","−Graph","−Dev.","Audit\nFull"])
    c.set_xlabel("Increase vs Audit Full (%)\nlogarithmic scale")
    c.set_yticklabels(["−WGAN","−Graph","−Deviation"])
    c.legend(loc="upper right",fontsize=9,ncol=1,handlelength=1,columnspacing=.7)
    d.set_title("Recovery vs free (%)", loc="left")
    d.set_yticklabels(["−WGAN","−Graph","−Deviation","Audit Full"])
    e.set_xlabel("Command RMS")
    e.set_ylabel("Mean time-resolved $W_1$")
    for text in list(e.texts):
        if isinstance(text, Annotation):
            text.remove()
    e.legend(e.collections,["−WGAN","−Graph","−Deviation","Audit Full"],
             loc="upper right",fontsize=9,handlelength=1,labelspacing=.35)
    e.set_ylim(.112,.32)
    f.set_ylabel("Contacts worse than Audit Full")
    f.set_ylim(0,43)
    f.legend(loc="lower center",bbox_to_anchor=(.5,1.015),fontsize=9,ncol=2,handlelength=1,columnspacing=.7)
    title(fig,"HUP060 run-02: component-removal audit (Audit Full)")
    return export(fig,"hup060_matched_mfc_ablation_summary_v2.pdf",before,.98)


def baselines():
    module = load_module(ARCHIVE / "taming-epilepsy-mfc-baseline/part3_mfc/figure_09_trivial_baselines.py", "frozen_baseline")
    module.SOURCE = PROJECT / "output/part3/source_data/figure_09_trivial_baselines"
    fig = capture(module,module.main)
    before = digest_artists(fig)
    fig.savefig(QA / "original_size_baselines_reproduction.pdf")
    fig.set_layout_engine(None)
    remove_figure_text(fig)
    fig.set_size_inches(170/25.4, 174/25.4)
    a,b,c,d1,d2=fig.axes
    for ax,pos in zip(fig.axes,([.115,.580,.350,.290],[.605,.580,.350,.290],
                    [.115,.105,.350,.290],[.605,.360,.350,.110],[.605,.105,.350,.135])):
        ax.set_position(pos)
    a.set_xticklabels(["Free","Noise 0.2","Constant","White","Damping","Full"],rotation=40,ha="right")
    a.set_ylim(0,1.20)
    a.legend(loc="upper left",ncol=1,fontsize=9,handlelength=1)
    b.set_xlabel("Command RMS")
    b.set_ylabel("Occupation $W_1$ (log)")
    handles, labels=b.get_legend_handles_labels()
    labels=["Scalar damping", "Primary Full", "RMS-matched damping"]
    b.legend(handles,labels,loc="upper left",fontsize=9,handlelength=1)
    for text in b.texts:
        if isinstance(text,Annotation):
            text.set_text("Best screened damping\n$K=0.15$, RMS=0.072")
            text.set_position((.08,.46))
    c.set_xlabel("Relative noise multiplier $\\kappa_\\sigma$")
    c.set_ylabel("Occupation $W_1$")
    handles,labels=c.get_legend_handles_labels()
    c.legend(handles,["To preictal reference","To recorded ictal"],loc="upper left",fontsize=9,handlelength=1)
    for text in c.texts:
        if isinstance(text,Annotation):
            if "prediction" in text.get_text():
                text.set_text("Frozen $\\kappa_\\sigma=1$"); text.set_position((.88,.18))
            else:
                text.set_text("Post hoc $\\kappa_\\sigma=0.2$"); text.set_position((.38,.056))
    d1.set_xlabel("Free → reference $W_1$ contribution")
    d1.legend(loc="lower center",bbox_to_anchor=(.50,1.12),ncol=3,fontsize=9,
              columnspacing=.7,handlelength=1)
    d1.set_xticks([0,.1,.2,.3])
    for text in list(d2.texts):
        if text.get_text() not in tuple("abcd"):
            text.remove()
    d2.set_ylabel("SD ratio")
    d2.set_xticklabels(["Reference /\nrecorded ictal","Reference /\nfree model"])
    d2.set_yticks([0,.5,1])
    title(fig,"HUP060 run-02: input audit (Primary Full) and noise sensitivity")
    return export(fig,"hup060_trivial_controller_baselines.pdf",before,.98)


def main():
    source_paths=[PROJECT/"part2_rc_sde/figure_02_rc_sde_prediction.py",
                  ROOT/"scripts/build_chaos_revision_figures.py",
                  ARCHIVE/"github_release_taming_epilepsy/part3_mfc/figure_07_mfc_ablation.py",
                  ARCHIVE/"taming-epilepsy-mfc-baseline/part3_mfc/figure_09_trivial_baselines.py"]
    data_paths=[]
    for folder in [PROJECT/"output/part2/source_data/figures_02_04",PROJECT/"output/part3/source_data/figure_06",
                   PROJECT/"output/part3/source_data/figure_07",PROJECT/"output/part3/source_data/figure_09_trivial_baselines"]:
        data_paths.extend(p for p in folder.glob("*.csv"))
    data_paths.append(PROJECT/"output/part3/source_data/figures_06_08/paired_comparison.npz")
    before_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths+data_paths}
    results=[prediction(),control(),ablation(),baselines()]
    after_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths+data_paths}
    assert before_hashes == after_hashes, "an archived generator or source file changed"
    binding=check_source_binding()
    audit={"contract":CONTRACT,"figures":results,
           "archived_generators_and_source_files_unchanged": True,
           "source_array_binding":binding,
           "plotting_generators":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths],
           "frozen_source_files":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in data_paths]}
    (QA/"round3_figure_source_and_type_qa.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2,ensure_ascii=False))


def check_source_binding():
    data=np.load(PROJECT/"output/part3/source_data/figures_06_08/paired_comparison.npz")
    names=data["channels"].astype(str)
    trajectories=pd.read_csv(PROJECT/"output/part3/source_data/figure_06/source_data_representative_trajectories.csv")
    prediction=pd.read_csv(PROJECT/"output/part2/source_data/figures_02_04/representative_trajectory_and_error.csv")
    series={"Observed ictal":data["observed_scaled"],
            "Free RC-SDE":data["uncontrolled_scaled"][0],
            "Interictal reference":data["reference_validation_scaled"][0],
            "Actor + WGAN-GP":data["candidate_controlled_scaled"][0]}
    differences=[]
    for index in data["representative_indices"].astype(int):
        name=names[index]
        for label, array in series.items():
            values=trajectories.loc[(trajectories.node==name)&(trajectories.series==label),"standardized_amplitude"].to_numpy()
            differences.append(float(np.max(np.abs(values-array[:,index]))))
        values=prediction.loc[prediction.channel==name,"predicted"].to_numpy()
        differences.append(float(np.max(np.abs(values-data["uncontrolled_scaled"][0,:,index]))))
    assert max(differences)<1e-12
    moment=pd.read_csv(PROJECT/"output/part3/source_data/figure_09_trivial_baselines/location_scale_moment_audit.csv")
    summation_error=float(np.max(np.abs(moment.shapley_location_contribution+moment.shapley_scale_contribution+moment.shape_residual_contribution-moment.free_to_reference_w1)))
    assert summation_error < 1e-12
    percentages=moment[["shapley_location_contribution","shapley_scale_contribution","shape_residual_contribution"]].mean()/moment.free_to_reference_w1.mean()*100
    return {"all_representative_csv_paths_bind_to_primary_npz":True,
            "maximum_csv_roundtrip_difference":max(differences),
            "location_scale_shape_sum_maximum_error":summation_error,
            "location_scale_shape_percent":percentages.to_dict()}


if __name__=="__main__":
    main()
