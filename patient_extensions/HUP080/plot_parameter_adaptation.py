"""Original-layout plots with honest completed parameter-adaptation bindings.

All rendering code is the unchanged layout snapshot. This wrapper replaces only
the completed-run data loader because warm-start metadata must not be mislabeled
as neutral fresh training. No loss plot is produced.
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hup080_original_layout_snapshot", HERE / "plot_external_layout_snapshot.py")
plot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plot
spec.loader.exec_module(plot)

def load_completed(subject, run):
    if subject != "HUP080":
        raise ValueError("This wrapper is for HUP080 only")
    receipt_path = run / "selection_receipt.json"
    report_path = run / "final_posthoc_report.json"
    if not receipt_path.is_file() or not report_path.is_file():
        raise ValueError("Finish and freeze both development arms and full diagnostics before reading curves")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if receipt["outer_used_for_selection"] is not False or receipt["ctx6_used_for_selection"] is not False:
        raise ValueError("Diagnostic outcomes cannot choose a checkpoint")
    evaluation = report["outer_posthoc_amendment"]
    if evaluation["context_bank_evaluations"] != 24 or evaluation["channels"] != 96 or evaluation["direct_actuators"] != 76:
        raise ValueError("Completed 8 x 3 / 96-channel / 76-actuator diagnostics required")
    if evaluation["outer_used_for_selection"] is not False or evaluation["ctx6_used_for_reselection"] is not False:
        raise ValueError("Outer / terminal diagnostics must not reselect")
    checkpoint = run / "frozen_actor.pt"
    digest = plot.sha(checkpoint)
    if digest != receipt["checkpoint_sha256"] or digest != evaluation["checkpoint_sha256"]:
        raise ValueError("Selection and diagnosis must use the same frozen controller")
    arm = next(row for row in receipt["arms"] if row["arm"] == receipt["selected_arm"])
    if arm["checkpoint_sha256"] != digest or arm["status"] != "completed" or arm["warm_start"] is not True:
        raise ValueError("Honest completed adaptation metadata required")
    metric_frame = pd.read_csv(run / "outer_posthoc_amendment/all_channel_context_bank_metrics.csv")
    if len(metric_frame) != 24 * 96 or metric_frame[["context_index", "crn_bank"]].drop_duplicates().shape[0] != 24:
        raise ValueError("All outer channel-context-bank cases must be present")
    display_path = run / "outer_posthoc_amendment/display_context_rollout.npz"
    with np.load(display_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if int(arrays["display_context_index"]) != 7 or int(arrays["display_crn_bank"]) != 0:
        raise ValueError("Keep original predeclared display context and bank")
    if arrays["observed_standardized"].shape != (256, 96):
        raise ValueError("Recorded ictal display shape changed")
    if arrays["free_standardized"].shape != (32, 256, 96) or arrays["controlled_standardized"].shape != (32, 256, 96):
        raise ValueError("Paired 32-particle original horizon required")
    if arrays["reference_standardized"].ndim != 3 or arrays["reference_standardized"].shape[1:] != (256, 96):
        raise ValueError("Original patient-specific reference shape changed")
    for style in plot.SERIES.values():
        if not np.isfinite(arrays[style[0]]).all():
            raise ValueError("Nonfinite displayed law")
    if arrays["direct_mask"].shape != (96,) or arrays["direct_mask"].astype(bool).sum() != 76:
        raise ValueError("Frozen actuator count changed")
    if not np.array_equal(np.flatnonzero(arrays["direct_mask"]), arrays["selected_indices"]):
        raise ValueError("Selection mask and index binding failed")
    if len(set(arrays["channels"].astype(str))) != 96:
        raise ValueError("Duplicate channel labels")
    training = dict(status="completed_parameter_adaptation", subject=subject,
        trained_updates=arm["added_actor_updates"], selected_update=arm["selected_added_update"],
        warm_start=True, starting_fresh_selected_update=200, earlier_fresh_training_budget=1000,
        source_arm=receipt["selected_arm"], selection_rule="common development-only dimensionless metric and revised budget")
    return arrays, training, evaluation, dict(checkpoint_sha256=digest,
        display_npz_sha256=plot.sha(display_path), selection_receipt_sha256=plot.sha(receipt_path),
        final_posthoc_report_sha256=plot.sha(report_path),
        rendering_layout_snapshot_sha256=plot.sha(HERE / "plot_external_layout_snapshot.py"))

plot.load_completed = load_completed

if __name__ == "__main__":
    result = plot.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    run = Path(sys.argv[sys.argv.index("--run-directory") + 1])
    receipt = json.loads((run / "selection_receipt.json").read_text(encoding="utf-8"))
    qa_path = output / "external_original_style_numeric_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa.update(training_classification="warm-start parameter adaptation, not neutral from-scratch training",
        warm_start=True, previous_fresh_training_budget=1000, starting_selected_update=200,
        selected_added_update=receipt["selected_added_update"], selected_arm=receipt["selected_arm"],
        revised_RMS_cap=.45, old_RMS_cap=.405, new_loss_plot_created=False,
        published_external_results_replaced=False, human_visual_review=False)
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(result)
