"""Refit calibration curves at fixed plateau-decay lengths, then assess transfer."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_condition_transfer import (
    fit_schedule, load_curves, objective_weights, write_json,
)

BASE = ROOT/"results/length_transfer_shape"
OUTPUT = ROOT/"results/length_onset_profile"
PARAMETER = "length_log_onset_decay_mm"


def scenario_name(scale):
    return f"decay_{scale:g}mm"


def prepare_config(scale, workers):
    if scale <= 0:
        raise ValueError("The onset-decay length must be positive")
    path = ROOT/"configs/length_onset_profile"/f"{scenario_name(scale)}.json"
    if path.exists():
        raise FileExistsError(f"Existing profile configuration {path}; use --resume or a new scale")
    selected = json.loads((BASE/"selected_fit.json").read_text(encoding="utf-8"))
    config = deepcopy(selected["config"])
    config["initial_parameters"] = selected["theta"].copy()
    index = selected["parameter_names"].index(PARAMETER)
    config["initial_parameters"][index] = float(np.log(scale))
    config["fixed_parameters"][PARAMETER] = float(np.log(scale))
    config["workers"] = workers
    config["output_directory"] = f"results/length_onset_profile/{scenario_name(scale)}"
    config["evaluation_kind"] = "calibration_profile_with_adaptive_transfer_assessment"
    config["profile"] = {
        "parameter": PARAMETER, "fixed_decay_length_mm": scale,
        "initial_fit_ref": "results/length_transfer_shape/selected_fit.json",
        "question": "Can weakly constrained plateau decay materially change intermediate-length peaks after calibration-only refitting?",
        "selection": "Fixed scales span an effectively absent intermediate plateau and increasingly persistent plateaus. Every scenario starts from the same saved calibration fit; no prediction residual enters fitting or selection.",
        "interpretation": "Deterministic calibrated scenarios, not a confidence or prediction interval; no observational noise model is available.",
    }
    write_json(path, config)
    return path


def response(directory, trial_id):
    with (directory/"curves"/trial_id/"response.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.asarray([row[key] for row in rows], dtype=float) for key in rows[0]}


def late_weakening(directory, trial_id, material):
    rows = response(directory, trial_id)
    fields = np.load(directory/"curves"/trial_id/"fields.npz")["fields"]
    reading = {}
    for displacement in [60, 80]:
        index = int(np.argmin(np.abs(rows["measured_displacement_mm"]-displacement)))
        kappa = fields[index, :, 5]
        excess = (material["peak"]-material["residual"])*np.exp(
            -np.maximum(kappa-material["softening_onset"], 0)/material["critical_slip"])
        reading[str(displacement)] = {
            "accumulated_slip_range_mm": [float(kappa.min()), float(kappa.max())],
            "maximum_degrading_excess_to_residual_ratio": float(np.max(excess/material["residual"])),
        }
    return reading


def summarize(scales):
    curves = load_curves(ROOT, "calibration", ["length_transfer"])
    initial = json.loads((BASE/"selected_fit.json").read_text(encoding="utf-8"))
    reference_paths = {c["record"]["trial_id"]: response(BASE, c["record"]["trial_id"]) for c in curves}
    records = []
    for directory in [BASE]+[OUTPUT/scenario_name(s) for s in scales]:
        selected = json.loads((directory/"selected_fit.json").read_text(encoding="utf-8"))
        assessment = json.loads((directory/"assessment.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        index = selected["parameter_names"].index(PARAMETER)
        difference_costs = []
        for curve in curves:
            trial = curve["record"]["trial_id"]
            current = response(directory, trial)
            reference = reference_paths[trial]
            schedule = fit_schedule(curve)
            difference = (np.interp(schedule[1:], current["measured_displacement_mm"], current["reaction_kN"])
                          - np.interp(schedule[1:], reference["measured_displacement_mm"], reference["reaction_kN"]))
            weighted = difference*objective_weights(schedule, curve["record"]["image_peak_kN"])
            difference_costs.append(float(weighted@weighted))
        paths = [response(directory, r["trial_id"]) for r in assessment["records"]]
        records.append({
            "name": directory.name,
            "fixed_decay_length": directory != BASE,
            "decay_length_mm": float(np.exp(selected["theta"][index])),
            "selected_fit_ref": (directory/"selected_fit.json").relative_to(ROOT).as_posix(),
            "assessment_ref": (directory/"assessment.json").relative_to(ROOT).as_posix(),
            "execution_ref": (directory/"execution.json").relative_to(ROOT).as_posix(),
            "calibration_replay": assessment["calibration_replay"],
            "calibration_response_difference_from_reference_weighted_rms": float(np.sqrt(np.mean(difference_costs))),
            "optimizer_message": selected["optimizer_message"],
            "active_bounds": selected["active_bounds"],
            "fitted_parameter_names": selected["fitted_parameter_names"],
            "free_parameter_changes_from_initial": {
                name: value-initial["theta"][i]
                for i, (name, value) in enumerate(zip(selected["parameter_names"], selected["theta"]))
                if name not in selected["fixed_parameters"]
            },
            "calibration_records": selected["calibration_records"],
            "curves": [{key: r[key] for key in [
                "trial_id", "role", "embedment_mm", "rmse_kN", "peak_predicted_kN", "peak_observed_kN",
                "peak_error_percent", "peak_displacement_predicted_mm", "peak_displacement_observed_mm",
                "late_predicted_kN", "late_observed_kN", "end_predicted_kN", "end_observed_kN",
            ]} | {"softening_onset_mm": metrics["parameters"][r["trial_id"]]["material"]["softening_onset"]}
                for r in assessment["records"]],
            "physical_consistency": {
                "maximum_path_balance_error_kN": max(float(np.max(np.abs(p["balance_error_kN"]))) for p in paths),
                "maximum_observation_error_mm": max(float(np.max(np.abs(p["observation_error_mm"]))) for p in paths),
                "minimum_incremental_work_MPa_mm": min(r["minimum_incremental_work_MPa_mm"] for r in assessment["records"]),
                "minimum_history_increment_mm": min(r["minimum_history_increment_mm"] for r in assessment["records"]),
            },
            "intermediate_length_late_weakening": {
                r["trial_id"]: late_weakening(directory, r["trial_id"], metrics["parameters"][r["trial_id"]]["material"])
                for r in assessment["records"] if r["role"] == "prediction"
            },
        })
    payload = {
        "kind": "length_onset_profile",
        "parameter": PARAMETER,
        "question": "How much does the weak plateau-decay scale change intermediate-length transfer after refitting the original calibration curves?",
        "calibration": "Original 100/300/500 mm records, unchanged objective and solver settings; eleven remaining shape coefficients are refitted. Reference compliance remains fixed at the existing diagnostic value.",
        "assessment": "Selected parameters are saved before loading exposed 200/400 mm responses; all intermediate-length results remain adaptive assessments.",
        "interpretation": "Conditional deterministic local-fit scenarios, with no claim of global profile minima. Their spread is not a statistical interval, and scenarios are compared using actual calibration support rather than an invented equivalence threshold. Saved free-parameter changes distinguish accepted refits from cases where the optimizer retained its initial coefficients.",
        "records": records,
    }
    write_json(OUTPUT/"profile.json", payload)
    for row in records:
        peaks = {r["embedment_mm"]: round(r["peak_predicted_kN"], 4) for r in row["curves"] if r["role"] == "prediction"}
        print(row["name"], row["calibration_replay"], peaks)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=float, nargs="+", default=[10, 50, 100])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.scales)
    else:
        for scale in args.scales:
            config_path = (ROOT/"configs/length_onset_profile"/f"{scenario_name(scale)}.json"
                           if args.resume else prepare_config(scale, args.workers))
            command = [sys.executable, str(ROOT/"scripts/run_condition_transfer.py"), "--config", str(config_path)]
            if args.resume:
                command.append("--resume")
            subprocess.run(command, cwd=ROOT, check=True)
            subprocess.run([sys.executable, str(ROOT/"scripts/assess_condition_transfer.py"),
                            "--directory", str(OUTPUT/scenario_name(scale))], cwd=ROOT, check=True)
