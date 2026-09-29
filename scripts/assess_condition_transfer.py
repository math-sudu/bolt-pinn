"""Read completed condition-transfer paths at their experimental support."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_condition_transfer import fit_schedule, load_curves, objective_weights, write_json


def assess(directory):
    selected = json.loads((directory/"selected_fit.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
    by_id = {r["trial_id"]: r for r in metrics["records"]}
    records = []
    calibration_costs = []
    experiments = selected["config"].get("experiment_ids")
    for curve in load_curves(ROOT, "calibration", experiments)+load_curves(ROOT, "prediction", experiments):
        record = curve["record"]
        trial_id = record["trial_id"]
        with (directory/"curves"/trial_id/"response.csv").open(encoding="utf-8", newline="") as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        fields = np.load(directory/"curves"/trial_id/"fields.npz")["fields"]
        predicted = np.array([r["reaction_kN"] for r in rows])
        displacement = np.array([r["measured_displacement_mm"] for r in rows])
        truth = np.interp(displacement[1:], curve["x"], curve["y"])
        if record["role"] == "calibration":
            schedule = fit_schedule(curve)
            residual = (np.interp(schedule[1:], displacement, predicted)
                        - np.interp(schedule[1:], curve["x"], curve["y"]))
            weighted = residual*objective_weights(schedule, record["image_peak_kN"])
            calibration_costs.append(float(weighted@weighted))
        peak_index = int(predicted.argmax())
        late_index = int(np.abs(displacement-60).argmin())
        checkpoints = {}
        for name, i in [("predicted_peak", peak_index), ("late", late_index)]:
            checkpoints[name] = {k: rows[i][k] for k in [
                "measured_displacement_mm", "reaction_kN", "collar_displacement_mm",
                "deep_end_displacement_mm", "balance_error_kN", "observation_error_mm",
                "cumulative_interface_dissipation_J"]}
            checkpoints[name].update(
                slip_range_mm=[float(fields[i,:,1].min()), float(fields[i,:,1].max())],
                shear_range_MPa=[float(fields[i,:,3].min()), float(fields[i,:,3].max())])
        m = by_id[trial_id]
        stages = {}
        for name, mask in [
            ("through_peak_region", displacement[1:] <= 16),
            ("early_post_peak", (displacement[1:] > 16) & (displacement[1:] <= 40)),
            ("late", displacement[1:] > 40)]:
            error = predicted[1:][mask]-truth[mask]
            stages[name] = {"rmse_kN": float(np.sqrt(np.mean(error**2))),
                            "mean_error_kN": float(error.mean())}
        area = np.pi*record["bolt_diameter_mm"]*record["embedment_mm"]
        records.append({**m, "embedment_mm": record["embedment_mm"],
            "peak_error_kN": m["peak_predicted_kN"]-m["peak_observed_kN"],
            "peak_error_percent": 100*(m["peak_predicted_kN"]/m["peak_observed_kN"]-1),
            "peak_displacement_error_mm": m["peak_displacement_predicted_mm"]-m["peak_displacement_observed_mm"],
            "late_error_kN": m["late_predicted_kN"]-m["late_observed_kN"],
            "observed_average_peak_transfer_MPa": m["peak_observed_kN"]*1000/area,
            "predicted_average_peak_transfer_MPa": m["peak_predicted_kN"]*1000/area,
            "stages": stages, "checkpoints": checkpoints,
            "minimum_incremental_work_MPa_mm": min(r["min_incremental_dissipation_MPa_mm"] for r in rows),
            "minimum_history_increment_mm": float(np.diff(fields[:,:,5], axis=0).min())})
    families = []
    for family in ["pressure_transfer", "length_transfer"]:
        for role in ["calibration", "prediction"]:
            subset = [r for r in records if r["experiment_id"] == family and r["role"] == role]
            if not subset:
                continue
            families.append({"experiment_id": family, "role": role,
                "curve_mean_rmse_kN": float(np.mean([r["rmse_kN"] for r in subset])),
                "curve_mean_absolute_peak_error_percent": float(np.mean([abs(r["peak_error_percent"]) for r in subset])),
                "curve_mean_absolute_late_error_kN": float(np.mean([abs(r["late_error_kN"]) for r in subset]))})
    sensitivity = selected["local_sensitivity"]
    weakest = sorted(zip(selected.get("fitted_parameter_names", selected["parameter_names"]),
                         sensitivity["weakest_scaled_direction"], strict=True),
                     key=lambda item: abs(item[1]), reverse=True)
    report = {"kind": "condition_response_assessment", "run_ref": directory.relative_to(ROOT).as_posix(),
        "support": "Each full curve is assessed on its own measured displacement support. Original digitized peaks and the shared 60 mm point define the physical comparisons.",
        "families": families, "records": records,
        "calibration_replay": {
            "fit_solver_weighted_relative_rms": float(np.sqrt(selected["cost"])),
            "final_solver_weighted_relative_rms": float(np.sqrt(np.mean(calibration_costs))),
            "role": "Same objective evaluated from saved final responses; exposes material changes due to path or network resolution."},
        "sensitivity": {"singular_values": sensitivity["singular_values"],
            "weakest_scaled_direction": dict(weakest),
            "C_500_mm_per_kN": selected["theta"][-1]},
        "field_role": "Inferred internal fields. Peak and late checkpoints diagnose global response and the observation boundary; no local field measurements are available."}
    write_json(directory/"assessment.json", report)
    for row in records:
        print(f"{row['trial_id']:18s} {row['role']:11s} RMSE={row['rmse_kN']:7.3f} "
              f"peak_error={row['peak_error_percent']:+7.2f}% "
              f"peak_x_error={row['peak_displacement_error_mm']:+6.2f} "
              f"F60_error={row['late_error_kN']:+7.3f} kN")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"results/condition_transfer")
    args = parser.parse_args()
    assess(args.directory.resolve())
