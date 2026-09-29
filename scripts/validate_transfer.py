"""Validate observed rankings, effective contact, and observation assumptions."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_condition_transfer import (load_curves, solve_job, assess_curve,
                                           write_json, init_worker)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def observed_rankings(curves):
    rows = []
    for curve in curves:
        record = curve["record"]
        rows.append({"trial_id": record["trial_id"], "experiment_id": record["experiment_id"],
            "role": record["role"], "vertical_MPa": record["pressure_vertical_MPa"],
            "horizontal_MPa": record["pressure_horizontal_MPa"], "embedment_mm": record["embedment_mm"],
            "peak_kN": record["image_peak_kN"], "peak_displacement_mm": record["image_peak_displacement_mm"],
            "late_forces_kN": {str(x): float(np.interp(x, curve["x"], curve["y"])) for x in [40, 60, 80]}})
    groups = []
    for pressure in [1, 2, 3]:
        group = [row for row in rows if row["experiment_id"] == "pressure_transfer" and row["vertical_MPa"] == pressure]
        groups.append({"vertical_MPa": pressure,
            "peak_order_low_to_high": [r["trial_id"] for r in sorted(group, key=lambda r: r["peak_kN"])],
            "late_order_low_to_high": {str(x): [r["trial_id"] for r in sorted(group,
                key=lambda r: r["late_forces_kN"][str(x)])] for x in [40, 60, 80]}})
    return {"records": rows, "pressure_rankings": groups,
        "source_refs": ["data/pressure_curves_digitized.csv", "data/length_curves_digitized.csv",
                        "data/digitization.json", "data/length_digitization.json", "data/record_aliases.csv"],
        "interpretation": "Condition-curve rankings, not independent specimen statistics or a separated causal pressure effect. The same nominal 500 mm length/diameter baseline is counted once."}


def validate(directory, output):
    selected = json.loads((directory/"selected_fit.json").read_text(encoding="utf-8"))
    config = selected["config"]
    theta = np.array(selected["theta"])
    curves = load_curves(ROOT, "calibration")+load_curves(ROOT, "prediction")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"observed_rankings.json", observed_rankings(curves))

    # The mechanical states are unchanged by re-labelling their measured
    # displacement. delta_constant = delta_equal - C_extra*F is exact at
    # each saved state; no extra fitted parameter or boundary solve is needed.
    observation_scenarios = []
    fields_readings = []
    for curve in curves:
        record = curve["record"]
        rows = read_rows(directory/"curves"/record["trial_id"]/"response.csv")
        peak = max(rows, key=lambda row: row["reaction_kN"])
        late = min(rows, key=lambda row: abs(row["measured_displacement_mm"]-60))
        fields_readings.append({"trial_id": record["trial_id"],
            "peak_balance_error_kN": peak["balance_error_kN"],
            "late_balance_error_kN": late["balance_error_kN"],
            "minimum_incremental_work_MPa_mm": min(r["min_incremental_dissipation_MPa_mm"] for r in rows),
            "peak_observation_error_mm": peak["observation_error_mm"]})
        if record["experiment_id"] == "length_transfer":
            extra = record["assumed_extra_free_compliance_mm_per_kN"]
            alternative = np.array([r["measured_displacement_mm"]-extra*r["reaction_kN"] for r in rows])
            observation_scenarios.append({"trial_id": record["trial_id"],
                "equal_total_length_peak_mm": peak["measured_displacement_mm"],
                "constant_external_compliance_peak_mm": peak["measured_displacement_mm"]-extra*peak["reaction_kN"],
                "peak_shift_mm": -extra*peak["reaction_kN"],
                "minimum_reparameterized_increment_mm": float(np.diff(alternative).min()),
                "role": "Fixed material parameters, exact boundary reparameterization at the same mechanical states."})

    # The shortest and longest lengths distinguish a material tail from the
    # much stronger fractional contact loss of a short embedment.
    representatives = [c for c in curves if c["record"]["trial_id"] in {"D36_L100_P5_5", "D36_L500_P5_5"}]
    no_contact = {**config, "contact_evolution": False}
    jobs = [(theta, c["record"], no_contact, np.r_[0, c["x"]], True,
             output/"fixed_contact"/c["record"]["trial_id"]) for c in representatives]
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker, initargs=(1,)) as pool:
        replays = list(pool.map(solve_job, jobs))
    contact_comparison = []
    for curve, (rows, _) in zip(representatives, replays):
        active = read_rows(directory/"curves"/curve["record"]["trial_id"]/"response.csv")
        contact_comparison.append({"trial_id": curve["record"]["trial_id"],
            "diminishing_contact": assess_curve(curve, active), "fixed_contact": assess_curve(curve, rows)})
    write_json(output/"physical_checks.json", {
        "adapted_run_ref": directory.relative_to(ROOT).as_posix(),
        "field_role": "Model-inferred internal states; no local measurements are available.",
        "representative_field_checks": fields_readings,
        "observation_scenarios": observation_scenarios,
        "contact_replays": contact_comparison,
        "contact_replay_policy": "Freeze the fitted finite-column material parameters and remove only the overlap factor. The elastic stiffness and critical slip are retained from the contact-on decode.",
        "purpose": "Quantify contact loss at the two embedment extremes and determine whether the alternative external free-length assumption changes the peak-displacement assessment."})

    first = json.loads((ROOT/"results/condition_transfer_repaired/assessment.json").read_text(encoding="utf-8"))
    adapted = json.loads((directory/"assessment.json").read_text(encoding="utf-8"))
    old = {r["trial_id"]: r for r in first["records"]}
    keys = ["rmse_kN", "peak_error_percent", "peak_displacement_error_mm", "late_error_kN"]
    comparison = [{"trial_id": r["trial_id"], "role": r["role"],
        "first_prediction_model": {k: old[r["trial_id"]][k] for k in keys},
        "adapted_model": {k: r[k] for k in keys}} for r in adapted["records"]]
    write_json(output/"model_comparison.json", {
        "evaluation_kind": "adaptive_assessment_against_preserved_first_prediction_set",
        "records": comparison,
        "pressure_prediction_rankings": [{"run_ref": report["run_ref"],
            "peak_order_low_to_high": [r["trial_id"] for r in sorted(
                [r for r in report["records"] if r["experiment_id"] == "pressure_transfer" and r["role"] == "prediction"],
                key=lambda r: r["peak_predicted_kN"])],
            "late_60mm_order_low_to_high": [r["trial_id"] for r in sorted(
                [r for r in report["records"] if r["experiment_id"] == "pressure_transfer" and r["role"] == "prediction"],
                key=lambda r: r["late_predicted_kN"])]} for report in [first, adapted]]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"results/condition_transfer_contact")
    parser.add_argument("--output", type=Path, default=ROOT/"results/validation")
    args = parser.parse_args()
    validate(args.directory.resolve(), args.output.resolve())
