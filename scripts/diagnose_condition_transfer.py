"""Separate response-family error from condition-sharing error using real paths.

Independent fits are diagnostic only, including the already inspected P2_0.
They do not overwrite the saved condition-transfer fits. The P2_0 fit uses its
response and therefore is not a condition prediction.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bolt_pinn.condition_model import decode
from bolt_pinn.constitutive import Interface
from bolt_pinn.incremental_solver import solve_path
from scripts.check_transfer_method import pilot_material
from scripts.run_condition_transfer import (load_curves, fit_schedule, objective_weights,
    assess_curve, init_worker, write_json)


def fit_curve(curve):
    length_family = curve["record"]["experiment_id"] == "length_transfer"
    initial_ref = ("results/condition_transfer_contact/selected_fit.json" if length_family
                   else "results/condition_transfer_repaired/selected_fit.json")
    initial_fit = json.loads((ROOT/initial_ref).read_text(encoding="utf-8"))
    config = initial_fit["config"]
    material, geometry, observation = decode(initial_fit["theta"], curve["record"], config)
    base_stiffness = material.stiffness-(material.peak/geometry.length_mm if length_family else 0)
    theta = np.array([np.log(base_stiffness), np.log(material.peak),
        np.log(material.residual/(material.peak-material.residual)),
        np.log(material.critical_slip-(material.peak-material.residual)/base_stiffness)])
    schedule = fit_schedule(curve)
    observed = np.interp(schedule[1:], curve["x"], curve["y"])
    weights = objective_weights(schedule, curve["record"]["image_peak_kN"])
    calls, started = 0, time.monotonic()

    def local_material(parameters):
        base = pilot_material(parameters)
        if not length_family:
            return base
        return Interface(base.stiffness+base.peak/geometry.length_mm, base.peak,
                         base.residual, base.critical_slip, "exponential", geometry.length_mm)

    def objective(parameters):
        nonlocal calls
        rows, _, _ = solve_path(schedule, local_material(parameters), geometry, observation,
            **config["fit_solver"], seed=config["random_seed"], verbose=False)
        residual = (np.array([r["reaction_kN"] for r in rows[1:]])-observed)*weights
        calls += 1
        if calls % 10 == 1:
            print(curve["record"]["trial_id"], calls, float(np.linalg.norm(residual)), flush=True)
        return residual

    settings = config["optimizer"]
    fit = least_squares(objective, theta, diff_step=0.015,
        bounds=([-4.6, -2.3, -8, -2.3], [2.3, 3, 8, 5.7]),
        ftol=settings["ftol"], xtol=settings["xtol"], gtol=settings["gtol"])
    if not fit.success:
        raise RuntimeError(fit.message)
    output = ROOT/("results/length_diagnosis" if length_family else "results/pressure_diagnosis")/curve["record"]["trial_id"]
    rows, _, _ = solve_path(np.r_[0, curve["x"]], local_material(fit.x), geometry, observation,
        **config["final_solver"], seed=config["random_seed"], output=output, verbose=False)
    result = {**assess_curve(curve, rows), "material": asdict(local_material(fit.x)),
        "theta": fit.x.tolist(), "observation": asdict(observation),
        "weighted_relative_rms": float(np.linalg.norm(fit.fun)),
        "optimizer_message": fit.message, "active_bounds": fit.active_mask.tolist(),
        "path_solves": calls+1, "runtime_seconds": time.monotonic()-started,
        "evaluation_kind": "same_curve_diagnostic_fit", "initial_fit_ref": initial_ref,
        "purpose": "Separate local response-family error from shared condition relationships; this is not prediction performance."}
    write_json(output/"diagnosis.json", result)
    return result


def summarize(records):
    if records[0]["experiment_id"] == "length_transfer":
        selected = json.loads((ROOT/"results/condition_transfer_contact/selected_fit.json").read_text(encoding="utf-8"))
        inputs = {c["record"]["trial_id"]: c["record"] for c in load_curves(ROOT, "calibration")}
        comparisons = []
        for result in records:
            record = inputs[result["trial_id"]]
            shared, _, _ = decode(selected["theta"], record, selected["config"])
            length = record["embedment_mm"]
            shared_base = shared.stiffness-shared.peak/length
            comparisons.append({"trial_id": result["trial_id"], "embedment_mm": length,
                "independent_fit": {"base_stiffness_MPa_per_mm": float(np.exp(result["theta"][0])),
                    "strength_scale_MPa": result["material"]["peak"],
                    "residual_fraction": result["material"]["residual"]/result["material"]["peak"],
                    "positive_width_mm": float(np.exp(result["theta"][3])),
                    "critical_slip_mm": result["material"]["critical_slip"]},
                "shared_fit": {"base_stiffness_MPa_per_mm": shared_base,
                    "strength_scale_MPa": shared.peak, "residual_fraction": shared.residual/shared.peak,
                    "positive_width_mm": shared.critical_slip-(shared.peak-shared.residual)/shared_base,
                    "critical_slip_mm": shared.critical_slip}})
        return {"role": "Independent calibration-length diagnostic fits at fixed contact-run compliance.",
            "shared_fit_ref": "results/condition_transfer_contact/assessment.json",
            "records": records,
            "parameter_relationships": comparisons,
            "parameter_role": "Effective parameters from these local fits, not uniquely identified or measured material properties.",
            "question": "Can the same contact/exponential law represent each calibration curve, and do the required stiffness and decay widths support the current shared length relationships?"}
    widths = {r["trial_id"]: float(np.exp(r["theta"][3])) for r in records}
    endpoint_midpoint = float(np.sqrt(widths["P1_0"]*widths["P3_0"]))
    return {
        "role": "Diagnostic fits only; the P2_0 response is used and these results are not condition predictions.",
        "first_prediction_ref": "results/condition_transfer_repaired/assessment.json",
        "records": records,
        "width_relationship": {
            "diagnostic_widths_mm": widths,
            "endpoint_log_linear_midpoint_mm": endpoint_midpoint,
            "middle_to_endpoint_midpoint_ratio": widths["P2_0"]/endpoint_midpoint,
            "constraint": "At zero horizontal pressure the shared log-width law enforces width(2)^2 = width(1)*width(3).",
            "reading": "The independently fitted midpoint requires a wider decay than the two endpoint fits. A term proportional to 1-((p_v-2 MPa)/(1 MPa))^2 is zero at both calibration levels and thus cannot be learned from those levels."}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["pressure", "length"], default="pressure")
    family = parser.parse_args().family
    ids = ({"P1_0", "P2_0", "P3_0"} if family == "pressure"
           else {"D36_L100_P5_5", "D36_L300_P5_5", "D36_L500_P5_5"})
    curves = [c for role in ["calibration", "prediction"] for c in load_curves(ROOT, role)
              if c["record"]["trial_id"] in ids]
    with ProcessPoolExecutor(max_workers=3, initializer=init_worker, initargs=(1,)) as pool:
        records = list(pool.map(fit_curve, curves))
    write_json(ROOT/f"results/{family}_diagnosis/diagnosis.json", summarize(records))
