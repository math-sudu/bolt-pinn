"""Test the short-anchor plateau, long-anchor mobilization, and C/K coupling.

All fits use original calibration curves. These are local response diagnostics,
not shared length laws or condition predictions. Saved condition predictions
are not read by this script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
import json
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bolt_pinn.constitutive import Interface
from bolt_pinn.incremental_solver import solve_path
from bolt_pinn.observation_operator import Observation
from bolt_pinn.pinn_fields import Geometry
from scripts.run_condition_transfer import (assess_curve, fit_schedule, init_worker,
    load_curves, objective_weights, write_json)


CASES = {
    "short_delayed": ("D36_L100_P5_5", "delayed", "contact"),
    "long_mobilized": ("D36_L500_P5_5", "mobilized", "contact"),
    "long_compliance": ("D36_L500_P5_5", "exponential", "repaired"),
    "long_mobilized_compliance": ("D36_L500_P5_5", "mobilized", "repaired"),
}


def material_from_parameters(theta, length, variant):
    base, peak, width = np.exp(theta[[0, 1, 3]])
    residual = peak*expit(theta[2])
    return Interface(float(base+peak/length), float(peak), float(residual),
        float(width+(peak-residual)/base), "exponential", length,
        mobilization_slip=float(np.exp(theta[4])) if variant == "mobilized" else None,
        yield_fraction=float(expit(theta[5])) if variant == "mobilized" else 1.0,
        softening_onset=float(np.exp(theta[4])) if variant == "delayed" else 0.0)


def fit_case(case, resume=False):
    trial_id, variant, compliance_source = CASES[case]
    started = time.monotonic()
    curve = next(c for c in load_curves(ROOT, "calibration") if c["record"]["trial_id"] == trial_id)
    initial_ref = f"results/length_diagnosis/{trial_id}/diagnosis.json"
    initial = json.loads((ROOT/initial_ref).read_text(encoding="utf-8"))
    shared_ref = f"results/condition_transfer_{compliance_source}/selected_fit.json"
    shared = json.loads((ROOT/shared_ref).read_text(encoding="utf-8"))
    config = shared["config"]
    geometry = Geometry(curve["record"]["embedment_mm"], curve["record"]["bolt_diameter_mm"],
                        config["youngs_modulus_MPa"])
    reference_compliance = shared["theta"][-1]
    observation = Observation.for_embedment(reference_compliance, geometry,
        config["reference_embedment_mm"], config["equal_total_length"])
    theta = np.array(initial["theta"])
    lower, upper = [-4.6, -2.3, -8, -2.3], [2.3, 3, 8, 5.7]
    names = ["log_base_stiffness", "log_strength_scale", "logit_residual_fraction", "log_positive_width"]
    if variant == "delayed":
        # The calibration trace has a plateau through about 24 mm. The onset
        # is a plastic-slip coordinate, so elastic slip is removed initially.
        theta[3] = np.log(2.0)
        theta = np.r_[theta, np.log(20.0)]
        lower += [-2.3]
        upper += [np.log(80.0)]
        names += ["log_softening_onset"]
    elif variant == "mobilized":
        theta = np.r_[theta, np.log(6.0), 0.0]
        lower += [-2.3, -8]
        upper += [np.log(80.0), 8]
        names += ["log_mobilization_slip", "logit_yield_fraction"]
    schedule = fit_schedule(curve)
    observed = np.interp(schedule[1:], curve["x"], curve["y"])
    weights = objective_weights(schedule, curve["record"]["image_peak_kN"])
    output = ROOT/"results/length_shape"/case
    output.mkdir(parents=True, exist_ok=True)
    calls, best = 0, float("inf")
    if resume:
        theta = np.array(json.loads((output/"best_candidate.json").read_text(encoding="utf-8"))["theta"])
    last_theta, last_residual = None, None

    def objective(parameters):
        nonlocal calls, best, last_theta, last_residual
        if last_theta is not None and np.array_equal(parameters, last_theta):
            return last_residual.copy()
        material = material_from_parameters(parameters, geometry.length_mm, variant)
        rows, _, _ = solve_path(schedule, material, geometry, observation,
            **config["fit_solver"], seed=config["random_seed"], verbose=False)
        residual = (np.array([r["reaction_kN"] for r in rows[1:]])-observed)*weights
        cost = float(residual@residual)
        calls += 1
        if cost < best:
            best = cost
            write_json(output/"best_candidate.json", {"theta": parameters.tolist(), "cost": cost,
                "case": case, "path_solves": calls})
        if calls % 10 == 1:
            print(case, calls, float(np.sqrt(cost)), flush=True)
        last_theta, last_residual = parameters.copy(), residual.copy()
        return residual

    def jacobian(parameters):
        baseline = objective(parameters)
        columns = []
        for i in range(len(parameters)):
            # Relative differences collapse near zero logits/log parameters.
            # Match the full-path, absolute step used by shared calibration.
            step = config["optimizer"]["log_parameter_difference"]
            if parameters[i]+step > upper[i]:
                step = -step
            perturbed = parameters.copy()
            perturbed[i] += step
            columns.append((objective(perturbed)-baseline)/step)
        return np.column_stack(columns)

    settings = config["optimizer"]
    fit = least_squares(objective, theta, bounds=(lower, upper), jac=jacobian,
        ftol=settings["ftol"], xtol=settings["xtol"], gtol=settings["gtol"])
    if not fit.success:
        raise RuntimeError(fit.message)
    material = material_from_parameters(fit.x, geometry.length_mm, variant)
    write_json(output/"selected_fit.json", {"case": case, "trial_id": trial_id,
        "theta": fit.x.tolist(), "parameter_names": names, "material": asdict(material),
        "observation": asdict(observation), "geometry": asdict(geometry),
        "config": config, "initial_ref": initial_ref, "compliance_ref": shared_ref,
        "C_500_mm_per_kN": reference_compliance,
        "fit_cost": float(fit.fun@fit.fun), "optimizer_message": fit.message,
        "active_bounds": fit.active_mask.tolist(), "bounds": {"lower": lower, "upper": upper},
        "parameter_difference": {"kind": "absolute_full_path", "step": settings["log_parameter_difference"]},
        "role": "Same-curve diagnostic on an original calibration record, not a condition prediction."})
    rows, fields, _ = solve_path(np.r_[0, curve["x"]], material, geometry, observation,
        **config["final_solver"], seed=config["random_seed"], output=output, verbose=False)
    x = np.array([r["measured_displacement_mm"] for r in rows])
    force = np.array([r["reaction_kN"] for r in rows])
    replay = (np.interp(schedule[1:], x, force)-observed)*weights
    checkpoints = [4, 8, 12, 16, 20, 24, 28, 32, 40, 60, 80]
    result = {**assess_curve(curve, rows), "case": case, "variant": variant,
        "material": asdict(material), "observation": asdict(observation),
        "C_500_mm_per_kN": reference_compliance,
        "fit_weighted_relative_rms": float(np.linalg.norm(fit.fun)),
        "replay_weighted_relative_rms": float(np.linalg.norm(replay)),
        "original_independent_fit_rmse_kN": initial["rmse_kN"],
        "checkpoints": [{"displacement_mm": d, "predicted_kN": float(np.interp(d, x, force)),
                         "observed_kN": float(np.interp(d, curve["x"], curve["y"]))} for d in checkpoints],
        "max_balance_error_kN": max(abs(r["balance_error_kN"]) for r in rows),
        "max_observation_error_mm": max(abs(r["observation_error_mm"]) for r in rows),
        "minimum_history_increment_mm": float(np.diff(fields[:, :, 5], axis=0).min()),
        "minimum_incremental_work_MPa_mm": min(r["min_incremental_dissipation_MPa_mm"] for r in rows),
        "active_bounds": fit.active_mask.tolist(), "path_solves": calls+1,
        "runtime_seconds": time.monotonic()-started,
        "evaluation_kind": "same_curve_diagnostic_fit"}
    write_json(output/"diagnosis.json", result)
    (output/"best_candidate.json").unlink()
    print(case, "completed", json.dumps(result), flush=True)
    return result


def summarize():
    records = [json.loads((ROOT/"results/length_shape"/case/"diagnosis.json").read_text(encoding="utf-8"))
               for case in CASES]
    directory = ROOT/"results/length_diagnosis/D36_L500_P5_5"
    original = json.loads((directory/"diagnosis.json").read_text(encoding="utf-8"))
    rows = np.genfromtxt(directory/"response.csv", delimiter=",", names=True)
    curve = next(c for c in load_curves(ROOT, "calibration")
                 if c["record"]["trial_id"] == "D36_L500_P5_5")
    reference_compliance = next(r["C_500_mm_per_kN"] for r in records if r["case"] == "long_compliance")
    displacement = (rows["measured_displacement_mm"]
        +(reference_compliance-original["observation"]["compliance_mm_per_kN"])*rows["reaction_kN"])
    if not np.all(np.diff(displacement) > 0):
        raise ValueError("The changed-compliance path folds in measured displacement")
    peak = int(rows["reaction_kN"].argmax())
    error = np.interp(curve["x"], displacement, rows["reaction_kN"])-curve["y"]
    refitted = next(r for r in records if r["case"] == "long_compliance")
    short_original = json.loads((ROOT/"results/length_diagnosis/D36_L100_P5_5/diagnosis.json").read_text(encoding="utf-8"))
    short_delayed = next(r for r in records if r["case"] == "short_delayed")
    result = {"kind": "length_response_shape_diagnosis",
        "question": "Can plateau onset and mobilization express the calibration-visible shapes, and can reference compliance alone explain the long-anchor peak?",
        "data_role": "Original 100 and 500 mm calibration curves only. No condition-prediction responses enter these fits.",
        "records": records,
        "unchanged_state_compliance_reparameterization": {
            "source_ref": "results/length_diagnosis/D36_L500_P5_5/response.csv",
            "equation": "delta_new = delta_old + (C_new - C_old) * F_kN",
            "C_old_mm_per_kN": original["observation"]["compliance_mm_per_kN"],
            "C_new_mm_per_kN": reference_compliance,
            "peak_displacement_old_mm": original["peak_displacement_predicted_mm"],
            "peak_displacement_new_mm": float(displacement[peak]),
            "peak_displacement_observed_mm": curve["record"]["image_peak_displacement_mm"],
            "rmse_on_original_samples_kN": float(np.sqrt(np.mean(error**2))),
            "original_rmse_kN": original["rmse_kN"],
            "role": "Exact observation change on unchanged states; compare with the refitted compliance cases above."},
        "compliance_stiffness_coupling": [{
            "case": name, "C_500_mm_per_kN": r["observation"]["compliance_mm_per_kN"],
            "stiffness_MPa_per_mm": r["material"]["stiffness"], "rmse_kN": r["rmse_kN"],
            "peak_measured_displacement_mm": r["peak_displacement_predicted_mm"],
            "peak_collar_displacement_mm": r["peak_displacement_predicted_mm"]
                -r["observation"]["compliance_mm_per_kN"]*r["peak_predicted_kN"]}
            for name, r in [("original_exponential", original), ("refitted_compliance", refitted)]],
        "short_anchor_parameter_coupling": [{
            "case": name, "residual_fraction": r["material"]["residual"]/r["material"]["peak"],
            "critical_slip_mm": r["material"]["critical_slip"], "rmse_kN": r["rmse_kN"]}
            for name, r in [("original_exponential", short_original), ("delayed_softening", short_delayed)]],
        "parameter_interpretation": "Curve-specific effective fits distinguish representation from sharing; they are not measured material properties or a validated length law.",
        "implementation_refs": ["bolt_pinn/constitutive.py", "tests/test_mechanics.py"],
        "reproduce_command": "python scripts/diagnose_length_shape.py"}
    write_json(ROOT/"results/length_shape/diagnosis.json", result)
    print(json.dumps([{k: r[k] for k in ["case", "rmse_kN", "peak_predicted_kN",
          "peak_displacement_predicted_mm", "replay_weighted_relative_rms"]} for r in records], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue saved candidates for selected unfinished cases")
    args = parser.parse_args()
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=len(args.cases), initializer=init_worker, initargs=(1,)) as pool:
            list(pool.map(partial(fit_case, resume=args.resume), args.cases))
    if set(args.cases) == set(CASES):
        summarize()
