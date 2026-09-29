"""Check the nonlinear return in a real PINN path and exercise shared fitting."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bolt_pinn.constitutive import Interface
from bolt_pinn.incremental_solver import solve_path
from bolt_pinn.observation_operator import Observation
from scripts.run_condition_transfer import (Calibration, assess_curve, fingerprints, fit_schedule,
                                           init_worker, load_curves, objective_weights, write_json)


def pilot_material(theta):
    stiffness, peak, width = np.exp(theta[[0, 1, 3]])
    residual = peak*expit(theta[2])
    return Interface(float(stiffness), float(peak), float(residual),
                     float(width+(peak-residual)/stiffness), "exponential")


def main(pilot_only=False):
    init_worker(2)
    config_path = ROOT/"configs/condition_transfer.json"
    config = json.loads(config_path.read_text())
    curves = load_curves(ROOT, "calibration")
    curve = next(c for c in curves if c["record"]["trial_id"] == "P1_0.5")
    schedule = fit_schedule(curve)
    observed = np.interp(schedule[1:], curve["x"], curve["y"])
    weights = objective_weights(schedule, curve["record"]["image_peak_kN"])
    calls = 0
    started = time.monotonic()

    def objective(theta):
        nonlocal calls
        rows, _, _ = solve_path(schedule, pilot_material(theta), observation=Observation(0.01),
            **config["fit_solver"], seed=config["random_seed"], verbose=False)
        pred = np.array([r["reaction_kN"] for r in rows[1:]])
        calls += 1
        residual = (pred-observed)*weights
        print(f"pilot_evaluation={calls} weighted_relative_rms={np.linalg.norm(residual):.6f}", flush=True)
        return residual

    initial = np.array([np.log(0.35), np.log(2.3), np.log(0.6/1.7), np.log(16)])
    settings = config["optimizer"]
    fit = least_squares(objective, initial, diff_step=0.02,
        ftol=settings["ftol"], xtol=settings["xtol"], gtol=settings["gtol"])
    if not fit.success:
        raise RuntimeError(fit.message)
    output = ROOT/"results/method/exponential_pilot"
    rows, fields, meta = solve_path(np.r_[0.0, curve["x"]], pilot_material(fit.x),
        observation=Observation(0.01), **config["final_solver"], seed=config["random_seed"],
        output=output, verbose=False)
    metrics = assess_curve(curve, rows)
    x = np.array([r["measured_displacement_mm"] for r in rows])
    f = np.array([r["reaction_kN"] for r in rows])
    tail = x >= 40
    metrics.update(tail_rmse_kN=float(np.sqrt(np.mean((f[tail]-np.interp(x[tail], curve["x"], curve["y"]))**2))),
        material=meta["material"], observation=meta["observation"], optimizer_message=fit.message,
        optimizer_evaluations=calls, runtime_seconds=time.monotonic()-started,
        max_observation_error_mm=max(abs(r["observation_error_mm"]) for r in rows),
        minimum_history_increment_mm=float(np.diff(fields[:, :, 5], axis=0).min()),
        min_incremental_dissipation_MPa_mm=min(r["min_incremental_dissipation_MPa_mm"] for r in rows),
        role="Method development on an existing calibration record. C_500=0.01 is a declared scenario.")
    write_json(output/"calibration.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)

    if pilot_only:
        return

    # Run the actual multi-condition objective and its finite-difference path at
    # the starting configuration. This exercises both families before E3 fitting.
    from concurrent.futures import ProcessPoolExecutor
    with tempfile.TemporaryDirectory(prefix="bolt_method_") as temporary, ProcessPoolExecutor(
            max_workers=config["workers"], initializer=init_worker,
            initargs=(config["threads_per_worker"],)) as pool:
        calibration = Calibration(curves, config, pool, Path(temporary), fingerprints(ROOT, config_path))
        theta = np.array(config["initial_parameters"])
        residual = calibration(theta)
        jacobian = calibration.jacobian(theta)
    if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
        raise FloatingPointError("Non-finite shared fitting response or sensitivity")
    write_json(ROOT/"results/method/verification.json", {
        "pilot_ref": "results/method/exponential_pilot/calibration.json",
        "shared_initial_weighted_relative_rms": float(np.linalg.norm(residual)),
        "calibration_trial_ids": [c["record"]["trial_id"] for c in curves],
        "parameter_count": len(theta), "residual_count": len(residual),
        "jacobian_shape": list(jacobian.shape), "path_solves": calibration.path_solves,
        "role": "Real-path method execution; prediction curves are not used."})
    print("Shared objective and finite-difference paths completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-only", action="store_true")
    main(parser.parse_args().pilot_only)
