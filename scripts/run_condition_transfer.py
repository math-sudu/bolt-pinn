"""Calibrate shared parameters, freeze them, and replay excluded conditions.

Each parameter evaluation starts a fresh incremental PINN at virgin history.
Workers solve independent paths; finite differences only rerun affected curves.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bolt_pinn.condition_model import parameter_names, decode, describe
from bolt_pinn.incremental_solver import solve_path

LOG = logging.getLogger("condition_transfer")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def fingerprints(root, config_path):
    paths = ["data/modelling_assumptions.json", "data/model_inputs.json", "data/response_samples.csv",
             "data/pressure_curves_digitized.csv", "data/length_curves_digitized.csv"]
    data = hashlib.sha256()
    for name in paths:
        data.update(name.encode())
        data.update((root/name).read_bytes())
    return {"config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "data_fingerprint": data.hexdigest()}


def load_curves(root, role, experiment_ids=None):
    """Only materialize responses for the requested role."""
    inputs = json.loads((root/"data/model_inputs.json").read_text(encoding="utf-8"))
    records = [r for r in inputs["records"] if r["role"] == role
               and (experiment_ids is None or r["experiment_id"] in experiment_ids)]
    ids = {r["trial_id"] for r in records}
    with (root/inputs["samples_ref"]).open(encoding="utf-8", newline="") as stream:
        selected = {key: [] for key in ids}
        for row in csv.DictReader(stream):
            if row["trial_id"] in selected:
                selected[row["trial_id"]].append(row)
    return [{"record": r,
             "x": np.array([float(p["displacement_mm"]) for p in selected[r["trial_id"]]]),
             "y": np.array([float(p["force_kN"]) for p in selected[r["trial_id"]]])}
            for r in records]


def fit_schedule(curve):
    x = curve["x"]
    candidates = np.r_[np.arange(0.5, 16.1, 0.5), np.arange(18, 40.1, 2),
                       np.arange(44, 100, 4), x[-1]]
    return np.r_[0.0, np.unique(candidates[(candidates >= x[0]) & (candidates <= x[-1])])]


def objective_weights(schedule, peak):
    """Displacement-quadrature weights prevent sampling density setting importance."""
    x = schedule[1:]
    weights = np.zeros_like(x)
    labels = np.where(x <= 16, 0, np.where(x <= 40, 1, 2))
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        if len(ids) == 1:
            mass = np.ones(1)
        else:
            mass = np.r_[(x[ids][1]-x[ids][0])/2,
                         (x[ids][2:]-x[ids][:-2])/2,
                         (x[ids][-1]-x[ids][-2])/2]
        weights[ids] = np.sqrt(mass/mass.sum()/len(np.unique(labels)))/peak
    return weights


def init_worker(threads):
    torch.set_num_threads(threads)


def solve_job(job):
    theta, record, config, schedule, final, output = job
    material, geometry, observation = decode(theta, record, config)
    try:
        rows, _, metadata = solve_path(schedule, material, geometry, observation,
            seed=config["random_seed"], output=output, verbose=False,
            **config["final_solver" if final else "fit_solver"])
    except (RuntimeError, FloatingPointError) as error:
        error.add_note(f"trial_id={record['trial_id']} theta={np.asarray(theta).tolist()}")
        raise
    return rows, metadata


def map_jobs(pool, jobs):
    return list(map(solve_job, jobs)) if pool is None else list(pool.map(solve_job, jobs))


class Calibration:
    def __init__(self, curves, config, pool, output, provenance):
        self.curves, self.config, self.pool = curves, config, pool
        self.output, self.provenance = output, provenance
        self.schedules = [fit_schedule(c) for c in curves]
        self.observed = [np.interp(s[1:], c["x"], c["y"]) for c, s in zip(curves, self.schedules)]
        self.weights = [objective_weights(s, c["record"]["image_peak_kN"])/np.sqrt(len(curves))
                        for c, s in zip(curves, self.schedules)]
        self.slices = []
        offset = 0
        for obs in self.observed:
            self.slices.append(slice(offset, offset+len(obs)))
            offset += len(obs)
        self.size = offset
        self.calls, self.path_solves = 0, 0
        self.best_cost = float("inf")
        self.last_theta, self.last_residual = None, None

    def jobs(self, theta, indices):
        return [(theta, self.curves[i]["record"], self.config, self.schedules[i], False, None)
                for i in indices]

    def residual_for(self, rows, index):
        pred = np.array([r["reaction_kN"] for r in rows[1:]])
        return (pred-self.observed[index])*self.weights[index]

    def __call__(self, theta):
        if self.last_theta is not None and np.array_equal(theta, self.last_theta):
            return self.last_residual.copy()
        values = map_jobs(self.pool, self.jobs(theta, range(len(self.curves))))
        self.path_solves += len(values)
        residual = np.concatenate([self.residual_for(rows, i) for i, (rows, _) in enumerate(values)])
        cost = float(residual@residual)
        self.calls += 1
        LOG.info("evaluation=%d weighted_relative_rms=%.6f C_500=%.6f path_solves=%d",
                 self.calls, np.sqrt(cost), theta[-1], self.path_solves)
        with (self.output/"evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"evaluation": self.calls, "cost": cost,
                                     "theta": theta.tolist()})+"\n")
        if cost < self.best_cost:
            self.best_cost = cost
            write_json(self.output/"best_candidate.json", {**self.provenance,
                "theta": theta.tolist(), "parameter_names": parameter_names(self.config),
                "cost": cost, "evaluation": self.calls})
        self.last_theta, self.last_residual = theta.copy(), residual.copy()
        return residual

    def jacobian(self, theta, columns=None):
        baseline = self(theta)
        jobs, locations = [], []
        for col, name in enumerate(parameter_names(self.config)):
            if columns is not None and col not in columns:
                continue
            compliance = name == "C_500_mm_per_kN"
            step = self.config["optimizer"]["compliance_difference_mm_per_kN" if compliance
                                             else "log_parameter_difference"]
            candidate = theta.copy()
            candidate[col] += step
            for i, curve in enumerate(self.curves):
                pressure = curve["record"]["experiment_id"] == "pressure_transfer"
                if compliance or name.startswith("shared_") or (name.startswith("pressure_") and pressure) or (name.startswith("length_") and not pressure):
                    jobs.extend(self.jobs(candidate, [i]))
                    locations.append((col, i, step))
        values = map_jobs(self.pool, jobs)
        self.path_solves += len(values)
        jac = np.zeros((self.size, len(theta)))
        for (col, i, step), (rows, _) in zip(locations, values):
            jac[self.slices[i], col] = (self.residual_for(rows, i)-baseline[self.slices[i]])/step
        return jac


def assess_curve(curve, rows):
    x = np.array([r["measured_displacement_mm"] for r in rows])
    force = np.array([r["reaction_kN"] for r in rows])
    use = (x >= curve["x"][0]) & (x <= curve["x"][-1])
    truth = np.interp(x[use], curve["x"], curve["y"])
    peak = int(force.argmax())
    record = curve["record"]
    return {"trial_id": record["trial_id"], "role": record["role"],
        "experiment_id": record["experiment_id"],
        "rmse_kN": float(np.sqrt(np.mean((force[use]-truth)**2))),
        "peak_predicted_kN": float(force[peak]), "peak_observed_kN": record["image_peak_kN"],
        "peak_displacement_predicted_mm": float(x[peak]),
        "peak_displacement_observed_mm": record["image_peak_displacement_mm"],
        "late_displacement_mm": 60.0,
        "late_predicted_kN": float(np.interp(60, x, force)),
        "late_observed_kN": float(np.interp(60, curve["x"], curve["y"])),
        "end_displacement_mm": float(x[-1]), "end_predicted_kN": float(force[-1]),
        "end_observed_kN": float(curve["y"][-1])}


def run(config_path, resume=False):
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT/config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output/"best_candidate.json").exists():
        raise FileExistsError(f"Existing fit in {output}; use --resume or a distinct output directory")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output/"run.log", encoding="utf-8")])
    provenance = fingerprints(ROOT, config_path)
    LOG.info("run_started=%s resume=%s config=%s", started_at, resume, config_path)
    experiments = config.get("experiment_ids")
    curves = load_curves(ROOT, "calibration", experiments)
    lower, upper = np.array(config["bounds_lower"]), np.array(config["bounds_upper"])
    # Positive collar slip requires C_total <= delta/F. Only calibration forces
    # constrain this apparatus parameter; no measured force transforms an input.
    compliance_bounds = []
    for curve in curves:
        positive = curve["y"] > 0
        extra = curve["record"]["assumed_extra_free_compliance_mm_per_kN"] if config["equal_total_length"] else 0
        compliance_bounds.extend(curve["x"][positive]/curve["y"][positive]-extra)
    upper[-1] = min(upper[-1], min(compliance_bounds))
    if upper[-1] <= 0:
        raise ValueError("Calibration response conflicts with the assumed positive-slip observation")
    theta = np.array(config["initial_parameters"])
    if theta[-1] > upper[-1]:
        theta[-1] = upper[-1]/2
    names = parameter_names(config)
    fixed = config.get("fixed_parameters", {})
    for name, value in fixed.items():
        theta[names.index(name)] = value
    if np.any(theta < lower) or np.any(theta > upper):
        raise ValueError("Initial or fixed parameters fall outside calibration bounds")
    active = np.array([i for i, name in enumerate(names) if name not in fixed])
    init_worker(config["threads_per_worker"])
    with ProcessPoolExecutor(max_workers=config["workers"], initializer=init_worker,
                             initargs=(config["threads_per_worker"],)) as pool:
        selected_path = output/"selected_fit.json"
        if resume and selected_path.exists():
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            if any(selected[key] != value for key, value in provenance.items()):
                raise ValueError("Frozen fit differs from the active data/configuration")
            theta = np.array(selected["theta"])
        else:
            if resume:
                saved = json.loads((output/"best_candidate.json").read_text(encoding="utf-8"))
                if any(saved[key] != value for key, value in provenance.items()):
                    raise ValueError("Candidate differs from the active data/configuration")
                theta = np.array(saved["theta"])
            objective = Calibration(curves, config, pool, output, provenance)
            settings = config["optimizer"]
            scale = np.r_[np.ones(len(theta)-1), 0.02]
            def expand(values):
                full = theta.copy()
                full[active] = values
                return full
            fit = least_squares(lambda values: objective(expand(values)), theta[active],
                jac=lambda values: objective.jacobian(expand(values), active)[:, active],
                bounds=(lower[active], upper[active]), x_scale=scale[active],
                ftol=settings["ftol"], xtol=settings["xtol"], gtol=settings["gtol"])
            if not fit.success:
                raise RuntimeError(f"Calibration failed: {fit.message}")
            theta = expand(fit.x)
            _, singular, directions = np.linalg.svd(fit.jac*scale[active], full_matrices=False)
            selected = {**provenance, "parameter_names": parameter_names(config), "theta": theta.tolist(),
                "config": config, "cost": float(fit.fun@fit.fun),
                "optimizer_message": fit.message, "optimizer_evaluations": objective.calls,
                "optimizer_settings": settings,
                "path_solves": objective.path_solves, "active_bounds": fit.active_mask.tolist(),
                "fitted_parameter_names": [names[i] for i in active], "fixed_parameters": fixed,
                "C_500_upper_from_calibration": float(upper[-1]),
                "calibration_records": [c["record"]["trial_id"] for c in curves],
                "local_sensitivity": {"parameter_scale": scale[active].tolist(),
                    "singular_values": singular.tolist(), "weakest_scaled_direction": directions[-1].tolist(),
                    "role": "Local sensitivity of calibration response; assess consequential C/K ambiguity before physical interpretation."},
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "prediction_status": "Parameters frozen before loading prediction responses"}
            write_json(selected_path, selected)
        assessments = []
        # Prediction responses become available only after the selected fit exists.
        for role in ["calibration", "prediction"]:
            phase_curves = curves if role == "calibration" else load_curves(ROOT, role, experiments)
            jobs = [(theta, c["record"], config, np.r_[0.0, c["x"]], True,
                     output/"curves"/c["record"]["trial_id"]) for c in phase_curves]
            for curve, (rows, _) in zip(phase_curves, map_jobs(pool, jobs)):
                metric = assess_curve(curve, rows)
                assessments.append(metric)
                LOG.info("%s %s RMSE=%.3f kN peak=%.3f kN at %.2f mm", role,
                         metric["trial_id"], metric["rmse_kN"], metric["peak_predicted_kN"],
                         metric["peak_displacement_predicted_mm"])
        write_json(output/"metrics.json", {"evaluation_kind": config.get("evaluation_kind", "retrospective_condition_transfer"),
            "records": assessments, "parameters": {c["record"]["trial_id"]: describe(theta, c["record"], config)
                for c in curves+load_curves(ROOT, "prediction", experiments)}})
    command = f"python scripts/run_condition_transfer.py --config {config_path.relative_to(ROOT).as_posix()}"
    if resume:
        command += " --resume"
    write_json(output/"execution.json", {**provenance, "command": command, "exit_code": 0,
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic()-started, "transcript_ref": f"{config['output_directory']}/run.log",
        "run_output_paths": [f"{config['output_directory']}/{name}" for name in
                             ["selected_fit.json", "metrics.json", "execution.json"]]
            + [f"{config['output_directory']}/curves/{metric['trial_id']}/{name}"
               for metric in assessments
               for name in ["response.csv", "fields.npz", "run.json", "final_state.pt"]]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/condition_transfer.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args.config.resolve(), args.resume)
    except Exception:
        LOG.exception("Run failed")
        raise
