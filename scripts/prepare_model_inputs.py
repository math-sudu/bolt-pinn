"""Sample declared condition-transfer inputs within the observed figure support."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def prepare(root: Path = ROOT):
    assumptions = json.loads((root / "data/modelling_assumptions.json").read_text(encoding="utf-8"))
    with (root / "data/test_registry.csv").open(encoding="utf-8", newline="") as stream:
        registry = {r["trial_id"]: r for r in csv.DictReader(stream)}
    records, samples, assigned = [], [], set()
    for experiment in assumptions["experiments"]:
        with (root / experiment["source_csv"]).open(encoding="utf-8", newline="") as stream:
            source = list(csv.DictReader(stream))
        calibration = experiment["calibration_records"]
        prediction = experiment["prediction_records"]
        if set(calibration) & set(prediction):
            raise ValueError(f"Calibration/prediction overlap in {experiment['experiment_id']}")
        schedule = np.arange(experiment["sample_start_mm"],
                             experiment["sample_stop_mm"] + experiment["sample_step_mm"] / 2,
                             experiment["sample_step_mm"])
        for role, identifiers in [("calibration", calibration), ("prediction", prediction)]:
            for trial_id in identifiers:
                if trial_id in assigned:
                    raise ValueError(f"Repeated analysis record assignment: {trial_id}")
                assigned.add(trial_id)
                trial = registry[trial_id]
                curve = [r for r in source if r["trial_id"] == trial_id]
                x = np.array([float(r["displacement_mm"]) for r in curve])
                if len(x) < 2 or not np.all(np.diff(x) > 0):
                    raise ValueError(f"Unusable displacement coordinates for {trial_id}")
                if schedule[0] < x[0] or schedule[-1] > x[-1]:
                    raise ValueError(f"Sampling would extrapolate {trial_id}")
                interpolated = {key: np.interp(schedule, x, [float(r[key]) for r in curve])
                                for key in ["force_kN", "force_low_kN", "force_high_kN"]}
                peak = max(curve, key=lambda r: float(r["force_kN"]))
                length = float(trial["embedment_mm"])
                diameter = float(trial["bolt_diameter_mm"])
                rigidity = assumptions["mechanics"]["steel_youngs_modulus_MPa"] * np.pi * diameter**2 / 4
                free_compliance = 1000 * (assumptions["observation"]["reference_embedment_mm"] - length) / rigidity
                records.append(dict(experiment_id=experiment["experiment_id"], trial_id=trial_id,
                    role=role, inferred_family=experiment["family_id"],
                    source_csv=experiment["source_csv"], source_image=curve[0]["source_image"],
                    source_anchors=trial["source_anchors"], embedment_mm=length,
                    bolt_diameter_mm=diameter, borehole_diameter_mm=float(trial["borehole_diameter_mm"]),
                    pressure_vertical_MPa=float(trial["pressure_vertical_MPa"]),
                    pressure_horizontal_MPa=float(trial["pressure_horizontal_MPa"]),
                    reported_BQ=trial["reported_BQ"],
                    assumed_extra_free_compliance_mm_per_kN=float(free_compliance),
                    image_peak_kN=float(peak["force_kN"]),
                    image_peak_displacement_mm=float(peak["displacement_mm"]),
                    physical_identity_status="unknown", prior_pilot_calibration=trial_id == "P1_0.5"))
                for j, delta in enumerate(schedule):
                    right = int(np.clip(np.searchsorted(x, delta), 1, len(x) - 1))
                    samples.append(dict(experiment_id=experiment["experiment_id"], trial_id=trial_id,
                        role=role, displacement_mm=float(delta),
                        **{key: float(values[j]) for key, values in interpolated.items()},
                        source_image=curve[0]["source_image"],
                        source_left_pixel_x=int(curve[right - 1]["pixel_x"]),
                        source_right_pixel_x=int(curve[right]["pixel_x"]),
                        evidence_kind="interpolated_digitized_report_figure"))
    output = root / "data/response_samples.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    payload = dict(kind="condition_transfer_inputs", assumptions_ref="data/modelling_assumptions.json",
                   samples_ref="data/response_samples.csv", records=records,
                   sample_count=len(samples), record_count=len(records),
                   evaluation_kind="retrospective_condition_transfer")
    (root / "data/model_inputs.json").write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    return payload


if __name__ == "__main__":
    prepared = prepare()
    print(f"Prepared {prepared['record_count']} curves and {prepared['sample_count']} response samples")
    for role in ["calibration", "prediction"]:
        print(role, [r["trial_id"] for r in prepared["records"] if r["role"] == role])
