"""Finite, shared condition relationships for retrospective transfer.

Pressure inputs are apparatus labels, not measured interface normal stresses.
The length series has its own effective material because its preparation and
5:5 pressure regime are not established continuations of the pressure series.
The adapted length law represents surrounding-medium size effects and contact
loss with shared effective parameters, without measured local normal stresses.
"""
from dataclasses import asdict

import numpy as np
from scipy.special import expit

from .constitutive import Interface
from .observation_operator import Observation
from .pinn_fields import Geometry


PARAMETER_NAMES = [
    "pressure_log_K", "pressure_log_K_a",
    "pressure_log_peak", "pressure_log_peak_a", "pressure_log_peak_ratio",
    "pressure_logit_residual", "pressure_logit_residual_a",
    "pressure_logit_residual_ratio", "pressure_logit_residual_hump",
    "pressure_log_width", "pressure_log_width_a", "pressure_log_width_hump",
    "length_log_K", "length_log_peak", "length_logit_residual", "length_log_width",
    "C_500_mm_per_kN",
]


def parameter_names(config):
    model = config.get("length_model", "homogeneous")
    if model == "homogeneous":
        names = PARAMETER_NAMES[:-1]
    elif model == "relaxation_contact":
        names = PARAMETER_NAMES[:-1]+["length_log_relaxation_mm"]
    elif model == "shape_contact":
        names = PARAMETER_NAMES[:-1]+[
            "length_log_relaxation_mm", "length_residual_log_length_slope",
            "length_width_power", "length_log_mobilization_mm",
            "length_mobilization_power", "length_logit_yield_fraction",
            "length_log_onset_mm", "length_log_onset_decay_mm",
        ]
    else:
        raise ValueError(f"Unknown length model: {model}")
    if config.get("mobilization_model") == "shared":
        names = names+["shared_log_mobilization_mm", "shared_logit_yield_fraction"]
    elif config.get("mobilization_model") is not None:
        raise ValueError(f"Unknown mobilization model: {config['mobilization_model']}")
    experiments = config.get("experiment_ids", ["pressure_transfer", "length_transfer"])
    names = [name for name in names
             if not (name.startswith("pressure_") and "pressure_transfer" not in experiments)
             and not (name.startswith("length_") and "length_transfer" not in experiments)]
    return names+[PARAMETER_NAMES[-1]]


def decode(theta, record, config):
    """No response force, observed peak, BQ, or trial ID enters this mapping."""
    p = dict(zip(parameter_names(config), theta, strict=True))
    if record["experiment_id"] == "pressure_transfer":
        a = record["pressure_vertical_MPa"]-2.0
        ratio = record["pressure_horizontal_MPa"]/record["pressure_vertical_MPa"]
        hump = 4*ratio*(1-ratio)
        log_k = p["pressure_log_K"]+p["pressure_log_K_a"]*a
        log_peak = p["pressure_log_peak"]+p["pressure_log_peak_a"]*a+p["pressure_log_peak_ratio"]*ratio
        logit_r = (p["pressure_logit_residual"]+p["pressure_logit_residual_a"]*a
                   +p["pressure_logit_residual_ratio"]*ratio+p["pressure_logit_residual_hump"]*hump)
        log_width = p["pressure_log_width"]+p["pressure_log_width_a"]*a+p["pressure_log_width_hump"]*hump
    elif record["experiment_id"] == "length_transfer":
        log_k, log_peak, logit_r, log_width = (p[name] for name in PARAMETER_NAMES[12:16])
        if config.get("length_model") == "shape_contact":
            logit_r += p["length_residual_log_length_slope"]*np.log(record["embedment_mm"]/300)
    else:
        raise ValueError(f"Unknown experiment: {record['experiment_id']}")
    stiffness, peak, width = np.exp([log_k, log_peak, log_width])
    return_stiffness = stiffness
    contact_length = None
    if record["experiment_id"] == "length_transfer" and config.get("length_model") in {"relaxation_contact", "shape_contact"}:
        length = record["embedment_mm"]
        relaxation = np.exp(p["length_log_relaxation_mm"])
        # Average of depth recovery 1-exp(-z/ell) over the embedded column.
        fraction = 1+np.expm1(-length/relaxation)*relaxation/length
        peak *= fraction
        # Unit power recovers the effective column-strain hypothesis. The
        # shape model estimates this exponent from calibration responses.
        width *= (length/config["reference_embedment_mm"])**p.get("length_width_power", 1.0)
        contact_length = length if config.get("contact_evolution", True) else None
        stiffness += peak/length
    residual = peak*expit(logit_r)
    # Positive width is the excess above the unique-return lower bound.
    critical_slip = width+(peak-residual)/return_stiffness
    mobilization_slip, yield_fraction, softening_onset = None, 1.0, 0.0
    if config.get("mobilization_model") == "shared":
        mobilization_slip = float(np.exp(p["shared_log_mobilization_mm"])*record["embedment_mm"]/config["reference_embedment_mm"])
        yield_fraction = float(expit(p["shared_logit_yield_fraction"]))
    elif record["experiment_id"] == "length_transfer" and config.get("length_model") == "shape_contact":
        ratio = record["embedment_mm"]/config["reference_embedment_mm"]
        mobilization_slip = float(np.exp(p["length_log_mobilization_mm"])*ratio**p["length_mobilization_power"])
        yield_fraction = float(expit(p["length_logit_yield_fraction"]))
        # A surface-associated plateau contribution decays with embedment.
        # This is a fitted effective relationship, not a measured damage depth.
        softening_onset = float(np.exp(p["length_log_onset_mm"]
            -(record["embedment_mm"]-100)/np.exp(p["length_log_onset_decay_mm"])))
    material = Interface(float(stiffness), float(peak), float(residual),
                         float(critical_slip), config["weakening"], contact_length,
                         mobilization_slip, yield_fraction, softening_onset)
    geometry = Geometry(record["embedment_mm"], record["bolt_diameter_mm"],
                        config["youngs_modulus_MPa"])
    observation = Observation.for_embedment(p["C_500_mm_per_kN"], geometry,
        config["reference_embedment_mm"], config["equal_total_length"])
    return material, geometry, observation


def describe(theta, record, config):
    material, geometry, observation = decode(theta, record, config)
    return {"material": asdict(material), "geometry": asdict(geometry),
            "observation": asdict(observation)}
