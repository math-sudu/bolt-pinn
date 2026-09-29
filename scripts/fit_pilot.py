"""Fit one effective interface to one report curve under declared boundary assumptions.

Every objective evaluation solves/replays the complete incremental PINN path.
This calibrates the implementation pilot, not independently validated material properties.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bolt_pinn.constitutive import Interface
from bolt_pinn.incremental_solver import solve_path


def decode(theta):
    stiffness,peak,width=np.exp(theta[[0,1,3]])
    residual=peak*expit(theta[2])
    return Interface(float(stiffness),float(peak),float(residual),float(width+(peak-residual)/stiffness))


if __name__=="__main__":
    torch.set_num_threads(2)
    data=[r for r in csv.DictReader((ROOT/"data/pressure_curves_digitized.csv").open(encoding="utf-8")) if r["trial_id"]=="P1_0.5"]
    x=np.array([float(r["displacement_mm"]) for r in data]); y=np.array([float(r["force_kN"]) for r in data])
    schedule=np.r_[0,np.arange(1,100,2)]
    observed=np.interp(schedule[1:],x,y)
    calls=0
    def objective(theta):
        global calls
        calls+=1
        rows,_,_=solve_path(schedule,decode(theta),points=33,check_points=129,width=12,verbose=False)
        residual=np.array([r["reaction_kN"] for r in rows[1:]])-observed
        print(f"evaluation={calls} rms_kN={np.sqrt(np.mean(residual**2)):.4f} parameters={decode(theta)}",flush=True)
        return residual
    initial=np.array([np.log(.33),np.log(2.3),np.log(.6/(2.3-.6)),np.log(24)])
    fit=least_squares(objective,initial,diff_step=0.02,ftol=0.002,xtol=0.002,gtol=0.002)
    material=decode(fit.x)
    final_schedule=np.unique(np.r_[np.arange(0,100),8.46])
    output=ROOT/"results/prototype/calibrated"
    rows,_,meta=solve_path(final_schedule,material,points=65,check_points=257,output=output)
    delta=np.array([r["measured_displacement_mm"] for r in rows])
    pred=np.array([r["reaction_kN"] for r in rows])
    truth=np.interp(delta,x,y)
    use=delta>=x.min()
    report={"trial_id":"P1_0.5","observation_source":"data/pressure_curves_digitized.csv",
        "calibration_window_mm":[1,99],"validation_status":"same_curve_calibration_only",
        "optimizer_success":bool(fit.success),"optimizer_message":fit.message,"objective_evaluations":calls,
        "material":meta["material"],"rms_kN":float(np.sqrt(np.mean((pred[use]-truth[use])**2))),
        "max_abs_error_kN":float(np.max(np.abs(pred[use]-truth[use]))),
        "model_peak_kN":float(pred.max()),"model_peak_displacement_mm":float(delta[pred.argmax()]),
        "image_peak_kN":float(y.max()),"image_peak_displacement_mm":float(x[y.argmax()]),
        "assumptions":["steel E=200000 MPa", "fixed 500 mm effective interface perimeter pi*25 mm", "rigid surrounding medium", "reported displacement equals collar slip; zero extra compliance"],
        "interpretation":"Effective parameters under assumed boundary; not intrinsic interface constants. Constant residual and fixed contact are tested assumptions, not established physical facts."}
    (output/"calibration.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
