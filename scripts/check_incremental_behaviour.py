"""Check the boundary scenario and unloading history that affect feasibility."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bolt_pinn.constitutive import Interface
from bolt_pinn.incremental_solver import solve_path
from bolt_pinn.observation_operator import Observation

if __name__=="__main__":
    torch.set_num_threads(2)
    fitted=json.loads((ROOT/"results/prototype/calibrated/calibration.json").read_text())
    material=Interface(**fitted["material"])
    base=ROOT/"results/prototype"
    boundary,_,_=solve_path(np.arange(0,22),material,observation=Observation(0.02),output=base/"compliance_scenario")
    cycle,fields,_=solve_path([0,2,4,6,8,10,12,11,10,11,12,14,18,30],material,output=base/"history_check")
    baseline=list(csv.DictReader((base/"calibrated/response.csv").open()))
    def envelope(rows):
        return {"max_balance_error_kN":max(abs(float(r["balance_error_kN"])) for r in rows),
            "max_observation_error_mm":max(abs(float(r["observation_error_mm"])) for r in rows),
            "max_equilibrium_rms_kN":max(float(r["equilibrium_rms_kN"]) for r in rows),
            "max_connection_rms_mm":max(float(r["connection_rms_mm"]) for r in rows),
            "min_incremental_dissipation_MPa_mm":min(float(r["min_incremental_dissipation_MPa_mm"]) for r in rows)}
    report={"calibrated_path":envelope(baseline),"compliance_scenario":envelope(boundary),
        "history_check":envelope(cycle),
        "compliance_scenario_peak_displacement_mm":max(boundary,key=lambda r:r["reaction_kN"])["measured_displacement_mm"],
        "compliance_scenario_peak_kN":max(r["reaction_kN"] for r in boundary),
        "unload_reload_force_difference_kN":cycle[10]["reaction_kN"]-cycle[6]["reaction_kN"],
        "unloading_max_history_change_mm":float(np.abs(fields[8,:,5]-fields[6,:,5]).max()),
        "minimum_history_increment_mm":float(np.diff(fields[:,:,5],axis=0).min()),
        "interpretation":"C=0.02 mm/kN is an assumed sensitivity scenario, not measured fixture compliance. The synthetic unload/reload checks state management; the experimental records contain no resolved unloading path."}
    (base/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
