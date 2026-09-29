"""Warm-started load increments with pure trial states and accepted-step commits.

Workflow adapted from H23 Algorithm 1 and DEM_Lib.py:404-464; mixed
first-order residuals replace the J2 energy functional and FE gradients.
"""
from __future__ import annotations
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .constitutive import Interface,State,return_map
from .pinn_fields import Geometry,MixedFields
from .observation_operator import Observation


def derivative(value,coordinate):
    return torch.autograd.grad(value,coordinate,torch.ones_like(value),create_graph=True)[0]


def evaluate(model,xi,delta,state,material):
    x=xi.detach().clone().requires_grad_(True)
    u,force=model(x,delta)
    tau,trial,diss=return_map(u,state,material)
    g=model.geometry
    equilibrium=(derivative(force,x)-g.perimeter_mm*g.length_mm*tau)/model.force_scale_N
    connection=(derivative(u,x)-g.length_mm*force/g.axial_rigidity_N)/model.axial_displacement_scale_mm
    return u,force,tau,trial,diss,equilibrium,connection


def solve_path(schedule,material: Interface,geometry=None,observation=None,*,points=65,check_points=257,width=16,max_iter=400,seed=7,output=None,verbose=True):
    """Replay from virgin history on every call, including after parameter changes.

    Check-grid material histories follow the entire path independently of the
    training grid; they are not interpolated from final plastic variables.
    """
    geometry=geometry or Geometry(); observation=observation or Observation()
    schedule=np.asarray(schedule,dtype=float)
    if schedule.ndim!=1 or not np.isfinite(schedule).all() or len(schedule)==0 or schedule[0]!=0:
        raise ValueError("Provide a finite one-dimensional schedule starting at zero")
    torch.manual_seed(seed)
    model=MixedFields(geometry,observation,width).double()
    xi=torch.linspace(0,1,points,dtype=torch.float64).reshape(-1,1)
    xc=torch.linspace(0,1,check_points,dtype=torch.float64).reshape(-1,1)
    state=State.zero(xi); check_state=State.zero(xc)
    cumulative_diss=torch.zeros_like(xc)
    rows=[]; fields=[]; start=time.monotonic()
    for step,delta in enumerate(schedule):
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=max_iter,
            tolerance_grad=1e-8,tolerance_change=1e-11,line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            *_,eq,cn=evaluate(model,xi,float(delta),state,material)
            loss=torch.mean(eq**2)+torch.mean(cn**2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite objective at step {step}")
            loss.backward()
            return loss
        optimizer.step(closure)
        iterations=optimizer.state[next(iter(model.parameters()))]["n_iter"]
        if iterations>=max_iter:
            uf,nf,tf,_,_,ef,cf=evaluate(model,xc,float(delta),check_state,material)
            diagnostics=dict(step=step,displacement_mm=float(delta),iterations=iterations,
                reaction_kN=float(nf[-1].detach())/1000,
                collar_displacement_mm=float(uf[-1].detach()),
                equilibrium_rms_kN=float(ef.detach().square().mean().sqrt())*model.force_scale_N/1000,
                connection_rms_mm=float(cf.detach().square().mean().sqrt())*model.axial_displacement_scale_mm,
                balance_error_kN=float((nf[-1]-torch.trapezoid(tf[:,0],xc[:,0])*geometry.length_mm*geometry.perimeter_mm).detach())/1000)
            raise RuntimeError(f"Optimizer exhausted; history not committed: {json.dumps(diagnostics)}")
        u,f,tau,trial,diss,eq,cn=evaluate(model,xc,float(delta),check_state,material)
        _,_,_,train_trial,_,_,_=evaluate(model,xi,float(delta),state,material)
        cumulative_diss=cumulative_diss+diss.detach()
        reaction=float(f[-1].detach())
        integral=float((torch.trapezoid(tau[:,0],xc[:,0])*geometry.length_mm*geometry.perimeter_mm).detach())
        row=dict(step=step,measured_displacement_mm=float(delta),reaction_kN=reaction/1000,
            collar_displacement_mm=float(u[-1].detach()),deep_end_displacement_mm=float(u[0].detach()),
            balance_error_kN=(reaction-integral)/1000,
            equilibrium_rms_kN=float(torch.mean(eq**2).sqrt().detach())*model.force_scale_N/1000,
            connection_rms_mm=float(torch.mean(cn**2).sqrt().detach())*model.axial_displacement_scale_mm,
            observation_error_mm=float(observation.measured_displacement(u[-1],f[-1]).detach())-float(delta),
            deep_end_force_kN=float(f[0].detach())/1000,
            min_incremental_dissipation_MPa_mm=float(diss.min().detach()),
            cumulative_interface_dissipation_J=float(torch.trapezoid(cumulative_diss[:,0],xc[:,0]))*geometry.length_mm*geometry.perimeter_mm/1000,
            max_accumulated_slip_mm=float(trial.accumulated_slip.max().detach()),iterations=iterations)
        rows.append(row)
        fields.append(np.column_stack([xc[:,0].numpy()*geometry.length_mm,u.detach().numpy()[:,0],f.detach().numpy()[:,0]/1000,
            tau.detach().numpy()[:,0],trial.plastic_slip.detach().numpy()[:,0],trial.accumulated_slip.detach().numpy()[:,0]]))
        # Exactly one history commit per accepted physical load increment.
        state=train_trial.committed(); check_state=trial.committed()
        if verbose and (step%10==0 or step==len(schedule)-1):
            print(f"step={step:3d} delta={delta:7.3f} F={row['reaction_kN']:8.3f} kN balance={row['balance_error_kN']:.4g} kN",flush=True)
    metadata=dict(material=asdict(material),geometry=asdict(geometry),observation=asdict(observation),
        schedule_mm=schedule.tolist(),points=points,check_points=check_points,width=width,seed=seed,
        max_iter=max_iter,runtime_seconds=time.monotonic()-start,torch_version=torch.__version__,
        field_columns=["x_mm","u_mm","force_kN","tau_MPa","plastic_slip_mm","accumulated_slip_mm"],
        evidence_kind="conditional_PINN_solution",history_policy="immutable_trial_commit_after_convergence")
    if output is not None:
        output=Path(output); output.mkdir(parents=True,exist_ok=True)
        with (output/"response.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (output/"run.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        np.savez_compressed(output/"fields.npz",fields=np.stack(fields))
        torch.save({"network":model.state_dict(),"training_state":state,"check_state":check_state,"metadata":metadata},output/"final_state.pt")
    return rows,np.stack(fields),metadata
