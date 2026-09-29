# Incremental PINN for bolt load transfer

Python implementation accompanying **Loading-end displacement and irreversible
load transfer in sandstone bolt pull-out: An incremental physics-informed
model**, by Pengcheng Zhu and Tielin Chen.

The model couples the loading-end displacement to collar slip through the
predicted reaction and a series compliance. Separate displacement and axial-force
networks enforce compatibility and equilibrium. The interface return map advances
irreversible history after each accepted displacement increment. Calibration
re-solves the complete path for every material-parameter evaluation.

## Installation

Use Python 3.10 or later. The research environment used Python 3.13,
PyTorch 2.8.0, NumPy 2.4.4 and SciPy 1.16.2. Computations run on the CPU.

```sh
python -m pip install -r requirements.txt
```

## Run the standalone example

```sh
python examples/run_forward.py
python -m unittest discover -s tests -p test_mechanics.py -v
```

The example uses illustrative material and loading inputs and exercises loading,
unloading and reloading. It writes a response table, field histories, solver
metadata and a final network/state checkpoint under `results/example/`.

Units are mm for displacement and length, MPa for interface stress and Young's
modulus, N for internal axial force, kN for reported reaction, and mm/kN for
series compliance.

## Implementation

| Location | Purpose |
| --- | --- |
| `bolt_pinn/constitutive.py` | Linear/exponential weakening, mobilization, delayed weakening, contact loss and irreversible return mapping |
| `bolt_pinn/pinn_fields.py` | Separate displacement/force networks and embedded boundary conditions |
| `bolt_pinn/observation_operator.py` | Reaction-dependent loading-end observation and free-rod extension |
| `bolt_pinn/incremental_solver.py` | Incremental L-BFGS solves and accepted history updates |
| `bolt_pinn/condition_model.py` | Shared pressure and embedment-length parameter relationships |
| `scripts/prepare_model_inputs.py` | Sample condition curves within their source support |
| `scripts/run_condition_transfer.py` | Whole-path calibration and condition assessment |
| `scripts/assess_condition_transfer.py` | Response-stage and field evaluation |
| `scripts/diagnose_*.py`, `scripts/validate_transfer.py`, `scripts/profile_length_onset.py` | Response diagnostics, contact/compliance checks and onset sensitivity |
| `tests/` | Constitutive, observation, condition-transfer and input checks |

## Run with experimental inputs

Place the experimental input files in a local `data/` directory at the repository
root. The preprocessing entry point expects:

- `modelling_assumptions.json`: mechanics, observation assumptions, experiment
  families, calibration/prediction identifiers and sampling schedules;
- `test_registry.csv`: record identifiers, geometry, apparatus pressures and
  source anchors;
- `pressure_curves_digitized.csv` and `length_curves_digitized.csv`: displacement,
  force, digitization bounds and source coordinates.

The exact field names are defined by `scripts/prepare_model_inputs.py`.
Preprocessing produces `data/model_inputs.json` and `data/response_samples.csv`.

```sh
python scripts/prepare_model_inputs.py
python -m unittest discover -s tests -v
python scripts/run_condition_transfer.py --config configs/condition_transfer_repaired.json
python scripts/assess_condition_transfer.py --directory results/condition_transfer_repaired
python scripts/run_condition_transfer.py --config configs/length_transfer_shape.json
python scripts/assess_condition_transfer.py --directory results/length_transfer_shape
```

The full test command includes tests that read the experimental inputs. The
mechanics-only command above and the standalone example run without those files.

`condition_transfer_repaired.json` supplies the initial pressure/length model
with the solver capacity used in the manuscript. `length_transfer_shape.json`
supplies the length-adapted model. The contact and mobilization configurations
support the intermediate variants. Configuration files contain numerical solver
settings, parameter initialization/bounds and the output location. Adjust
`workers` and `threads_per_worker` to the available CPU resources.

Each fit writes `selected_fit.json` before evaluating the prediction conditions.
Use `--resume` to continue an existing fit or replay its selected parameters;
keep its configuration and input files unchanged. For a new fit, choose a new
`output_directory` in the configuration.

This repository distributes the implementation, configurations and tests.
Experimental inputs, generated outputs, manuscripts and figures are excluded
from version control.
