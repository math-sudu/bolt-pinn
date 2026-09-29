"""Run an illustrative loading, unloading and reloading path on the CPU."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bolt_pinn.constitutive import Interface
from bolt_pinn.incremental_solver import solve_path
from bolt_pinn.observation_operator import Observation
from bolt_pinn.pinn_fields import Geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/example")
    args = parser.parse_args()
    torch.set_num_threads(2)

    # These are illustrative inputs for exercising the implementation.
    material = Interface(10.0, 2.0, 0.4, 1.0, weakening="exponential")
    geometry = Geometry(length_mm=100.0, diameter_mm=25.0)
    observation = Observation(compliance_mm_per_kN=0.02)
    schedule = np.r_[np.linspace(0.0, 3.0, 13), 2.5, 2.0, 2.5, 3.0, 4.0]
    rows, _, metadata = solve_path(
        schedule, material, geometry, observation,
        points=33, check_points=65, width=12, max_iter=1000,
        output=args.output,
    )
    print(json.dumps({
        "output_directory": str(args.output.resolve()),
        "steps": len(rows),
        "peak_force_kN": max(row["reaction_kN"] for row in rows),
        "runtime_seconds": metadata["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
