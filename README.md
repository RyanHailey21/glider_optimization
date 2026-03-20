# Glider Optimization

Two-phase optimization workflow for a passive, drop-launched RC glider.

This version of the repository uses a point-mass trajectory model in Phase 2, with alpha represented as an optimized schedule (not rigid-body-emergent alpha dynamics).

## What The Code Does

The solver runs in two phases:

1. Phase 1 static MDO (per candidate wing airfoil)
- Builds or loads NeuralFoil aerodynamic surrogate grids.
- Solves a static trim NLP for minimum sink rate.
- Outputs candidate designs with geometry, mass, and trim metrics.

2. Phase 2 trajectory NLP
- Advances top-k Phase 1 designs (ranked by sink rate).
- Uses `DynamicsPointMass2DSpeedGamma` in AeroSandbox.
- Optimizes total flight time from 60 ft release to touchdown.
- Treats alpha as a low-order piecewise-linear control schedule, with alpha-rate and smoothness penalties.

## Repository Structure

- `main.py`: End-to-end pipeline and candidate selection.
- `config.py`: Constants, bounds, and tuning parameters.
- `mdo_solver.py`: Phase 1 static optimization and surrogate generation/cache.
- `trajectory.py`: Phase 2 point-mass trajectory optimization.
- `mass_model.py`: Structural mass and structural moment estimate.
- `visualization.py`: Figure generation plus CSV data export.
- `test_nf.py`: NeuralFoil/spline sanity script.
- `requirements.txt`: Python dependencies.

## Current Phase 2 Model (Important)

`trajectory.py` currently uses:
- Point-mass states: `x_e`, `z_e`, `speed`, `gamma`
- Indirect control variable: `alpha`
- Aerodynamic loads from `AeroBuildup(...).run()`
- Wind-axis force application: `dyn.add_force(*aero["F_w"], axes="wind")`

Alpha is parameterized by control points (`TRAJ_ALPHA_CTRL_POINTS`) and linearly interpolated in time. This is a pragmatic optimal-control formulation and is not a full rigid-body passive pitch simulation.

## Selection Logic

`main.py` evaluates the top `TOP_K_PHASE2` designs and compares trajectory outcomes.

Current behavior:
- Feasible solutions are preferred over infeasible ones.
- If all trajectories are infeasible, the script can still select the best available fallback trajectory (from debug values) by time.

## Outputs

On each run, the code produces:

- Figure files:
  - `glider_optimization_v2.png`
  - `glider_optimization_v2.pdf`

- Data exports:
  - `results/trajectory_timeseries.csv`
  - `results/design_summary.csv`

- Terminal summary:
  - Winner airfoil and build guide text

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
python main.py
```

## Key Configuration Parameters

Main knobs in `config.py`:

- Workflow
  - `TOP_K_PHASE2`
  - `N_STARTS_MDO`

- Surrogate generation/cache
  - `SURROGATE_USE_CACHE`
  - `SURROGATE_CACHE_DIR`
  - `SURROGATE_LOCAL_REFINEMENT`
  - `REFINE_*`

- Trajectory schedule and regularization
  - `TRAJ_ALPHA_CTRL_POINTS`
  - `TRAJ_ALPHA_DOT_MAX_DEG_S`
  - `TRAJ_ALPHA_SMOOTH_WEIGHT`
  - `TRAJ_CM_PENALTY_WEIGHT`

## Caching

NeuralFoil surrogate grids are cached in:

- `.cache/neuralfoil`

Cache filenames include a hash of the alpha/Re grid. If refined grid points change, new cache files are expected.

## Troubleshooting

- Missing NeuralFoil or import errors:
  - Verify the active interpreter is `.venv`.
  - Reinstall with `pip install -r requirements.txt`.

- Trajectory solve struggles or noisy alpha profile:
  - Increase `TRAJ_ALPHA_CTRL_POINTS` cautiously.
  - Increase `TRAJ_ALPHA_SMOOTH_WEIGHT`.
  - Re-check alpha-rate limit (`TRAJ_ALPHA_DOT_MAX_DEG_S`).

- Unexpected new cache files:
  - Usually caused by refined surrogate grids changing from run to run; this is normal.

## Dependencies

- aerosandbox
- neuralfoil
- casadi
- matplotlib
