# Real Bayesian Optimisation (Julia)

This repository implements a **Bayesian Optimisation (BO) framework** in Julia, leveraging Gaussian Processes (GPs) for surrogate modelling and the Expected Improvement (EI) acquisition function. It is applied to a calibration problem defined in `calibration.jl`.

---

## Features
- Gaussian Process regression using [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl).
- Support for maximisation and minimisation tasks.
- Utility functions for:
  - Scaling/unscaling candidate points.
  - Random sampling of initial candidates.
  - Surrogate modelling with hyperparameter optimisation.
  - Expected Improvement acquisition strategy.
- Visualisation of:
  - Posterior mean surface.
  - Uncertainty and error maps.
  - Best-so-far trace.
- Optional animated `.gif` output to track optimisation progress.

## Repository Structure

Files:
- RealBO.jl         Main module implementing Bayesian Optimisation
- calibration.jl    Calibration routines (external functions: Q, ideal, etc.)
- main.jl           Example script applying RealBO to the calibration problem
- README.md         Project documentation

In detail:
- **`RealBO.jl`**  
  Contains the `RealBO` module with Bayesian Optimisation routines:
  - `fit_gp!`: Fit a Gaussian Process surrogate.
  - `gp_predict_f`: Predict posterior mean/variance at new points.
  - `ei`: Expected Improvement acquisition function.
  - `bayesopt`: Main optimisation loop.

- **`calibration.jl`**  
  Defines the domain-specific calibration problem (functions such as `Q` and `ideal`).

- **`main.jl`**  
  Runs an example optimisation:
  1. Sets up the calibration problem.
  2. Defines a scaled objective function `scaled_Q`.
  3. Calls `RealBO.bayesopt` with user-defined bounds and hyperparameters.
  4. Saves an animated `.gif` of the optimisation process.

---

## Installation
Ensure that [Julia](https://julialang.org/) ≥ 1.6 is installed.  
Required packages:

```julia
using Pkg
Pkg.add([
    "Random",
    "LinearAlgebra",
    "Distributions",
    "Plots",
    "GaussianProcesses",
    "Statistics"
])
```

Additional Instructions are needed and described in installation.md


