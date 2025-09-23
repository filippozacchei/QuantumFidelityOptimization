# Installation Guide

This document explains how to set up and run the repository in a **reproducible Julia environment**, including applying the required patch to **IonSim.jl**.

---

## Requirements
- Julia ≥ 1.6
- Git (to clone repositories)

---

## Step 1: Clone the Repository
```bash
git clone git@github.com:filippozacchei/QuantumFidelityOptimization.git
cd QuantumFidelityOptimization
```

## Step 2: Activate Julia Environment

This project comes with a Project.toml and Manifest.toml to ensure reproducibility.

Open Julia in the project folder and run:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Step 3: Patch IonSim.jl

This project requires a small fix to IonSim.jl, related to the optimisation tolerances.

Put IonSim in development mode. 

```julia
using Pkg
Pkg.develop("IonSim")
```

This creates an editable copy of IonSim in:

```bash
~/.julia/dev/IonSim
```

Apply the modification

Navigate to:

```bash
~/.julia/dev/IonSim/src/iontraps.jl
```

At line 495 (or search for Optim.Options), replace the line with:

```bash
Optim.Options(g_tol=1e-6, x_abstol=1e-12, x_reltol=1e-6,
              f_abstol=1e-12, f_reltol=1e-6)
``` 

Save the file.

Rebuild IonSim

Back in Julia:

```julia
using Pkg
Pkg.build("IonSim")
```

## Step 4: Verify the Patch

Check that your modification is in place:

```bash
grep -n "Optim.Options" ~/.julia/dev/IonSim/src/iontraps.jl
```

You should see:
```bash
Optim.Options(g_tol=1e-6, x_abstol=1e-12, x_reltol=1e-6,
              f_abstol=1e-12, f_reltol=1e-6)
```

## Step 5: Run the Repository

Execute the main script:
```bash
julia main.jl
```

This will:
	•	Run Bayesian Optimisation with the RealBO module.
	•	Generate progress plots.
	•	Save an animation (if specified in main.jl).

⸻

📝 Notes
	•	If you later want to undo your IonSim modifications, simply run:

```julia
using Pkg
Pkg.free("IonSim")
```
