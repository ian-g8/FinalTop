# top_optim

**top_optim** is a research fork of [FEniTop](https://github.com/missionlab/fenitop), an open-source topology optimization framework built on [FEniCSx](https://fenicsproject.org).

This version extends FEniTop for topology optimization of **hard-magnetic soft materials (hMSMs)**. The code supports hyperelastic finite-element models, magneto-mechanical coupling, magnetic material distribution optimization, remanence-direction optimization, and model-comparison evaluation.

## Project Overview

The main goal of this code is to optimize magnetic soft material designs by controlling:

- **rho**: mechanical material density
- **phi**: magnetic material fraction / magnetic density distribution
- **theta**: remanent magnetization direction

The current formulation supports different active design-variable choices. For example, a run may optimize only `phi` while keeping `rho = 1`, or may optimize combinations of `rho`, `phi`, and `theta`.

The code is currently focused on compliance, displacement, and rotation-based objectives under applied magnetic fields and/or mechanical tractions.

## Main Modules

- **topopt.py**  
  Runs the topology optimization loop, manages active design variables, applies filters, handles multi-load-case aggregation, calls sensitivities, updates the design with MMA, and writes output files.

- **fem.py**  
  Builds the finite-element problem in FEniCSx, including hyperelastic material models, magneto-mechanical energy terms, boundary conditions, load cases, objective forms, constraint forms, and derivative forms.

- **sensitivity.py**  
  Computes objective and constraint sensitivities using adjoint solves. Supports gradients with respect to `rho`, `phi`, and `theta`.

- **parameterize.py**  
  Provides the original FEniTop density filter and Heaviside projection tools.

- **optimize.py**  
  Provides the original FEniTop OC and MMA optimizers.

- **utility.py**  
  Provides plotting, communication, linear-problem utilities, and the added `WrapNonlinearProblem` class for nonlinear finite-element solves.

- **evaluate.py**  
  Evaluates a fixed optimized design across different material models and exports displacement/compliance comparison data.

## hMSM Formulation

The magneto-mechanical model uses a hyperelastic mechanical energy combined with a magnetic energy term. The effective magnetic material fraction is represented using

```text
phi_eff = rho * phi