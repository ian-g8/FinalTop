import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from fenitop.topopt import topopt

# ============================================================
# Mesh
# ============================================================

# Simple 2D cantilever: 100 × 15 rectangle
mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [100.0, 15.0]],
    [150, 22],
    cell_type=CellType.quadrilateral,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [100.0, 15.0]],
        [150, 22],
        CellType.quadrilateral
    )
else:
    mesh_serial = None

# ============================================================
# FEM Parameters
# ============================================================
fem_params = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,

    # Mechanical model
    "shear_modulus": 100.0,      # base shear modulus G0
    "poisson's ratio": 0.49,      # only used for Kerner model if selected
    "hyperelastic": True,
    "hyperModel": "neoHookean2",

    # Shear modulus microstructure model
    # options: "default", "guth", "mooney", "kerner"
    "G_model": "mooney",

    # Boundary conditions
    # Fix left edge (x=0)
    "disp_bc": lambda x: np.isclose(x[0], 0.0),

    # External body force
    "body_force": (0.0, 0.0),

    "quadrature_degree": 2,

    # Magnetic parameters
    "mu0": 1.256e3,                 # magnetic permeability
    "B_rem_mag": 250.0,            # remanent field magnitude
    "B_rem_dir": (1.0, 0.0),        # direction of remanent field (x-direction)
    "B_app_mag": 0.0,
    "B_app_dir": (0.0, 1.0),

    # Traction boundary conditions
    "traction_bcs": [
        {
            # Right boundary (used for objectives and optional tractions)
            "name": "out_right",
            "traction_max": (0.0, 0.0),
            "on_boundary": lambda x: np.isclose(x[0], 100.0),
        },
    ],

    "load_cases": [
        {
            "name": "B_up_MooneyN2",
            "weight": 1.0,
            "B_app_mag": 500.0,
            "B_app_dir": (0.0, 1.0),
            "tractions": {},   # none
        },
    ],

    # Load stepping
    "load_steps": 30,

    # PETSc solver
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "snes_max_it": "500",
        "snes_error_if_not_converged": None,
    },
}

# ============================================================
# Optimization Parameters
# ============================================================
opt = {
    "max_iter": 100,
    "opt_tol": 1e-5,

    # Volume fraction contraint for density
    "vol_frac_rho": 0.50,

    # Magnetic material volume fraction constraints
    "vol_frac_phi": 0.10,
    "phi_cap": 0.30,

    # Passive zones (none)
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone":  lambda x: np.full(x.shape[1], False),

    # SIMP Parameters
    "penalty": 3.0,
    "epsilon": 1e-6,

    # Filter parameters
    "filter_radius": 1.5,
    "beta_interval": 100,
    "beta_max": 32.0,

    # Optimizer
    "use_oc": False,
    "move": 0.01,   # original 0.02

    # Stress constraint
    "stress_constraint": False,
    "stress_pnorm": 12,
    "sigma_max": 0.15,   

    # Strain-energy constraint
    "strain_constraint": False,

    # Base (fallback if ramping disabled)
    "U_max": 0.15,

    # Strain constraint ramping
    "strain_ramp": {
        "enabled": False,        # master toggle
        "U_start": 0.35,         # initial U_max
        "U_end": 0.15,           # final U_max (hard cap)
        "start_iter": 1,        # iteration to start ramping
        "end_iter": 100,         # iteration to finish ramping
        "schedule": "linear",   # "linear" or "exp"
    },

    # Compliance constraint 
    "compliance_constraint": False,
    "compliance_ref": 0.7, 
    "compliance_gamma": 1.0,  

    # Objective
    "objective_type": "max_disp_norm", # compliance, max_disp, disp_track
    "enforce_volume_equality": True,

    # Output
    "output_dir": "./results_MAXX_MooneyN2/",
    "sim_output_interval": 5,
    "sim_image_output_interval": 5,
}

# ============================================================
# Design Variables
# ============================================================
design_variables = {
    "rho": {
        "active": False,
    },
    "phi": {
        "active": True, 
    },
    "theta": {
        "active": False,  
    },
}

# ============================================================
# Run Optimization
# ============================================================
if __name__ == "__main__":
    topopt(fem_params, opt, design_variables=design_variables)