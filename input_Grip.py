import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from fenitop.topopt import topopt

# ============================================================
#  MESH
# ============================================================

# Import U-gripper mesh (gmsh-based)
from mesh_U_gripper import mesh, mesh_serial, facet_tags



# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "facet_tags": facet_tags,


    # --- Mechanical model ---
    "shear_modulus": 100.0,      # base shear modulus G0
    "poisson's ratio": 0.49,      # only used for Kerner model if selected
    "hyperelastic": True,
    "hyperModel": "neoHookean2", # or stVenant, neoHookean2

    # --- Shear modulus microstructure model ---
    # options: "default", "guth", "mooney", "kerner"
    "G_model": "mooney",

    # --- Boundary conditions ---
    "disp_bc": lambda x: np.isclose(x[1], 0.0),



    # --- Body force (none) ---
    "body_force": (0.0, 0.0),

    "quadrature_degree": 2,

    # ============================================================
    #  MAGNETIC PARAMETERS
    # ============================================================
    "mu0": 1.256e3,                 # magnetic permeability
    "B_rem_mag": 60.0,              # remanent field magnitude
    "B_rem_dir": (1.0, 0.0),        # direction of remanent field (x-direction)
    "B_app_mag": 0.0,
    "B_app_dir": (0.0, 1.0),

    # Initial remanent direction for theta initialization (ONLY affects initial guess)
    # Does NOT affect physics once optimization starts
    "theta_init_dir": (0.0, 1.0),   # vertical initialization


    # ============================================================
    #  TRACTION BOUNDARY CONDITIONS (mechanical load)
    # ============================================================
    # No mechanical tractions (pure magnetic actuation)
    "traction_bcs": [],



    "load_cases": [
        {
            "name": "B_vertical_MooneyN2",
            "weight": 1.0,
            "B_app_mag": 60.0,
            "B_app_dir": (0.0, 1.0),
            "tractions": {},
        },
    ],


    # Load stepping
    "load_steps": 1,

    # PETSc solver
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "snes_max_it": "500",
        "snes_error_if_not_converged": None,
    },
}

# ============================================================
#  OPTIMIZATION PARAMETERS
# ============================================================

opt = {
    "max_iter": 100,
    "opt_tol": 1e-5,

    # Volume fraction for density
    "vol_frac_rho": 0.50,

    # Magnetic material volume fraction
    "vol_frac_phi": 0.125,
    "phi_cap": 0.30,

    # Passive zones (none)
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone":  lambda x: np.full(x.shape[1], False),

    # Penalty
    "penalty": 3.0,
    "epsilon": 1e-6,

    # FILTERING
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


    # ------------------------------------------------------------
    # OBJECTIVE REGULARIZATION (NOT A CONSTRAINT)
    # ------------------------------------------------------------

    # Compliance constraint 
    "compliance_constraint": False,
    "compliance_ref": 0.7,  # start 0.255
    "compliance_gamma": 1.0,  

    # Reference compliance for OBJECTIVE scaling ONLY
    # Use iteration-1 compliance (~3.4e-03 from your logs)
    "compliance_ref": 3.4e-03,

    # Objective
    # Directional boundary displacement objective (gripper closure)
    "objective_type": "boundary_disp",

    # ------------------------------------------------------------
    # OBJECTIVE BOUNDARIES (gripper jaws)
    # ------------------------------------------------------------
    "objective_bcs": [
        {
            "name": "left_jaw_inner",
            "marker": 20,                 # LEFT_JAW_INNER
            "direction": ( 1.0, 0.0),     # inward
            "weight": 1.1,
            "on_boundary": lambda x: False,
        },
        {
            "name": "right_jaw_inner",
            "marker": 21,                 # RIGHT_JAW_INNER
            "direction": (-1.0, 0.0),     # inward
            "weight": 1.0,
            "on_boundary": lambda x: False,
        },
    ],


    "enforce_volume_equality": False,

    # Output
    "output_dir": "./results_GRIP_MooneyN2/",
    "sim_output_interval": 20,
    "sim_image_output_interval": 20,
}

# ============================================================
#  DESIGN VARIABLE TOGGLES
# ============================================================

design_variables = {
    "rho": {
        "active": False,
        "type": "density",  
    },
    "phi": {
        "active": True,
        "type": "scalar",    
    },
    "theta": {
        "active": True,
        "type": "angle",     
    },
}


# ============================================================
#  RUN
# ============================================================
if __name__ == "__main__":
    topopt(fem_params, opt, design_variables=design_variables)