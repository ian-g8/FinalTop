import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from fenitop.topopt import topopt

# ============================================================
#  MESH
# ============================================================

from dolfinx.mesh import create_rectangle, meshtags
import numpy as np

from mesh_Lbracket import mesh
mesh_serial = mesh if MPI.COMM_WORLD.rank == 0 else None

# Border thickness (grows inward from outer geometry)
border_thickness = 6.0



# Serial mesh (for plotting)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = mesh
else:
    mesh_serial = None



# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,

    # --- Mechanical model ---
    "shear_modulus": 100.0,      # base shear modulus G0
    "poisson's ratio": 0.49,      # only used for Kerner model if selected
    "hyperelastic": True,
    "hyperModel": "neoHookean2",

    # --- Shear modulus microstructure model ---
    # options: "default", "guth", "mooney", "kerner"
    "G_model": "guth",

    # --- Boundary conditions ---
    # Fix left vertical edge AND bottom horizontal edge
    "disp_bc": lambda x: np.logical_or(
        np.isclose(x[0], 0.0),
        np.isclose(x[1], 0.0)
    ),


    # --- Body force (none) ---
    "body_force": (0.0, 0.0),

    "quadrature_degree": 2,

    # ============================================================
    #  MAGNETIC PARAMETERS
    # ============================================================
    "mu0": 1.256e3,                 # magnetic permeability
    "B_rem_mag": 00.0,              # remanent field magnitude
    "B_rem_dir": (0.0, 0.0),        # direction of remanent field (x-direction)
    "B_app_mag": 0.0,
    "B_app_dir": (0.0, 1.0),


    # ============================================================
    #  TRACTION BOUNDARY CONDITIONS (mechanical load)
    # ============================================================
    "traction_bcs": [
        {
            "name": "corner_load_edge",
            "traction_max": (0.0, 0.0),
            # Exterior boundary: right edge, lower half only (matches "L" arm)
            "on_boundary": lambda x: np.logical_and(
                np.isclose(x[1], 50.0),
                x[0] >= 50.0
            ),
        },
    ],



    "load_cases": [
        # Baseline (0 deg)
        {
            "name": "down_0deg",
            "weight": .20,
            "B_app_mag": 0.0,
            "B_app_dir": (0.0, 0.0),
            "tractions": {
                "corner_load_edge": (0.0, -0.5),
            },
        },

        # +15 deg
        {
            "name": "down_p15deg",
            "weight": .20,
            "B_app_mag": 0.0,
            "B_app_dir": (0.0, 0.0),
            "tractions": {
                "corner_load_edge": (0.1294, -0.4829),
            },
        },

        # -15 deg
        {
            "name": "down_m15deg",
            "weight": .20,
            "B_app_mag": 0.0,
            "B_app_dir": (0.0, 0.0),
            "tractions": {
                "corner_load_edge": (-0.1294, -0.4829),
            },
        },

        # +30 deg
        {
            "name": "down_p30deg",
            "weight": .20,
            "B_app_mag": 0.0,
            "B_app_dir": (0.0, 0.0),
            "tractions": {
                "corner_load_edge": (0.25, -0.4330),
            },
        },

        # -30 deg
        {
            "name": "down_m30deg",
            "weight": .20,
            "B_app_mag": 0.0,
            "B_app_dir": (0.0, 0.0),
            "tractions": {
                "corner_load_edge": (-0.25, -0.4330),
            },
        },
    ],



    # Load stepping
    "load_steps": 5,

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
    "max_iter": 200,
    "opt_tol": 1e-5,

    # Volume fraction for density
    "vol_frac_rho": 0.50,

    # Magnetic material volume fraction
    "vol_frac_phi": 0.10,
    "phi_cap": 0.30,

    # Passive solid zones (enforce realistic boundary attachment)
    "solid_zone": lambda x: np.logical_or.reduce((
        # Left edge border: x = 0, y ∈ [25, 100], grow inward (+x)
        np.logical_and.reduce((
            x[0] >= 0.0,
            x[0] <= border_thickness,
            x[1] >= 25.0,
            x[1] <= 100.0,
        )),

        # Top edge border: y = 100, x ∈ [0, 50], grow inward (-y)
        np.logical_and.reduce((
            x[1] <= 100.0,
            x[1] >= 100.0 - border_thickness,
            x[0] >= 0.0,
            x[0] <= 50.0,
        )),

        # Inner vertical border: x = 50, y ∈ [50, 100], grow inward (-x)
        np.logical_and.reduce((
            x[0] <= 50.0,
            x[0] >= 50.0 - border_thickness,
            x[1] >= 50.0,
            x[1] <= 100.0,
        )),

        # Bottom edge border: y = 0, x ∈ [25, 75], grow inward (+y)
        np.logical_and.reduce((
            x[1] >= 0.0,
            x[1] <= border_thickness,
            x[0] >= 25.0,
            x[0] <= 75.0,
        )),
    )),



    "void_zone": lambda x: np.full(x.shape[1], False),


    # Penalty
    "penalty": 3.0,
    "epsilon": 1e-6,

    # FILTERING
    "filter_radius": 3,
    "beta_interval": 25,
    "beta_max": 64.0,

    # Optimizer
    "use_oc": False,
    "move": 0.02,   # original 0.02

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

    # Tip displacement constraint
    "disp_constraint": False,

    # Base (fallback if ramping disabled)
    "u_min": 0.05,

    # Displacement constraint ramping
    "disp_ramp": {
        "enabled": False,
        "u_start": 0.5,
        "u_end": 5,
        "start_iter": 1,
        "end_iter": 75,
        "schedule": "linear",
    },

    # Compliance constraint 
    "compliance_constraint": False,
    "compliance_ref": 0.7,  # start 0.255
    "compliance_gamma": 1.0,  

    # Objective
    "objective_type": "compliance", # compliance, max_disp, disp_track
    "enforce_volume_equality": False,

    # Output
    "output_dir": "./results_Lbracket_pert/",
    "sim_output_interval": 25,
    "sim_image_output_interval": 25,
}

# ============================================================
#  DESIGN VARIABLE TOGGLES
# ============================================================

design_variables = {
    "rho": {
        "active": True,
        "type": "density",  
    },
    "phi": {
        "active": False,
        "type": "scalar",    
    },
    "theta": {
        "active": False,
        "type": "angle",     
    },
}


# ============================================================
#  RUN
# ============================================================
if __name__ == "__main__":
    topopt(fem_params, opt, design_variables=design_variables)