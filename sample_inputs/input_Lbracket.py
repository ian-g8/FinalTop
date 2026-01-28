import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from fenitop.topopt import topopt

# ============================================================
#  MESH
# ============================================================

from dolfinx.mesh import create_rectangle, meshtags
import numpy as np

# Border thickness (grows inward from outer geometry)
from mesh_Lbracket import h_elem as mesh_h
border_thickness = 6.0 * mesh_h  # robust: multiple cell layers regardless of mesh resolution

from mesh_Lbracket import mesh
mesh_serial = mesh if MPI.COMM_WORLD.rank == 0 else None

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
    # Clamp:
    #  - full left edge: x = 0, y ∈ [0, 100]
    #  - top edge from x ∈ [0, 50] at y = 100
    # Clamped = u = (0, 0)
    "disp_bc": lambda x: np.logical_or(
        np.isclose(x[0], 0.0),
        np.logical_and(
            np.isclose(x[1], 100.0),
            x[0] <= 50.0
        )
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
            "name": "top_right_corner_load",
            "traction_max": (0.0, -.10),
            # Inner corner edge at y = 50, x ∈ [50, 100]
            "on_boundary": lambda x: np.logical_and(
                np.isclose(x[1], 50.0),
                x[0] >= 50.0
            ),
        },
    ],




    "load_cases": [
        {
            "name": "L_bracket_down",
            "weight": 1.0,
            "B_app_mag": 0.0,
            "B_app_dir": (0.0, 0.0),
            "tractions": {
                "top_right_corner_load": (0.0, -.10),
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
    "max_iter": 350,
    "opt_tol": 1e-5,

    # Volume fraction for density
    "vol_frac_rho": 0.60,

    # Magnetic material volume fraction
    "vol_frac_phi": 0.10,
    "phi_cap": 0.30,


    "rho_solid_value": 0.995,
    "rho_void_value": 0.05,

    # Simple enforced solid strips (DG0-friendly)

    "solid_zone": lambda x: np.logical_or(

        # --------------------------------------------
        # Left vertical strip: x in [0, 5]
        # Full height
        # --------------------------------------------
        (x[0] >= 0.0) & (x[0] <= 5.0),

        # --------------------------------------------
        # Top horizontal strip: y in [95, 100], x in [0, 50]
        # --------------------------------------------
        (x[1] >= 95.0) & (x[1] <= 100.0) & (x[0] <= 50.0)

    ),

    "void_zone": lambda x: np.full(x.shape[1], False),


    # Penalty
    "penalty": 3.0,
    "epsilon": 1e-6,

    # FILTERING
    "filter_radius": 3,
    "beta_interval": 50,
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
    "output_dir": "./results_Lbracket_baseline/",
    "sim_output_interval": 1,
    "sim_image_output_interval": 1,
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

centers = mesh.geometry.x.T
mask = opt["solid_zone"](centers)
print("Solid-zone cells:", np.count_nonzero(mask))

# DEBUG: check solid-zone coverage on DG0 centers
from dolfinx import fem
centers_dbg = fem.functionspace(mesh, ("DG", 0)).tabulate_dof_coordinates().T
mask_dbg = opt["solid_zone"](centers_dbg)
print("[DEBUG] solid_zone DG0 count:", np.count_nonzero(mask_dbg))

# ============================================================
#  RUN
# ============================================================
if __name__ == "__main__":
    topopt(fem_params, opt, design_variables=design_variables)