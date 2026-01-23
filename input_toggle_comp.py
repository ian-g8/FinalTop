import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from fenitop.topopt import topopt

# ============================================================
#  MESH
# ============================================================

# Simple 2D cantilever: 100 × 20 rectangle
mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [100.0, 20.0]],
    [200, 40],
    cell_type=CellType.quadrilateral,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [100.0, 20.0]],
        [200, 40],
        CellType.quadrilateral
    )
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
    # Fix left edge (x=0)
    "disp_bc": lambda x: np.isclose(x[0], 0.0),

    # --- Body force (none) ---
    "body_force": (0.0, 0.0),

    "quadrature_degree": 2,

    # ============================================================
    #  MAGNETIC PARAMETERS
    # ============================================================
    "mu0": 1.256e3,                 # magnetic permeability
    "B_rem_mag": 40.0,              # remanent field magnitude
    "B_rem_dir": (1.0, 0.0),        # direction of remanent field (x-direction)
    "B_app_mag": 0,               # applied field magnitude ZERO
    "B_app_dir": (0.0, 1.0),        # applied field in +y direction

    # ============================================================
    #  TRACTION BOUNDARY CONDITIONS (mechanical load)
    # ============================================================
    "traction_bcs" : [
        {
            "name": "right_edge",
            "on_boundary": lambda x: np.isclose(x[0], 100.0),
            "traction_max": (0.0, 0.0),  # overridden per load case
        }
    ],

    "load_cases": [
        {
            "name": "downward_pull",
            "weight": 1.0,
            "B_app_mag": 0.0,
            "B_app_dir": (0.0, 1.0),
            "tractions": {
                "right_edge": (0.0, -0.01),
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
    "max_iter": 300,
    "opt_tol": 1e-5,

    # Volume fraction for density
    "vol_frac_rho": 0.50,

    # Magnetic material volume fraction
    "vol_frac_phi": 0.10,
    "phi_cap": 0.30,

    # Passive zones (none)
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone":  lambda x: np.full(x.shape[1], False),

    # Penalty
    "penalty": 3.0,
    "epsilon": 1e-6,

    # FILTERING
    "filter_radius": 1.2,
    "beta_interval": 50,
    "beta_max": 32.0,

    # Optimizer
    "use_oc": False,
    "move": 0.02,

    "stress_constraint": False,
    "strain_constraint": False,     

    # Objective
    "objective_type": "compliance", # compliance, max_disp, disp_track

    # Output
    "output_dir": "./toggle_test_comp_results/",
    "sim_output_interval": 25,
    "sim_image_output_interval": 25,
}

# ============================================================
#  DESIGN VARIABLE TOGGLES
# ============================================================

design_variables = {
    "rho": {
        "active": True,
        "type": "density",   # placeholder, unused for now
    },
    "phi": {
        "active": False,
        "type": "scalar",    # placeholder, unused for now
    },
    "theta": {
        "active": False,
        "type": "angle",     # placeholder, unused for now
    },
}


# ============================================================
#  RUN
# ============================================================
if __name__ == "__main__":
    topopt(fem_params, opt, design_variables=design_variables)