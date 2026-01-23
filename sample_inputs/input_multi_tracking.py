import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from fenitop.topopt import topopt

# ============================================================
#  MESH
# ============================================================

# Thin worm-like beam: 100 × 4 rectangle
mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, -2.0], [100.0, 2.0]],
    [300, 12],
    cell_type=CellType.quadrilateral,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, -2.0], [100.0, 2.0]],
        [300, 12],
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
    "B_app_mag": 8.0,
    "B_app_dir": (0.0, 1.0),


    # ============================================================
    #  TRACTION BOUNDARY CONDITIONS (mechanical load)
    # ============================================================
    "traction_bcs" : [
    ],

    "load_cases": [
        {
            "name": "magnetic_up",
            "weight": 1.0,
            "B_app_mag": 20.0,
            "B_app_dir": (0.0, 1.0),
            "tractions": {},
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
    "max_iter": 600,
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
    "beta_interval": 100,
    "beta_max": 32.0,

    # Optimizer
    "use_oc": False,
    "move": 0.02,

    "stress_constraint": False,
    "strain_constraint": False,     

    # Objective
    "objective_type": "disp_track", # compliance, max_disp, disp_track

    # ============================================================
    #  MULTI-POINT DISPLACEMENT TRACKING (worm arc)
    # ============================================================

    "disp_track": [
        {
            "point": (30.0, 0.0),
            "target": (0.0, 1.0),
            "components": ("y",),
            "sigma": 10.0,
            "weight": 1.0,
        },
        {
            "point": (60.0, 0.0),
            "target": (0.0, 2.5),
            "components": ("y",),
            "sigma": 10.0,
            "weight": .5,
        },
        {
            "point": (100.0, 0.0),
            "target": (0.0, 4.0),
            "components": ("y",),
            "sigma": 10.0,
            "weight": 2.0,
        },
    ],



    # Output
    "output_dir": "./trackingFINAL_results/",
    "sim_output_interval": 25,
    "sim_image_output_interval": 25,
}

# ============================================================
#  RUN
# ============================================================
if __name__ == "__main__":
    topopt(fem_params, opt)