import os
import sys

# Add project root (FinalTop) to python path so fenitop can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from mpi4py import MPI

from dolfinx.mesh import create_rectangle, CellType
from dolfinx import fem

# Import evaluation driver
from fenitop.evaluate import evaluate


# ============================================================
#  EXPERIMENT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ============================================================
#  MESH  (must match optimization mesh exactly)
# ============================================================

mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [100.0, 5.0]],
    [150, 8],
    cell_type=CellType.quadrilateral,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
    [[0.0, 0.0], [100.0, 5.0]],
    [150, 8],
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

    # Base material
    "shear_modulus": 100.0,
    "poisson's ratio": 0.49,

    "hyperelastic": True,
    "hyperModel": "stVenant",   # overridden in sweep
    "G_model": "default",       # overridden in sweep

    # BCs
    "disp_bc": lambda x: np.isclose(x[0], 0.0),


    # Loads
    "body_force": (0.0, 0.0),

    "traction_bcs": [
        {
            "name": "out_right",
            "traction_max": (0.0, 0.0),
            "on_boundary": lambda x: np.isclose(x[0], 100.0),
        },
    ],

    "load_cases": [
        {
            "name": "B_up",
            "weight": 1.0,
            "B_app_mag": 100.0,
            "B_app_dir": (0.0, 1.0),
            #"B_app_dir": (1.0, 0),
            "tractions": {},
        },
    ],

    "load_steps": 8,

    # Quadrature
    "quadrature_degree": 2,

    # Magnetic
    "mu0": 1.256e3,
    "B_rem_mag": 50.0,
    "B_rem_dir": (1.0, 0.0),   # fallback if theta inactive
    "B_app_mag": 0.0,
    "B_app_dir": (0.0, 1.0),

    # PETSc
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "snes_max_it": "500",
        "snes_error_if_not_converged": None,
    },
}


# ============================================================
#  EVALUATION CONFIG
# ============================================================

eval_config = {

    "G_models": ["default", "guth", "mooney", "kerner"],
    "hyperelastic_models": ["stVenant", "neoHookean1", "neoHookean2"],

    "output_dir": RESULTS_DIR,

    "write_bp": True,
    "write_csv": True,
    "csv_name": "snake_model_comparison.csv",

    "compute_max_disp": True,
    "compute_compliance": False,
}


# ============================================================
#  DESIGN BUILDER  (Snake magnetization)
# ============================================================

def build_snake_design(mesh):

    # CG1 scalar function space (same as evaluate expects)
    V = fem.functionspace(mesh, ("CG", 1))

    ndofs = V.dofmap.index_map.size_local

    coords = V.tabulate_dof_coordinates()

    x = coords[:, 0]

    L = np.max(x)

    # --------------------------------------------------------
    # rho field (constant 1)
    # --------------------------------------------------------

    rho = np.ones(ndofs)

    # --------------------------------------------------------
    # phi field (constant = phi_cap)
    # --------------------------------------------------------

    phi_cap = 0.30
    phi = np.full(ndofs, phi_cap)

    # --------------------------------------------------------
    # theta field (snake rotation)
    #
    # theta(x) = π/2 + 2π x/L
    #
    # produces:
    # (0,1) -> (-1,0) -> (0,-1) -> (1,0) -> (0,1)
    # --------------------------------------------------------

    #theta = np.pi/2 + 2*np.pi * x / L  #one rotation
    #theta = np.pi/2 + 4*np.pi * x / L    # two rotation
    theta = 2*np.pi * x / L #one rotation, shifted 

    return rho, phi, theta


design_source = {
    "type": "callable",
    "builder": build_snake_design,
}


# ============================================================
#  RUN EVALUATION
# ============================================================

if __name__ == "__main__":
    evaluate(fem_params, eval_config, design_source)