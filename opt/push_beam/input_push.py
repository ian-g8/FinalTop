import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fenitop.topopt import topopt

# ============================================================
#  MESH
# ============================================================

L = 10.0
H = 3.0

plate_width = 0.5
plate_x0 = L - plate_width

phi_buffer = 0.35
phi_void_x0 = plate_x0 - phi_buffer

mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [L, H]],
    [180, 54],
    cell_type=CellType.quadrilateral,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [L, H]],
        [180, 54],
        CellType.quadrilateral,
    )
else:
    mesh_serial = None


# ============================================================
#  GEOMETRIC HELPERS
# ============================================================

def left_clamp(x):
    return np.isclose(x[0], 0.0)

def right_edge(x):
    return np.isclose(x[0], L)

def output_plate_zone(x):
    return x[0] >= plate_x0

def phi_void_plate_buffer_zone(x):
    return x[0] >= phi_void_x0

def phi_active_left_zone(x):
    return x[0] <= 0.55 * L


def phi_void_zone(x):
    return ~phi_active_left_zone(x)


def false_zone(x):
    return np.full(x.shape[1], False)


# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,

    "shear_modulus": 100.0,
    "poisson's ratio": 0.49,
    "hyperelastic": True,
    "hyperModel": "neoHookean1",

    "G_model": "mooney",

    # Clamp left edge
    "disp_bc": left_clamp,

    "body_force": (0.0, 0.0),
    "quadrature_degree": 2,

    # Magnetic parameters
    "mu0": 1.256e3,
    "B_rem_mag": 50.0,
    #"B_rem_dir": (1.0, 0.0),
    "B_rem_dir": (0, 1.0),
    "theta_init_dir": (0.0, 1.0),

    # overwritten by load case
    "B_app_mag": 0.0,
    "B_app_dir": (-1.0, 0.0),

    # Dummy traction marker only
    "traction_bcs": [
        {
            "name": "dummy_right",
            "traction_max": (0.0, 0.0),
            "on_boundary": right_edge,
        },
    ],

    "load_cases": [
        {
            "name": "plate_push_Bx",
            "weight": 1.0,
            "B_app_mag": 100.0,
            "B_app_dir": (-1.0, 0.0),
            "tractions": {
                "dummy_right": (0.0, 0.0),
            },
        },
    ],

    "load_steps": 25,

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

    "vol_frac_rho": 0.40,

    "vol_frac_phi": 0.12,
    "phi_cap": 0.30,

    # Backward-compatible defaults
    "solid_zone": false_zone,
    "void_zone": false_zone,

    # Rho passive zones:
    # right output plate is mechanically solid
    "rho_solid_zone": output_plate_zone,
    "rho_void_zone": false_zone,

    # Phi passive zones:
    # no magnetic material in or near the output plate
    "phi_solid_zone": false_zone,
    "phi_void_zone": phi_void_zone,

    "penalty": 3.0,
    "epsilon": 1e-6,

    "filter_radius": 0.20,
    "beta_interval": 100,
    "beta_max": 4.0,

    "use_oc": False,
    "move": 0.01,

    "stress_constraint": False,
    "stress_pnorm": 12,
    "sigma_max": 1.0,

    "strain_constraint": False,
    "U_max": 1.0,
    "strain_ramp": {
        "enabled": False,
        "U_start": 1.0,
        "U_end": 1.0,
        "start_iter": 1,
        "end_iter": 100,
        "schedule": "linear",
    },

    "compliance_constraint": False,
    "compliance_ref": 1.0,
    "compliance_gamma": 1.0,

    "disp_constraint": False,

    # ============================================================
    #  OBJECTIVE: push right output plate in +x direction
    # ============================================================
    "objective_type": "boundary_disp",

    "objective_bcs": [
        {
            "name": "right_output_edge",
            "on_boundary": right_edge,
            "direction": (1.0, 0.0),
            "weight": 1.0,
        },
    ],

    "enforce_volume_equality": True,

    "output_dir": str(Path(__file__).resolve().parent / "results_TonguePush_Bleft_RhoPhiTheta"),
    "sim_output_interval": 1,
    "sim_image_output_interval": 5,
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
        "active": True,
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