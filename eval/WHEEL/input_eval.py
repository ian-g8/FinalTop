import os
import sys

# Add project root (FinalTop) to python path so fenitop can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from mpi4py import MPI

from dolfinx import fem

# Import evaluation driver
from fenitop.evaluate import evaluate


# ============================================================
#  WHEEL GEOMETRY PARAMETERS
# ============================================================

wheel = {
    "R": 1.0,
    "lc": 0.02,

    "r_inner": 0.9,
    "t": 0.0125,
    "r_hub": None,

    "phi_cap": 0.30,
}

if wheel["r_inner"] is None:
    wheel["r_inner"] = 0.9 * wheel["R"]

if wheel["t"] is None:
    wheel["t"] = 0.025 * wheel["R"]

if wheel["r_hub"] is None:
    wheel["r_hub"] = wheel["t"]


# ============================================================
#  BUILD WHEEL + SPOKES MESH
# ============================================================

def build_wheel_spokes_mesh(R=1.0, lc=0.05, comm=MPI.COMM_WORLD):

    import gmsh

    rank = comm.rank

    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("wheel_spokes")

        P1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
        P2 = gmsh.model.geo.addPoint(0.0, R,   0.0, lc)
        P3 = gmsh.model.geo.addPoint(R,   0.0, 0.0, lc)

        r = wheel["r_inner"]
        t = wheel["t"]

        P4 = gmsh.model.geo.addPoint(t, t, 0.0, lc)
        P5 = gmsh.model.geo.addPoint(t, r, 0.0, lc)
        P6 = gmsh.model.geo.addPoint(r, t, 0.0, lc)

        L1 = gmsh.model.geo.addLine(P1, P3)
        L2 = gmsh.model.geo.addCircleArc(P3, P1, P2)
        L3 = gmsh.model.geo.addLine(P2, P1)

        L4 = gmsh.model.geo.addLine(P4, P5)
        L5 = gmsh.model.geo.addCircleArc(P5, P1, P6)
        L6 = gmsh.model.geo.addLine(P6, P4)

        OuterLoop = gmsh.model.geo.addCurveLoop([L1, L2, L3])
        InnerLoop = gmsh.model.geo.addCurveLoop([L4, L5, L6])

        Surface = gmsh.model.geo.addPlaneSurface([OuterLoop, InnerLoop])

        surfs = [Surface]

        for k in [1, 2, 3]:
            cp = gmsh.model.geo.copy([(2, Surface)])
            gmsh.model.geo.rotate(cp, 0, 0, 0, 0, 0, 1, k*np.pi/2)
            surfs.append(cp[0][1])

        gmsh.model.geo.synchronize()

        gmsh.model.geo.removeAllDuplicates()
        gmsh.model.geo.synchronize()

        all_surfs = [tag for (dim, tag) in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, all_surfs, 1)
        gmsh.model.setPhysicalName(2, 1, "domain")

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

    from dolfinx.io.gmshio import model_to_mesh

    mesh, cell_tags, facet_tags = model_to_mesh(
        gmsh.model,
        comm,
        0,
        gdim=2
    )

    if rank == 0:
        gmsh.finalize()

    return mesh


mesh = build_wheel_spokes_mesh(R=wheel["R"], lc=wheel["lc"], comm=MPI.COMM_WORLD)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = build_wheel_spokes_mesh(R=wheel["R"], lc=wheel["lc"], comm=MPI.COMM_SELF)
else:
    mesh_serial = None


# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {

    "mesh": mesh,
    "mesh_serial": mesh_serial,

    "shear_modulus": 100.0,
    "poisson's ratio": 0.49,

    "hyperelastic": True,
    "hyperModel": "stVenant",
    "G_model": "default",

    "interior_BC": True,

    # Clamp hub region
    "disp_bc": lambda x: (np.abs(x[0]) <= 0.0125) & (np.abs(x[1]) <= 0.0125),

    "body_force": (0.0, 0.0),
    "traction_bcs": [],

    "load_cases": [
        {
            "name": "B_up",
            "weight": 1.0,
            "B_app_mag": 500.0,
            "B_app_dir": (0.0, 1.0),
            "tractions": {},
        },
    ],

    "load_steps": 100,

    "quadrature_degree": 2,

    "mu0": 1.256e3,
    "B_rem_mag": 200.0,
    "B_rem_dir": (1.0, 0.0),

    "B_app_mag": 500.0,
    "B_app_dir": (0.0, 1.0),

    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "snes_max_it": "500",
        "snes_error_if_not_converged": None,
    },
}


# ============================================================
#  EVALUATION SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

eval_config = {

    "G_models": ["default", "guth", "mooney", "kerner"],
    "hyperelastic_models": ["stVenant", "neoHookean1", "neoHookean2"],

    "output_dir": RESULTS_DIR,

    "write_bp": True,
    "write_csv": True,
    "csv_name": "wheel_model_comparison.csv",

    "compute_max_disp": True,
    "compute_compliance": False,
}


# ============================================================
#  DESIGN BUILDER (WHEEL MAGNETIZATION)
# ============================================================

def build_wheel_design(mesh):

    V = fem.functionspace(mesh, ("CG", 1))

    coords = V.tabulate_dof_coordinates()

    x = coords[:, 0]
    y = coords[:, 1]

    r = np.sqrt(x**2 + y**2)

    R = wheel["R"]
    r_inner = wheel["r_inner"]

    ndofs = coords.shape[0]

    rho = np.ones(ndofs)

    phi = np.zeros(ndofs)
    phi[(r > r_inner) & (r <= R)] = wheel["phi_cap"]

    # RADIAL MAGNETIZATION
    theta = np.arctan2(y, x)

    return rho, phi, theta

design_source = {
    "type": "callable",
    "builder": build_wheel_design,
}



# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    evaluate(fem_params, eval_config, design_source)