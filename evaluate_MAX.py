import os
import csv
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.io
import ufl
from dolfinx import fem

# Your FEM builder
from fenitop.fem import form_fem

# ============================================================
# USER INPUTS
# ============================================================

# --- Mesh import (same style as input_Grip.py) ---
#from mesh_U_gripper import mesh, mesh_serial, facet_tags

# ============================================================
#  MESH  (must MATCH input_MAX.py exactly)
# ============================================================

from dolfinx.mesh import create_rectangle, CellType


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


# ------------------------------------------------------------
# Frozen design fields (PHYSICAL fields, CG1)
#
# Semantics:
# - active=False  -> do NOT load; fem.py will freeze internally
# - active=True   -> load from file and assign to *_phys_field
# - theta fallback_dir used if theta inactive OR no file supplied
# ------------------------------------------------------------
design_fields = {
    "rho": {
        "active": True,                  # matches input_MAX.py
        "file": "final_rho_phys.npy",     # will NOT be loaded when inactive
    },
    "phi": {
        "active": True,
        "file": "final_phi_phys.npy",
    },
    "theta": {
        "active": False,
        "file": "final_theta_phys.npy",   # optional, ignored when inactive
        "fallback_dir": (1.0, 0.0),
    },
}



# ------------------------------------------------------------
# FEM parameters (shared)
# NOTE: G_model and hyperModel will be overridden in loops
# ------------------------------------------------------------
fem_params = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,
   # "facet_tags": facet_tags,  for input messhes with BC tagging

    # Base material
    "shear_modulus": 100.0,
    "poisson's ratio": 0.49,

    "hyperelastic": True,
    "hyperModel": "stVenant",   # placeholder, overridden
    "G_model": "default",       # placeholder, overridden

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
            "name": "B_up_KernerN2",
            "weight": 1.0,
            "B_app_mag": 100.0,
            "B_app_dir": (0.0, 1.0),
            "tractions": {},   # none
        },
    ],

    "load_steps": 8,

    # Quadrature
    "quadrature_degree": 2,

    # Magnetic
    "mu0": 1.256e3,                 # magnetic permeability
    "B_rem_mag": 60.0,            # remanent field magnitude
    "B_rem_dir": (1.0, 0.0),        # direction of remanent field (x-direction)
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


# ------------------------------------------------------------
# Evaluation controls
# ------------------------------------------------------------
evaluation = {
    "G_models": ["default", "guth", "mooney", "kerner"],
    "hyperelastic_models": ["stVenant", "neoHookean1", "neoHookean2"],

    # Outputs
    "output_dir": "./results/evaluate_compliance/",
    "write_bp": True,
    "write_csv": True,
    "csv_name": "model_comparison.csv",
}

# ------------------------------------------------------------
# Analysis reference models (explicit, documented)
# ------------------------------------------------------------
analysis_ref = {
    "G_model": "kerner",
    "HyperModel": "neoHookean2",
}


# ============================================================
# HELPERS
# ============================================================

def _ensure_serial(comm):
    # Your .npy saving in topopt was done on rank 0 only, so it is only safe in serial.
    # If you later want parallel-safe design import, we should switch to XDMF checkpointing.
    if comm.size != 1:
        raise RuntimeError(
            "evaluate_design.py currently expects serial (mpirun -n 1).\n"
            "Reason: your saved .npy arrays are rank-local from the optimization run.\n"
            "If you want MPI>1 evaluation, we should switch design export/import to XDMF."
        )

def _load_npy_or_raise(path):
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Design field file not found: {path}")
    arr = np.load(path)
    return arr

def _assign_cg1_field(func: fem.Function, arr: np.ndarray, name: str):
    # dolfinx stores CG1 function dofs in func.x.array (local, no ghosts in serial)
    if arr is None:
        raise RuntimeError(f"Attempted to assign {name} but array is None.")

    local = func.x.array
    if arr.shape[0] != local.shape[0]:
        raise ValueError(
            f"{name} size mismatch:\n"
            f"  file array: {arr.shape[0]}\n"
            f"  function dofs: {local.shape[0]}\n"
            "Make sure the .npy was saved from the same mesh and same function space (CG1)."
        )
    local[:] = arr[:]
    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                mode=PETSc.ScatterMode.FORWARD)

def _theta_from_dir(theta_func: fem.Function, fallback_dir, name="theta_phys"):
    d = np.array(fallback_dir, dtype=float)
    nrm = np.linalg.norm(d)
    if nrm <= 0:
        raise ValueError("theta fallback_dir must be a nonzero vector, e.g. (1.0, 0.0).")
    d /= nrm
    theta0 = float(np.arctan2(d[1], d[0]))

    with theta_func.x.petsc_vec.localForm() as loc:
        loc.set(theta0)
    theta_func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                       mode=PETSc.ScatterMode.FORWARD)

def _safe_unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n > 0:
        return v / n
    return np.zeros_like(v)


# ============================================================
# MAIN
# ============================================================

def main():
    comm = mesh.comm
    _ensure_serial(comm)

    if comm.rank == 0:
        os.makedirs(evaluation["output_dir"], exist_ok=True)
        print("[evaluate_design] output_dir:", evaluation["output_dir"], flush=True)

    # Build model permutation list (deterministic ordering)
    G_models = list(evaluation["G_models"])
    H_models = list(evaluation["hyperelastic_models"])

    # CSV rows accumulator
    rows = []

    # Design toggles consistent with your topology code
    design_variables = {
        "rho":   {"active": bool(design_fields["rho"]["active"])},
        "phi":   {"active": bool(design_fields["phi"]["active"])},
        "theta": {"active": bool(design_fields["theta"]["active"])},
    }

    # Pre-load arrays once (fast, and avoids repeated disk IO)
    # If inactive -> None (handled later as constants)
    rho_arr = None
    phi_arr = None
    theta_arr = None

    if design_variables["rho"]["active"]:
        rho_arr = _load_npy_or_raise(design_fields["rho"].get("file", None))

    if design_variables["phi"]["active"]:
        phi_arr = _load_npy_or_raise(design_fields["phi"].get("file", None))

    if design_variables["theta"]["active"]:
        theta_arr = _load_npy_or_raise(design_fields["theta"].get("file", None))

    for hyperModel in H_models:
        for G_model in G_models:

            # --- override constitutive choices ---
            fem_params_local = dict(fem_params)
            fem_params_local["hyperModel"] = hyperModel
            fem_params_local["G_model"] = G_model

            # --- minimal opt dict required by fem.py ---
            opt = {
                # required for rho penalization + stress relaxation
                "penalty": 3.0,
                "epsilon": 1e-6,

                # design toggles consumed by fem.py
                "design_variables": design_variables,

                # objective can be anything; we are NOT optimizing
                "objective_type": "compliance",
                "objective_bcs": [],
            }



            # ------------------------------------------------------------
            # Per-load-case solve + output
            # ------------------------------------------------------------

            load_cases = fem_params_local.get("load_cases", [{"name": "single"}])
            N_steps = int(fem_params_local.get("load_steps", 1))

            for lc in load_cases:
                lc_name = lc.get("name", "unnamed")

                # ------------------------------------------------------------
                # Initialize traction BCs for this load case
                # ------------------------------------------------------------
                for bc_dict in fem_params_local.get("traction_bcs", []):
                    name = bc_dict.get("name", None)
                    if name is not None and name in lc.get("tractions", {}):
                        bc_dict["traction_max"] = lc["tractions"][name]
                    else:
                        bc_dict["traction_max"] = (0.0, 0.0)


                # ------------------------------------------------------------
                # Set magnetic parameters FOR THIS LOAD CASE
                # (must be set before form_fem)
                # ------------------------------------------------------------

                # ------------------------------------------------------------
                # Applied magnetic field (external) — LOAD STEPPED
                # ------------------------------------------------------------

                # Target applied magnetic field
                B_app_mag_target = float(lc.get(
                    "B_app_mag",
                    fem_params_local.get("B_app_mag", 0.0)
                ))
                B_app_dir_target = np.array(_safe_unit(
                    lc.get("B_app_dir", fem_params_local.get("B_app_dir", (0.0, 0.0)))
                ), dtype=float)

                # Initialize applied magnetic field to ZERO (will be ramped)
                fem_params_local["B_app_mag"] = 0.0
                fem_params_local["B_app_dir"] = tuple(B_app_dir_target)


                # Remanent field (internal)
                # - If theta inactive: fem.py uses B_rem_dir
                # - If theta active: fem.py overrides internally
                fem_params_local["B_rem_mag"] = float(
                    fem_params_local.get("B_rem_mag", 0.0)
                )

                # ------------------------------------------------------------
                # Build FEM problem (per load case)
                # ------------------------------------------------------------
                (
                    femProblem,
                    u_field,
                    lambda_field,
                    rho_field,
                    rho_phys_field,
                    phi_field,
                    phi_phys_field,
                    phi_eff_field,
                    theta_phys_field,
                    traction_constants,
                    ds,
                ) = form_fem(fem_params_local, opt)


                # ------------------------------------------------------------
                # Assign frozen physical fields (CG1)
                # ------------------------------------------------------------

                # rho_phys
                if design_variables["rho"]["active"]:
                    if rho_arr is None:
                        raise RuntimeError(
                            "rho.active=True but no rho field file was provided.\n"
                            "Either set active=False or provide final_rho_phys.npy"
                        )
                    _assign_cg1_field(rho_phys_field, rho_arr, "rho_phys")
                # else: fem.py freezes rho_phys = 1

                # phi_phys
                if design_variables["phi"]["active"]:
                    if phi_arr is None:
                        raise RuntimeError(
                            "phi.active=True but no phi field file was provided.\n"
                            "Either set active=False or provide final_phi_phys.npy"
                        )
                    _assign_cg1_field(phi_phys_field, phi_arr, "phi_phys")
                # else: fem.py freezes phi_phys = 0

                # theta_phys
                if design_variables["theta"]["active"]:
                    if theta_arr is not None:
                        _assign_cg1_field(theta_phys_field, theta_arr, "theta_phys")
                    else:
                        _theta_from_dir(theta_phys_field, design_fields["theta"]["fallback_dir"])
                else:
                    fem_params_local["B_rem_dir"] = tuple(
                        _safe_unit(design_fields["theta"]["fallback_dir"])
                    )

                # ------------------------------------------------------------
                # phi_eff = rho_phys * phi_phys
                # ------------------------------------------------------------
                phi_eff_field.x.array[:] = (
                    rho_phys_field.x.array * phi_phys_field.x.array
                )
                phi_eff_field.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD
                )

                # Reset tractions for this case
                for t_const in traction_constants:
                    t_const.value = np.zeros_like(t_const.value)

                # Incremental load stepping (same structure as topopt.py)
                for step in range(1, N_steps + 1):

                    # ------------------------------------------------------------
                    # Ramp applied magnetic field
                    # ------------------------------------------------------------
                    alpha = step / N_steps
                    opt["B_app"].value[:] = alpha * B_app_mag_target * B_app_dir_target

                    # Increment traction loads (if any)
                    for t_const, bc_dict in zip(traction_constants, fem_params_local.get("traction_bcs", [])):
                        # traction_max should already be set per load case
                        t_max = np.array(bc_dict.get("traction_max", (0.0, 0.0)), dtype=float)
                        t_const.value += (1.0 / N_steps) * t_max


                    femProblem.solve_fem()

                # Max displacement
                u_array = u_field.x.array
                max_disp = float(np.max(np.abs(u_array)))

                if comm.rank == 0:
                    print(f"[{G_model:>7s} | {hyperModel:>11s} | {lc_name}]  max|u| = {max_disp:.6e}", flush=True)

                # BP output
                if evaluation.get("write_bp", True):
                    # deterministic filename
                    bp_name = f"disp_{G_model}_{hyperModel}_{lc_name}.bp"
                    bp_path = os.path.join(evaluation["output_dir"], bp_name)

                    fields_to_write = [rho_phys_field, phi_phys_field, phi_eff_field, u_field]

                    # also output theta_phys if active
                    if design_variables["theta"]["active"]:
                        fields_to_write.append(opt["theta_phys_field"])

                        # and m_eff (helpful for debugging)
                        V_vec_cg = fem.functionspace(
                            mesh,
                            ("CG", 1, (mesh.geometry.dim,))
                        )
                        m_eff = fem.Function(V_vec_cg, name="m_eff")
                        theta_phys = opt["theta_phys_field"]
                        m_expr = ufl.as_vector((
                            phi_eff_field * ufl.cos(theta_phys),
                            phi_eff_field * ufl.sin(theta_phys)
                        ))
                        m_eff.interpolate(fem.Expression(m_expr, V_vec_cg.element.interpolation_points()))
                        fields_to_write.append(m_eff)

                    writer = dolfinx.io.VTXWriter(mesh.comm, bp_path, fields_to_write, engine="BP4")
                    writer.write(0.0)
                    writer.close()

                # Record CSV row
                rows.append({
                    "G_model": G_model,
                    "HyperModel": hyperModel,
                    "LoadCase": lc_name,
                    "MaxDisplacement": max_disp,
                })

    # ------------------------------------------------------------
    # Post-process results for clean model sensitivity analysis
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Sensitivity across G models (holding HyperModel fixed)
    # ------------------------------------------------------------
    by_H = {}
    for r in rows:
        key = (r["HyperModel"], r["LoadCase"])
        by_H.setdefault(key, []).append(r)

    for key, group in by_H.items():
        disps = np.array([r["MaxDisplacement"] for r in group], dtype=float)
        mean_u = float(np.mean(disps))
        spread_percent = 100.0 * (np.max(disps) - np.min(disps)) / mean_u if mean_u > 0 else 0.0

        for r in group:
            r["G_model_spread_percent"] = spread_percent


    # ------------------------------------------------------------
    # Sensitivity across hyperelastic models (holding G_model fixed)
    # ------------------------------------------------------------
    by_G = {}
    for r in rows:
        key = (r["G_model"], r["LoadCase"])
        by_G.setdefault(key, []).append(r)

    for key, group in by_G.items():
        disps = np.array([r["MaxDisplacement"] for r in group], dtype=float)
        mean_u = float(np.mean(disps))
        spread_percent = 100.0 * (np.max(disps) - np.min(disps)) / mean_u if mean_u > 0 else 0.0

        for r in group:
            r["HyperModel_spread_percent"] = spread_percent


    # Write CSV
    if comm.rank == 0 and evaluation.get("write_csv", True):
        csv_path = os.path.join(evaluation["output_dir"], evaluation.get("csv_name", "model_comparison.csv"))
        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "G_model",
                "HyperModel",
                "LoadCase",

                # Primary result
                "MaxDisplacement",              # units: length (same as u)

                # Sensitivity metrics (percent variation)
                "G_model_spread_percent",       # variation across G models (%)
                "HyperModel_spread_percent",    # variation across hyperelastic models (%)
            ]


            w = csv.DictWriter(f, fieldnames=fieldnames)

            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"[evaluate_design] wrote CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()


