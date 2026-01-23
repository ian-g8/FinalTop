import os
import numpy as np
import time
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.linalg import solve
import dolfinx.io
from dolfinx import fem, mesh
from dolfinx.mesh import create_box, CellType, locate_entities_boundary, meshtags, create_rectangle
from dolfinx.fem import (Function, Constant, dirichletbc, locate_dofs_topological, 
                        form, assemble_scalar, functionspace)
from dolfinx import la
from dolfinx.fem.petsc import (create_vector, create_matrix, assemble_vector, assemble_matrix, set_bc)
import pyvista
pyvista.set_jupyter_backend('html')
import basix
from basix.ufl import element
from ufl import variable, inner, grad, det, tr, Identity, outer, dev, sym

from fenitop.utility import create_mechanism_vectors
from fenitop.utility import LinearProblem
from fenitop.utility import WrapNonlinearProblem

def form_fem(fem_params, opt):
    """Form an FEA problem."""
    # Function spaces and functions
    mesh = fem_params["mesh"]
    element = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    V = fem.functionspace(mesh, element)
    S0 = fem.functionspace(mesh, ("DG", 0))
    S = fem.functionspace(mesh, ("CG", 1))
    v = ufl.TestFunction(V)
    u_field = Function(V, name="u") 
    lambda_field = Function(V, name="lambda")  

    # Initialize material density field
    rho_field = Function(S0, name="rho")  
    rho_phys_field = Function(S, name="rho_phys")  
    
    # Initialize magnetic density field
    phi_field = Function(S0, name="phi")  
    phi_phys_field = Function(S, name="phi_phys")  

    # Initialize theta (remanence direction angle) field
    # theta_field: design variable (DG0)
    # theta_phys_field: filtered physical field (CG1)
    theta_field = Function(S0, name="theta")
    theta_phys_field = Function(S, name="theta_phys")

    # Expose handles for other modules (topopt/sensitivity) without changing return signature
    opt["theta_field"] = theta_field
    opt["theta_phys_field"] = theta_phys_field

    # ============================================================
    # Freeze inactive design variables (initial state)
    # ============================================================

    dv_cfg = opt.get("design_variables", {})

    # rho inactive → rho_phys = 1 (CG1)
    if not dv_cfg.get("rho", {}).get("active", True):
        with rho_phys_field.x.petsc_vec.localForm() as loc:
            loc.set(1.0)
        rho_phys_field.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD
        )

    # phi inactive → phi_phys = 0 (CG1)
    if not dv_cfg.get("phi", {}).get("active", True):
        with phi_phys_field.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        phi_phys_field.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD
        )


    # theta inactive → theta_phys = angle implied by B_rem_dir (CG1)
    # (When theta is active, topopt will update theta_phys each iteration via filtering.)
    if not dv_cfg.get("theta", {}).get("active", False):
        # Default remanence direction from input file (unit vector)
        _B_dir = np.array(fem_params.get("B_rem_dir", (1.0, 0.0)), dtype=np.float64)
        _n = np.linalg.norm(_B_dir)
        if _n > 0:
            _B_dir /= _n
        theta0 = float(np.arctan2(_B_dir[1], _B_dir[0]))

        with theta_phys_field.x.petsc_vec.localForm() as loc:
            loc.set(theta0)
        theta_phys_field.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD
        )

    # Material interpolation
    G0 = fem_params["shear_modulus"]
    nu = fem_params["poisson's ratio"]  # only needed for Kerner model

    # Rho penalization 
    p, eps = opt["penalty"], opt["epsilon"]
    rho_penalty = eps + (1 - eps) * rho_phys_field**p
    G0 = G0 * rho_penalty

    # ============================================================
    # WE-weighted void-penalty infrastructure (frozen weights)
    # ============================================================
    # DG0 weight field w(x) used in penalty:  P = ∫ w(x) (1 - rho)^2 dx
    # This will be populated/updated in topopt.py from strain-energy density.
    w_void = Function(S0, name="w_void")

    # Normalization constant P0 (set once in topopt.py when weights are first frozen)
    P0_void = Constant(mesh, PETSc.ScalarType(1.0))

    # Store handles so topopt.py can update them
    opt["we_voidpen_weight_field"] = w_void
    opt["we_voidpen_P0_const"] = P0_void
   

    model = fem_params.get("G_model", "default")

    # --- Material shear modulus models (all use physical φ ∈ [0, 0.3]) ---
    if model == "default":
        mu = G0
    elif model == "guth":
        mu = G0 * (1 + 2.5*phi_phys_field + 14.1*phi_phys_field**2)
    elif model == "mooney":
        mu = G0 * ufl.exp(2.5*phi_phys_field / (1.0 - 1.35*phi_phys_field))
    elif model == "kerner":
        A = 15*(1 - nu) / (8 - 10*nu)
        mu = G0 * (1 + (A*phi_phys_field) / (1.0 - phi_phys_field))
    else:
        raise ValueError(f"Unknown G_model: {model}")

    # Bulk modulus 
    K = 1000 * G0
    
    # Magnetic parameters (now prescribed by input file)
    mu_0_val = fem_params.get("mu0", 1.256e+3)  # mN/(kA)^2
    mu0 = Constant(mesh, PETSc.ScalarType(mu_0_val))

    # --- Remanent magnetic field (B_rem) ---
    # If theta is active, it overrides B_rem_dir and enters as:
    #   B_rem = B_rem_mag * [cos(theta_phys), sin(theta_phys)]
    # If theta is inactive, we use the prescribed B_rem_dir.
    B_rem_mag = float(fem_params.get("B_rem_mag", 50.0))

    element2 = basix.ufl.element("DG", mesh.basix_cell(), 0, shape=(mesh.geometry.dim,))
    S0_vector = fem.functionspace(mesh, element2)
    B_rem_field = Function(S0_vector, name="B_rem")  # for output/diagnostics

    theta_active = dv_cfg.get("theta", {}).get("active", False)

    if theta_active:
        # UFL expression (updates automatically when theta_phys_field changes)
        B_rem = B_rem_mag * ufl.as_vector((ufl.cos(theta_phys_field), ufl.sin(theta_phys_field)))

        # NOTE:
        # When theta is active, B_rem_expr (UFL) is used in physics.
        # B_rem_field is for diagnostics / output only and will be
        # updated in topopt.py for visualization.

        # Use current theta_phys_field value if present (it is already frozen when inactive; when active it starts at 0)
        # We'll just set field = B_rem_mag * (1,0) initially for robustness.
        def _mag_flux_density_init(x):
            f = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=np.float64)
            f[0, :] = B_rem_mag * 1.0
            f[1, :] = B_rem_mag * 0.0
            return f
        B_rem_field.interpolate(_mag_flux_density_init)

    else:
        # Prescribed direction from input (unit vector)
        B_rem_dir = np.array(fem_params.get("B_rem_dir", (1.0, 0.0)), dtype=np.float64)
        nrm = np.linalg.norm(B_rem_dir)
        if nrm > 0:
            B_rem_dir /= nrm

        def mag_flux_density(x):
            f = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=np.float64)
            f[0, :] = B_rem_mag * B_rem_dir[0]
            f[1, :] = B_rem_mag * B_rem_dir[1]
            return f

        B_rem_field.interpolate(mag_flux_density)
        B_rem = B_rem_field  # physics uses the interpolated field when theta is inactive

    # Expose both for downstream use
    opt["B_rem_field"] = B_rem_field
    opt["B_rem_expr"] = B_rem


    # --- Applied field (B_app) ---
    B_app_mag = fem_params.get("B_app_mag", 5.0)
    B_app_dir = np.array(fem_params.get("B_app_dir", (0.0, 1.0)), dtype=np.float64)
    B_app_dir /= np.linalg.norm(B_app_dir)
    B_app = Constant(mesh, B_app_mag * B_app_dir)
 
    # Boundary conditions
    dim = mesh.topology.dim
    fdim = dim - 1
    disp_facets = locate_entities_boundary(mesh, fdim, fem_params["disp_bc"])
    bc = dirichletbc(Constant(mesh, np.full(dim, 0.0)),
                     locate_dofs_topological(V, fdim, disp_facets), V)

    # Tractions
    facets, markers, traction_constants, tractions = [], [], [], [] 

    for marker, bc_dict in enumerate(fem_params["traction_bcs"]):
        traction_max = np.array(bc_dict["traction_max"], dtype=float)
        traction_func = bc_dict["on_boundary"]    
        traction_const = Constant(mesh, np.zeros_like(traction_max))  # start zero, update externally
        traction_constants.append(traction_const)
        current_facets = locate_entities_boundary(mesh, fdim, traction_func)
        facets.extend(current_facets)
        markers.extend([marker,]*len(current_facets))

    facets = np.array(facets, dtype=np.int32)
    markers = np.array(markers, dtype=np.int32)
    _, unique_indices = np.unique(facets, return_index=True)
    facets, markers = facets[unique_indices], markers[unique_indices]
    sorted_indices = np.argsort(facets)
    facet_tags = meshtags(mesh, fdim, facets[sorted_indices], markers[sorted_indices])
    
    metadata = {"quadrature_degree": fem_params["quadrature_degree"]}
    dx = ufl.Measure("dx", metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags)
    b = Constant(mesh, np.array(fem_params["body_force"], dtype=float))

    # Kinematics
    I = ufl.Identity(dim)          #Identity Matrix
    F = ufl.variable(ufl.Identity(dim) + ufl.grad(u_field)) # Deformation gradient
    C = F.T * F                    # Right Cauchy-Green tensor
    #Ic = ufl.tr(C)                 # First invariant
    Ic = ufl.tr(C) + 1.0                # First invariant
    J = ufl.det(F)                 # Jacobian determinant

    # Stored strain energy density 
    if fem_params["hyperModel"] == "neoHookean1":
        W_elastic = (mu / 2) * (J**(-2/3)*Ic - 3) + (K/2) * (J - 1)**2
    elif fem_params["hyperModel"] == "neoHookean2":
        W_elastic = (mu / 2) * (Ic - 3 - 2*ufl.ln(J)) + (K/2) * (J - 1)**2
    elif fem_params["hyperModel"] == "stVenant":
        Egreen = (C - I) / 2
        W_elastic = (K/2) * (ufl.tr(Egreen))**2 + mu * ufl.tr(Egreen * Egreen)

    # Magnetic energy density
    W_magnetic = -(1/mu0) * inner(F * B_rem, B_app)

    # rho/phi coupling
    phi_eff = phi_phys_field * rho_phys_field

    # For post-processing: magnetization-like vector field
    # m(x) = phi_eff * [cos(theta_phys), sin(theta_phys)]
    opt["m_expr"] = phi_eff * ufl.as_vector((ufl.cos(theta_phys_field), ufl.sin(theta_phys_field)))

    W_magnetic = phi_eff * W_magnetic

    # Total energy
    W = W_elastic + W_magnetic

    P = ufl.diff(W, F)             # First Piola-Kirchhoff stress tensor

    # Assemble Residual
    a = inner(grad(v), P)*dx     # Semilinear form
    L = inner(v, b)*dx           # Linear form
    for marker, t in enumerate(traction_constants):   # Add tractions
        L += inner(v, t)*ds(marker)
    R = L - a 

    # Wrap nonlinear problem
    femProblem = WrapNonlinearProblem(u_field, R, [bc], fem_params["petsc_options"])

    # ============================================================
    # GENERIC OBJECTIVE FORMS
    # ============================================================

    # Internal force 
    opt["f_int"] = ufl.derivative(W * dx, u_field, v)

    # Determine objective type
    obj_type = opt.get("objective_type", "compliance")

    if obj_type == "compliance":
        J = inner(u_field, b) * dx
        for marker, t in enumerate(traction_constants):
            J += inner(u_field, t) * ds(marker)

    elif obj_type == "max_disp":
        # --------------------------------------------------------
        # Maximize vertical displacement on right boundary
        # This uses the same boundary marker as the traction load
        # Marker index = 0 (first traction BC region)
        # Objective: J = -∫ u_y ds  (negative: maximize disp)
        # --------------------------------------------------------
        target_marker = 0
        J = - ufl.inner(u_field, ufl.as_vector((0.0, 1.0))) * ds(target_marker)

    elif obj_type == "max_disp_we_voidpen":
        # --------------------------------------------------------
        # Maximize vertical displacement on right boundary
        # PLUS WE-weighted void penalty (frozen weights)
        #
        # J = (-∫ u_y ds) + alpha * (P / P0)
        # P = ∫ w(x) (1 - rho_phys)^2 dx
        #
        # Notes:
        # - w(x) is a DG0 Function updated in topopt.py from W_elastic
        # - P0 is a Constant set once in topopt.py when w(x) is first frozen
        # --------------------------------------------------------
        target_marker = 0
        J_tip = - ufl.inner(u_field, ufl.as_vector((0.0, 1.0))) * ds(target_marker)

        alpha = float(opt.get("we_voidpen_alpha", 0.01))

        # penalty form (rho-only), w_void treated as constant during differentiation
        P_void = opt["we_voidpen_weight_field"] * (1.0 - rho_phys_field)**2 * dx

        # store for diagnostics (optional)
        opt["we_voidpen_form"] = P_void

        # normalized penalty contribution
        J = J_tip + alpha * (1.0 / opt["we_voidpen_P0_const"]) * P_void
 
    elif obj_type == "disp_track":
        # --------------------------------------------------------
        # Displacement tracking objective (multi-point, vector)
        #
        # Supports:
        #   opt["disp_track"] = dict  (single point, backward compatible)
        #   opt["disp_track"] = list of dicts (multi-point)
        #
        # Each point dict supports:
        #   "point":  (x, y)
        #   "target": (ux_t, uy_t)   <-- vector target (Option A)
        #   "sigma":  float          <-- Gaussian radius
        #   "weight": float          <-- optional per-point weight
        #
        # Objective:
        #   J = Σ_i weight_i ∫ w_i(x) ||u(x) - u_target_i||^2 dx
        # --------------------------------------------------------

        disp_cfg = opt["disp_track"]

        # Normalize to list for backward compatibility
        if isinstance(disp_cfg, dict):
            disp_cfg_list = [disp_cfg]
        else:
            disp_cfg_list = list(disp_cfg)

        # Spatial coordinate
        X = ufl.SpatialCoordinate(mesh)

        # Displacement vector field
        u_vec = ufl.as_vector((u_field[0], u_field[1]))

        # Build objective as sum over points
        J = 0

        # Store per-point diagnostics (optional, for printing in topopt.py)
        opt["track_terms"] = []

        for cfg in disp_cfg_list:
            # Required
            x_target = cfg["point"]  # (x, y)

            # Target vector: accept tuple/list/np array
            tgt = cfg["target"]
            ux_t = float(tgt[0])
            uy_t = float(tgt[1])
            u_tgt = ufl.as_vector((ux_t, uy_t))

            # Optional knobs
            sigma = float(cfg.get("sigma", 1.0))
            w_pt  = float(cfg.get("weight", 1.0))

            # Gaussian localization weight
            r2 = (X[0] - x_target[0])**2 + (X[1] - x_target[1])**2
            w  = ufl.exp(-r2 / sigma**2)

            # --------------------------------------------
            # Component-selective displacement tracking
            # --------------------------------------------
            components = cfg.get("components", ("x", "y"))

            term = 0

            if "x" in components:
                diff_x = u_field[0] - ux_t
                term += diff_x**2

            if "y" in components:
                diff_y = u_field[1] - uy_t
                term += diff_y**2

            J += w_pt * w * term * dx

            # Diagnostics:
            #   weighted-average displacement vector over the local region
            #   u_avg = (∫ u w dx) / (∫ w dx)
            opt["track_terms"].append(
                {
                    # scalar component integrals (UFL-safe)
                    "ux_form": u_field[0] * w * dx,
                    "uy_form": u_field[1] * w * dx,
                    "w_form":  w * dx,

                    "target": (ux_t, uy_t),
                    "point":  (float(x_target[0]), float(x_target[1])),
                    "sigma":  sigma,
                    "weight": w_pt,
                }
            )

    else:
        raise RuntimeError(f"Unknown objective_type: {obj_type}")

    # Store objective form
    opt["objective_form"] = J

    # Derivatives of objective wrt fields
    opt["dObj_du_form"]   = ufl.derivative(J, u_field)
    opt["dObj_drho_form"] = ufl.derivative(J, rho_phys_field)
    opt["dObj_dphi_form"] = ufl.derivative(J, phi_phys_field)
    opt["dObj_dtheta_form"] = ufl.derivative(J, theta_phys_field)

    # ============================================================
    # COMPLIANCE CONSTRAINT FORMS (for normalized compliance constraint)
    #
    # Compliance (general): C = ∫ u·b dx + Σ ∫ u·t ds
    # Notes:
    # - In your current test cases, body_force is zero, so only traction terms contribute.
    # - We store both the scalar form and partial derivatives so Sensitivity can compute
    #   TOTAL derivatives via an adjoint solve (per load case).
    # ============================================================

    C_form = inner(u_field, b) * dx
    for marker, t in enumerate(traction_constants):
        C_form += inner(u_field, t) * ds(marker)

    opt["compliance_form"] = C_form

    # Partials needed for total derivative (adjoint-based)
    opt["dC_du_form"]   = ufl.derivative(C_form, u_field)
    opt["dC_drho_form"] = ufl.derivative(C_form, rho_phys_field)
    opt["dC_dphi_form"] = ufl.derivative(C_form, phi_phys_field)
    opt["dC_dtheta_form"] = ufl.derivative(C_form, theta_phys_field)

    # ============================================================
    # STRESS CONSTRAINT (Zhao 2022 p-norm von-Mises constraint)
    # ============================================================

    # ---- material parameters ----
    pnorm = opt.get("stress_pnorm", 12)
    sigma_max = opt.get("sigma_max", 1e6)  # override in input script

    # ---- compute Cauchy stress ----
    P = ufl.diff(W, F)
    J_det = ufl.det(F)
    sigma_cauchy = (1.0 / J_det) * (P * ufl.transpose(F))

    # ---- von-Mises ----
    sigma_dev = ufl.dev(ufl.sym(sigma_cauchy))
    sigma_vm = ufl.sqrt(1.5 * ufl.inner(sigma_dev, sigma_dev))

    # ---- stress relaxation weight ----
    q_rho = float(opt.get("stress_q_rho", 1.0/3.0))
    eps = opt["epsilon"]
    w_sigma = eps + (1.0 - eps) * (rho_phys_field ** q_rho)

    # ---- p-norm integrand ----
    stress_integrand = (w_sigma * sigma_vm)**pnorm

    # store the p-power form ( ∫ f^p dx )
    opt["stress_power_form"] = stress_integrand * dx

    # actual p-norm G = ( ∫ f^p dx )^(1/p) will be computed in Sensitivity.py
    opt["stress_form"] = opt["stress_power_form"]

    # ---- derivatives wrt fields ----
    opt["dStress_du_form"]   = ufl.derivative(opt["stress_power_form"], u_field)
    opt["dStress_drho_form"] = ufl.derivative(opt["stress_power_form"], rho_phys_field)
    opt["dStress_dphi_form"] = ufl.derivative(opt["stress_power_form"], phi_phys_field)
    opt["dStress_dtheta_form"] = ufl.derivative(opt["stress_power_form"], theta_phys_field)

    # Store raw von-Mises expression for post-processing/output
    opt["sigma_vm_expr"] = sigma_vm

    # ============================================================
    # OPTIONAL STRAIN-ENERGY CONSTRAINT (pure mechanical energy)
    # ============================================================

    # Store the elastic strain-energy integrand for constraint use
    opt["strain_energy_form"] = W_elastic * dx

    # Derivatives for adjoint computation (only used if enabled)
    opt["dU_du_form"]   = ufl.derivative(W_elastic * dx, u_field)
    opt["dU_drho_form"] = ufl.derivative(W_elastic * dx, rho_phys_field)
    opt["dU_dphi_form"] = ufl.derivative(W_elastic * dx, phi_phys_field)
    opt["dU_dtheta_form"] = ufl.derivative(W_elastic * dx, theta_phys_field)

    # For ParaView visualization (optional): local strain energy density
    opt["W_elastic_expr"] = W_elastic

    # Define global optimization-related variables 
    opt["volume_phi"] = phi_phys_field*dx
    opt["volume_rho"] = rho_phys_field * dx
    opt["total_volume"] = Constant(mesh, 1.0)*dx
    
    phi_eff_field = Function(S, name="phi_eff")

    return femProblem, u_field, lambda_field, rho_field, rho_phys_field, phi_field, phi_phys_field, phi_eff_field, traction_constants, ds