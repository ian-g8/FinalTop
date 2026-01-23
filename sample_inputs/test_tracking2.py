import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.fem import Function, Constant, form, assemble_scalar
from dolfinx.fem.petsc import assemble_vector


def main():

    comm = MPI.COMM_WORLD

    # --------------------------------------------------
    # Mesh (same geometry as your optimization problem)
    # --------------------------------------------------
    mesh_ = create_rectangle(
        comm,
        [[0.0, 0.0], [100.0, 20.0]],
        [40, 8],
        CellType.quadrilateral
    )

    gdim = mesh_.geometry.dim

    # --------------------------------------------------
    # Function space: vector CG1
    # --------------------------------------------------
    V = fem.functionspace(mesh_, ("CG", 1, (gdim,)))
    u = Function(V, name="u")

    # --------------------------------------------------
    # Prescribed control point + target displacement
    # --------------------------------------------------
    x_target = np.array([100.0, 20.0])
    u_target_y = 10.0

    # --------------------------------------------------
    # Define localized weight function w(x)
    # Gaussian bump centered at target point
    # --------------------------------------------------
    X = ufl.SpatialCoordinate(mesh_)
    r2 = (X[0] - x_target[0])**2 + (X[1] - x_target[1])**2

    # Controls localization radius (smaller = sharper)
    sigma2 = Constant(mesh_, 1.0)   # try 0.5, 1.0, 2.0

    w = ufl.exp(-r2 / sigma2)

    # --------------------------------------------------
    # Zhao-style displacement tracking objective (scalar)
    # J = ∫ w(x) * (u_y - u_target)^2 dx
    # --------------------------------------------------
    uy = u[1]

    J = w * (uy - u_target_y)**2 * ufl.dx

    # --------------------------------------------------
    # Assemble objective value
    # --------------------------------------------------
    J_val = comm.allreduce(assemble_scalar(form(J)), op=MPI.SUM)

    if comm.rank == 0:
        print("Objective value J =", J_val)

    # --------------------------------------------------
    # Compute derivative w.r.t. displacement
    # --------------------------------------------------
    dJ_du = ufl.derivative(J, u)

    dJ_du_vec = fem.petsc.create_vector(form(dJ_du))
    assemble_vector(dJ_du_vec, form(dJ_du))
    dJ_du_vec.ghostUpdate(addv=1, mode=1)

    if comm.rank == 0:
        print("||dJ/du||_inf =", np.max(np.abs(dJ_du_vec.array)))

    # --------------------------------------------------
    # Sanity check: zero displacement case
    # --------------------------------------------------
    u.x.array[:] = 0.0

    J_zero = comm.allreduce(assemble_scalar(form(J)), op=MPI.SUM)

    if comm.rank == 0:
        print("J with u = 0 :", J_zero)

    # --------------------------------------------------
    # Perturb displacement near target to verify response
    # --------------------------------------------------

    # --------------------------------------------------
    # Prescribe displacement FIELD to verify objective
    # --------------------------------------------------
    u.interpolate(
        lambda x: np.vstack((
            np.zeros_like(x[0]),                 # u_x = 0
            u_target_y * np.ones_like(x[0])       # u_y = target everywhere
        ))
    )

    J_hit = comm.allreduce(assemble_scalar(form(J)), op=MPI.SUM)

    if comm.rank == 0:
        print("J after matching target field :", J_hit)



if __name__ == "__main__":
    main()
