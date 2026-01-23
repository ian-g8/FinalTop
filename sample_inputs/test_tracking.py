import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from dolfinx import fem

def main():
    # --- Create mesh (same style as your code) ---
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [100.0, 20.0]],
        [10, 2],
        CellType.quadrilateral
    )

    # --- Vector CG1 space ---
    V = fem.functionspace(mesh, ("CG", 1, (mesh.geometry.dim,)))

    # --- Target physical point ---
    x_target = np.array([100.0, 20.0])

    # --- Tabulate DOF coordinates ---
    dof_coords = V.tabulate_dof_coordinates()

    # dof_coords shape is (num_dofs, gdim)
    if MPI.COMM_WORLD.rank == 0:
        print("dof_coords shape:", dof_coords.shape)

    # --- Find closest DOF ---

    dists = np.linalg.norm(dof_coords[:, :2] - x_target, axis=1)

    dof_closest = np.argmin(dists)

    if MPI.COMM_WORLD.rank == 0:
        print("closest DOF index:", dof_closest)
        print("closest DOF coord:", dof_coords[dof_closest])

    # --- Check component ---
    # In dolfinx vector CG spaces:
    # dofs are interleaved: [u0x, u0y, u1x, u1y, ...]
    comp = dof_closest % mesh.geometry.dim

    if MPI.COMM_WORLD.rank == 0:
        print("component index (0=x, 1=y):", comp)


if __name__ == "__main__":
    main()
