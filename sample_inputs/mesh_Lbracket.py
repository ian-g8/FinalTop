import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio

gmsh.initialize()
gmsh.model.add("L_bracket")

# ------------------------------------------------------------
# Mesh resolution control
# ------------------------------------------------------------
h_elem = 2.0   # dev resolution (coarser/faster); border logic will scale with this

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_elem)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_elem)


# Geometry parameters
L = 100.0
h = 50.0

# Points
p1 = gmsh.model.geo.addPoint(0,   0,   0)
p2 = gmsh.model.geo.addPoint(L,   0,   0)
# Corner split for load application
p3a = gmsh.model.geo.addPoint(L,   h,   0)
p4 = gmsh.model.geo.addPoint(h,   h,   0)
p5 = gmsh.model.geo.addPoint(h,   L,   0)
p6 = gmsh.model.geo.addPoint(0,   L,   0)

# Lines
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3a)    # right outer edge
l_load = gmsh.model.geo.addLine(p3a, p4)
l3     = gmsh.model.geo.addLine(p4, p5)
l5 = gmsh.model.geo.addLine(p5, p6)
l6 = gmsh.model.geo.addLine(p6, p1)

# Curve loop + surface
cl = gmsh.model.geo.addCurveLoop([l1, l2, l_load, l3, l5, l6])
s = gmsh.model.geo.addPlaneSurface([cl])

# Physical groups (VERY IMPORTANT)
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(2, [s], tag=1)        # domain
gmsh.model.addPhysicalGroup(1, [l_load], tag=2)  # load edge at (50,50)
gmsh.model.addPhysicalGroup(1, [l1, l6], tag=3)   # fixed edges

gmsh.model.mesh.generate(2)

mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=2
)

gmsh.finalize()
