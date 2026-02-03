import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio

# ============================================================
#  U-GRIPPER GEOMETRY (TRUE "U" AS A SINGLE CONCAVE SURFACE)
# ============================================================

gmsh.initialize()
gmsh.model.add("U_gripper")

# ------------------------------------------------------------
# Mesh resolution
# ------------------------------------------------------------
h_elem = 1.5
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_elem)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_elem)

# ------------------------------------------------------------
# Geometry parameters (edit these to taste)
# ------------------------------------------------------------
t = 6.0        # arm thickness
gap = 20.0     # gap between arms
base = 8.0     # base thickness
arm_h = 55.0   # arm height above base

W = 2.0 * t + gap
H = base + arm_h

# Convenience
x0 = 0.0
x1 = t
x2 = t + gap
x3 = W

y0 = 0.0
y1 = base
y2 = H

# ------------------------------------------------------------
# Points for a concave "U" polygon boundary (counter-clockwise)
#
# Trace the *solid* boundary:
# bottom -> outer right -> top of right arm -> inner right down -> inner base
# -> inner left up -> top of left arm -> outer left down -> close
# ------------------------------------------------------------
p1 = gmsh.model.geo.addPoint(x0, y0, 0)  # bottom-left outer
p2 = gmsh.model.geo.addPoint(x3, y0, 0)  # bottom-right outer
p3 = gmsh.model.geo.addPoint(x3, y2, 0)  # top-right outer
p4 = gmsh.model.geo.addPoint(x2, y2, 0)  # top-right arm inner corner (x = t+gap)
p5 = gmsh.model.geo.addPoint(x2, y1, 0)  # inner right bottom (on base top)
p6 = gmsh.model.geo.addPoint(x1, y1, 0)  # inner left bottom (on base top)
p7 = gmsh.model.geo.addPoint(x1, y2, 0)  # top-left arm inner corner (x = t)
p8 = gmsh.model.geo.addPoint(x0, y2, 0)  # top-left outer

# ------------------------------------------------------------
# Lines (match the point trace order)
# ------------------------------------------------------------
l_bottom = gmsh.model.geo.addLine(p1, p2)   # (y=0) fixed base

l_right_outer = gmsh.model.geo.addLine(p2, p3)

l_top_right_arm = gmsh.model.geo.addLine(p3, p4)  # objective: right arm top segment

l_inner_right = gmsh.model.geo.addLine(p4, p5)

l_inner_base = gmsh.model.geo.addLine(p5, p6)     # inner base (top of base)

l_inner_left = gmsh.model.geo.addLine(p6, p7)

l_top_left_arm = gmsh.model.geo.addLine(p7, p8)   # objective: left arm top segment

l_left_outer = gmsh.model.geo.addLine(p8, p1)

# ------------------------------------------------------------
# Surface (single concave loop, SIMPLE and robust)
# ------------------------------------------------------------
loop = gmsh.model.geo.addCurveLoop([
    l_bottom,
    l_right_outer,
    l_top_right_arm,
    l_inner_right,
    l_inner_base,
    l_inner_left,
    l_top_left_arm,
    l_left_outer
])
surface = gmsh.model.geo.addPlaneSurface([loop])


gmsh.model.geo.synchronize()

# ------------------------------------------------------------
# Physical groups
# ------------------------------------------------------------
gmsh.model.addPhysicalGroup(2, [surface], tag=1)


gmsh.model.setPhysicalName(2, 1, "DOMAIN")

# Fixed base
gmsh.model.addPhysicalGroup(1, [l_bottom], tag=10)
gmsh.model.setPhysicalName(1, 10, "FIXED_BASE")

# Objective boundaries = top faces of the arms
gmsh.model.addPhysicalGroup(1, [l_top_left_arm], tag=20)
gmsh.model.setPhysicalName(1, 20, "LEFT_JAW_TOP")

gmsh.model.addPhysicalGroup(1, [l_top_right_arm], tag=21)
gmsh.model.setPhysicalName(1, 21, "RIGHT_JAW_TOP")

# ------------------------------------------------------------
# Mesh
# ------------------------------------------------------------

gmsh.model.mesh.generate(2)


mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=2
)

gmsh.finalize()

# ------------------------------------------------------------
# Serial mesh (for plotting)
# ------------------------------------------------------------
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = mesh
else:
    mesh_serial = None
