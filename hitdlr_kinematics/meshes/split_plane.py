import trimesh
import numpy as np


mesh = trimesh.load('./hitdlr_hand_refine/distal.stl')
plane_normal = (1., 0, 0)
plane_origin = (-0.005, 0, 0.00)
new_mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal, plane_origin)
new_mesh.show()
# new_mesh = trimesh.creation.box(extents=(0.0005, 0.0005, 0.0005))
# new_mesh.show()
new_mesh.export('./hitdlr_hand_tmp/distal.stl')