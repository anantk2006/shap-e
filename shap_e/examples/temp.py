import open3d as o3d
import numpy as np
import sys
args = sys.argv[1:]

mesh = o3d.io.read_triangle_mesh(args[0])
mesh.compute_vertex_normals()
np.printoptions(suppress=True)

np.savetxt(args[1], np.concatenate([np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)], axis = 1), "%.8f")
#print(str(list(np.concatenate([np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)], axis = 1))).replace("]), array([", "\n").replace(", ", "\t" ).replace(",\n", "\t"))