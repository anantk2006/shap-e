import open3d as o3d
import numpy as np 

# mesh = o3d.io.read_triangle_mesh("old_2.ply")
# print(mesh.vertices, mesh.faces)
file = open("cakeo2.txt", "r").read()
file = [float(i) for i in file.replace("\n", "\t").split("\t") if i != ""]
file = np.asarray(file).reshape(-1, 6)
print(file)

# alpha = 0.05
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
# o3d.io.write_triangle_mesh("hull.ply", mesh)

# points = np.asarray(mesh.vertices)
# p = o3d.geometry.PointCloud()

# pcd = o3d.geometry.TriangleMesh()
# pcd.create_from_point_cloud_alpha_shape(p, 0.1)
# pcd.compute_vertex_normals()
# pcd.compute_triangle_normals()
# o3d.io.write_triangle_mesh("sync.ply", pcd)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(file[:, :3])
pcd.normals = o3d.utility.Vector3dVector(file[:, 3:])


nn_distances = pcd.compute_nearest_neighbor_distance()
avg_nn_distance = np.mean(nn_distances)
radius_of_ball = 3 * avg_nn_distance
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius_of_ball, radius_of_ball])
                                                                        
                                                                         )

#o3d.visualization.draw_geometries([bpa_mesh])
# import pyvista as pv


# point_cloud = pv.PolyData(file[:, :3])
print("C")
# mesh = point_cloud.reconstruct_surface()
# mesh.save('mesh.stl')
# #pcd = o3d.io.read_point_cloud("old_2.ply")
o3d.io.write_triangle_mesh("sync.ply", bpa_mesh)


