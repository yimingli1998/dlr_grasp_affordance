import trimesh
import glob

#ycb_072-b_toy_airplane_scaled,gd_banana_poisson_003_scaled
files = glob.glob('picked_obj/gd_banana_poisson_003_scaled.obj')
for f in files:
    print(f)
    scene = trimesh.Scene()
    obj = trimesh.load(f)
    pc =trimesh.PointCloud(obj.vertices,colors = [0,255,0])
    scene.add_geometry(pc)
    scene.show()