import glob
import os
import trimesh
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

LABEL_DIR = 'affordance_labels/'
obj_files = glob.glob(os.path.join('train_dataset/lm/models', 'obj_000002.ply'))


def fps(points, npoint):
    """
    Input:
        mesh: input mesh
        npoint: target point number to sample
    Return:
        centroids: sampled pointcloud index, [npoint]
    """

    N, C = points.shape
    centroids = np.zeros(npoint, dtype=int)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest

        centroid = points[farthest, :].reshape(1, 3)

        dist = np.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = np.argmax(distance, -1)
    return centroids


def inside_test(points, cube3d):
    """
    cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
    points = array of points with shape (N, 3).

    Returns the indices of the points array which are inside the cube3d
    """
    b1, b2, b3, b4, t1, t2, t3, t4 = cube3d

    dir1 = (t1 - b1)
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / size1

    dir2 = (b2 - b1)
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / size2

    dir3 = (b3 - b1)
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / size3

    cube3d_center = (b1 + t4) / 2.0

    dir_vec = points - cube3d_center

    res1 = np.where((np.absolute(np.dot(dir_vec, dir1)) * 2) < size1)[0]
    res2 = np.where((np.absolute(np.dot(dir_vec, dir2)) * 2) < size2)[0]
    res3 = np.where((np.absolute(np.dot(dir_vec , dir3)) * 2) < size3)[0]

    return list(set(res1).intersection(res2).intersection(res3))


print(f'num objs:{len(obj_files)}')
cnt = 0
for i, src in enumerate(obj_files):
    scene = trimesh.Scene()
    obj_name = src.split('/')[-1].split('.')[0]
    print(obj_name)
    obj = trimesh.load(src)
    points, _ = trimesh.sample.sample_surface(mesh=obj, count=5000)
    idx = fps(points, 3000)
    points = points[idx]
    PC = trimesh.PointCloud(points, colors=[255,215,0])
    scene = trimesh.Scene([PC])
    # PC.show()
    # exit()
    # scene.add_geometry(obj)
    with open(LABEL_DIR + '/' + obj_name + '.json', 'r') as file:
        boxes = json.load(file)
    # print(boxes['objects'])
    # if boxes['objects'] != []:
    #     cnt += 1
    for box in boxes['objects']:
        x, y, z = box['centroid']['x'], box['centroid']['y'], box['centroid']['z']
        l, w, h = box['dimensions']['length'], box['dimensions']['width'], box['dimensions']['height']
        # print(l,w,h)9
        r_x, r_y, r_z = box['rotations']['x'], box['rotations']['y'], box['rotations']['z']
        r_m = np.eye(4)
        r_m[:3, :3] = R.from_euler('xyz', [r_x, r_y, r_z],degrees=True).as_matrix()
        r_m[:3, 3] = [x, y, z]
        # print(r_m)
        new_box = trimesh.primitives.Box(extents=[l, w, h], transform=r_m)
        new_box.visual.face_colors = [0, 0, 0, 100]

        # scene.add_geometry(new_box)
        inside_idx = inside_test(points, new_box.vertices)
        inside_points = points[inside_idx]
        pc = trimesh.PointCloud(inside_points, colors=[255, 0, 0])
        scene.add_geometry([pc])
        # scene.show()
        # exit()

    scene.show()
    # exit()
