import numpy as np
import trimesh
import os
import json
import pickle
from tqdm import tqdm
from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import common_util, scene_utils,pc_utils
from scipy.spatial.transform import Rotation
import torch
import copy
import time
import yaml
with open('config/base_config.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

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
    res3 = np.where((np.absolute(np.dot(dir_vec, dir3)) * 2) < size3)[0]

    return list(set(res1).intersection(res2).intersection(res3))

def generate_scene_affordance(img_id,save_path):
    scene =trimesh.Scene()
    point,sem = scene_utils.load_scene_pointcloud(img_id, use_base_coordinate=cfg['use_base_coordinate'])
    pc1 = trimesh.PointCloud(point, colors=[0, 255, 0])
    scene.add_geometry([pc1])
    scene_mesh,gt_objs,transform_list= scene_utils.load_scene(img_id)
    # scene.add_geometry(scene_mesh)
    affordance_mask = -np.ones(len(point))
    boxes_list,trans_list,sem_afford_list = [],[],[]
    affordance_idx_list = []
    for i,(obj,trans) in enumerate(zip(gt_objs, transform_list)):
        with open(f"affordance_labels/obj_{str(obj['obj_id']).zfill(6)}.json", 'r') as file:
            f = json.load(file)
            if f['objects'] != []:
                boxes_list += f['objects']
                trans_list.append(trans)
                sem_afford_list.append(i+1)

    for sem_id,box,trans in zip(sem_afford_list,boxes_list,trans_list):
        x, y, z = box['centroid']['x'], box['centroid']['y'], box['centroid']['z']
        l, w, h = box['dimensions']['length'], box['dimensions']['width'], box['dimensions']['height']
        # print(l,w,h)9
        r_x, r_y, r_z = box['rotations']['x'], box['rotations']['y'], box['rotations']['z']
        r_m = np.eye(4)
        r_m[:3, :3] = Rotation.from_euler('xyz', [r_x, r_y, r_z],degrees=True).as_matrix()
        r_m[:3, 3] = [x, y, z]
        # print(r_m)
        new_box = trimesh.primitives.Box(extents=[l, w, h], transform=r_m)
        new_box.apply_transform(trans)
        new_box.visual.face_colors = [0, 0, 0, 100]

        # scene.add_geometry(new_box)
        # scene.show()
        inside_idx = inside_test(point, new_box.vertices)
        inside_mask = np.zeros(len(point),dtype=np.bool)
        inside_mask[inside_idx]=1
        sem_mask = (sem==sem_id)
        mask_1 = inside_mask & sem_mask
        mask_1_idx = list(np.where(mask_1==1)[0])
        affordance_idx_list += mask_1_idx

    # exit()
    # pc2 = tz
    # exit()

    np.save(f'{save_path}/scene_{str(img_id).zfill(6)}_affordance.npy',np.asarray(affordance_idx_list))
    print(f'{save_path}/scene_{str(img_id).zfill(6)}_affordance.npy finished!')

def parallel_generate_scene_affordance(proc,save_path):
    from multiprocessing import Pool
    p = Pool(processes = proc)
    res_list = []
    for i in range(0,cfg['num_images']):
        res_list.append(p.apply_async(generate_scene_affordance, (i,save_path,)))
    p.close()
    p.join()
    for res in tqdm(res_list):
        res.get()

if  __name__ =='__main__':
    save_path = 'scene_affordance_mask'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    # generate_scene_affordance(0, save_path)
    parallel_generate_scene_affordance(6,save_path)

    # test save file
