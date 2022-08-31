import numpy as np
import trimesh
import os
import json
import pickle
from tqdm import tqdm
from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import common_util, scene_utils,pc_utils
import torch
import copy
import time
from scipy import spatial
import yaml
with open('config/base_config.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

def generate_scene_pointcloud(img_id,save_path):
    scene = trimesh.Scene()
    object_meshes,gt_objs,transform_list = scene_utils.load_scene(img_id,return_obj_mesh=True,add_plannar=False)
    affordance_mask =[]
    affordance_points,affordance_normals = [],[]
    for obj,obj_mesh, trans in zip(gt_objs,object_meshes,transform_list):
        point_list,normal_list = cal_obj_affordance_label(obj,obj_mesh,trans)
        if point_list!=[]:
            affordance_points.append(point_list)
            affordance_normals.append(normal_list)
    affordance_points = np.concatenate([ap for ap in affordance_points],axis = 0)
    affordance_normals = np.concatenate([an for an in affordance_normals],axis = 0)
    print(affordance_points.shape)
    print(affordance_normals.shape)
    scene_mesh =np.sum(m for m in object_meshes)
    normals = scene_mesh.face_normals
    scene_points, scene_points_idx = trimesh.sample.sample_surface_even(scene_mesh,10000)
    scene_normals = normals[scene_points_idx]
    pc1 = trimesh.PointCloud(scene_points,colors = cfg['color']['pointcloud'])
    pc2 = trimesh.PointCloud(affordance_points,colors = cfg['color']['affordance_point'])
    scene.add_geometry([pc1,pc2])
    scene.show()
    # ray_visualize = trimesh.load_path(np.hstack((scene_points, scene_points + scene_normals / 100)).reshape(-1, 2, 3))
    # pc = trimesh.PointCloud(scene_points,colors = cfg['color']['pointcloud'])
    # scene.add_geometry([ray_visualize,pc])
    # scene.show()
    obj_points = np.concatenate([scene_points,scene_normals],axis = -1)
    obj_afford_points = np.concatenate([affordance_points,affordance_normals],axis = -1)
    np.save(f'{save_path}/scene_{str(img_id//4).zfill(6)}_obj_point.npy',obj_points)
    np.save(f'{save_path}/scene_{str(img_id//4).zfill(6)}_obj_afford_point.npy',obj_afford_points)
    print(f'{save_path}/scene_{str(img_id//4).zfill(6)}_obj_point.npy finished, num obj points:{len(obj_points)}, num obj afford points:{len(obj_afford_points)}')

def parallel_generate_scene_pointcloud(proc,save_path):
    from multiprocessing import Pool
    p = Pool(processes = proc)
    res_list = []
    for i in range(0,cfg['num_images'],4):
        res_list.append(p.apply_async(generate_scene_pointcloud, (i,save_path,)))
    p.close()
    p.join()
    for res in tqdm(res_list):
        res.get()

def cal_obj_affordance_label(obj,obj_mesh,trans):
    from generate_scene_affordance import inside_test
    normal = obj_mesh.face_normals
    point, points_idx = trimesh.sample.sample_surface_even(obj_mesh,1024)
    normal = normal[points_idx]
    inside_list = []
    point_list,normal_list = [],[]
    with open(f"affordance_labels/obj_{str(obj['obj_id']).zfill(6)}.json", 'r') as file:
        f = json.load(file)
        if f['objects'] != []:
            for box in f['objects']:
                x, y, z = box['centroid']['x'], box['centroid']['y'], box['centroid']['z']
                l, w, h = box['dimensions']['length'], box['dimensions']['width'], box['dimensions']['height']
                # print(l,w,h)9
                r_x, r_y, r_z = box['rotations']['x'], box['rotations']['y'], box['rotations']['z']
                r_m = np.eye(4)
                from scipy.spatial.transform import Rotation
                r_m[:3, :3] = Rotation.from_euler('xyz', [r_x, r_y, r_z],degrees=True).as_matrix()
                r_m[:3, 3] = [x, y, z]
                # print(r_m)
                new_box = trimesh.primitives.Box(extents=[l, w, h], transform=r_m)
                new_box.apply_transform(trans)
                new_box.visual.face_colors = [0, 0, 0, 100]

                inside_list += inside_test(point, new_box.vertices)
            if inside_list != []:
                point_list = np.asarray(point[inside_list])
                normal_list = np.asarray(normal[inside_list])
    return point_list,normal_list

if  __name__ =='__main__':
    save_path = 'scene_obj_points'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    generate_scene_pointcloud(10, save_path)
    # parallel_generate_scene_pointcloud(6,save_path)

    # test save file
