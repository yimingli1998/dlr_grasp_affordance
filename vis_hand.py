import trimesh
import glob
import torch
import numpy as np
import os
from utils import scene_utils
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
import yaml

with open('config/base_config.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

pos = np.asarray([0,0,0])
# quat = np.asarray([0.9155427,0.3815801,0.1271934,0])
quat = np.asarray([-0.707,0,0,0.707])
# dlr_init = np.asarray(grasp_dict_20f['DLR_init']['joint_init'])*np.pi/180.
# hand = scene_utils.load_hand(pos,quat,dlr_init)
# hand.show()
types = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
# types = ['Pen_Pinch']
for k in types:
    joint_init = np.asarray(grasp_dict_20f[k]['joint_init'])*np.pi/180.
    joint_end = np.asarray(grasp_dict_20f[k]['joint_final'])*np.pi/180.
    joint = np.linspace(joint_init,joint_end,5)
    for i,j in enumerate(joint):
        scene = trimesh.Scene()

        hand = scene_utils.load_hand(pos,quat,j)
        v = hand.vertices
        f = hand.faces
        v,f = trimesh.remesh.subdivide(v,f)
        new_hand = trimesh.Trimesh(v,f)
        new_hand.show()
        print(new_hand.vertices.shape)
        # hand = scene_utils.load_hand(pos,quat,np.asarray([-5, 20, 20, 20,
        #                                                   5,  20,  20,  20,
        #                                                   0,  5,  5,  5,
        #                                                   -5,  0,  0,  0,
        #                                                   -10,  0,  -0,  -0])*np.pi/180.)
        # v = hand.vertices
        # n = hand.vertex_normals
        # ray_visualize = trimesh.load_path(np.hstack((v, v + n / 100)).reshape(-1, 2, 3))
        # ray_visualize.face_colors = [255,0,0]
        scene.add_geometry([hand])
        scene.show()