import trimesh
import glob
import torch
import numpy as np
import os
from utils import scene_utils
import yaml
with open('config/base_config.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

np.random.seed(24003)
files = glob.glob('train_dataset/lm/models/obj_000177.ply')
grasp_path = 'grasp_dataset'
for f in files:

    print(f)
    obj = trimesh.load(f)
    obj.visual.face_colors = [255,215,0]
    vertices,faces= trimesh.remesh.subdivide_to_size(obj.vertices,obj.faces,max_edge=0.01)
    new_obj = trimesh.Trimesh(vertices,faces)
    pc =trimesh.PointCloud(obj.vertices)
    # obj.show()
    # continue
    obj_name = f.split('/')[-1].split('.')[0]
    grasp_label = np.load(os.path.join(grasp_path,f'{obj_name}.npy'),allow_pickle=True).item()
    tax_grasp = {
        'Parallel_Extension':   [],
        'Pen_Pinch':            [],
        'Palmar_Pinch':         [],
        'Precision_Sphere':     [],
        'Large_Wrap':           []

    }
    for p in grasp_label.keys():
        for t in tax_grasp.keys():
            if grasp_label[p][t] != []:
                tax_grasp[t].append(grasp_label[p][t])
    # scene = trimesh.Scene()
    # scene.add_geometry(obj)
    # scene.show()
    # print(dir(obj))
    # obj = obj.subdivide_to_size(0.02)
    # pc = trimesh.PointCloud(obj.vertices,colors=[255,215,0])
    # scene.add_geometry(pc)
    # scene.show()
    show_single_hand = True
    if show_single_hand:
        for t in tax_grasp.keys():
            if t == 'Pen_Pinch':
            # if True:
                if len(tax_grasp[t]) > 0:
                    tax_grasp[t] = np.concatenate(tax_grasp[t], axis=0)
                    choice = np.random.choice(len(tax_grasp[t]), 100, replace=True)
                    hand_conf = tax_grasp[t][choice]
                    for hand_conf in np.asarray(hand_conf,dtype=np.float32):
                        j = np.array([hand_conf[8],hand_conf[9]-0.5,hand_conf[10],
                                     hand_conf[12],hand_conf[13]-0.2,hand_conf[14]-0.6,
                                     hand_conf[16],hand_conf[17]-0.2,hand_conf[18]-0.6,
                                     hand_conf[20],hand_conf[21],hand_conf[22],
                                     hand_conf[24],hand_conf[25],hand_conf[26]])
                        hand,v = scene_utils.load_hand_15f(hand_conf[:3]+np.asarray([0.01,0.03,0.01]),hand_conf[3:7],j)
                        # hand.show()
                        # hand = scene_utils.load_hand(hand_conf[:3],hand_conf[3:7],hand_conf[8:])

                        # Hand =trimesh.proximity.ProximityQuery(hand)
                        # dist = Hand.signed_distance(obj.vertices)
                        # print(dist)
                        # color = np.zeros((len(obj.vertices), 3))
                        # mask = dist>-0.03
                        # dist[dist>0] = 0
                        # color[mask,1] = (-dist[mask]/0.03)*255
                        # color[mask,0] = 255
                        #
                        # print(dist.max(),dist.min())
                        # # color[:,0]= (1-(dist - dist.min())/(dist.max()-dist.min()))*255.0
                        # print(color)
                        #
                        # pc =trimesh.PointCloud(obj.vertices, color)
                        scene = trimesh.Scene()
                        scene.add_geometry(hand)
                        scene.add_geometry(obj)
                        scene.add_geometry(v)
                        # scene.add_geometry(pc)
                        scene.show()

    else:
        scene = trimesh.Scene()
        scene.add_geometry(obj)

        hc_1 = np.asarray([8.6097438e-03, 2.7245662e-01,  6.5467112e-02 , 1.8216725e-02,
         6.4257562e-01 ,- 2.2037500e-01 , 7.3362088e-01 ,           0,
         1.4935853e-01 , 2.7545655e-01 , 8.4980920e-02 , 8.6316153e-02,
         9.5371060e-02 , 1.4170347e-01 , 7.2689998e-01,  7.2509253e-01,
         2.3599467e-03  ,1.4775647e-01 , 8.1081271e-01 , 8.0891126e-01,
         - 2.3444118e-05,  1.0458014e+00 , 1.1340668e+00 , 1.1340631e+00,
         - 8.7256685e-02 , 1.0457983e+00,  1.1340679e+00 , 1.1340643e+00])
        hc_2 = np.asarray([-0.01757636,  0.14939258, - 0.17167296 , 0.5615669,   0.52098745, - 0.30980405,
         - 0.5632372,          0,  0.26149803 , 0.17430758 , 0.504167 ,   0.50311095,
         0.14703287,  0.22144137,  0.37478504,  0.3672616,   0.10868329,  0.30978984,
         0.52395445,  0.51938397, - 0.09338,     0.16175511 , 0.3058447 ,  0.29850373,
         - 0.2616122,   0.78299373,  0.82899255,  0.8219447])
        hc_3 = np.asarray([-0.00907238,  0.25623888, - 0.00909263,  0.14419535,  0.9286006, - 0.05659493,
         0.33720273 ,        0,  0.16775666,  0.20676778,  0.50483745,  0.50381184,
         - 0.10245391,  0.08762102,  0.86485577,  0.8548785, - 0.10983229 , 0.52900904,
         0.74552846 , 0.7370384, - 0.17431396,  0.5225278,   0.87007743,  0.86010504,
         - 0.17438026 , 0.4355414 ,  0.8611621,   0.85190654])
        hand_1 = scene_utils.load_hand(hc_1[:3],hc_1[3:7],hc_1[8:],color = [100,100,100,220])
        hand_2 = scene_utils.load_hand(hc_2[:3],hc_2[3:7],hc_2[8:],color = cfg['color']['Large_Wrap'])
        hand_3 = scene_utils.load_hand(hc_3[:3],hc_3[3:7],hc_3[8:],color = cfg['color']['Precision_Sphere'])
        scene.add_geometry(hand_1)
        # scene.add_geometry(hand_2)
        # scene.add_geometry(hand_3)

        scene.show()

