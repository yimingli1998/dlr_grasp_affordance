import numpy as np
import random
import trimesh
import os
import json
import pickle
from tqdm import tqdm
from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import common_util,pc_utils
import torch
import copy
import yaml
import glob
from scipy.spatial.transform import Rotation

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, '../config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

taxonomies = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']

def load_obj(obj_name,color = [255,0,0],transform = None):

    obj = trimesh.load(obj_name)
    if transform:
        obj.apply_transform(transform)
    # obj.visual.face_colors = color
    return obj

def load_init_hand(pos, quat, init_hand,color = [0,0,255]):
    hand_mesh = copy.deepcopy(init_hand)
    T_hand = trimesh.transformations.translation_matrix(pos)
    R_hand = trimesh.transformations.quaternion_matrix(quat)
    matrix_hand = trimesh.transformations.concatenate_matrices(T_hand,R_hand)
    hand_mesh.apply_transform(matrix_hand)
    hand_mesh.visual.face_colors = color
    return hand_mesh

def load_hand(pos, quat, joint_configuration,color = [0,0,255]):

    hit_hand = HitdlrLayer()
    theta_tensor = torch.from_numpy(joint_configuration).reshape(-1, 20)
    # theta_tensor = torch.from_numpy(joint_configuration)
    pose_tensor = torch.from_numpy(np.identity(4)).reshape(-1, 4, 4).float()
    hand_mesh = hit_hand.get_forward_hand_mesh(pose_tensor, theta_tensor, save_mesh=False)
    hand_mesh = np.sum(hand_mesh)
    T_hand = trimesh.transformations.translation_matrix(pos)
    R_hand = trimesh.transformations.quaternion_matrix(quat)
    matrix_hand = trimesh.transformations.concatenate_matrices(T_hand,R_hand)
    hand_mesh.apply_transform(matrix_hand)
    hand_mesh.visual.face_colors = color
    return hand_mesh

def load_hand_15f(pos, quat, joint_configuration,color = [0,0,255]):
    from hitdlr_kinematics.hitdlr_layer import hitdlr_layer_15f
    hit = hitdlr_layer_15f.HitdlrLayer(device='cpu')
    theta_tensor = torch.from_numpy(joint_configuration).reshape(-1, 15)
    # theta_tensor = torch.from_numpy(joint_configuration)
    pose_tensor = torch.from_numpy(np.identity(4)).reshape(-1, 4, 4).float()

    hand_mesh = hit.get_forward_hand_mesh(pose_tensor, theta_tensor, save_mesh=False)[0]
    # hand_mesh.show()
    hand_verts, hand_normal = hit.get_forward_vertices(pose_tensor, theta_tensor)
    # hand_verts = hand_verts[0]
    # hand_faces = hit.get_faces()
    # hand_mesh = trimesh.Trimesh(hand_verts,hand_faces)
    # hand_mesh = np.sum(hand_mesh)
    T_hand = trimesh.transformations.translation_matrix(pos)
    R_hand = trimesh.transformations.quaternion_matrix(quat)
    matrix_hand = trimesh.transformations.concatenate_matrices(T_hand,R_hand)
    hand_mesh.apply_transform(matrix_hand)
    from hitdlr_kinematics.hitdlr_layer.hitdlr_layer_15f import TaxPoints
    v = trimesh.PointCloud(hand_verts[0][TaxPoints['Pen_Pinch']],[255,0,0])
    v.apply_transform(matrix_hand)
    # v =[]
    # hand_mesh.visual.face_colors = color
    return hand_mesh,v

def load_scene_pointcloud(img_id, use_base_coordinate=True,split = 'train'):
    if (split =='train' or split =='val'):
        file_path = os.path.join(dir_path,'../train_dataset/output/bop_data/lm/train_pbr',str(img_id//1000).zfill(6))
    else:
        file_path = os.path.join(dir_path,'../test_dataset/output/bop_data/lm/train_pbr',str(img_id//1000).zfill(6))
        # print('***')
    with open(os.path.join(file_path,'../../camera.json')) as f:
        intrinsics = json.load(f)
    depth_file = os.path.join(file_path,f'depth/{str(img_id%1000).zfill(6)}.png')
    mask_files = glob.glob(os.path.join(file_path,f'mask_visib/{str(img_id%1000).zfill(6)}_*.png'))
    points,sem = pc_utils.depth_to_pointcloud(depth_file,intrinsics,mask_files)
    if use_base_coordinate:
        # load camera to base pose
        with open(os.path.join(file_path,'scene_camera.json')) as f:
            camera_config = json.load(f)[str(img_id%1000)]
        R_w2c = np.asarray(camera_config['cam_R_w2c']).reshape(3,3)
        t_w2c = np.asarray(camera_config['cam_t_w2c'])*0.001
        c_w = common_util.inverse_transform_matrix(R_w2c,t_w2c)
        points = common_util.transform_points(points,c_w)

        # print(np.sum(sem==0))
        table_mask = points[:,2]<0.001
        # print(np.sum(table_ma sk))
        sem[table_mask] = 0
        # print(np.sum(sem==0))
    return points,sem

def load_scene(img_id,use_base_coordinate=True,use_simplified_model=False,split='train',return_obj_mesh=False,add_plannar=True):
    meshes = []
    if (split =='train' or split =='val'):
        file_path = os.path.join(dir_path, '../train_dataset/output/bop_data/lm/train_pbr', str(img_id//1000).zfill(6))
        # print('2',split)
    else:
        file_path = os.path.join(dir_path,'../test_dataset/output/bop_data/lm/train_pbr',str(img_id//1000).zfill(6))
        # print('***')
    # load obj poses
    with open(os.path.join(file_path,'scene_gt.json')) as f:
        gt_objs = json.load(f)[str(img_id%1000)]
    # load camera to base pose
    with open(os.path.join(file_path,'scene_camera.json')) as f:
        camera_config = json.load(f)[str(img_id%1000)]
    R_w2c = np.asarray(camera_config['cam_R_w2c']).reshape(3,3)
    t_w2c = np.asarray(camera_config['cam_t_w2c'])*0.001
    c_w = common_util.inverse_transform_matrix(R_w2c,t_w2c)

    # create plannar
    planar = trimesh.creation.box([1,1,0.01])
    planar.visual.face_colors = cfg['color']['plannar']
    if not use_base_coordinate:
        planar.apply_transform(common_util.rt_to_matrix(R_w2c,t_w2c))
    if add_plannar:
        meshes.append(planar)
    transform_list = []
    for obj in gt_objs:
        if (split =='train' or split =='val'):
            if use_simplified_model:
                mesh = trimesh.load(
                    os.path.join(dir_path, '../train_dataset/lm/simplified_models', 'obj_' + str(obj['obj_id']).zfill(6) + '_simplified.ply'))
            else:
                mesh = trimesh.load(os.path.join(dir_path,'../train_dataset/lm/models','obj_' + str(obj['obj_id']).zfill(6)+'.ply'))
        else:
            mesh = trimesh.load(os.path.join(dir_path,'../test_dataset/lm/models','obj_' + str(obj['obj_id']).zfill(6)+'.ply'))
        T_obj = trimesh.transformations.translation_matrix(np.asarray(obj['cam_t_m2c'])*0.001)
        quat_obj = trimesh.transformations.quaternion_from_matrix(np.asarray(obj['cam_R_m2c']).reshape(3,3))
        R_obj = trimesh.transformations.quaternion_matrix(quat_obj)
        matrix_obj = trimesh.transformations.concatenate_matrices(T_obj,R_obj)
        mesh.apply_transform(matrix_obj)
        transform = matrix_obj
        # boxes = load_affordance_box(obj['obj_id'], matrix_obj)
        if use_base_coordinate:
            mesh.apply_transform(c_w)
            transform = np.dot(c_w,transform)
            # boxes_copy = []
            # for box in boxes:
            #     box.apply_transform(c_w)
                # boxes_copy.append(box)
            # boxes = boxes_copy
        # boxes_list += boxes
        transform_list.append(transform)
        mesh.visual.face_colors = cfg['color']['object']
        meshes.append(mesh)
    if return_obj_mesh:
        return meshes,gt_objs,transform_list
    else:
        scene_mesh = np.sum(m for m in meshes)
        return scene_mesh,gt_objs,transform_list
    # return scene_mesh,gt_objs,transform_list,boxes_list

def load_scene_grasp(img_id,taxonomy):
    scene_idx = img_id//cfg['num_images_per_scene']
    file_path = os.path.join(dir_path,'../scene_grasp_affordance',f'scene_grasp_{str(scene_idx).zfill(4)}.npy')
    scene_grasp = np.load(file_path,allow_pickle=True).item()
    ungraspable_points = scene_grasp[taxonomy]['0']
    print('ungraspable_points:', ungraspable_points.shape)
    graspable_points = scene_grasp[taxonomy]['1']
    print('graspable_points:', graspable_points.shape)
    hand_meshes = []
    if len(graspable_points)>1:
        choice = np.random.choice(len(graspable_points),2,replace = False)
        graspable_points = graspable_points[choice]
        for i,gp in enumerate(graspable_points):
            gp = np.asarray(gp, dtype=np.float32)
            hand_mesh = load_hand(gp[3:6],gp[6:10],gp[11:],color = cfg['color'][taxonomy])
            hand_meshes.append(hand_mesh)
        hand_meshes = np.sum(hand_meshes)
    return hand_meshes

def decode_pickle(obj_name):
    R_hand = np.load(f'{dir_path}/R_hand.npy')
    # sampled_points = np.load(f'sampled_points/{obj_name}_sampled_points.npy')
    sampled_points = np.load(f'new_sampled_points/{obj_name}_sampled_points.npy')
    taxonomies = grasp_dict_20f.keys()
    single_obj_grasp_dict = {}
    for i,s_p in enumerate(sampled_points):
        single_obj_grasp_dict[i] = {}
        single_obj_grasp_dict[i]['point'] = s_p
        for taxonomy in taxonomies:
            single_obj_grasp_dict[i][taxonomy] = []
    for taxonomy in taxonomies:

        # grasp_file = os.path.join(f'tmp/pickle_obj/{obj_name}_{taxonomy}_final.pickle')
        grasp_file = os.path.join(f'pickle_512/{obj_name}_{taxonomy}_final.pickle')
        if os.path.exists(grasp_file):
            with open(grasp_file, 'rb') as f:
                grasp_dicts = pickle.load(f)
            for i,grasp_dict in enumerate(grasp_dicts):
                if not grasp_dict:
                    continue
                metric = np.asarray([grasp_dict['metric']])
                joint_configuration = np.asarray(grasp_dict['joint_configuration'])
                pos = np.asarray(grasp_dict['pos'])
                quat = np.asarray(grasp_dict['quat'])

                R = trimesh.transformations.quaternion_matrix(quat)
                t = trimesh.transformations.translation_matrix(pos)
                R_obj = trimesh.transformations.concatenate_matrices(t, R)
                inv_R_obj = trimesh.transformations.inverse_matrix(R_obj)
                hand_in_obj = trimesh.transformations.concatenate_matrices(inv_R_obj, R_hand)
                translation = copy.deepcopy(hand_in_obj[:3, 3])
                quaternion = trimesh.transformations.quaternion_from_matrix(hand_in_obj)

                hand = np.concatenate([translation,quaternion,metric,joint_configuration],axis = -1)
                point = grasp_dict['point']
                # print('point',point)
                # print('pos',pos)
                # exit()

                dist = np.linalg.norm(sampled_points[:,:3] - point[:3],axis =1)
                index = np.argmin(dist)

                if dist[index]==0:
                    single_obj_grasp_dict[index][taxonomy].append(hand)
                else:
                    print('***')

    for i in single_obj_grasp_dict.keys():
        single_obj_grasp_dict[i]['tax_name'] = []
        for taxonomy in taxonomies:
            if single_obj_grasp_dict[i][taxonomy]:
                if taxonomy not in single_obj_grasp_dict[i]['tax_name']:
                    single_obj_grasp_dict[i]['tax_name'].append(taxonomy)
                single_obj_grasp_dict[i][taxonomy] = np.asarray(single_obj_grasp_dict[i][taxonomy])
    return single_obj_grasp_dict

def vis_grasp_dataset(index,cfg):
    point,_ = load_scene_pointcloud(index, use_base_coordinate=cfg['use_base_coordinate'])
    grasp = np.load(f'{dir_path}/../point_grasp_data_with_affordance/scene_{str(index).zfill(6)}_label.npy',allow_pickle=True).item()
    taxonomies = grasp_dict_20f.keys()
    scene = trimesh.Scene()
    if cfg['vis']['vis_scene']:
        scene_mesh,_,_ = load_scene(index)
        # scene.add_geometry(scene_mesh)
        # scene.show()
    if cfg['vis']['vis_pointcloud']:
        pc = trimesh.PointCloud(point,colors = cfg['color']['pointcloud'])
        scene.add_geometry(pc)
    if cfg['vis']['vis_obj_pointcloud']:
        print(index)
        obj_points = np.load(os.path.join(dir_path,f'../scene_obj_points/scene_{str(index//4).zfill(6)}_obj_point.npy'))
        obj_pc = trimesh.PointCloud(obj_points,colors = cfg['color']['object'])
        scene.add_geometry(obj_pc)
    for taxonomy in taxonomies:
        print(taxonomy)
        if taxonomy == 'DLR_init':
            bad_points_index = list(grasp[taxonomy].keys())
            bad_point = point[bad_points_index]
            if cfg['vis']['vis_pointcloud']:
                bad_pc =trimesh.PointCloud(bad_point,colors = cfg['color']['bad_point'])
                scene.add_geometry(bad_pc)
        else:
            if grasp[taxonomy]:
                good_points_index = list(grasp[taxonomy].keys())
                good_point = point[good_points_index]
                if cfg['vis']['vis_pointcloud']:
                    good_pc =trimesh.PointCloud(good_point,colors = cfg['color']['good_point'])
                    scene.add_geometry(good_pc)
                if cfg['vis']['vis_handmesh']:
                    for index in good_points_index:
                        hand = grasp[taxonomy][index][3:]
                        hand = np.asarray(hand,dtype = np.float32)
                        hand_mesh = load_hand(hand[:3],hand[3:7],hand[8:],color = cfg['color']['hand_mesh'])
                        scene.add_geometry(hand_mesh)
                        break
        scene.show()

def decode_prediction(point, pred_hand, taxonomy, img_id, cfg,vis = True):
    '''
    :param pred_hand: size:(N*(2+1+4))
    :return:
    '''
    # print(pred_hand.shape)
    R_hand = np.load(os.path.join(dir_path,'R_hand.npy'))
    graspable,depth,quat = pred_hand[:,:2],pred_hand[:,2],pred_hand[:,3:]
    out = np.argmax(graspable,1)
    mask = (out == 1)
    depth, quat = depth[mask], quat[mask]
    good_point = point[mask]

    mat = trimesh.transformations.quaternion_matrix(quat)
    approach = mat[:,:3,2]
    offset = (depth*(approach.T)).T
    pos = good_point + offset
    mat[:,:3,3] = pos

    # right dot R_hand_inv
    new_mat = np.dot(mat,R_hand)
    pos = new_mat[:,:3,3]
    R = new_mat[:,:3,:3]
    quat = common_util.matrix_to_quaternion(R)

    if vis:
        scene = trimesh.Scene()
        pc = trimesh.PointCloud(point,colors = cfg['color']['pointcloud'])
        scene.add_geometry(pc)
        good_pc = trimesh.PointCloud(good_point,colors = cfg['color']['good_point'])
        scene.add_geometry(good_pc)

        good_mask = good_point[:,2] > 0.01 #filter plannar
        pos,quat = pos[good_mask], quat[good_mask]
        init_hand = trimesh.load(f'dir_path/../hand_taxonomy_mesh/{taxonomy}.stl')

        scene_mesh, _, _ = load_scene(img_id,use_base_coordinate = True)
        scene.add_geometry(scene_mesh)

        choice = np.random.choice(len(pos),5,replace=True)
        pos,quat = pos[choice],quat[choice]
        for p,q in zip(pos,quat):
            hand_mesh = load_init_hand(p, q, init_hand,color = cfg['color']['hand_mesh'])
            scene.add_geometry(hand_mesh)
        scene.show()


    return pos,quat

def decode_pred_new(pos,R,joint,tax):
    R_hand = np.load(os.path.join(dir_path,'R_hand.npy'))
    mat = np.tile(np.eye(4),[R.shape[0],1,1])
    mat[:,:3,:3] = R
    mat[:,:3,3] = pos
    # print(R)
    mask = R[:,2,2] > 0
    mat,pos,R,joint = mat[mask],pos[mask],R[mask],joint[mask]
    new_mat = np.dot(mat,R_hand)
    # new_mat = mat
    pos = new_mat[:,:3,3]
    R = new_mat[:,:3,:3]
    quat = common_util.matrix_to_quaternion(R)

    joint_init =  np.asarray(grasp_dict_20f[tax]['joint_init'])*np.pi/180.0
    joint_final =  np.asarray(grasp_dict_20f[tax]['joint_final'])*np.pi/180.0
    joint = joint*(joint_final-joint_init)+joint_init
    # joint = joint*180.0/np.pi
    return pos,quat,joint,mask

def decode_groundtruth(index):
    point = np.load(f'{dir_path}/../point_grasp_data/scene_{str(index).zfill(6)}_point.npy')
    grasp = np.load(f'{dir_path}/../point_grasp_data/scene_{str(index).zfill(6)}_label.npy',allow_pickle=True).item()
    taxonomy_list = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch']
    all_hands = {}
    for taxonomy in taxonomy_list:
        if grasp[taxonomy]:
            hand = grasp[taxonomy].values()
            # print(hand)
    return all_hands

def add_scene_cloud(scene,point):
    bg_point = point[point[:,2]<0.01]
    fg_point = point[point[:,2]>=0.01]
    # print(bg_point.shape)
    # print(fg_point.shape)
    bg_pc =trimesh.PointCloud(bg_point,colors = cfg['color']['plannar'])
    # bg_pc =trimesh.PointCloud(bg_point,colors = cfg['color']['object'])
    fg_pc =trimesh.PointCloud(fg_point,colors = cfg['color']['object'])
    scene.add_geometry(bg_pc)
    scene.add_geometry(fg_pc)
    return scene

def add_point_cloud(scene,point,color = [0,255,0]):
    pc =trimesh.PointCloud(point,colors = color)
    scene.add_geometry(pc)
    return scene

def vis_sem_grasp_data(point_data,cfg,index=None):
    R_hand = np.load(os.path.join(dir_path,'R_hand.npy'))
    scene = trimesh.Scene()
    point = point_data['point']
    fg = point[point[:,2] >0.001]
    pc =trimesh.PointCloud(fg,colors = cfg['color']['pointcloud'])
    scene.add_geometry(pc)
    table = point[point[:,2] <0.001]
    pc_table =trimesh.PointCloud(table,colors = [0,100,0,255])
    # planar = trimesh.creation.box([1,1,0.001])
    # planar.visual.face_colors = [100,0,0,100]
    scene.add_geometry(pc_table)
    if index:
        scene_mesh,_,_ = load_scene(index)
        scene.add_geometry(scene_mesh)
    # scene.show()

    label = point_data['label']
    bad_point = point[label[:,0]==0]
    bad_pc = trimesh.PointCloud(bad_point)
    bad_pc = trimesh.PointCloud(bad_point,colors = cfg['color']['bad_point'])
    # scene.add_geometry(bad_pc)
    good_point_index = (label[:,0]==1)
    good_point = point[good_point_index]
    good_pc = trimesh.PointCloud(good_point,colors = cfg['color']['good_point'])
    scene.add_geometry(good_pc)

    hand_conf = label[good_point_index]
    print(hand_conf[0])
    for i,lb in enumerate(hand_conf):
        if cfg['train']['use_bin_loss']:
            pass
        else:
            tax_idx,depth,quat,joint = np.asarray(lb[1],dtype=np.int),lb[2]/100.0,lb[3:7],lb[7:]
            tax = taxonomies[tax_idx]
            joint_init =  np.asarray(grasp_dict_20f[tax]['joint_init'])*np.pi/180.0
            joint_final =  np.asarray(grasp_dict_20f[tax]['joint_final'])*np.pi/180.0
            joint = joint*(joint_final-joint_init)+joint_init

            R = trimesh.transformations.quaternion_matrix(quat)[:3,:3]
            approach = R[:3,2]
            offset = depth * approach
            pos = good_point[i] + offset

            # right dot R_hand
            mat = common_util.rt_to_matrix(R,pos)
            new_mat = np.dot(mat,R_hand)
            new_pos = new_mat[:3,3]
            new_quat = trimesh.transformations.quaternion_from_matrix(new_mat)
            hand_mesh = load_hand(new_pos,new_quat,joint,color = cfg['color'][tax])
            scene.add_geometry(hand_mesh)
            # print(good_point[i])
            # print(label[i,1:4])
            if i>5:
                break
    scene.show()

def vis_point_data(point_data,cfg,index=None):
    R_hand = np.load(os.path.join(dir_path,'R_hand.npy'))
    scene = trimesh.Scene()
    point = point_data['point']
    fg = point[point[:,2] >0.001]
    pc =trimesh.PointCloud(fg,colors = cfg['color']['object'])
    scene.add_geometry(pc)
    table = point[point[:,2] <0.001]
    pc_table =trimesh.PointCloud(table,colors = cfg['color']['plannar'])
    # planar = trimesh.creation.box([1,1,0.001])
    # planar.visual.face_colors = [100,0,0,100]
    scene.add_geometry(pc_table)
    # scene.show()
    afford_label = point_data['affordance_label']
    afford_point = point[afford_label==1]
    afford_pc = trimesh.PointCloud(afford_point,colors = [250,60,0])
    scene.add_geometry(afford_pc)
    scene.show()

    return

    # obj_points = point_data['object_point']
    # obj_pc = trimesh.PointCloud(obj_points[:,:3],colors = [250,60,0])
    # scene.add_geometry(obj_pc)
    # scene.show()
    # return
    if index:
        scene_mesh,_,_ = load_scene(index)
        scene_mesh_for_show = scene.add_geometry(scene_mesh)
        # scene.show()
    # afford_pc = trimesh.PointCloud(point[point_data['affordance_label']==1],color = cfg['color']['affordance_point'])
    # scene.add_geometry(afford_pc)
    sem = point_data['sem']
    hand_nodes = []
    for c in np.unique(sem):
        if c> 0.1:
            tax_idx = random.choice([0,1,2,3,4])
            k = taxonomies[tax_idx]

            label = point_data[k]
            bad_point = point[label[:,0]==0]
            bad_pc = trimesh.PointCloud(bad_point)
            bad_pc = trimesh.PointCloud(bad_point,colors = cfg['color']['bad_point'])
            # scene.add_geometry(bad_pc)
            good_point_index = (label[:,0]==1) & (sem ==c)
            good_point = point[good_point_index]
            # good_pc = trimesh.PointCloud(good_point,colors = cfg['color']['good_point'])
            # scene.add_geometry(good_pc)

            # affordance_point_index = (label[:,0]==2)
            # affordance_point = point[affordance_point_index]
            # affordance_pc = trimesh.PointCloud(affordance_point,colors = cfg['color']['affordance_point'])
            # scene.add_geometry(affordance_pc)
            hand_grasp = label[good_point_index]
            shuffle_idx = list(range(len(hand_grasp)))
            random.shuffle(shuffle_idx)
            hand_grasp = hand_grasp[shuffle_idx]
            # print(hand_grasp)
            for i,lb in enumerate(hand_grasp):
                if cfg['train']['use_bin_loss']:
                    pass
                else:
                    depth,quat,joint = (lb[1]*8.+20)/100.,lb[2:6],lb[7:]
                    joint_init =  np.asarray(grasp_dict_20f[k]['joint_init'])*np.pi/180.0
                    joint_final =  np.asarray(grasp_dict_20f[k]['joint_final'])*np.pi/180.0
                    joint = joint*(joint_final-joint_init)+joint_init
                    R = trimesh.transformations.quaternion_matrix(quat)[:3,:3]
                    approach = R[:3,2]
                    offset = depth * approach
                    pos = good_point[i] + offset

                    # right dot R_hand
                    mat = common_util.rt_to_matrix(R,pos)
                    new_mat = np.dot(mat,R_hand)
                    new_pos = new_mat[:3,3]
                    new_quat = trimesh.transformations.quaternion_from_matrix(new_mat)
                    # test hitdlr_layer_15f
                    # hand_mesh1 = load_hand_15f(new_pos,new_quat,np.take(joint,[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18]),color = cfg['color'][k])
                    hand_mesh2 = load_hand(new_pos,new_quat,joint,color = cfg['color'][k])
                    hand_nodes.append(scene.add_geometry([hand_mesh2]))
                    break
                    # scene.show()
                    # print(good_point[i])
                    # print(label[i,1:4])
    scene.show()
    for hn in hand_nodes:
        scene.delete_geometry(hn)
    scene.show()
    # scene.add_geometry(afford_pc)
    # scene.show()

def load_affordance_box(obj, trans):
    with open(os.path.join(dir_path,'../affordance_labels/' + 'obj_' + str(obj).zfill(6) + '.json'), 'r') as file:
        boxes = json.load(file)
    box_meshes = []
    for box in boxes['objects']:
        x, y, z = box['centroid']['x'], box['centroid']['y'], box['centroid']['z']
        l, w, h = box['dimensions']['length'], box['dimensions']['width'], box['dimensions']['height']
        # print(l,w,h)
        r_x, r_y, r_z = box['rotations']['x'], box['rotations']['y'], box['rotations']['z']
        r_m = np.eye(4)
        r_m[:3, :3] = Rotation.from_euler('xyz', [r_x, r_y, r_z],degrees=True).as_matrix()
        r_m[:3, 3] = [x, y, z]
        # print(r_m)
        new_box = trimesh.primitives.Box(extents=[l, w, h], transform=r_m)
        new_box.visual.face_colors = [255, 0, 0, 100]
        new_box.apply_transform(trans)
        box_meshes.append(new_box)
    return box_meshes

def vis_obj_pointnormal(img_id):
    scene_mesh,_,_,= load_scene(img_id)
    scene = trimesh.Scene()
    obj_points = np.load(f"scene_obj_points/scene_{str(img_id//cfg['num_images_per_scene']).zfill(6)}_obj_point.npy")
    obj_points_choice = np.random.choice(len(obj_points),10000,replace=True)
    obj_points = obj_points[obj_points_choice]
    v = trimesh.PointCloud(obj_points[:,:3])
    ray_visualize = trimesh.load_path(np.hstack((obj_points[:,:3], obj_points[:,:3] + obj_points[:,3:] / 100)).reshape(-1, 2, 3))
    scene.add_geometry([v, ray_visualize,scene_mesh])
    scene.show()

def color_map(point, value):
    point_color = np.zeros((len(point), 3))
    bg_point_mask = point[:,2]<0.01
    mask = value>=0
    point_color[mask,0] = 255
    point_color[mask,1] = 215-(value[mask])*215
    # point_color[mask,2] = (value[mask])*255
    point_color[bg_point_mask] = cfg['color']['plannar']
    return point_color


if  __name__ =='__main__':
    scene_mesh,_,_,box = load_scene(1)
    scene = trimesh.Scene()
    scene.add_geometry(scene_mesh)
    print(scene_mesh.vertices.shape)
    # scene.add_geometry(box)
    scene.show()