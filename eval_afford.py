import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import argparse
import time
import copy
from semdataset import GraspDataset
from affordance_model import backbone_pointnet2
import yaml
import loss_utils
from utils import scene_utils,grasp_utils,eval_utils,common_util
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
import trimesh
from dlr_mujoco_grasp import MujocoEnv
with open('config/base_config_2022.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def test(output_path = 'output',split = 'test_easy'):
    model = backbone_pointnet2().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(f"{cfg['eval']['model_path']}/model_{str(cfg['eval']['epoch']).zfill(3)}.pth")))
    model = model.eval()
    print('1',split)
    test_data = GraspDataset('point_grasp_data',split)
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size = 1,
                                                  shuffle = False,
                                                  num_workers = 1)
    taxonomy = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)
    # torch.cuda.synchronize()
    # t0 = time.time()
    for i, (data,index) in enumerate(tqdm(test_dataloader)):
        bat_point = copy.deepcopy(data['point'])
        bat_sem = copy.deepcopy(data['sem'])
        bat_index = index
        # print(bat_index)
        for k in data.keys():
            data[k] = data[k].cuda().float()
        bat_pred_afford,bat_pred_graspable,bat_pred_pose,bat_pred_joint = \
            model(data['point'],data['norm_point'].transpose(1, 2),data,cal_loss=False)
        # torch.cuda.synchronize()
        # t1 = time.time()
        # print(t1-t0)
        for point,sem,afford,gp,pose,joint in zip(bat_point,bat_sem,bat_pred_afford,bat_pred_graspable,bat_pred_pose,bat_pred_joint):
            afford_mask = torch.argmax(afford,dim=1).detach().cpu().numpy()
            afford_map = torch.softmax(afford,dim=1)[:,1].detach().cpu().numpy()
            afford_mask = np.asarray(afford_mask,dtype=np.bool)
            point_cuda = point.cuda()
            index= index.item()
            output_hand_grasp = {}
            for t in range(gp.size(-1)): # for each taxonomy
                tax = taxonomy[t]
                scene = trimesh.Scene()
                scene_mesh,_,_ = scene_utils.load_scene(index,split=split)
                scene.add_geometry(scene_mesh)
                scene = scene_utils.add_point_cloud(scene,point,color = cfg['color']['pointcloud'])
                tax_gp,tax_pose,tax_joint = gp[:,:,t],pose[:,:,t],joint[:,:,t]
                out_gp = torch.argmax(tax_gp,dim = 1).bool()
                if torch.sum(out_gp)>0:
                    if cfg['train']['use_bin_loss']:
                        out_gp,out_pos,out_R,out_joint,out_score = loss_utils.decode_pred(point_cuda,tax_gp,tax_pose,tax_joint)
                        out_pos,out_R,out_joint,out_score = out_pos.detach().cpu().numpy(),\
                                                  out_R.detach().cpu().numpy(),\
                                                  out_joint.detach().cpu().numpy(),\
                                                  out_score.detach().cpu().numpy()
                    else:
                        gp_score = F.softmax(tax_gp,dim = 1)[:,1].detach().cpu().numpy()
                        # score = gp_score + afford_map
                        score = gp_score
                        tax_gp,tax_pose,tax_joint = tax_gp.detach().cpu().numpy(),tax_pose.detach().cpu().numpy(),tax_joint.detach().cpu().numpy()
                        depth,quat = tax_pose[:,0],tax_pose[:,1:]
                        depth = depth*8.0+20
                        out_gp = np.argmax(tax_gp,1)
                        final_mask = (out_gp==1) & afford_mask

                        mat = trimesh.transformations.quaternion_matrix(quat)
                        out_R = mat[:,:3,:3]
                        approach = mat[:,:3,2]
                        offset = (depth/100.0*(approach.T)).T
                        out_pos = point + offset

                        out_pos,out_R,out_joint,out_score = out_pos[final_mask==1],out_R[final_mask==1],tax_joint[final_mask==1],score[final_mask==1]

                    out_sem = sem[final_mask==1]
                    out_sem = out_sem.detach().cpu().numpy()
                    # test gt
                    out_pos,out_quat,out_joint,select_mask = scene_utils.decode_pred_new(out_pos,out_R,out_joint,tax)
                    out_sem = out_sem[select_mask]
                    # print(type(out_sem))
                    out_score = out_score[select_mask]

                    for i,(p,q,j) in enumerate(zip(out_pos,out_quat,out_joint)):
                        hand_mesh = scene_utils.load_hand(p, q, j, color=cfg['color'][taxonomy[t]])
                        # hand_meshes.append(hand_mesh)
                        # hand_mesh = scene_utils.load_init_hand(p, q, init_hand, color=cfg['color'][taxonomy[t]])
                        scene.add_geometry(hand_mesh)
                        # scene.show()
                        break
                    # print(out_sem.shape,out_pos.shape,out_quat.shape,out_joint.shape,out_score.shape)
                    output_hand_grasp[tax] = {
                        'pos':            out_pos,
                        'quat':           out_quat,
                        'joint':          out_joint,
                        'obj':            out_sem,
                        'score':          out_score
                    }
                    # print(type(out_sem))
                    np.save(os.path.join(output_path,f'img_{index}.npy'),output_hand_grasp)
                    # break


def evaluate(path,img_id):
    print('img id:',img_id)
    scene = trimesh.Scene()
    scene_mesh,gt_objs,_ = scene_utils.load_scene(img_id,split='train')
    # scene_mesh.show()
    num_objs = len(gt_objs)
    env = MujocoEnv()
    scene_xml = env.create_scene_obj('mujoco_objects/objects_xml',img_id)
    env.update_scene_model(scene_xml)
    state = env.get_env_state()
    init_obj_height = state.qpos[-5:-5-7*(num_objs):-7]

    grasp = np.load(os.path.join(path,f'img_{img_id}.npy'),allow_pickle= True).item()
    # print(grasp.keys())
    penetration = {}
    for k in grasp.keys():
        init_hand = trimesh.load(f'hand_taxonomy_mesh/{k}.stl')
        pos,quat,joint,sem,score = grasp[k]['pos'],grasp[k]['quat'],grasp[k]['joint'],grasp[k]['obj'],grasp[k]['score']
        R = trimesh.transformations.quaternion_matrix(quat)[:,:3,:3]
        depth_list,volume_list,success_list,obj_list = [],[],[],[]
        for c in np.unique(sem):
            # print('c',c)
            if c> 0.1: # 0 is background
                ins_pos,ins_R,ins_joint,ins_score = pos[sem==c], \
                                                    R[sem==c], \
                                                    joint[sem==c], \
                                                    score[sem==c]
                if len(ins_pos) >0:
                    order = np.argsort(ins_score)
                    p,_R,j,s = ins_pos[order[-1]],ins_R[order[-1]],ins_joint[order[-1]],ins_score[order[-1]]
                    # ins_pos,ins_R,ins_joint = grasp_utils.grasp_nms(ins_pos,ins_R,ins_joint,ins_score)
                    # if len(ins_pos) > 1: # each objec choice two grasp
                    # ins_pos,ins_R,ins_joint = ins_pos[0],ins_R[0],ins_joint[0]
                    q = common_util.matrix_to_quaternion(_R)
                    # ins_pos,ins_quat,ins_joint,mask= scene_utils.decode_pred_new(ins_pos,ins_R,ins_joint,k)
                    # print(j)
                    hand_mesh = scene_utils.load_hand(p, q,j, color=cfg['color'][k])
                    # hand_mesh = scene_utils.load_init_hand(p, q,init_hand, color=cfg['color'][k])
                    depth, volume_sum = eval_utils.calculate_metric(hand_mesh,scene_mesh)
                    # hand_mesh = scene_utils.load_init_hand(p, q,init_hand, color=cfg['color'][k])
                    # depth, volume_sum = eval_utils.calculate_metric(hand_mesh,scene_mesh)
                    depth_list.append(depth)
                    volume_list.append(volume_sum)
                    init_joint = np.asarray(grasp_dict_20f[k]['joint_init'])*np.pi/180.
                    final_joint = np.asarray(grasp_dict_20f[k]['joint_final'])*np.pi/180.
                    env.set_hand_pos(j=init_joint, quat=q, pos=p)
                    env.step(100)
                    env.nie(j)
                    env.step(300)
                    env.qi()
                    for _ in range (200):
                        env.step(5)
                        cur_state = env.get_env_state().qpos
                        obj_height = cur_state[-5:-5-7*(num_objs):-7]
                        lift_height = obj_height-init_obj_height
                        if np.max(lift_height) >0.2:
                            success =1
                            obj_list.append(c)
                            break
                        success = 0
                        # if cur_state
                    env.set_env_state(state)
                    success_list.append(success)

                        # scene.add_geometry(hand_mesh)
        mean_depth = np.mean(depth_list)
        mean_volume = np.mean(volume_list)
        mean_success = np.mean(success_list)
        completion = len(list(set(obj_list)))/float(num_objs)
        penetration[k] = {
            'depth':        mean_depth,
            'volume':       mean_volume,
            'success':      mean_success,
            'completion':   completion,
            'obj_list':     obj_list

        }
        print(img_id,penetration)
        np.save(f'output_res/{img_id}_res_test_medium.npy',penetration)
        # np.save()
    # env.close()
    return penetration

def evaluate_without_tax(path,img_id):
    print('img id:',img_id)
    taxonomy = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
    scene = trimesh.Scene()
    scene_mesh,gt_objs,_ = scene_utils.load_scene(img_id,split='test')

    num_objs = len(gt_objs)
    env = MujocoEnv()
    scene_xml = env.create_scene_obj('mujoco_objects/objects_xml',img_id)
    env.update_scene_model(scene_xml)
    state = env.get_env_state()
    init_obj_height = state.qpos[-5:-5-7*(num_objs):-7]

    grasp = np.load(os.path.join(path,f'img_{img_id}_quat.npy'),allow_pickle= True).item()
    # print(grasp.keys())
    penetration = {}
    all_grasp = []
    for i,k in enumerate(grasp.keys()):
        pos,quat,joint,sem,score = grasp[k]['pos'],grasp[k]['quat'],grasp[k]['joint'],grasp[k]['obj'],grasp[k]['score']
        all_grasp.append(np.concatenate([pos,quat,joint,sem[:,np.newaxis],score[:,np.newaxis],np.asarray([i]*len(pos))[:,np.newaxis]],axis = -1))
    all_grasp = np.concatenate(all_grasp,axis = 0)
    pos,quat,joint,sem,score,tax = all_grasp[:,:3],all_grasp[:,3:7],all_grasp[:,7:27],all_grasp[:,27],all_grasp[:,28],all_grasp[:,29]
    R = trimesh.transformations.quaternion_matrix(quat)[:,:3,:3]
    depth_list,volume_list,success_list,obj_list = [],[],[],[]
    for c in np.unique(sem):
        # print('c',c)
        if c> 0.1: # 0 is background
            ins_pos,ins_R,ins_joint,ins_score,ins_tax = pos[sem==c], \
                                                R[sem==c], \
                                                joint[sem==c], \
                                                score[sem==c], \
                                                tax[sem==c]
            if len(ins_pos) >0:
                ins_pos,ins_R,ins_joint,ins_tax = grasp_utils.grasp_nms(ins_pos,ins_R,ins_joint,ins_score,ins_tax)
                if len(ins_pos) > 2: # each objec choice two grasp
                    ins_pos,ins_R,ins_joint,ins_tax = ins_pos[:2],ins_R[:2],ins_joint[:2],ins_tax[:2]
                ins_quat = common_util.matrix_to_quaternion(ins_R)
                # ins_pos,ins_quat,ins_joint,mask= scene_utils.decode_pred_new(ins_pos,ins_R,ins_joint,k)
                for i,(p,q,j,t) in enumerate(zip(ins_pos,ins_quat,ins_joint,ins_tax)):
                    # print('i',i)
                    k = taxonomy[int(t)]
                    init_hand = trimesh.load(f'hand_taxonomy_mesh/{k}.stl')
                    # hand_mesh = scene_utils.load_hand(p, q, j, color=cfg['color'][k])
                    # depth, volume_sum = eval_utils.calculate_metric(hand_mesh,scene_mesh)
                    hand_mesh = scene_utils.load_init_hand(p, q,init_hand, color=cfg['color'][k])
                    depth, volume_sum = eval_utils.calculate_metric(hand_mesh,scene_mesh)
                    depth_list.append(depth)
                    volume_list.append(volume_sum)
                    init_joint = np.asarray(grasp_dict_20f[k]['joint_init'])*np.pi/180.
                    final_joint = np.asarray(grasp_dict_20f[k]['joint_final'])*np.pi/180.
                    env.set_hand_pos(j=init_joint, quat=q, pos=p)
                    env.step(100)
                    env.nie(final_joint)
                    env.step(300)
                    env.qi()
                    for _ in range (200):
                        env.step(5)
                        cur_state = env.get_env_state().qpos
                        obj_height = cur_state[-5:-5-7*(num_objs):-7]
                        lift_height = obj_height-init_obj_height
                        if np.max(lift_height) >0.2:
                            success =1
                            obj_list.append(c)
                            break
                        success = 0
                        # if cur_state
                    env.set_env_state(state)
                    success_list.append(success)

                        # scene.add_geometry(hand_mesh)
            mean_depth = np.mean(depth_list)
            mean_volume = np.mean(volume_list)
            mean_success = np.mean(success_list)
            completion = len(list(set(obj_list)))/float(num_objs)
            penetration = {
                'depth':        mean_depth,
                'volume':       mean_volume,
                'success':      mean_success,
                'completion':   completion,
                'obj_list':     obj_list

            }
    print(penetration)
    np.save(f'output_res/{img_id}_res_quat.npy',penetration)
    return penetration


def parallel_evaluate(path='output',split = 'test_easy',proc = 8):
    from multiprocessing import Pool
    if split =='val':
        imgIds = list(range(16000,16400))
    elif  split =='test_easy':
        imgIds = list(range(800,810))
    elif split =='test_medium':
        imgIds = list(range(400,800))
    elif split =='test_hard':
        imgIds = list(range(0,400))
    else:
        imgIds = list(range(0,1200))

    from multiprocessing import Pool
    p = Pool(processes = proc)
    res_list = []
    for img_id in imgIds:
        # res_list.append(p.apply_async(evaluate_without_tax, (path,img_id,)))
        res_list.append(p.apply_async(evaluate, (path,img_id,)))
    p.close()
    p.join()
    # output = []
    # for res in tqdm(res_list):
    #     output.append(res.get())
    # return output

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])
    # test(output_path = 'output',split ='test')
    # for i in range(0,100):
    #     evaluate('output',i)
    evaluate('output',36)
    # parallel_evaluate('output',split = 'test_medium',proc = 16)

    # # evaluate without tax
    # import glob
    # # # res = glob.glob('output_res/*_test_medium.npy')
    # res = glob.glob('output_res/*_test_medium.npy')
    # output = [np.load(r,allow_pickle =True).item() for r in res]
    # mean_depth = np.mean([out['depth'] for out in output])
    # mean_volume = np.mean([out['volume'] for out in output])
    # mean_success = np.mean([out['success'] for out in output])
    # mean_complete = np.mean([out['completion'] for out in output])
    # print(f'depth:{mean_depth}')
    # print(f'volume:{mean_volume}')
    # print(f'success:{mean_success}')
    # print(f'complete:{mean_complete}')

    # # evaluate
    # import glob
    # # # res = glob.glob('output_res/*_quat.npy')
    # res = glob.glob('output_res/*_test_medium.npy')
    # output = [np.load(r,allow_pickle =True).item() for r in res]
    # taxonomy = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
    #
    # for k in taxonomy:
    #     print(k)
    #     mean_depth = np.mean([out[k]['depth'] for out in output if k in out.keys()])
    #     mean_volume = np.mean([out[k]['volume'] for out in output if k in out.keys()])
    #     mean_success = np.mean([out[k]['success'] for out in output if k in out.keys()])
    #     mean_complete = np.mean([out[k]['completion'] for out in output if k in out.keys()])
    #     print(f'depth:{mean_depth}')
    #     print(f'volume:{mean_volume}')
    #     print(f'success:{mean_success}')
    #     print(f'complete:{mean_complete}')

