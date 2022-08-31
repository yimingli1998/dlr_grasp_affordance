import os
import shutil
import sys
import time

import torch.nn as nn
import torch
import numpy as np
torch.set_printoptions(threshold=1000)
import torch.nn.functional as F
from collections import Counter
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)  # model
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
import pointnet2.pointnet2_utils as pointnet2_utils
import pointnet2.pytorch_utils as pt_utils
import loss_utils
import trimesh
import yaml
from utils import distance_loss, scene_utils
from hitdlr_kinematics.hitdlr_layer import hitdlr_layer_15f

with open('config/base_config_2022.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

azimuth_scope = cfg['bin_loss']['azimuth_scope']
azimuth_bin_size = cfg['bin_loss']['azimuth_bin_size']
elevation_scope = cfg['bin_loss']['elevation_scope']
elevation_bin_size = cfg['bin_loss']['elevation_bin_size']
grasp_angle_scope = cfg['bin_loss']['grasp_angle_scope']
grasp_angle_bin_size = cfg['bin_loss']['grasp_angle_bin_size']
depth_scope = cfg['bin_loss']['depth_scope']
depth_bin_size = cfg['bin_loss']['depth_bin_size']

per_azimuth_bin_num = int(azimuth_scope / azimuth_bin_size)
per_elevation_bin_num = int(elevation_scope / elevation_bin_size)
per_grasp_angle_bin_num = int(grasp_angle_scope / grasp_angle_bin_size)
per_depth_bin_num = int(depth_scope / depth_bin_size)



class backbone_pointnet2(nn.Module):
    def __init__(self):
        super(backbone_pointnet2, self).__init__()
        if cfg['dataset']['use_norm_points']:
            self.sa1 = PointnetSAModule(mlp=[3, 32, 32, 64], npoint=1024, radius=0.1, nsample=32, bn=True)
        else:
            self.sa1 = PointnetSAModule(mlp=[0, 32, 32, 64], npoint=1024, radius=0.1, nsample=32, bn=True)
        self.sa2 = PointnetSAModule(mlp=[64, 64, 64, 128], npoint=256, radius=0.2, nsample=64, bn=True)
        self.sa3 = PointnetSAModule(mlp=[128, 128, 128, 256], npoint=64, radius=0.4, nsample=128, bn=True)
        self.sa4 = PointnetSAModule(mlp=[256, 256, 256, 512], npoint=None, radius=None, nsample=None, bn=True)

        self.fp4 = PointnetFPModule(mlp=[768, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[384, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[320, 256, 128])
        if cfg['dataset']['use_norm_points']:
            self.fp1 = PointnetFPModule(mlp=[128+6, 128, 128])
        else:
            self.fp1 = PointnetFPModule(mlp=[128+3, 128, 128])

        # fc layer
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        if cfg['train']['use_bin_loss']:
            self.grasp_channels = 2
            self.grasp_channels += per_azimuth_bin_num * 2
            self.grasp_channels += per_elevation_bin_num * 2
            self.grasp_channels += per_grasp_angle_bin_num * 2
            self.grasp_channels += per_depth_bin_num * 2
            self.grasp_channels += 20
            grasp_channels = self.grasp_channels*cfg['dataset']['num_taxonomies']
        else:
            self.grasp_channels = 27
            grasp_channels = self.grasp_channels*cfg['dataset']['num_taxonomies']
        self.affordance_channels = 2 # grasp affordance segmentation
        self.conv3 = nn.Conv1d(128, grasp_channels, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, self.affordance_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.gp_weight = torch.tensor(cfg['train']['gp_weight']).float()
        self.afford_weight = torch.tensor(cfg['train']['afford_weight']).float()
        self.R_hand = torch.from_numpy(np.load(os.path.join('utils/R_hand.npy'))).float()

    def forward(self, xyz, points, data, cal_loss=True):
        l1_xyz, l1_points = self.sa1(xyz.contiguous(), points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        feature = self.fp1(xyz.contiguous(), l1_xyz, torch.cat((xyz.transpose(1, 2), points), dim=1), l1_points)
        feature = F.leaky_relu(self.bn1(self.conv1(feature)),negative_slope = 0.2)
        feature = self.drop1(feature)
        feature = F.leaky_relu(self.bn2(self.conv2(feature)),negative_slope = 0.2)
        feature = self.drop2(feature)
        bat_pred_grasp = self.conv3(feature).permute(0,2,1).view(xyz.size(0),-1,self.grasp_channels,cfg['dataset']['num_taxonomies']).contiguous()   # B,N,C,5
        bat_pred_afford = self.conv4(feature).permute(0,2,1).contiguous()
        bat_pred_graspable, bat_pred_pose,bat_pred_joint = bat_pred_grasp[:,:,:2,:],bat_pred_grasp[:,:,2:-20,:],bat_pred_grasp[:,:,-20:,:]
        bat_pred_joint = torch.clamp(bat_pred_joint, -1e-4, 1 + 1e-4)
        # print(bat_pred_afford.shape, bat_pred_graspable.shape, bat_pred_pose.shape, bat_pred_joint.shape)
        # pred = bat_pred_afford, bat_pred_graspable, bat_pred_pose, bat_pred_joint
        if cal_loss:
            loss_dict, acc_dict = self.get_loss(bat_pred_afford, bat_pred_graspable, bat_pred_pose, bat_pred_joint,data)
            return bat_pred_afford, bat_pred_graspable, bat_pred_pose, bat_pred_joint,loss_dict, acc_dict
        else:
            return bat_pred_afford, bat_pred_graspable, bat_pred_pose, bat_pred_joint

    def get_loss(self,bat_pred_afford, bat_pred_graspable, bat_pred_pose, bat_pred_joint,data):
        gt_list = []
        tax_list = []
        loss = 0
        # acc_dict = {}
        # print(bat_pred_graspable.shape,bat_pred_pose.shape,bat_pred_joint.shape)
        afford_label = data['affordance_label'].long()
        point = data['point']
        sem = data['sem']
        obj_points_contact = data['object_point_contact'].unsqueeze(1).expand(-1,point.size(1),-1,-1)
        obj_points_collision = data['object_point_collision'].unsqueeze(1).expand(-1,point.size(1),-1,-1)
        img_ids = data['id'].unsqueeze(1).expand(-1,point.size(1)).long()
        # print(afford_label.shape,bat_pred_afford.shape)
        # exit()
        # print(afford_label.shape)
        afford_mask = torch.where(afford_label > -1)
        # print(bat_pred_afford[afford_mask].shape)
        # print(afford_label[afford_mask].shape)
        # exit()
        afford_loss = nn.CrossEntropyLoss(self.afford_weight.cuda())(bat_pred_afford[afford_mask],afford_label[afford_mask])
        out_bat_afford = torch.argmax(bat_pred_afford,dim =2)
        afford_acc = self.two_class_acc(out_bat_afford,afford_label)
        # print(afford_acc)
        for k in data.keys():
            if data[k].size(-1) == 27: # graspable 1, depth 1,quat 4 ,metric 1 ,joint 20
                gt_list.append(data[k])
                tax_list.append(k)
        assert len(gt_list) == bat_pred_graspable.size(-1)
        loss_dict = {}
        mean_pose_acc = 0
        for i,gt in enumerate(gt_list):
            finger_points_idx = hitdlr_layer_15f.TaxPoints[tax_list[i]]
            if cfg['train']['use_bin_loss']:
                graspable, pose_label, joint = gt[:,:,0].long(), gt[:,:,1:5],gt[:,:,7:]
            else:
                graspable, pose_label, joint = gt[:,:,0].long(), gt[:,:,1:6],gt[:,:,7:]
            pred_graspable,pred_pose,pred_joint = bat_pred_graspable[:,:,:,i],bat_pred_pose[:,:,:,i],bat_pred_joint[:,:,:,i]
            if_gp_mask = torch.where(graspable > -1)
            gp_mask = torch.where(graspable > 0)
            gp_loss = nn.CrossEntropyLoss(self.gp_weight.cuda())(pred_graspable[if_gp_mask],graspable[if_gp_mask])
            # out_gp = torch.argmax(pred_graspable, dim=2)
            # gp_acc = self.two_class_acc(out_gp[if_gp_mask],graspable[if_gp_mask])
            # acc_dict[tax_list[i]] = {
            #     'TP':        gp_acc[0],
            #     'FP':        gp_acc[1],
            #     'TN':        gp_acc[2],
            #     'FN':        gp_acc[3],
            #     'acc':       gp_acc[4],
            #     'p':         gp_acc[5],
            #     'r':         gp_acc[6],
            #     'F1':        gp_acc[7],
            # }
            gp,pose_label,joint_lable,pred_pose,pred_joint,obj_point_contact,obj_point_collision,img_id = point[gp_mask],\
                                                             pose_label[gp_mask],\
                                                             joint[gp_mask],\
                                                             pred_pose[gp_mask],\
                                                             pred_joint[gp_mask], \
                                                             obj_points_contact[gp_mask], \
                                                             obj_points_collision[gp_mask], \
                                                             img_ids[gp_mask], \



            # scene_mesh,_,_ = scene_utils.load_scene(img_id[0].item())
            if len(pose_label)<=0:
                continue
            Qloss = quat_loss()
            pose_loss,pose_acc = Qloss.forward(pred_pose[:,1:],pose_label[:,1:])
            mean_pose_acc += pose_acc
            # acc_dict[tax_list[i]] = {'pose': pose_acc}
            # print(pose_loss,pose_acc)
            pred_depth = torch.clamp(pred_pose[:,0],-1e-4,1+1e-4)
            depth_loss = torch.nn.MSELoss()(pred_depth,pose_label[:,0])*10.0
            joint_loss = torch.nn.MSELoss()(pred_joint,joint_lable)*10.0

            # contact and collision loss
            pose_mat,joint = self.decode_pose_and_joint(pred_pose,pred_joint,gp,tax_list[i])
            # hand_conf_afford_mask = (afford_label[gp_mask]==1)
            # pose_mat_afford,joint_afford = pose_mat[hand_conf_afford_mask],joint[hand_conf_afford_mask]
            # if len(pose_mat_afford) > 50:
            #     hit = hitdlr_layer_15f.HitdlrLayer()
            #     hand_verts, hand_normal = hit.get_forward_vertices(pose_mat_afford,joint_afford)
            #     select_hand_index = torch.multinomial(torch.ones(len(hand_verts)),50,replacement=True)
            #     hand_verts, hand_normal,obj_point_contact,obj_point_collision = hand_verts[select_hand_index], hand_normal[select_hand_index],obj_point_contact[select_hand_index],obj_point_collision[select_hand_index]
            # else:
            hit = hitdlr_layer_15f.HitdlrLayer()
            hand_verts, hand_normal = hit.get_forward_vertices(pose_mat,joint)
            select_hand_index = torch.multinomial(torch.ones(len(hand_verts)),50,replacement=True)
            hand_verts, hand_normal,obj_point_contact,obj_point_collision = hand_verts[select_hand_index], hand_normal[select_hand_index],obj_point_contact[select_hand_index],obj_point_collision[select_hand_index]
            # print(obj_points_contact.shape,obj_points_collision.shape)
            # selected_hand_verts,selected_hand_normal = hand_verts[:,finger_points_idx], hand_normal[:,finger_points_idx]
            # joint_loss = torch.tensor(0)

            s_h ,h_s = distance_loss.get_distance(hand_verts,obj_point_contact[:,:,:3],hand_normal,obj_point_contact[:,:,3:])
            cont_loss, _ = distance_loss.get_distance_loss(s_h,h_s,finger_points_idx)

            s_h ,h_s = distance_loss.get_distance(hand_verts,obj_point_collision[:,:,:3],hand_normal,obj_point_collision[:,:,3:])
            _, coll_loss = distance_loss.get_distance_loss(s_h,h_s,finger_points_idx)

            cont_loss = 5.0*cont_loss
            # print(cont_loss, coll_loss
            loss_dict[tax_list[i]] = {
                'gp_loss':          gp_loss,
                'depth_loss':       depth_loss,
                'pose_loss':        pose_loss,
                'joint_loss':       joint_loss,
                'cont_loss':        cont_loss,
                'coll_loss':        coll_loss,
                'loss':             gp_loss + depth_loss + joint_loss + pose_loss + cont_loss + coll_loss
                # 'loss':             gp_loss + joint_loss
            }

        loss += afford_loss
        loss_dict['afford_loss'] = afford_loss
        for tax in tax_list:
            if tax in loss_dict.keys():
                loss += (loss_dict[tax]['loss'])
            else:
                # only for run code
                loss_dict[tax] = {
                    'gp_loss':          torch.tensor(0.).cuda(),
                    'depth_loss':       torch.tensor(0.).cuda(),
                    'pose_loss':        torch.tensor(0.).cuda(),
                    'joint_loss':       torch.tensor(0.).cuda(),
                    'cont_loss':        torch.tensor(0.).cuda(),
                    'coll_loss':        torch.tensor(0.).cuda(),
                    'loss':             torch.tensor(0.).cuda(),
                }
        acc_dict = {
            'TP':        afford_acc[0],
            'FP':        afford_acc[1],
            'TN':        afford_acc[2],
            'FN':        afford_acc[3],
            'acc':       afford_acc[4],
            'p':         afford_acc[5],
            'r':         afford_acc[6],
            'F1':        afford_acc[7],
            'pose':      mean_pose_acc/float(len(gt_list))
        }
        loss_dict['total_loss'] = loss
        return loss_dict, acc_dict

    def decode_pose_and_joint(self,pose,joint,gp,tax):

        N,k =pose.size()
        p = joint.size(1)

        joint_init = torch.tensor(grasp_dict_20f[tax]['joint_init']).cuda()*np.pi/180.0
        joint_final = torch.tensor(grasp_dict_20f[tax]['joint_final']).cuda()*np.pi/180.0
        joint = joint*(joint_final-joint_init)+joint_init
        if p == 20:
            joint = torch.index_select(joint,1,torch.tensor([0,1,2,4,5,6,8,9,10,12,13,14,16,17,18]).cuda())

        pred_rot_mat = quaternion_to_matrix(pose[:,1:])
        depth = torch.clamp(pose[:,0],-1e-4,1+1e-4)
        depth = depth*8.0+20
        approach = pred_rot_mat[:,:3,2]
        pos = gp + (approach *(depth[:,np.newaxis])/100.)
        pose_mat = torch.from_numpy(np.identity(4)).cuda().reshape(-1, 4, 4).float().repeat(pos.shape[0],1,1)
        # print(pose_mat.device())
        pose_mat[:,:3,:3] = pred_rot_mat
        pose_mat[:,:3,3] = pos
        pose_mat = torch.matmul(pose_mat,self.R_hand.cuda())
        # print(pose_mat[0],joint[0])
        return pose_mat,joint


    # def get_loss(self,pred,data):
    #     gt_list = []
    #     tax_list = []
    #     bat_pred_graspable,bat_pred_pose,bat_pred_joint = pred
    #     for k in data.keys():
    #         if data[k].size(-1) == 27: # graspable 1, depth 1,quat 4 ,metric 1 ,joint 20
    #             gt_list.append(data[k])
    #             tax_list.append(k)
    #     assert len(gt_list) == bat_pred_graspable.size(-1)
    #     loss_dict, acc_dict = {},{}
    #     for i,gt in enumerate(gt_list):
    #         graspable, depth, quat, joint = gt[:,:,0].long(), gt[:,:,1], gt[:,:,2:6],gt[:,:,7:]
    #         pred_graspable, pred_depth, pred_quat, pred_joint = bat_pred_graspable[:,:,:,i],bat_pred_pose[:,:,0,i],bat_pred_pose[:,:,1:,i],bat_pred_joint[:,:,:,i]
    #         if_gp_mask = torch.where(graspable > -1)
    #         gp_mask = torch.where(graspable > 0)
    #         gp_loss = nn.CrossEntropyLoss(self.gp_weight)(pred_graspable[if_gp_mask],graspable[if_gp_mask])
    #         out_gp = torch.argmax(pred_graspable[if_gp_mask], dim=1)
    #         gp_acc = self.two_class_acc(out_gp,graspable[if_gp_mask])
    #
    #         depth_loss = nn.SmoothL1Loss()(pred_depth[gp_mask],depth[gp_mask])
    #         depth_acc_mask = torch.abs(pred_depth[gp_mask]-depth[gp_mask]) < cfg['eval']['dist_thresh']
    #         depth_acc = torch.sum(depth_acc_mask)/float(pred_depth[gp_mask].size(0))
    #         rot_loss,rot_acc = quat_loss()(pred_quat[gp_mask],quat[gp_mask])
    #         joint_loss = torch.nn.MSELoss()(pred_joint,joint)
    #         loss_dict[tax_list[i]] = {
    #             'gp_loss':        gp_loss,
    #             'depth_loss':     depth_loss,
    #             'rot_loss':       rot_loss,
    #             'joint_loss':     joint_loss,
    #             'loss':           gp_loss + depth_loss + rot_loss + joint_loss
    #         }
    #         acc_dict[tax_list[i]] = {
    #             'TP':        gp_acc[0],
    #             'FP':        gp_acc[1],
    #             'TN':        gp_acc[2],
    #             'FN':        gp_acc[3],
    #             'acc':        gp_acc[4],
    #             'p':        gp_acc[5],
    #             'r':        gp_acc[6],
    #             'F1':        gp_acc[7],
    #             'depth_acc': depth_acc.item(),
    #             'quat_acc':   rot_acc.item(),
    #         }
    #
    #     loss = 0
    #     for tax in tax_list:
    #         loss += (loss_dict[tax]['loss'])
    #     loss_dict['total_loss'] = loss
    #     return loss_dict, acc_dict

    def two_class_acc(self,out,gt):
        TP = torch.sum((out == 1) & (gt == 1)).float()
        FP = torch.sum((out == 1) & (gt == 0)).float()
        TN = torch.sum((out == 0) & (gt == 0)).float()
        FN = torch.sum((out == 0) & (gt == 1)).float()
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        acc_list = [TP,FP,TN,FN,acc,p,r,F1]
        return acc_list

class offset_loss(nn.Module):
    def __init__(self):
        super(offset_loss, self).__init__()

    def forward(self,pred_off,gt_off):
        cosine = torch.abs(torch.sum(pred_off*gt_off,dim = 1))
        cosine_loss = -torch.sum(cosine)/float(pred_off.size(0))
        position_loss = torch.nn.SmoothL1Loss()(pred_off,gt_off)

        cos_acc_mask = cosine > np.cos(cfg['eval']['angle_thresh']/180*np.pi)
        pos_acc_mask = torch.norm((pred_off-gt_off),dim=1)<cfg['eval']['dist_thresh']
        acc_mask = cos_acc_mask & pos_acc_mask
        offset_acc = torch.sum(acc_mask)/float(pred_off.size(0))
        return cosine_loss + position_loss * 1000, offset_acc

class quat_loss(nn.Module):
    def __init__(self):
        super(quat_loss, self).__init__()

    def forward(self,pred_quat,gt_quat):
        position_loss = torch.nn.SmoothL1Loss()(pred_quat,gt_quat)
        norm_pred_quat = F.normalize(pred_quat,dim = 1)
        pred_R = quaternion_to_matrix(norm_pred_quat)
        gt_R = quaternion_to_matrix(gt_quat)
        cosine_angle = self.so3_relative_angle(pred_R,gt_R,cos_angle =True)
        angle_loss = -torch.sum(cosine_angle)/float(pred_R.size(0))
        quat_acc = torch.sum(cosine_angle > np.cos(cfg['eval']['quat_thresh']/180*np.pi))/float(pred_R.size(0))
        return angle_loss + position_loss, quat_acc

    def so3_relative_angle(self,R1, R2, cos_angle: bool = True):
        R12 = torch.bmm(R1, R2.permute(0, 2, 1))
        return self.so3_rotation_angle(R12, cos_angle=cos_angle)

    def so3_rotation_angle(self,R, eps: float = 1e-4, cos_angle: bool = True):
        N, dim1, dim2 = R.shape
        if dim1 != 3 or dim2 != 3:
            raise ValueError("Input has to be a batch of 3x3 Tensors.")

        rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
            raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

        # clamp to valid range
        rot_trace = torch.clamp(rot_trace, -1.0 - eps, 3.0 + eps)

        # phi ... rotation angle
        phi = 0.5 * (rot_trace - 1.0)

        if cos_angle:
            return phi
        else:
            # pyre-fixme[16]: `float` has no attribute `acos`.
            return phi.acos()

def quaternion_to_matrix(quaternions):

    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


if  __name__ =='__main__':
    data1 = torch.rand(1,5000,3).cuda()
    data2 = torch.rand(1,5000,3).permute(0,2,1).cuda()
    model = backbone_pointnet2().cuda()
    model(data1,data2)

