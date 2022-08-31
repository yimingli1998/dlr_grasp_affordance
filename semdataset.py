import numpy as np
np.set_printoptions(threshold=np.inf)
import trimesh
from tqdm import tqdm
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import common_util, scene_utils,pc_utils
import torch
import copy
import yaml
import random
with open('config/base_config.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

class GraspDataset:
    def __init__(self,data_path,split='train'):
        self.data_path = data_path
        self.num_imgs = cfg['num_images']
        self.taxonomies = ['Parallel_Extension','Pen_Pinch','Palmar_Pinch','Precision_Sphere','Large_Wrap']
        self.num_points = cfg['dataset']['num_points']
        self.R_hand = np.load('utils/R_hand.npy')
        self.R_hand_inv = common_util.inverse_transform_matrix(self.R_hand[:3,:3],self.R_hand[:3,3])
        self.vis = False
        self.split = split
        if split =='train':
            self.imgIds = list(range(0,16000))
        elif split =='val':
            self.imgIds = list(range(16000,16400))
        elif split =='test_easy':
            self.imgIds = list(range(0,400))
        elif split =='test_medium':
            self.imgIds = list(range(400,800))
        elif split =='test_hard':
            self.imgIds = list(range(800,1200))
        elif split =='test':
            self.imgIds = list(range(0,1200))
    def __getitem__(self, index):
        # index = 2560
        print(index)
        img_id = self.imgIds[index]
        # while img_id==7299:  # len(affordance index) ==0
            # img_id = random.choice(self.imgIds)
        affordance_index = np.load(f"scene_affordance_mask/scene_{str(img_id).zfill(6)}_affordance.npy")
        while len(affordance_index) == 0:
            img_id = random.choice(self.imgIds)
            affordance_index = np.load(f"scene_affordance_mask/scene_{str(img_id).zfill(6)}_affordance.npy")

        # scene_utils.vis_obj_pointnormal(img_id)
        obj_points = np.load(f"scene_obj_points/scene_{str(img_id//cfg['num_images_per_scene']).zfill(6)}_obj_point.npy")
        obj_points_choice = np.random.choice(len(obj_points),8192,replace=True)
        obj_points_collision = obj_points[obj_points_choice]

        obj_affordance_points = np.load(f"scene_obj_points/scene_{str(img_id//cfg['num_images_per_scene']).zfill(6)}_obj_afford_point.npy")
        obj_affordance_points_choice = np.random.choice(len(obj_affordance_points),2048,replace=True)
        obj_affordance_points = obj_affordance_points[obj_affordance_points_choice]

        # obj_points_choice = np.random.choice(len(obj_points),2048,replace=True)
        # obj_points = obj_points[obj_points_choice]
        obj_points_contact = np.concatenate([obj_points_collision,obj_affordance_points],axis=0)

        # print('affordance_index:',len(affordance_index),'img_id',img_id)
        point, sem = scene_utils.load_scene_pointcloud(img_id, use_base_coordinate=cfg['use_base_coordinate'],split=self.split)
        affordance_point = point[affordance_index]
        affordance_label = -np.ones(len(point),dtype=np.long)
        # random select planar points as negative data
        random_choice = np.random.choice(len(point),
                                         int(cfg['dataset']['sample_planar_rate']*len(point)),
                                         replace=False)
        affordance_label[random_choice] = 0
        affordance_label[affordance_index] = 1
        for s in np.unique(sem[affordance_label==1]):
            unafford_mask = (sem==s) & (affordance_label==-1)
            affordance_label[unafford_mask] = 0
        # scene_mesh,_,_ = scene_utils.load_scene(img_id)

        # point_choice =np.random.choice(len(point),40000,replace=True)
        # scene_utils.add_scene_cloud(scene,point[point_choice])
        # scene_utils.add_point_cloud(scene,obj_points[:,:3],color=[255,0,0])
        # scene_utils.add_point_cloud(scene,point[affordance_label==1],color=[255,0,0])
        # scene_utils.add_point_cloud(scene,point[affordance_label==0],color=[0,255,0])
        # scene_utils.add_point_cloud(scene,point[affordance_label==-1],color=[0,0,255])
        # scene.show()
        # exit()


        raw_grasp = np.load(f'{self.data_path}/scene_{str(img_id).zfill(6)}_label.npy',allow_pickle=True).item()
        if self.vis:
            scene_utils.vis_grasp_dataset(img_id,cfg)

        point_data = {}
        point_data['point'] = point
        point_data['id'] = img_id

        point_data['affordance_label'] = affordance_label
        point_data['object_point_contact'] = obj_points_contact
        point_data['object_point_collision'] = obj_points_collision
        # if (self.split=='train' or self.split=='val'):
        #     point_data['sem'] = np.zeros(len(point))
        # else:
        point_data['sem'] = sem
        center = np.mean(point,axis=0)
        point_data['norm_point'] = point - center

        grasp_point_label = -np.ones(len(point))

        # random select planar points as negative data
        grasp_point_label[random_choice] = 0

        grasp_point_index = list(raw_grasp['DLR_init'].keys())
        grasp_point_label[grasp_point_index] = 0

        for taxonomy in self.taxonomies:
            point_label = copy.deepcopy(grasp_point_label)
            good_points_index = list(raw_grasp[taxonomy].keys())
            point_label[good_points_index] = 1

            hand_conf = np.zeros((len(point),26))   # N*26
            if raw_grasp[taxonomy]:
                hand_conf_label = np.concatenate(list(raw_grasp[taxonomy].values())).reshape(-1,31)[:,3:]   # N*28
                pos = hand_conf_label[:,:3]
                quat = hand_conf_label[:,3:7]
                mat = trimesh.transformations.quaternion_matrix(quat)
                if len(mat.shape) <3:
                    mat = mat[np.newaxis,:,:]
                mat[:,:3,3] = pos

                # right dot R_hand_inv
                new_mat = np.dot(mat,self.R_hand_inv)
                new_pos = new_mat[:,:3,3]
                new_R = new_mat[:,:3,:3]
                approach = new_R[:,:3,2] # the last column
                offset = new_pos - point[good_points_index]
                depth = (np.sum(offset*approach,axis =-1)*100-20)/8.0 # m to cm 20~28 ->0~1
                if cfg['train']['use_bin_loss']:
                    binormal = new_R[:,:3,1]
                    angle_ = np.arctan2(binormal[:,1], binormal[:,0])
                    angle_[angle_<0] = angle_[angle_<0] + np.pi*2.
                    grasp_angle = angle_ / np.pi * 180
                    azimuth_ = np.arctan2(approach[:,1], approach[:,0])
                    azimuth_[azimuth_<0] = azimuth_[azimuth_<0] + np.pi*2.
                    azimuth_angle = azimuth_ / np.pi * 180
                    elevation_ = np.arctan2(-approach[:,2], np.sqrt(approach[:,0]**2 + approach[:,1]**2)) + np.pi/2.
                    elevation_angle = elevation_ / np.pi * 180
                    # print(depth[0],azimuth_angle[0],elevation_angle[0],grasp_angle[0])
                    depth = depth - 20 # [20,28]->[0,8]
                    hand_conf[good_points_index,0] = depth
                    hand_conf[good_points_index,1] = azimuth_angle
                    hand_conf[good_points_index,2] = elevation_angle
                    hand_conf[good_points_index,3] = grasp_angle
                    # hand_conf[good_points_index,4] = None
                    hand_conf[good_points_index,5:] = hand_conf_label[:,7:]

                    # norm joint
                    joint_init =  np.asarray(grasp_dict_20f[taxonomy]['joint_init'])*np.pi/180.0
                    joint_final =  np.asarray(grasp_dict_20f[taxonomy]['joint_final'])*np.pi/180.0
                    hand_conf[good_points_index,6:] = np.clip(hand_conf[good_points_index,6:],joint_init,joint_final)
                    hand_conf[good_points_index,6:] = (hand_conf[good_points_index,6:]-joint_init)/(joint_final-joint_init+0.00001)
                    # print('joint_final',joint_final/np.pi*180.0)
                    hand_conf = hand_conf[:,:26]

                else:
                    new_quat = common_util.matrix_to_quaternion(new_R)

                    hand_conf[good_points_index,0] = depth
                    hand_conf[good_points_index,1:5] = new_quat

                    hand_conf[good_points_index,5:] = hand_conf_label[:,7:]
                    # depth,quat,metric,joint
                    # norm joint
                    joint_init =  np.asarray(grasp_dict_20f[taxonomy]['joint_init'])*np.pi/180.0
                    joint_final =  np.asarray(grasp_dict_20f[taxonomy]['joint_final'])*np.pi/180.0
                    hand_conf[good_points_index,6:] = np.clip(hand_conf[good_points_index,6:],joint_init,joint_final)
                    hand_conf[good_points_index,6:] = (hand_conf[good_points_index,6:]-joint_init)/(joint_final-joint_init+0.00001)

            label = np.concatenate((point_label[:,np.newaxis],hand_conf),axis =-1)
            # print(Counter(label[:,0]))
            point_data[taxonomy] = label
        crop_point,crop_index = pc_utils.crop_point(point)
        choice = np.random.choice(len(crop_point),self.num_points,replace=False)
        for k in point_data.keys():
            # print(k)
            if (k.startswith('object_point') is False) and k!= 'id':
                point_data[k] = point_data[k][crop_index][choice]
                # if k in self.taxonomies:
                #     print(k,Counter(point_data[k][:,0]))
        show_scene_point_and_object_point = False
        if show_scene_point_and_object_point:
            from scipy import spatial
            kdtree = spatial.KDTree(point_data['object_point_collision'][:,:3])
            points_query = kdtree.query_ball_point(point_data['point'],0.005)
            points_query = [item for sublist in points_query for item in sublist]
            points_query = list(set(points_query))
            point_for_vis = copy.deepcopy(point_data['object_point_collision'][:,:3])
            point_for_vis = np.delete(point_for_vis,points_query,axis=0)
            print(point_for_vis.shape)
            scene = trimesh.Scene()
            scene_utils.add_point_cloud(scene,point_for_vis[:,:3],color=[255,60,0])
            # scene.show()
            # scene = trimesh.Scene()
            scene_utils.add_scene_cloud(scene,point_data['point'])
            # scene.show()
        # scene_utils.vis_point_data(point_data,cfg,img_id)
        # scene_utils.vis_point_data(point_data,cfg)
        return point_data,img_id

    def __len__(self):
        return len(self.imgIds)
        # return 1

if  __name__ =='__main__': 
    data_path = 'point_grasp_data'
    gd = GraspDataset(data_path,split = 'train') 
    # gd.__getitem__(180)
    dataloader = torch.utils.data.DataLoader(gd, batch_size=1, shuffle=True,
                                             num_workers=1)
    for i,(data,id) in enumerate(tqdm(dataloader)):
        # for k in data.keys():
        #     print(k,data[k].shape)
        # print(data['object_point'].shape)
        pass
