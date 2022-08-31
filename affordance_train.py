import torch
import torch.optim as optim
import torch.utils.data
import os
import numpy as np
import argparse
import time
from semdataset import GraspDataset
from affordance_model import backbone_pointnet2
import yaml
from tqdm import tqdm

with open('config/base_config_2022.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser('Train DLR Grasp')
parser.add_argument('--batchsize', type=int, default=cfg['train']['batchsize'], help='input batch size')
parser.add_argument('--workers', type=int, default=cfg['train']['workers'], help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=cfg['train']['epoches'], help='number of epochs for training')
parser.add_argument('--gpu', type=str, default=cfg['train']['gpu'], help='specify gpu device')
parser.add_argument('--learning_rate', type=float, default=cfg['train']['learning_rate'], help='learning rate for training')
parser.add_argument('--optimizer', type=str, default=cfg['train']['optimizer'], help='type of optimizer')
parser.add_argument('--theme', type=str, default=cfg['train']['theme'], help='type of train')
parser.add_argument('--model_path', type=str, default=cfg['eval']['model_path'], help='type of train')
FLAGS = parser.parse_args()

theme = 'quat_loss' + str(FLAGS.batchsize)
time_now = str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
LOG_DIR = os.path.join(BASE_DIR, 'experiment', time_now + '_' + theme)
# LOG_DIR = os.path.join(BASE_DIR, 'experiment', theme)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FOUT_test = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT_test.write(str(FLAGS) + '\n')
LOG_FOUT_train = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT_train.write(str(FLAGS) + '\n')
save_model_dir = os.path.join(LOG_DIR, 'checkpoints')

if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

def log_string_test(out_str):
    LOG_FOUT_test.write(out_str + '\n')
    LOG_FOUT_test.flush()
    print(out_str)

def log_string_train(out_str):
    LOG_FOUT_train.write(out_str + '\n')
    LOG_FOUT_train.flush()
    print(out_str)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 10 epochs"""
    lr = max(FLAGS.learning_rate / (2 ** (epoch // 10)), 0.00001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(optimizer,num_epoches,train_dataloader,test_dataloader,model):
    model = model.train()
    for epoch in range(num_epoches):
        loss_list, acc_list = [],[]
        for i, (data,index) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            for k in data.keys():
                data[k] = data[k].cuda().float()
            # print(data['point'].shape,data['norm_point'].shape)
            bat_pred_afford, bat_pred_graspable, bat_pred_pose, bat_pred_joint, loss_dict, acc_dict = model(data['point'],data['norm_point'].transpose(1, 2),data)
            # print(pred.shape)
            if cfg['train']['use_bin_loss']:
                loss_dict,acc_dict = model.module.get_loss(pred,data)
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()
                loss_list.append([v.item() for k in loss_dict.keys() if k !='total_loss' for v in loss_dict[k].values()])
                acc_list.append([v for k in acc_dict.keys() for v in acc_dict[k].values()])
                each_tax_mean_loss = np.mean(loss_list,0)
                mean_loss = np.mean(each_tax_mean_loss.reshape(-1,4),0)
                each_tax_mean_acc = np.mean(acc_list,0)
                mean_acc = np.mean(each_tax_mean_acc.reshape(-1,8),0)
                if (i%cfg['train']['printfreq'] == 0) or (i == len(train_dataloader)-1):
                    print('epoch:%i\t iter:%.0f/%.0f\t gpts loss:%.3f\t pose loss:%.3f\t' 
                          'joint loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t' 
                          'TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t' 
                          'r:%.3f\t F1:%.3f\t '
                          %(epoch,i,len(train_dataloader),mean_loss[0],mean_loss[1],
                           mean_loss[2],mean_loss[3],mean_acc[0],mean_acc[1],
                           mean_acc[2],mean_acc[3],mean_acc[4],mean_acc[5],
                           mean_acc[6],mean_acc[7]
                           ))
            else:
                # loss_dict,acc_dict = model.module.get_loss(pred,data)
                loss = loss_dict['total_loss']
                # print(loss)
                loss.sum().backward()
                optimizer.step()
                loss_list.append([v.mean().item() for k in loss_dict.keys() if k !='total_loss' and k !='afford_loss'  for v in loss_dict[k].values()])
                acc_list.append([v.mean().item()  for v in acc_dict.values()])
                each_tax_mean_loss = np.mean(loss_list,0)
                mean_loss = np.mean(each_tax_mean_loss.reshape(-1,7),0)
                each_tax_mean_acc = np.mean(acc_list,0)
                mean_acc = np.mean(each_tax_mean_acc.reshape(-1,9),0)
                if (i%cfg['train']['printfreq'] == 0) or (i == len(train_dataloader)-1):
                    print('epoch:%i\t iter:%.0f/%.0f\t affordance loss:%.3f\tgpts loss:%.3f\t depth loss:%.3f\t pose loss:%.3f\t'
                             'joint loss:%.3f\t contact loss:%.3f\t collision loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t'
                             'TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t'
                             'r:%.3f\t F1:%.3f\t pose:%.3f\t '
                             %(epoch,i,len(train_dataloader),loss_dict['afford_loss'].mean().item(),mean_loss[0],mean_loss[1],
                               mean_loss[2],mean_loss[3],mean_loss[4],mean_loss[5],mean_loss[6],mean_acc[0],mean_acc[1],
                               mean_acc[2],mean_acc[3],mean_acc[4],mean_acc[5],
                               mean_acc[6],mean_acc[7],mean_acc[8]
                               ))
        if cfg['train']['use_bin_loss']:
            log_string_train('epoch:%i\t iter:%.0f/%.0f\t gpts loss:%.3f\t pose loss:%.3f\t'
                  'joint loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t'
                  'TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t'
                  'r:%.3f\t F1:%.3f\t '
                  %(epoch,i,len(train_dataloader),mean_loss[0],mean_loss[1],
                    mean_loss[2],mean_loss[3],mean_acc[0],mean_acc[1],
                    mean_acc[2],mean_acc[3],mean_acc[4],mean_acc[5],
                    mean_acc[6],mean_acc[7]
                    ))
        else:
            log_string_train('epoch:%i\t iter:%.0f/%.0f\t affordance loss:%.3f\tgpts loss:%.3f\t depth loss:%.3f\t pose loss:%.3f\t'
                  'joint loss:%.3f\t contact loss:%.3f\t collision loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t'
                  'TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t'
                  'r:%.3f\t F1:%.3f\t pose:%.3f\t '
                  %(epoch,i,len(train_dataloader),loss_dict['afford_loss'].mean().item(),mean_loss[0],mean_loss[1],
                    mean_loss[2],mean_loss[3],mean_loss[4],mean_loss[5],mean_loss[6],mean_acc[0],mean_acc[1],
                    mean_acc[2],mean_acc[3],mean_acc[4],mean_acc[5],
                    mean_acc[6],mean_acc[7],mean_acc[8]
                    ))
        torch.save(model.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'model', epoch))
        test(test_dataloader,model)

def test(test_dataloader,model):
    model = model.eval()
    with torch.no_grad():
        loss_list, acc_list = [],[]
        for i, (data,index) in enumerate(tqdm(test_dataloader)):
            for k in data.keys():
                data[k] = data[k].cuda().float()
            bat_pred_afford, bat_pred_graspable, bat_pred_pose, bat_pred_joint, loss_dict, acc_dict = model(data['point'],data['norm_point'].transpose(1, 2),data)
            if cfg['train']['use_bin_loss']:
                loss_dict,acc_dict = model.module.get_bin_loss(pred,data)
                loss_list.append([v.item() for k in loss_dict.keys() if k !='total_loss' for v in loss_dict[k].values()])
                acc_list.append([v for k in acc_dict.keys() for v in acc_dict[k].values()])
                each_tax_mean_loss = np.mean(loss_list,0)
                mean_loss = np.mean(each_tax_mean_loss.reshape(-1,4),0)
                each_tax_mean_acc = np.mean(acc_list,0)
                mean_acc = np.mean(each_tax_mean_acc.reshape(-1,8),0)
            else:
                # loss_dict,acc_dict = model.module.get_loss(pred,data)
                loss = loss_dict['total_loss']
                loss_list.append([v.mean().item() for k in loss_dict.keys() if k !='total_loss' and k !='afford_loss'  for v in loss_dict[k].values()])
                acc_list.append([v.mean().item()  for v in acc_dict.values()])
                each_tax_mean_loss = np.mean(loss_list,0)
                mean_loss = np.mean(each_tax_mean_loss.reshape(-1,7),0)
                each_tax_mean_acc = np.mean(acc_list,0)
                mean_acc = np.mean(each_tax_mean_acc.reshape(-1,9),0)
        if cfg['train']['use_bin_loss']:
            log_string_test('test:\tgpts loss:%.3f\t pose loss:%.3f\t'
                  'joint loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t'
                  'TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t'
                  'r:%.3f\t F1:%.3f\t'
                  %(mean_loss[0],mean_loss[1],
                    mean_loss[2],mean_loss[3],mean_acc[0],mean_acc[1],
                    mean_acc[2],mean_acc[3],mean_acc[4],mean_acc[5],
                    mean_acc[6],mean_acc[7]
                    ))
        else:
            log_string_test('test:\t affordance loss:%.3f\tgpts loss:%.3f\t depth loss:%.3f\t pose loss:%.3f\t'
                             'joint loss:%.3f\t contact loss:%.3f\t collision loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t'
                             'TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t'
                             'r:%.3f\t F1:%.3f\t pose:%.3f\t'
                             %(loss_dict['afford_loss'].mean().item(),mean_loss[0],mean_loss[1],
                               mean_loss[2],mean_loss[3],mean_loss[4],mean_loss[5],mean_loss[6], mean_acc[0],mean_acc[1],
                               mean_acc[2],mean_acc[3],mean_acc[4],mean_acc[5],
                               mean_acc[6],mean_acc[7],mean_acc[8]
                               ))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])
    dataset_path = 'point_grasp_data'
    train_data = GraspDataset(dataset_path,split='train')
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size = FLAGS.batchsize,
                                                   shuffle=True,
                                                   num_workers = FLAGS.workers)
    test_data = GraspDataset(dataset_path,split='val')
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                               batch_size = FLAGS.batchsize,
                                               shuffle=True,
                                               num_workers = FLAGS.workers)


    model = backbone_pointnet2()
    model = torch.nn.DataParallel(model,device_ids=cfg['train']['gpu']).cuda()
    if cfg['train']['pretrain']:
        model.load_state_dict(torch.load(os.path.join(f"{FLAGS.model_path}/model_{str(cfg['eval']['epoch']).zfill(3)}.pth")))
        print(f"load pretrained model:{FLAGS.model_path}/model_{str(cfg['eval']['epoch']).zfill(3)}.pth")

    optim_params = [
        {'params': model.parameters(),
         'lr': FLAGS.learning_rate,
         'betas': (0.9, 0.999),
         'eps': 1e-08},
    ]
    optimizer = optim.Adam(optim_params)
    train(optimizer,80,train_dataloader,test_dataloader,model)
    # model.load_state_dict(torch.load(os.path.join(f'{FLAGS.model_path}/model_079.pth')))
    # vis_model(model,test_dataloader)
