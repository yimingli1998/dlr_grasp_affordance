import torch
import matplotlib.pyplot as plt
from utils.train_tool import point2point_signed

def get_distance(pc_h, pc_s, n_h, n_s):
    s_h ,h_s, _, _= point2point_signed(pc_h, pc_s, n_h, n_s)
    return s_h ,h_s

def get_distance_loss(s_h ,h_s,hand_mask):
    h_s = h_s[:,hand_mask]/0.005
    h_s = 2*torch.sigmoid(2*torch.abs(h_s))-1
    contact_loss = torch.sum(h_s**2)/h_s.shape[1]/h_s.shape[0]
    s_h_mask = torch.where(s_h<0)
    collision_loss = -torch.sum(s_h[s_h<0])/(torch.sum(s_h<0)+0.00001)*100.0
    return contact_loss, collision_loss


if __name__ == '__main__':
    pc_1 = torch.rand([1, 10, 3], dtype=torch.float)
    pc_2 = torch.rand([1, 20, 3], dtype=torch.float)
    n_1 = torch.tensor([-1, 0, 0], dtype=torch.float).repeat(1, 20, 1)
    n_2 = torch.tensor([1, 0, 0], dtype=torch.float).repeat(1, 20, 1)
    loss = get_distance_loss(pc_1, pc_2, n_1, n_2)
    print(loss)
