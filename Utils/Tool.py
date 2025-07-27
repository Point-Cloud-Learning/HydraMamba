import json
import os
import random

import torch
import numpy as np
from torch import nn
from torch.functional import F


def set_save_seed(seed_dir):
    os.environ["PYTHONHASHSEED"] = str(0)
    # torch.backends.cudnn.deterministic = True  torch.backends.cudnn.benchmark = False  torch.backends.cudnn.enabled = False
    if os.path.exists(seed_dir):
        with open(seed_dir, 'r') as f:
            seed_values = json.load(f)
        seed_values["py_state"][1] = tuple(seed_values["py_state"][1])
        random.setstate(tuple(seed_values["py_state"]))
        seed_values["numpy_state"][1] = np.array(seed_values["numpy_state"][1])
        np.random.set_state(tuple(seed_values["numpy_state"]))
        torch.manual_seed(seed_values["torch_seed"])
        torch.cuda.manual_seed(seed_values["torch_cuda_seed"])
        torch.cuda.manual_seed_all(seed_values["torch_cuda_seed"])
    else:
        py_state = random.getstate()
        numpy_state = tuple(item if type(item) != np.ndarray else item.tolist() for item in np.random.get_state())
        torch_seed = torch.initial_seed()
        torch_cuda_seed = torch.cuda.initial_seed()
        seed_dic = {"torch_cuda_seed": torch_cuda_seed, "torch_seed": torch_seed, "numpy_state": numpy_state,
                    "py_state": py_state}
        with open(seed_dir, 'w') as f:
            json.dump(seed_dic, f)


def create_experiment_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Create successfully at %s' % path)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1).to(points.device)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def query_ball_point(radius, max_sample, xyz, query_xyz, dis_mats):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = query_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    group_idx[dis_mats > radius] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :max_sample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, max_sample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, in_channels_dec=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, in_channels_dec=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,in_channels_dec=-1)+sum(dst**2,in_channels_dec=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sqrt(torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1))


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, num_point):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((num_point,))
    distance = np.ones((N,)) * 1e10
    # farthest = np.random.randint(0, N)
    farthest = 0
    for i in range(num_point):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    # point = point[centroids.astype(np.int32)]
    return centroids.astype(np.int32)


class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, label_smooth=None, class_num=0):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of ModelNet40 output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12

        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot
            # label smoothing
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1), max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)
        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)).reshape(-1, 1) + torch.log(torch.exp(pred + eps).sum(dim=1)).reshape(-1, 1)

        return loss.mean()
