""" Utility functions for processing point clouds.
Author: Charles R. Qi, Hao Su
Date: November 2016
"""

import os
import sys
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations, Translate
import torch
import open3d as o3d
import torch.nn as nn
import numpy as np

EPS = 1e-10


def rotate(points, rot, device=torch.device('cpu'), return_trot=False, t=0.1, uniform=False):
    '''

    :param points:
        - A torch tensor of shape (B,3,N)
    :param rot:
        - String one of [z, so3]
    :return:
        - Rotated points
    '''
    trot = None
    if rot == 'z':
        trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Y", degrees=True).to(device)
    elif rot == 'so3':
        trot = Rotate(R=random_rotations(points.shape[0])).to(device)
    elif rot == 'se3':
        trot_R = Rotate(R=random_rotations(points.shape[0])).to(device)
        # if uniform:
        t_ = t * (2 * torch.rand(points.shape[0], 3, device=device) - 1)
        # else:
        #     t_ = t * torch.randn(points.shape[0], 3, device=device)
        trot_T = Translate(t_)
        trot = trot_R.compose(trot_T)
    if trot is not None:
        points = trot.transform_points(points.transpose(1, 2)).transpose(1, 2)

    if return_trot:
        return points, trot
    else:
        return points


def save_numpy_to_pcd(xyz, path_):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(path_, pcd)

def load_pcd_to_numpy(path_):
    pcd = o3d.io.read_point_cloud(path_)
    return np.asarray(pcd.points)

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def remove_knn(x, source_id_to_reomve, k=20, device=torch.device('cuda')):
    x = x.clone()
    batch_size = x.size(0)
    num_points = x.size(2)

    knn_idx = knn(x, k=k).view(batch_size*num_points, k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1) * num_points
    knn_ind_to_remove = knn_idx[idx_base + source_id_to_reomve, :].squeeze(1) + idx_base
    x = x.transpose(2,1)
    all_points_mask = torch.ones(batch_size*num_points, dtype=torch.bool, device=device)
    all_points_mask[knn_ind_to_remove.view(-1)] = 0
    x_ = x.contiguous().view(batch_size*num_points, -1)[all_points_mask, :].view(batch_size,(num_points-k),-1 )

    return x_.transpose(1,2)

def sample(pc, num_samples, device=torch.device('cuda')):
    #pc: Bx3xN
    #Sample the same loc across the batch
    id_to_keep = torch.randint(0, pc.size(2), (num_samples, ), device=device)
    pc_ = pc.clone().detach()[:, :, id_to_keep].clone().contiguous()
    return pc_

def to_rotation_mat(self, rot, which_rot='svd'):
    if which_rot == 'svd':
        u, s, v = torch.svd(rot)
        M_TM_pow_minus_half = torch.matmul(v / (s + EPS).unsqueeze(1), v.transpose(2, 1))
        rot_mat = torch.matmul(rot, M_TM_pow_minus_half)
        # If gradient trick is rqeuired:
        #rot_mat = (rot_mat - rot).detach() + rot
    else:
        # Gram–Schmidt
        rot_vec0 = rot[:,0,:]
        rot_vec1 = rot[:,1,:] - rot_vec0 * torch.sum(rot_vec0 *  rot[:,1,:], dim=-1, keepdim=True) \
                           / (torch.sum(rot_vec0 **2, dim=-1, keepdim=True) + EPS)

        rot_vec2 = rot[:,2,:] - rot_vec0 * torch.sum(rot_vec0 *  rot[:,2,:], dim=-1, keepdim=True) \
                           / (torch.sum(rot_vec0 **2, dim=-1, keepdim=True) + EPS)
        rot_vec2 = rot_vec2 - rot_vec1 * torch.sum(rot_vec1 * rot[:, 2, :], dim=-1, keepdim=True) \
                           / (torch.sum(rot_vec1 **2, dim=-1, keepdim=True) + EPS)
        rot_mat = torch.stack([rot_vec0, rot_vec1, rot_vec2], dim=1)
        rot_mat = rot_mat / torch.sqrt((torch.sum(rot_mat ** 2, dim=2, keepdim=True) + EPS))
    return rot_mat

def farthest_point_sample_xyz(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    fps_points = []
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        fps_points.append(centroid)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    fps_points = torch.cat(fps_points, dim=1)
    return centroids, fps_points

def batched_trace(mat):
    return mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def cal_angular_metric(R1, R2):
    M = torch.matmul(R1, R2.transpose(1, 2))
    dist = torch.acos(torch.clamp((batched_trace(M) - 1) / 2., -1 + EPS, 1 - EPS))
    dist = (180 / np.pi) * dist
    return dist

def to_rotation_mat(rot):
    u, s, v = torch.svd(rot)
    M_TM_pow_minus_half = torch.matmul(v / (s + EPS).unsqueeze(1), v.transpose(2, 1))
    rot_mat = torch.matmul(rot, M_TM_pow_minus_half)

    return rot_mat