import torch
import torch.nn as nn
import absl.flags as flags
import torch.nn.functional as F

device = 'cuda'
# from loss.loss import ChamferLoss, chamfer_distance_naive, chamfer_distance_kdtree

FLAGS = flags.FLAGS  # can control the weight of each term here

def feat_consistency_loss(x1, x2):
    x1 = F.normalize(x1, dim=1)  # [bs,1286]
    x2 = F.normalize(x2, dim=1)  # [bs,1286]
    loss = 2 - 2 * (x1 * x2).sum() / x1.shape[0]
    loss = FLAGS.feat_consist_w * loss
    return loss


def prop_sym_matching_loss(PC, PC_re, gt_R, gt_t, sym):
    """
    PC torch.Size([32, 1028, 3])
    PC_re torch.Size([32, 1028, 3])
    p_g_vec torch.Size([32, 3])
    p_r_vec torch.Size([32, 3])
    p_t torch.Size([32, 3])
    gt_R torch.Size([32, 3, 3])
    gt_t torch.Size([32, 3])
    """

    bs = PC.shape[0]
    points_re_cano = torch.bmm(gt_R.permute(0, 2, 1), (PC - gt_t.view(bs, 1, -1)).permute(0, 2, 1))
    points_re_cano = points_re_cano.permute(0, 2, 1)
    res_p_recon = get_p_recon_loss(points_re_cano, gt_t, gt_R, sym, PC_re, PC)
    return res_p_recon


def get_p_recon_loss(points_re_cano, gt_t, gt_R, sym, PC_re, PC):
    y_reflection_gt_PC = get_y_reflection_gt_pc(points_re_cano, gt_t, gt_R, sym)
    yx_reflection_gt_PC = get_yx_reflection_gt_pc(points_re_cano, gt_t, gt_R, sym)
    no_reflection_gt_pc = get_no_reflection_gt_pc(PC, sym)
    res_gt_PC = yx_reflection_gt_PC + y_reflection_gt_PC + no_reflection_gt_pc

    flag = torch.logical_and(sym[:, 0] == 1, torch.sum(sym[:, 1:], dim=-1) == 0).view(-1, 1, 1)
    pc_re = torch.where(flag, torch.zeros_like(PC_re), PC_re)
    res_p_recon = loss_func(res_gt_PC, pc_re)
    return res_p_recon


loss_func = nn.L1Loss()


def get_y_reflection_gt_pc(points_re_cano, gt_t, gt_R, sym):
    """
    For y axis reflection, can, bowl, bottle
    """
    # rotation 180 degree
    gt_re_points = points_re_cano * torch.tensor([-1, 1, -1], dtype=points_re_cano.dtype,
                                                 device=points_re_cano.device).reshape(-1, 3)
    gt_PC = (torch.matmul(gt_R, gt_re_points.transpose(-2, -1)) + gt_t.unsqueeze(-1)).transpose(-2, -1)
    flag = torch.logical_and(sym[:, 0] == 1, torch.sum(sym[:, 1:], dim=-1) > 0).view(-1, 1, 1)
    gt_PC = torch.where(flag, gt_PC, torch.zeros_like(gt_PC))
    return gt_PC


def get_yx_reflection_gt_pc(points_re_cano, gt_t, gt_R, sym):
    """
    For yx axis reflection, laptop, mug
    """
    gt_re_points = points_re_cano * torch.tensor([1, 1, -1], dtype=points_re_cano.dtype,
                                                 device=points_re_cano.device).reshape(-1, 3)
    gt_PC = (torch.matmul(gt_R, gt_re_points.transpose(-2, -1)) + gt_t.unsqueeze(-1)).transpose(-2, -1)
    flag = torch.logical_and(sym[:, 0] == 0, sym[:, 1] == 1).view(-1, 1, 1)
    gt_PC = torch.where(flag, gt_PC, torch.zeros_like(gt_PC))
    return gt_PC


def get_no_reflection_gt_pc(pc, sym):
    # get pc directly for objects with no reflection
    flag = torch.logical_and(sym[:, 0] == 0, sym[:, 1] != 1).view(-1, 1, 1)
    gt_PC = torch.where(flag, pc, torch.zeros_like(pc))
    return gt_PC
