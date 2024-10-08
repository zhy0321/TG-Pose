import torch
import torch.nn as nn
import absl.flags as flags
import torch.nn.functional as F
from losses.chamfer3D import dist_chamfer_3D
from tools.geom_utils import batch_dot
import math

device = 'cuda'

# from loss.loss import ChamferLoss, chamfer_distance_naive, chamfer_distance_kdtree

FLAGS = flags.FLAGS  # can control the weight of each term here


class TDA_loss(nn.Module):
    def __init__(self):
        super(TDA_loss, self).__init__()
        self.loss_MSE = nn.MSELoss()
        self.loss_MAE = nn.L1Loss(reduction='none')
        if FLAGS.fsnet_loss_type == 'l1':
            self.loss_func_t = nn.L1Loss()
            self.loss_func_s = nn.L1Loss()
            self.loss_func_Rot1 = nn.L1Loss()
            self.loss_func_Rot2 = nn.L1Loss()
            self.loss_func_r_con = nn.L1Loss()
            self.loss_func_Recon = nn.L1Loss()
            self.loss_func = nn.L1Loss()
        elif FLAGS.fsnet_loss_type == 'smoothl1':  # same as MSE
            self.loss_func_t = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_s = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Rot1 = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Rot2 = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_r_con = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Recon = nn.SmoothL1Loss(beta=0.3)

        else:
            raise NotImplementedError

    def forward(self, name_list, pred_list, gt_list, sym, gt_pred_flag=False):
        loss_list = dict()
        if "Rot1" in name_list:
            # Rot1 means the x axis; Rot
            loss_list["Rot1"] = FLAGS.rot_1_w * self.cal_loss_Rot1(pred_list["Rot1"], gt_list["Rot1"])

        if "Rot1_cos" in name_list:
            loss_list["Rot1_cos"] = FLAGS.rot_1_w * self.cal_cosine_dis(pred_list["Rot1"], gt_list["Rot1"])

        if "Rot2" in name_list:
            loss_list["Rot2"] = FLAGS.rot_2_w * self.cal_loss_Rot2(pred_list["Rot2"], gt_list["Rot2"], sym)

        if "Rot2_cos" in name_list:
            loss_list["Rot2_cos"] = FLAGS.rot_2_w * self.cal_cosine_dis_sym(pred_list["Rot2"], gt_list["Rot2"], sym)

        if "Rot_regular" in name_list:
            loss_list["Rot_r_a"] = FLAGS.rot_regular * self.cal_rot_regular_angle(pred_list["Rot1"],
                                                                                  pred_list["Rot2"], sym)

        if "Prop_sym" in name_list and (FLAGS.prop_sym_w > 0):
            Prop_sym_recon = self.prop_sym_matching_loss(gt_list['Recon'],
                                                         pred_list['Recon'],
                                                         pred_list['Rot1'],
                                                         pred_list['Rot2'],
                                                         pred_list['Tran'],
                                                         gt_list['R'],
                                                         gt_list['Tran'],
                                                         sym)
            loss_list["Prop_sym"] = FLAGS.prop_sym_w * Prop_sym_recon

        if "recon_completion" in name_list and (FLAGS.recon_w > 0):
            recon_completion = self.recon_completion_loss(gt_list['Recon'], pred_list['Recon'])
            loss_list["recon_completion"] = FLAGS.recon_w * recon_completion

        if "Tran" in name_list:
            loss_list["Tran"] = FLAGS.tran_w * self.cal_loss_Tran(pred_list["Tran"], gt_list["Tran"])

        if "Size" in name_list:
            loss_list["Size"] = FLAGS.size_w * self.cal_loss_Size(pred_list["Size"], gt_list["Size"])

        if "R_con" in name_list:
            loss_list["R_con"] = FLAGS.r_con_w * self.cal_loss_R_con(pred_list["Rot1"], pred_list["Rot2"],
                                                                     gt_list["Rot1"], gt_list["Rot2"],
                                                                     pred_list["Rot1_f"], pred_list["Rot2_f"], sym)


        if "TDA_h1_cate" in name_list:
            loss_list["TDA_h1_cate"] = self.ph_loss_fn_cate(pred_list["TDA_h1"], gt_list["pdh1_category"],
                                                            gt_list["h1"])

        if "TDA_h1" in name_list:
            loss_list["TDA_h1"] = FLAGS.h1_w * self.ph_loss_fn(pred_list["TDA_h1"], gt_list["h1"])

        if "TDA_h2_cate" in name_list:
            loss_list["TDA_h2_cate"] = self.ph_loss_fn_cate(pred_list["TDA_h2"], gt_list["pdh2_category"],
                                                            gt_list["h2"])

        if "TDA_h2" in name_list:
            loss_list["TDA_h2"] = FLAGS.h2_w * self.ph_loss_fn(pred_list["TDA_h2"], gt_list["h2"])

        if "R_DCD_cate_pred" in name_list:
            loss_list["R_DCD_cate_pred"] = FLAGS.DCD_align * self.R_DCD(gt_list["points_category"],
                                                                        pred_list["Recon"],
                                                                        gt_list["R"],
                                                                        pred_list["Rot1"],
                                                                        pred_list["Rot1_f"],
                                                                        pred_list["Rot2"],
                                                                        pred_list["Rot2_f"],
                                                                        pred_list["Tran"],
                                                                        pred_list["Size"],
                                                                        sym
                                                                        )
        return loss_list

    def feat_consist_loss(self, feat_global, feat_global_knn):
        feat_global = F.normalize(feat_global, dim=1)
        feat_global_knn = F.normalize(feat_global_knn, dim=1)
        loss = 2 - 2 * (feat_global * feat_global_knn).sum() / feat_global.shape[0]
        return loss

    def prop_sym_matching_loss(self, PC, PC_re, p_g_vec, p_r_vec, p_t, gt_R, gt_t, sym):
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
        res_p_recon = self.get_p_recon_loss(points_re_cano, gt_t, gt_R, sym, PC_re, PC)
        return res_p_recon

    def get_p_recon_loss(self, points_re_cano, gt_t, gt_R, sym, PC_re, PC):
        # calculate the symmetry pointcloud reconstruction loss
        # y axis reflection, can, bowl, bottle
        y_reflection_gt_PC = self.get_y_reflection_gt_pc(points_re_cano, gt_t, gt_R, sym)
        yx_reflection_gt_PC = self.get_yx_reflection_gt_pc(points_re_cano, gt_t, gt_R, sym)
        no_reflection_gt_pc = self.get_no_reflection_gt_pc(PC, sym)
        res_gt_PC = yx_reflection_gt_PC + y_reflection_gt_PC + no_reflection_gt_pc

        flag = torch.logical_and(sym[:, 0] == 1, torch.sum(sym[:, 1:], dim=-1) == 0).view(-1, 1, 1)
        pc_re = torch.where(flag, torch.zeros_like(PC_re), PC_re)
        res_p_recon = self.loss_func(res_gt_PC, pc_re)
        return res_p_recon

    def get_p_rt_loss(self, PC, p_t, p_g_vec, PC_re, sym, p_r_vec):
        y_reflec_pc_b, y_reflec_pc_re = self.get_y_reflection_pc_b(PC, p_t, p_g_vec, PC_re, sym)

        yx_reflec_pc_b, yx_reflec_pc_re = self.get_yx_reflection_pc_b(p_r_vec, p_g_vec, PC, PC_re, sym, p_t)
        res_p_rt = self.loss_func(y_reflec_pc_b + yx_reflec_pc_b, yx_reflec_pc_re + y_reflec_pc_re)
        return res_p_rt

    def get_y_reflection_pc_b(self, PC, p_t, p_g_vec, PC_re, sym):
        pc_t_res = PC - p_t.unsqueeze(-2)
        vec_along_p_g = torch.matmul(torch.matmul(pc_t_res, p_g_vec.unsqueeze(-1)),
                                     p_g_vec.unsqueeze(-2))  # bs x 1028 x 3
        a_to_1_2_b = vec_along_p_g - pc_t_res
        PC_b = PC + 2.0 * a_to_1_2_b
        flag = torch.logical_and(sym[:, 0] == 1, torch.sum(sym[:, 1:], dim=-1) > 0).view(-1, 1, 1)
        PC_b = torch.where(flag, PC_b, torch.zeros_like(PC_b))
        PC_re = torch.where(flag, PC_re, torch.zeros_like(PC_re))
        return PC_b, PC_re

    def get_yx_reflection_pc_b(self, p_r_vec, p_g_vec, PC, PC_re, sym, p_t):
        p_z = torch.cross(p_r_vec, p_g_vec)
        p_z = p_z / (torch.norm(p_z, dim=-1, keepdim=True) + 1e-8)
        t = -(torch.matmul(p_z.unsqueeze(-2), PC.transpose(-2, -1))
              - batch_dot(p_z, p_t).view(-1, 1, 1))  # 1 x  1028
        PC_b = PC + 2.0 * torch.matmul(p_z.unsqueeze(-1), t).transpose(-2, -1)
        flag = torch.logical_and(sym[:, 0] == 0, sym[:, 1] == 1).view(-1, 1, 1)
        PC_b = torch.where(flag, PC_b, torch.zeros_like(PC_b))
        PC_re = torch.where(flag, PC_re, torch.zeros_like(PC_re))
        return PC_b, PC_re

    def get_y_reflection_gt_pc(self, points_re_cano, gt_t, gt_R, sym):
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

    def get_yx_reflection_gt_pc(self, points_re_cano, gt_t, gt_R, sym):
        """
        For yx axis reflection, laptop, mug
        """
        gt_re_points = points_re_cano * torch.tensor([1, 1, -1], dtype=points_re_cano.dtype,
                                                     device=points_re_cano.device).reshape(-1, 3)
        gt_PC = (torch.matmul(gt_R, gt_re_points.transpose(-2, -1)) + gt_t.unsqueeze(-1)).transpose(-2, -1)
        flag = torch.logical_and(sym[:, 0] == 0, sym[:, 1] == 1).view(-1, 1, 1)
        gt_PC = torch.where(flag, gt_PC, torch.zeros_like(gt_PC))
        return gt_PC

    def get_no_reflection_gt_pc(self, pc, sym):
        # get pc directly for objects with no reflection
        flag = torch.logical_and(sym[:, 0] == 0, sym[:, 1] != 1).view(-1, 1, 1)
        gt_PC = torch.where(flag, pc, torch.zeros_like(pc))
        return gt_PC

    def cal_loss_R_con(self, p_rot_g, p_rot_r, g_rot_g, g_rot_r, p_g_con, p_r_con, sym):
        # confidence-aware pose regression loss
        dis_g = p_rot_g - g_rot_g
        dis_g_norm = torch.norm(dis_g, dim=-1)
        p_g_con_gt = torch.exp(-13.7 * dis_g_norm * dis_g_norm)  #
        res_g = self.loss_func_r_con(p_g_con_gt, p_g_con)
        res_r = 0.0
        bs = p_rot_g.shape[0]
        for i in range(bs):
            if sym[i, 0] == 0:
                dis_r = p_rot_r[i, ...] - g_rot_r[i, ...]
                dis_r_norm = torch.norm(dis_r)
                p_r_con_gt = torch.exp(-13.7 * dis_r_norm * dis_r_norm)
                res_r += self.loss_func_r_con(p_r_con_gt, p_r_con[i])
        res_r = res_r / bs
        return res_g + res_r

    def cal_loss_Rot1(self, pred_v, gt_v):
        res = self.loss_func_Rot1(pred_v, gt_v)
        return res

    def cal_loss_Rot2(self, pred_v, gt_v, sym):
        bs = pred_v.shape[0]
        res = torch.FloatTensor([0.0]).to(pred_v.device)
        valid = 0.0
        for i in range(bs):
            sym_now = sym[i, 0]
            if sym_now == 1:
                continue
            else:
                pred_v_now = pred_v[i, ...]
                gt_v_now = gt_v[i, ...]
                res += self.loss_func_Rot2(pred_v_now, gt_v_now)
                valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_cosine_dis(self, pred_v, gt_v):
        res = torch.mean((1 - torch.sum(pred_v * gt_v, dim=1)) * 2)
        return res

    def cal_cosine_dis_sym(self, pred_v, gt_v, sym):
        # pred_v  bs x 6, gt_v bs x 6
        bs = pred_v.shape[0]
        res = torch.FloatTensor([0.0]).to(pred_v.device)
        valid = 0.0
        for i in range(bs):
            sym_now = sym[i, 0]
            if sym_now == 1:
                continue
            else:
                pred_v_now = pred_v[i, ...]
                gt_v_now = gt_v[i, ...]
                res += (1.0 - torch.sum(pred_v_now * gt_v_now)) * 2.0
                valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_rot_regular_angle(self, pred_v1, pred_v2, sym):
        bs = pred_v1.shape[0]
        res = torch.FloatTensor([0.0]).to(pred_v1.device)
        pred_v3 = torch.cross(pred_v1, pred_v2)

        valid = 0.0
        for i in range(bs):
            if sym[i, 0] == 1:
                continue
            y_direction = pred_v1[i, ...]
            z_direction = pred_v2[i, ...]
            residual = torch.dot(y_direction, z_direction)
            res += torch.abs(residual)
            valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_loss_Recon(self, pred_recon, gt_recon):
        return self.loss_func_Recon(pred_recon, gt_recon)

    def cal_loss_Tran(self, pred_trans, gt_trans):
        return self.loss_func_t(pred_trans, gt_trans)

    def cal_loss_Size(self, pred_size, gt_size):
        return self.loss_func_s(pred_size, gt_size)


    def ph_loss_fn(self, ph, gt_ph):
        w = (torch.sum(gt_ph, dim=1, keepdim=True) > 0) * 1.0
        if has_nan_or_inf(ph) or has_nan_or_inf(gt_ph):
            loss = torch.mean(self.loss_MAE(ph, ph) * w)
        else:
            loss = torch.mean(self.loss_MAE(ph, gt_ph) * w)
        return loss


    def ph_loss_fn_cate(self, ph, gt_ph, cate_ph):
        w = (torch.sum(gt_ph, dim=1, keepdim=True) > 0) * 1.0
        if has_nan_or_inf(ph) or has_nan_or_inf(gt_ph):
            loss = torch.mean(self.loss_MAE(ph, ph) * w)
        else:
            loss = torch.mean(self.loss_MAE(ph, gt_ph) * w)
        omega = self.omega(gt_ph, cate_ph, 2, 1)
        loss = loss * omega
        return loss

    def omega(self, gt_h1, cate_h1, k, lam):
        w = (torch.sum(cate_h1, dim=1, keepdim=True) > 0) * 1.0
        if has_nan_or_inf(gt_h1) or has_nan_or_inf(cate_h1):
            loss = torch.mean(self.loss_MAE(gt_h1, gt_h1) * w)
        else:
            loss = torch.mean(self.loss_MAE(gt_h1, cate_h1) * w)
        omega = k * math.exp(-lam * loss)

        return omega


    def R_DCD(self, cate_ori, points, g_R, p_g_vec, f_g_vec, p_r_vec, f_r_vec, p_t, p_s, sym):
        near_zeros = torch.full(f_g_vec.shape, 1e-5, device=f_g_vec.device)
        new_y_sym, new_x_sym = get_vertical_rot_vec_in_batch(f_g_vec, near_zeros, p_g_vec, g_R[..., 0])
        new_y, new_x = get_vertical_rot_vec_in_batch(f_g_vec, f_r_vec, p_g_vec, p_r_vec)
        sym_flag = sym[:, 0].unsqueeze(-1) == 1
        new_y = torch.where(sym_flag, new_y_sym, new_y)
        new_x = torch.where(sym_flag, new_x_sym, new_x)
        p_R = get_rot_mat_y_first(new_y, new_x)
        points_re_n = torch.matmul(p_R.transpose(-2, -1), (points - p_t.unsqueeze(-2)).transpose(-2, -1)).transpose(-2,
                                                                                                                    -1)

        points_re_n = points_re_n * p_s.unsqueeze(-2)
        R_DCD_loss = calc_dcd(points_re_n, cate_ori, alpha=70, n_lambda=0.3, return_raw=False,
                              non_reg=False)
        R_DCD_loss = torch.mean(R_DCD_loss)

        return R_DCD_loss

    def recon_completion_loss(self, pcl_in, recon):
        recon_completion_loss = recon_completion(pcl_in, recon, alpha=70, n_lambda=0.3, return_raw=False,
                                                 non_reg=False)
        recon_completion_loss = torch.mean(recon_completion_loss)
        return recon_completion_loss


def get_rot_mat_y_first(y, x):
    # poses

    y = F.normalize(y, p=2, dim=-1)
    z = torch.cross(x, y, dim=-1)
    z = F.normalize(z, p=2, dim=-1)
    x = torch.cross(y, z, dim=-1)

    # (*,3)x3 --> (*,3,3)
    return torch.stack((x, y, z), dim=-1)


def R_recover(pred_v1, pred_v2):
    # bs = pred_v1.shape[0]
    pred_v3 = torch.cross(pred_v1, pred_v2, dim=-1)
    rotation_pred = torch.stack([pred_v2, pred_v1, pred_v3], dim=-1)
    return rotation_pred


def get_vertical_rot_vec_in_batch(c1, c2, y, z):
    c1 = c1.unsqueeze(-1)
    c2 = c2.unsqueeze(-1)
    ##  c1, c2 are weights
    ##  y, x are rotation vectors
    rot_x = torch.cross(y, z, dim=-1)
    rot_x = rot_x / (torch.norm(rot_x, dim=-1, keepdim=True) + 1e-8)
    # cal angle between y and z
    y_z_cos = torch.sum(y * z, dim=-1, keepdim=True)
    y_z_cos = torch.clamp(y_z_cos, -1 + 1e-6, 1 - 1e-6)
    y_z_theta = torch.acos(y_z_cos)
    theta_2 = c1 / (c1 + c2) * (y_z_theta - math.pi / 2)
    theta_1 = c2 / (c1 + c2) * (y_z_theta - math.pi / 2)

    # first rotate y
    c = torch.cos(theta_1)
    s = torch.sin(theta_1)
    rotmat_y = to_rot_matrix_in_batch(rot_x, s, c)
    new_y = torch.matmul(rotmat_y, y.unsqueeze(-1)).squeeze(-1)
    # then rotate z
    c = torch.cos(-theta_2)
    s = torch.sin(-theta_2)
    rotmat_z = to_rot_matrix_in_batch(rot_x, s, c)
    new_z = torch.matmul(rotmat_z, z.unsqueeze(-1)).squeeze(-1)
    return new_y, new_z


def to_rot_matrix_in_batch(rot_x, s, c):
    rx_0 = rot_x[:, 0].unsqueeze(-1)
    rx_1 = rot_x[:, 1].unsqueeze(-1)
    rx_2 = rot_x[:, 2].unsqueeze(-1)
    r1 = torch.cat([rx_0 * rx_0 * (1 - c) + c, rx_0 * rx_1 * (1 - c) - rx_2 * s, rx_0 * rx_2 * (1 - c) + rx_1 * s],
                   dim=-1).unsqueeze(-2)
    r2 = torch.cat([rx_1 * rx_0 * (1 - c) + rx_2 * s, rx_1 * rx_1 * (1 - c) + c, rx_1 * rx_2 * (1 - c) - rx_0 * s],
                   dim=-1).unsqueeze(-2)
    r3 = torch.cat([rx_0 * rx_2 * (1 - c) - rx_1 * s, rx_2 * rx_1 * (1 - c) + rx_0 * s, rx_2 * rx_2 * (1 - c) + c],
                   dim=-1).unsqueeze(-2)
    rotmat = torch.cat([r1, r2, r3], dim=-2)
    return rotmat


def calc_dcd(pred_recon, cate_gt, alpha=0.1, n_lambda=0.3, return_raw=False, non_reg=False):
    pred_recon = pred_recon.float()
    cate_gt = cate_gt.float()
    batch_size, n_pred, _ = pred_recon.shape
    batch_size, n_gt, _ = cate_gt.shape
    assert pred_recon.shape[0] == cate_gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_pred / n_gt)
        frac_21 = max(1, n_gt / n_pred)
    else:
        frac_12 = n_pred / n_gt
        frac_21 = n_gt / n_pred

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(pred_recon, cate_gt, return_raw=True)
    exp_dist1 = torch.exp(-dist1 * alpha)
    exp_dist2 = torch.exp(-dist2 * alpha)

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = loss1 + 0.5 * loss2

    res = loss
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res


def recon_completion(pred_recon, cate_gt, alpha=0.1, n_lambda=0.3, return_raw=False, non_reg=False):
    pred_recon = pred_recon.float()
    cate_gt = cate_gt.float()
    batch_size, n_pred, _ = pred_recon.shape
    batch_size, n_gt, _ = cate_gt.shape
    assert pred_recon.shape[0] == cate_gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_pred / n_gt)
        frac_21 = max(1, n_gt / n_pred)
    else:
        frac_12 = n_pred / n_gt
        frac_21 = n_gt / n_pred

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(pred_recon, cate_gt, return_raw=True)
    exp_dist1 = torch.exp(-dist1 * alpha)
    exp_dist2 = torch.exp(-dist2 * alpha)

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = 0.9 * torch.mean(loss1) + 0.1 * torch.mean(loss2)

    res = loss
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res


def calc_cd(pred, gt, return_raw=False, separate=False):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cham_loss(pred, gt)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0), dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res


def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()
