#   introduced from fs-net
import numpy as np
import cv2
import torch
import math
import absl.flags as flags
import time

FLAGS = flags.FLAGS


def pc_sampler(points, num):
    pc_idxs = np.arange(0, points.shape[0])
    np.random.shuffle(pc_idxs)
    points = points[pc_idxs[:num], :]
    return points


class PC_BasicAugment(object):
    def __init__(self):
        self.mode = 'train'

    def __call__(self, db):
        # aug_bb,

        PC = db['pcl_in'].unsqueeze(0)
        gt_R = db['rotation'].unsqueeze(0)
        gt_t = db['translation'].unsqueeze(0)
        gt_s = db['fsnet_scale'].unsqueeze(0)
        model_point = db['model_point'].unsqueeze(0)
        mean_shape = db['mean_shape'].unsqueeze(0)
        sym = db['sym_info'].unsqueeze(0)
        aug_bb = db['aug_bb'].unsqueeze(0)
        aug_rt_t = db['aug_rt_t'].unsqueeze(0)
        aug_rt_r = db['aug_rt_R'].unsqueeze(0)
        obj_ids = db['cat_id'].unsqueeze(0)
        nocs_scale = db['nocs_scale'].unsqueeze(0)

        # increase the dimension for the element of db

        # augmentation
        bs = PC.shape[0]

        prob_bb = torch.rand((bs, 1), device=PC.device)

        flag = prob_bb < FLAGS.aug_bb_pro
        PC, gt_s, model_point = aug_bb_with_flag(PC, gt_R, gt_t, gt_s, model_point, mean_shape, sym, aug_bb, flag)

        prob_rt = torch.rand((bs, 1), device=PC.device)
        flag = prob_rt < FLAGS.aug_rt_pro
        PC, gt_R, gt_t = aug_rt_with_flag(PC, gt_R, gt_t, aug_rt_t, aug_rt_r, flag)

        # only do bc for mug and bowl
        prob_bc = torch.rand((bs, 1), device=PC.device)
        flag = torch.logical_and(prob_bc < FLAGS.aug_bc_pro, torch.logical_or(obj_ids == 5, obj_ids == 1).unsqueeze(-1))
        PC, gt_s, _, _ = aug_3D_bc_with_flag(PC, gt_R, gt_t, gt_s, model_point, nocs_scale, mean_shape, flag)

        prob_pc = torch.rand((bs, 1), device=PC.device)
        flag = prob_pc < FLAGS.aug_pc_pro
        PC, _ = aug_pc_with_flag(PC, gt_t, flag, FLAGS.aug_pc_r)

        return PC, gt_R, gt_t, gt_s


class PcJitter(object):
    def __init__(self, std=0.01, clip=0.05, p=1):
        self.std = std
        self.clip = clip
        self.p = p

    def __call__(self, points):
        if np.random.uniform() > self.p:
            return points

        data = points.new(points.size(0), 3).normal_(mean=0, std=self.std).clamp_(-self.clip, self.clip)
        # noise = np.clip(self.std * np.random.randn(*points.shape), -1 * self.clip, self.clip)
        points[:, :3] = points[:, :3] + data

        return points


class PcRandomDropout(object):
    def __init__(self, max_dropout_ratio=0.875, p=1):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        # dropout_ratio = self.max_dropout_ratio
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()


class PcRandomCrop(object):
    def __init__(self, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1, min_num_points=4096, max_try_num=10):
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.p = p

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()

        isvalid = False
        try_num = 0
        while not isvalid:
            coord_min = np.min(points[:, :3], axis=0)
            coord_max = np.max(points[:, :3], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            new_coord_range = np.zeros(3)
            new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            new_coord_range[1] = new_coord_range[0] * ar
            new_coord_range[2] = new_coord_range[0] / ar
            # new_coord_range = np.where(new_coord_range>1, 1, new_coord_range)

            new_coord_min = np.random.uniform(0, 1 - new_coord_range)
            new_coord_max = new_coord_min + new_coord_range

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            new_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            new_indices = np.sum(new_indices, axis=1) == 3
            new_points = points[new_indices]

            # other_num = points.shape[0] - new_points.shape[0]
            # if new_points.shape[0] > 0:
            #     isvalid = True
            if new_points.shape[0] >= self.min_num_points and new_points.shape[0] < points.shape[0]:
                isvalid = True

            try_num += 1
            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()

        # other_indices = np.random.choice(np.arange(new_points.shape[0]), other_num)
        # other_points = new_points[other_indices]
        # new_points = np.concatenate([new_points, other_points], axis=0)

        # new_points[:,:3] = (new_points[:,:3] - new_coord_min) / (new_coord_max - new_coord_min) * coord_diff + coord_min
        return torch.from_numpy(new_points).float()


class PcRandomCutout(object):
    def __init__(self, ratio_min=0.3, ratio_max=0.6, p=1, min_num_points=4096, max_try_num=10):
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.p = p
        self.min_num_points = min_num_points
        self.max_try_num = max_try_num

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        points = points.numpy()
        try_num = 0
        valid = False
        while not valid:
            coord_min = np.min(points[:, :3], axis=0)
            coord_max = np.max(points[:, :3], axis=0)
            coord_diff = coord_max - coord_min

            cut_ratio = np.random.uniform(self.ratio_min, self.ratio_max, 3)
            new_coord_min = np.random.uniform(0, 1 - cut_ratio)
            new_coord_max = new_coord_min + cut_ratio

            new_coord_min = coord_min + new_coord_min * coord_diff
            new_coord_max = coord_min + new_coord_max * coord_diff

            cut_indices = (points[:, :3] > new_coord_min) & (points[:, :3] < new_coord_max)
            cut_indices = np.sum(cut_indices, axis=1) == 3

            # print(np.sum(cut_indices))
            # other_indices = (points[:, :3] < new_coord_min) | (points[:, :3] > new_coord_max)
            # other_indices = np.sum(other_indices, axis=1) == 3
            try_num += 1

            if try_num > self.max_try_num:
                return torch.from_numpy(points).float()

            # cut the points, sampling later

            if points.shape[0] - np.sum(cut_indices) >= self.min_num_points and np.sum(cut_indices) > 0:
                # print (np.sum(cut_indices))
                points = points[cut_indices == False]
                valid = True

        # if np.sum(other_indices) > 0:
        #     comp_indices = np.random.choice(np.arange(np.sum(other_indices)), np.sum(cut_indices))
        #     points[cut_indices] = points[comp_indices]
        return torch.from_numpy(points).float()


class PcUpSampling(object):
    def __init__(self, max_num_points, radius=0.1, nsample=5, centroid="random"):
        self.max_num_points = max_num_points
        # self.radius = radius
        self.centroid = centroid
        self.nsample = nsample

    def __call__(self, points):
        t0 = time.time()

        p_num = points.shape[0]
        if p_num > self.max_num_points:
            return points

        c_num = self.max_num_points - p_num

        if self.centroid == "random":
            cids = np.random.choice(np.arange(p_num), c_num)
        else:
            assert self.centroid == "fps"
            fps_num = c_num / self.nsample
            fps_ids = fps(points, fps_num)
            cids = np.random.choice(fps_ids, c_num)

        xyzs = points[:, :3]
        loc_matmul = torch.matmul(xyzs, xyzs.t())
        loc_norm = xyzs * xyzs
        r = torch.sum(loc_norm, -1, keepdim=True)

        r_t = r.t()  # 转置
        dist = r - 2 * loc_matmul + r_t
        # adj_matrix = torch.sqrt(dist + 1e-6)

        dist = dist[cids]
        # adj_sort = torch.argsort(adj_matrix, 1)
        adj_topk = torch.topk(dist, k=self.nsample * 2, dim=1, largest=False)[1]

        uniform = np.random.uniform(0, 1, (cids.shape[0], self.nsample * 2))
        median = np.median(uniform, axis=1, keepdims=True)
        # choice = adj_sort[:, 0:self.nsample*2][uniform > median]  # (c_num, n_samples)
        choice = adj_topk[uniform > median]  # (c_num, n_samples)

        choice = choice.reshape(-1, self.nsample)

        sample_points = points[choice]  # (c_num, n_samples, 3)

        new_points = torch.mean(sample_points, dim=1)
        new_points = torch.cat([points, new_points], 0)

        return new_points


class PcNormalize(object):
    def __init__(self, radius=1):
        self.radius = radius

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __call__(self, points):
        pc = points.numpy()
        pc[:, 0:3] = self.pc_normalize(pc[:, 0:3])
        return torch.from_numpy(pc).float()


class PcRemoveInvalid(object):
    def __init__(self, invalid_value=0):
        self.invalid_value = invalid_value

    def __call__(self, points):
        pc = points.numpy()
        valid = np.sum(pc, axis=1) != self.invalid_value
        pc = pc[valid, :]
        return torch.from_numpy(pc).float()


# class


def fps(points, num):
    cids = []
    cid = np.random.choice(points.shape[0])
    cids.append(cid)
    id_flag = np.zeros(points.shape[0])
    id_flag[cid] = 1

    dist = torch.zeros(points.shape[0]) + 1e4
    dist = dist.type_as(points)
    while np.sum(id_flag) < num:
        dist_c = torch.norm(points - points[cids[-1]], p=2, dim=1)
        dist = torch.where(dist < dist_c, dist, dist_c)
        dist[id_flag == 1] = 1e4
        new_cid = torch.argmin(dist)
        id_flag[new_cid] = 1
        cids.append(new_cid)
    cids = torch.Tensor(cids)
    return cids


# add noise to mask
def defor_2D(roi_mask, rand_r=2, rand_pro=0.3):
    '''

    :param roi_mask: 256 x 256
    :param rand_r: randomly expand or shrink the mask iter rand_r
    :return:
    '''
    roi_mask = roi_mask.copy().squeeze()
    if np.random.rand() > rand_pro:
        return roi_mask
    mask = roi_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_erode = cv2.erode(mask, kernel, rand_r)  # rand_r
    mask_dilate = cv2.dilate(mask, kernel, rand_r)
    change_list = roi_mask[mask_erode != mask_dilate]
    l_list = change_list.size
    if l_list < 1.0:
        return roi_mask
    choose = np.random.choice(l_list, l_list // 2, replace=False)
    change_list = np.ones_like(change_list)
    change_list[choose] = 0.0
    roi_mask[mask_erode != mask_dilate] = change_list
    roi_mask[roi_mask > 0.0] = 1.0
    return roi_mask


# point cloud based data augmentation
# augment based on bounding box
def defor_3D_bb(pc, R, t, s, sym=None, aug_bb=None):
    # pc  n x 3, here s must  be the original s
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    if sym[0] == 1:  # y axis symmetry
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]

        exz = (ex + ez) / 2
        pc_reproj[:, (0, 2)] = pc_reproj[:, (0, 2)] * exz
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        s[0] = s[0] * exz
        s[1] = s[1] * ey
        s[2] = s[2] * exz
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
        return pc_new, s
    else:
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]

        pc_reproj[:, 0] = pc_reproj[:, 0] * ex
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        pc_reproj[:, 2] = pc_reproj[:, 2] * ez
        s[0] = s[0] * ex
        s[1] = s[1] * ey
        s[2] = s[2] * ez
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
        return pc_new, s


def defor_3D_bb_in_batch(pc, model_point, R, t, s, sym=None, aug_bb=None):
    pc_reproj = torch.matmul(R.transpose(-1, -2), (pc - t.unsqueeze(-2)).transpose(-1, -2)).transpose(-1, -2)
    sym_aug_bb = (aug_bb + aug_bb[:, [2, 1, 0]]) / 2.0
    sym_flag = (sym[:, 0] == 1).unsqueeze(-1)
    new_aug_bb = torch.where(sym_flag, sym_aug_bb, aug_bb)
    pc_reproj = pc_reproj * new_aug_bb.unsqueeze(-2)
    model_point_new = model_point * new_aug_bb.unsqueeze(-2)
    pc_new = (torch.matmul(R, pc_reproj.transpose(-2, -1)) + t.unsqueeze(-1)).transpose(-2, -1)
    s_new = s * new_aug_bb
    return pc_new, s_new, model_point_new


def defor_3D_bc(pc, R, t, s, model_point, nocs_scale):
    # resize box cage along y axis, the size s is modified
    ey_up = torch.rand(1, device=pc.device) * (1.2 - 0.8) + 0.8
    ey_down = torch.rand(1, device=pc.device) * (1.2 - 0.8) + 0.8
    # for each point, resize its x and z linealy
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    per_point_resize = (pc_reproj[:, 1] + s[1] / 2) / s[1] * (ey_up - ey_down) + ey_down
    pc_reproj[:, 0] = pc_reproj[:, 0] * per_point_resize
    pc_reproj[:, 2] = pc_reproj[:, 2] * per_point_resize
    pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
    pc_new = pc_new.T

    model_point_resize = (model_point[:, 1] + s[1] / 2) / s[1] * (ey_up - ey_down) + ey_down
    model_point[:, 0] = model_point[:, 0] * model_point_resize
    model_point[:, 2] = model_point[:, 2] * model_point_resize

    lx = max(model_point[:, 0]) - min(model_point[:, 0])
    ly = max(model_point[:, 1]) - min(model_point[:, 1])
    lz = max(model_point[:, 2]) - min(model_point[:, 2])

    lx_t = lx * nocs_scale
    ly_t = ly * nocs_scale
    lz_t = lz * nocs_scale
    return pc_new, torch.tensor([lx_t, ly_t, lz_t], device=pc.device)


# point cloud based data augmentation
# augment based on bounding box
def deform_non_linear(pc, R, t, s, model_point, axis=0):
    # pc  n x 3, here s must  be the original s
    assert axis in [0, 1]
    r_max = torch.rand(1, device=pc.device) * 0.2 + 1.1
    r_min = -torch.rand(1, device=pc.device) * 0.2 + 0.9
    # for each point, resize its x and z
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    per_point_resize = r_min + 4 * (pc_reproj[:, axis] * pc_reproj[:, axis]) / (s[axis] ** 2) * (r_max - r_min)
    pc_reproj[:, axis] = pc_reproj[:, axis] * per_point_resize
    pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
    pc_new = pc_new.T

    norm_s = s / torch.norm(s)
    model_point_resize = r_min + 4 * (model_point[:, axis] * model_point[:, axis]) / (norm_s[axis] ** 2) * (
            r_max - r_min)
    model_point[:, axis] = model_point[:, axis] * model_point_resize

    lx = 2 * max(max(model_point[:, 0]), -min(model_point[:, 0]))
    ly = max(model_point[:, 1]) - min(model_point[:, 1])
    lz = max(model_point[:, 2]) - min(model_point[:, 2])

    lx_t = lx * torch.norm(s)
    ly_t = ly * torch.norm(s)
    lz_t = lz * torch.norm(s)
    size_new = torch.tensor([lx_t, ly_t, lz_t], device=pc.device)

    nocs_scale_aug = torch.norm(torch.tensor([lx, ly, lz]))
    model_point = model_point / nocs_scale_aug

    # nocs_resize = r_min + 4 * (nocs[:, axis] * nocs[:, axis]) / (norm_s[axis] ** 2) * (r_max - r_min)
    # nocs[:, axis] = nocs[:, axis] * nocs_resize
    # nocs = nocs / nocs_scale_aug
    return pc_new, size_new, model_point


def defor_3D_bc_in_batch(pc, R, t, s, model_point, nocs_scale):
    # resize box cage along y axis, the size s is modified
    bs = pc.size(0)
    ey_up = torch.rand((bs, 1), device=pc.device) * (1.2 - 0.8) + 0.8
    ey_down = torch.rand((bs, 1), device=pc.device) * (1.2 - 0.8) + 0.8
    pc_reproj = torch.matmul(R.transpose(-1, -2), (pc - t.unsqueeze(-2)).transpose(-1, -2)).transpose(-1, -2)

    s_y = s[..., 1].unsqueeze(-1)
    per_point_resize = (pc_reproj[..., 1] + s_y / 2.0) / s_y * (ey_up - ey_down) + ey_down
    pc_reproj[..., 0] = pc_reproj[..., 0] * per_point_resize
    pc_reproj[..., 2] = pc_reproj[..., 2] * per_point_resize
    pc_new = (torch.matmul(R, pc_reproj.transpose(-2, -1)) + t.unsqueeze(-1)).transpose(-2, -1)

    new_model_point = model_point * 1.0
    model_point_resize = (new_model_point[..., 1] + s_y / 2) / s_y * (ey_up - ey_down) + ey_down
    new_model_point[..., 0] = new_model_point[..., 0] * model_point_resize
    new_model_point[..., 2] = new_model_point[..., 2] * model_point_resize

    s_new = (torch.max(new_model_point, dim=1)[0] - torch.min(new_model_point, dim=1)[0]) * nocs_scale.unsqueeze(-1)
    return pc_new, s_new, ey_up, ey_down


# def defor_3D_pc(pc, r=0.05):
#     points_defor = torch.randn(pc.shape).to(pc.device)
#     pc = pc + points_defor * r * pc
#     return pc

def defor_3D_pc(pc, gt_t, r=0.2, points_defor=None, return_defor=False):
    if points_defor is None:
        points_defor = torch.rand(pc.shape).to(pc.device) * r
    new_pc = pc + points_defor * (pc - gt_t.unsqueeze(1))
    if return_defor:
        return new_pc, points_defor
    return new_pc


# point cloud based data augmentation
# random rotation and translation
def defor_3D_rt(pc, R, t, aug_rt_t, aug_rt_r):
    #  add_t
    dx = aug_rt_t[0]
    dy = aug_rt_t[1]
    dz = aug_rt_t[2]

    pc[:, 0] = pc[:, 0] + dx
    pc[:, 1] = pc[:, 1] + dy
    pc[:, 2] = pc[:, 2] + dz
    t[0] = t[0] + dx
    t[1] = t[1] + dy
    t[2] = t[2] + dz

    # add r
    '''
    Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
    Rm_tensor = torch.tensor(Rm, device=pc.device)
    pc_new = torch.mm(Rm_tensor, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm_tensor, R)
    R = R_new
    '''
    '''
    x_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    y_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    z_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    Rm = get_rotation_torch(x_rot, y_rot, z_rot)
    '''
    Rm = aug_rt_r
    pc_new = torch.mm(Rm, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm, R)
    R = R_new
    T_new = torch.mm(Rm, t.view(3, 1))
    t = T_new

    return pc, R, t


def defor_3D_rt_in_batch(pc, R, t, aug_rt_t, aug_rt_r):
    pc_new = pc + aug_rt_t.unsqueeze(-2)
    t_new = t + aug_rt_t
    pc_new = torch.matmul(aug_rt_r, pc_new.transpose(-2, -1)).transpose(-2, -1)

    R_new = torch.matmul(aug_rt_r, R)
    t_new = torch.matmul(aug_rt_r, t_new.unsqueeze(-1)).squeeze(-1)
    return pc_new, R_new, t_new


def get_rotation(x_, y_, z_):
    # print(math.cos(math.pi/2))
    x = float(x_ / 180) * math.pi
    y = float(y_ / 180) * math.pi
    z = float(z_ / 180) * math.pi
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]])

    R_y = np.array([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]])

    R_z = np.array([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x)).astype(np.float32)


def get_rotation_torch(x_, y_, z_):
    x = (x_ / 180) * math.pi
    y = (y_ / 180) * math.pi
    z = (z_ / 180) * math.pi
    R_x = torch.tensor([[1, 0, 0],
                        [0, math.cos(x), -math.sin(x)],
                        [0, math.sin(x), math.cos(x)]], device=x_.device)

    R_y = torch.tensor([[math.cos(y), 0, math.sin(y)],
                        [0, 1, 0],
                        [-math.sin(y), 0, math.cos(y)]], device=y_.device)

    R_z = torch.tensor([[math.cos(z), -math.sin(z), 0],
                        [math.sin(z), math.cos(z), 0],
                        [0, 0, 1]], device=z_.device)
    return torch.mm(R_z, torch.mm(R_y, R_x))


def aug_bb_with_flag(PC, gt_R, gt_t, gt_s, model_point, mean_shape, sym, aug_bb, flag):
    PC_new, gt_s_new, model_point_new = defor_3D_bb_in_batch(PC, model_point, gt_R, gt_t, gt_s + mean_shape,
                                                             sym, aug_bb)
    gt_s_new = gt_s_new - mean_shape  # shape [32, 3]
    PC = torch.where(flag.unsqueeze(-1), PC_new, PC)
    gt_s = torch.where(flag, gt_s_new, gt_s)
    model_point_new = torch.where(flag.unsqueeze(-1), model_point_new, model_point)
    return PC, gt_s, model_point_new


def aug_rt_with_flag(PC, gt_R, gt_t, aug_rt_t, aug_rt_r, flag):
    PC_new, gt_R_new, gt_t_new = defor_3D_rt_in_batch(PC, gt_R, gt_t, aug_rt_t, aug_rt_r)
    PC_new = torch.where(flag.unsqueeze(-1), PC_new, PC)
    gt_R_new = torch.where(flag.unsqueeze(-1), gt_R_new, gt_R)
    gt_t_new = torch.where(flag, gt_t_new, gt_t)
    return PC_new, gt_R_new, gt_t_new


def aug_3D_bc_with_flag(PC, gt_R, gt_t, gt_s, model_point, nocs_scale, mean_shape, flag):
    pc_new, s_new, ey_up, ey_down = defor_3D_bc_in_batch(PC, gt_R, gt_t, gt_s + mean_shape, model_point,
                                                         nocs_scale)
    pc_new = torch.where(flag.unsqueeze(-1), pc_new, PC)
    s_new = torch.where(flag, s_new - mean_shape, gt_s)
    return pc_new, s_new, ey_up, ey_down


def aug_pc_with_flag(PC, gt_t, flag, aug_pc_r):
    PC_new, defor = defor_3D_pc(PC, gt_t, aug_pc_r, return_defor=True)
    PC_new = torch.where(flag.unsqueeze(-1), PC_new, PC)
    return PC_new, defor
