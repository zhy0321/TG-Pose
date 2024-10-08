import torch
import torch.nn as nn
import absl.flags as flags
from absl import app
import numpy as np
import torch.nn.functional as F
import random

from network.fs_net_repo.PoseR import Rot_red, Rot_green
from network.fs_net_repo.PoseTs import Pose_Ts
from network.fs_net_repo.FaceRecon import FaceNet


device = 'cuda'

FLAGS = flags.FLAGS


class PoseNet9D(nn.Module):
    def __init__(self, only_encoder=False):
        super(PoseNet9D, self).__init__()
        self.only_encoder = only_encoder
        # Used the fsnet rot_green and rot_red directly

        if not only_encoder:
            self.face_all = FaceNet()  # encoder,decoder, phd_pred
            self.rot_green = Rot_green()
            self.rot_red = Rot_red()
            self.ts = Pose_Ts()
        else:
            self.face_enc = FaceNet()

    def forward(self, points, obj_id, enable_proj=False):
        bs, p_num = points.shape[0], points.shape[1]
        if self.only_encoder:
            # encoder and face reconstruction
            recon, _, feat_global_aug, _, _ = self.face_enc(points - points.mean(dim=1, keepdim=True),
                                                            obj_id,
                                                            enable_proj=enable_proj, pred_PH=False)
            feat_global_aug = feat_global_aug.max(2)[0]

            db_dict = dict()
            db_dict['feat_global'] = feat_global_aug
            db_dict['recon'] = recon
            return db_dict
        else:
            # encoder and face reconstruction
            recon, feat, feat_global, h1, h2, = self.face_all(points - points.mean(dim=1, keepdim=True),
                                                              obj_id, enable_proj=enable_proj, pred_PH=True)
            feat_global = feat_global.max(2)[0]

            #  rotation
            green_R_vec = self.rot_green(feat.permute(0, 2, 1))
            red_R_vec = self.rot_red(feat.permute(0, 2, 1))

            # normalization
            p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
            p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
            # sigmoid for confidence
            f_green_R = F.sigmoid(green_R_vec[:, 0])
            f_red_R = F.sigmoid(red_R_vec[:, 0])
            # translation and size
            feat_for_ts = torch.cat([feat, points - points.mean(dim=1, keepdim=True)], dim=2)
            T, s = self.ts(feat_for_ts.permute(0, 2, 1))
            Pred_T = T + points.mean(dim=1)
            Pred_s = s

            db_dict = dict()
            if FLAGS.train:
                # for training stage
                recon = recon + points.mean(dim=1, keepdim=True)
                db_dict['recon'] = recon
                db_dict['p_green_R'] = p_green_R
                db_dict['p_red_R'] = p_red_R
                db_dict['f_green_R'] = f_green_R
                db_dict['f_red_R'] = f_red_R
                db_dict['Pred_T'] = Pred_T
                db_dict['Pred_s'] = Pred_s
                db_dict['h1'] = h1
                db_dict['h2'] = h2
                db_dict['feat'] = feat
                db_dict['feat_global'] = feat_global
            else:
                # for testing stage
                db_dict['p_green_R'] = p_green_R
                db_dict['p_red_R'] = p_red_R
                db_dict['f_green_R'] = f_green_R
                db_dict['f_red_R'] = f_red_R
                db_dict['Pred_T'] = Pred_T
                db_dict['Pred_s'] = Pred_s
            return db_dict


def main(argv):
    classifier_seg3D = PoseNet9D()

    points = torch.rand(2, 1000, 3)
    import numpy as np
    obj_idh = torch.ones((2, 1))
    obj_idh[1, 0] = 5
    recon, f_n, f_d, f_f, r1, r2, c1, c2, t, s = classifier_seg3D(points, obj_idh)
    t = 1


if __name__ == "__main__":
    print(1)
    from config.config import *

    app.run(main)
