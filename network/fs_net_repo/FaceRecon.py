# Modified from FS-Net
import torch.nn as nn
import network.fs_net_repo.gcn3d as gcn3d
import torch
import torch.nn.functional as F
from absl import app
import absl.flags as flags

FLAGS = flags.FLAGS


class Face_Enc(nn.Module):
    def __init__(self):
        super(Face_Enc, self).__init__()
        self.neighbor_num = FLAGS.gcn_n_num
        self.support_num = FLAGS.gcn_sup_num
        self.output_channels = FLAGS.output_channels

        # 3D convolution for point cloud
        self.conv_0 = gcn3d.HSlayer_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = gcn3d.HS_layer(128, 128, support_num=self.support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d.HS_layer(128, 256, support_num=self.support_num)
        self.conv_3 = gcn3d.HS_layer(256, 256, support_num=self.support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.HS_layer(256, 512, support_num=self.support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.proj_layer = nn.Sequential(nn.Conv1d(1286, 1286, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(1286),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Conv1d(1286, 1286, kernel_size=1, bias=False))



    def forward(self, vertices, cat_id, enable_proj=False):
        """
        :param vertices: [B, N, 3]
        :param cat_id: [B,1]
        :param enable_proj: bool, default False
        """

        #  concate feature
        bs, vertice_num, _ = vertices.size()
        # cat_id to one-hot
        if cat_id.shape[0] == 1:
            obj_idh = cat_id.view(-1, 1).repeat(cat_id.shape[0], 1)
        else:
            obj_idh = cat_id.view(-1, 1)

        one_hot = torch.zeros(bs, FLAGS.obj_c).to(cat_id.device).scatter_(1, obj_idh.long(), 1)


        fm_0 = F.relu(self.conv_0(vertices, self.neighbor_num), inplace=True)
        fm_1 = F.relu(self.bn1(self.conv_1(vertices, fm_0, self.neighbor_num).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        fm_2 = F.relu(self.bn2(self.conv_2(v_pool_1, fm_pool_1, min(self.neighbor_num, v_pool_1.shape[1] // 8))
                               .transpose(1, 2)).transpose(1, 2), inplace=True)

        fm_3 = F.relu(self.bn3(self.conv_3(v_pool_1, fm_2, min(self.neighbor_num, v_pool_1.shape[1] // 8))
                               .transpose(1, 2)).transpose(1, 2), inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        fm_4 = self.conv_4(v_pool_2, fm_pool_2, min(self.neighbor_num, v_pool_2.shape[1] // 8))

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor_new(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor_new(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor_new(fm_4, nearest_pool_2).squeeze(2)

        one_hot = one_hot.unsqueeze(1).repeat(1, vertice_num, 1)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, one_hot], dim=2)

        feat_global = feat.permute(0, 2, 1)

        if enable_proj:
            feat_global_prj = self.proj_layer(feat_global)
        else:
            feat_global_prj = feat_global

        return feat, feat_global_prj


class Face_Dec(nn.Module):
    def __init__(self, dim_fuse):
        super(Face_Dec, self).__init__()
        self.recon_num = 3
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.recon_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.recon_num, 1),
        )

    def forward(self, x):
        # print('x.shape', x.shape)
        conv1d_out = self.conv1d_block(x)
        recon = self.recon_head(conv1d_out)

        return recon.permute(0, 2, 1)


class PH_Predictor(nn.Module):
    def __init__(self):
        super(PH_Predictor, self).__init__()
        self.output_channels = FLAGS.output_channels
        self.conv_5 = nn.Sequential(nn.Conv1d(128 + 128 + 256 + 256 + 512 + 6, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 1024, bias=False)


        self.bn5 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024, self.output_channels)
        self.linear3 = nn.Linear(1024, self.output_channels)
        self.linear4 = nn.Linear(self.output_channels, 1286)
        self.linear5 = nn.Linear(self.output_channels, 1286)
        self.ac2 = nn.Sigmoid()
        self.ac3 = nn.Sigmoid()

    def forward(self, feat):
        bs = feat.shape[0]
        vertice_num = feat.shape[1]

        feat_PH = feat.permute(0, 2, 1)
        feat_PH = self.conv_5(feat_PH)
        feat_PH1 = F.adaptive_max_pool1d(feat_PH, 1).view(bs, -1)
        feat_PH2 = F.adaptive_max_pool1d(feat_PH, 1).view(bs, -1)
        feat_All = torch.cat((feat_PH1, feat_PH2), 1)

        feat_All = F.leaky_relu(self.bn5(self.linear1(feat_All)), negative_slope=0.2)
        feat_All = self.dp1(feat_All)

        pi1 = self.linear2(feat_All)
        pi1_1 = self.linear4(pi1)
        pi1_1 = pi1_1.unsqueeze_(dim=-1)
        pi1_1 = pi1_1.repeat(1, 1, vertice_num)
        h1 = self.ac2(pi1)

        pi2 = self.linear3(feat_All)
        pi2_1 = self.linear5(pi2)
        pi2_1 = pi2_1.unsqueeze_(dim=-1)
        pi2_1 = pi2_1.repeat(1, 1, vertice_num)
        h2 = self.ac3(pi2)

        conv1d_input = feat.permute(0, 2, 1)
        feat_pi_with_recon = conv1d_input + pi1_1 + pi2_1

        return feat_pi_with_recon, h1, h2


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.encoder = Face_Enc()
        im_fuse = sum([1286])
        self.decoder = Face_Dec(im_fuse)
        self.ph_pred = PH_Predictor()

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                cat_id: "tensor (bs, 1)",
                enable_proj=False,
                pred_PH=True
                ):
        """
        :param vertices: (bs, vertice_num, 3)
        :param cat_id: (bs, 1)
        :param enable_proj: bool
        Return: (bs, vertice_num, class_num)
        """

        feat, feat_global = self.encoder(vertices, cat_id, enable_proj)
        feat_dec = feat.permute(0, 2, 1)
        if pred_PH:
            feat_ph, h1, h2 = self.ph_pred(feat)
            recon = self.decoder(feat_ph)
        else:
            recon = self.decoder(feat_dec)
            h1, h2 = None, None

        return recon, feat, feat_global, h1, h2




def main(argv):

    points = torch.rand(2, 1000, 3)
    import numpy as np
    obj_idh = torch.ones((2, 1))
    obj_idh[1, 0] = 5



if __name__ == "__main__":
    print(1)
    from config.config import *

    app.run(main)
