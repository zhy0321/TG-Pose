import torch
import absl.flags as flags
import time
import math
import random

FLAGS = flags.FLAGS

from network.fs_net_repo.PoseNet9D import PoseNet9D
from losses.TDA_loss_sym_recon import TDA_loss
from engine.organize_loss import control_loss
from losses.consistency_loss import feat_consistency_loss, prop_sym_matching_loss
from tools.training_utils import get_gt_v
from tools.training_utils import build_lr_rate, build_optimizer


def create_network(mode):
    if mode == 'RL_TDA':
        net1 = PoseNet9D()
        net2 = PoseNet9D(only_encoder=True)
        return net1, net2


class RT_TDA_Trainer(object):
    def __init__(self, logger):
        self.logger = logger
        self.device = torch.device('cuda:0')
        self.net1, self.net2 = None, None
        self.loss_tda_net = None
        self.optimizer = None
        self.scheduler = None

    def setup(self, mode):
        self.init_network(mode)
        self.init_loss()
        self.set_optimizer_scheduler()

    def print_model_info(self):
        self.logger.info('The parameters of net1:')
        for name, param in self.net1.named_parameters():
            print(name, param.shape)
        self.logger.info('The parameters of net2:')
        for name, param in self.net2.named_parameters():
            print(name, param.shape)

    def init_network(self, mode):
        self.net1, self.net2 = create_network(mode)
        if self.net1 is not None:
            self.net1 = self.net1.to(self.device)
        if self.net2 is not None:
            self.net2 = self.net2.to(self.device)

    def init_loss(self):
        self.loss_tda_net = TDA_loss()
        self.name_fs_list, self.name_recon_list, \
        self.name_geo_list, self.name_prop_list, self.name_TDA_list = control_loss('TDA')

    def set_optimizer_scheduler(self):
        params_list = self.build_params()
        self.optimizer = build_optimizer(params_list)
        self.scheduler = build_lr_rate(self.optimizer,
                                       total_iters=FLAGS.train_steps * FLAGS.total_epoch // FLAGS.accumulate)

    def init_RL_TDA_model(self, model_path):
        self.logger.info('[RT_TDA] loading model from {} '.format(model_path))
        checkpoint = torch.load(model_path)
        net1_dict = self.net1.state_dict()
        print('[RL] net1 state dict')
        for k, v in checkpoint['net1_state_dict'].items():
            print(k, v.shape)

        print('[TDA] net1 state dict')
        for k, v in net1_dict.items():
            print(k, v.shape)

        if self.net1 is not None:
            net1_updated_dict = {}
            for k, v in checkpoint['net1_state_dict'].items():
                if 'face_enc' in k and 'ph_pred' not in k:
                    new_k = k.replace('face_enc', 'face_all')
                    if new_k in net1_dict:
                        net1_updated_dict[new_k] = v
                        print('update {} to {}'.format(k, new_k))

            net1_dict.update(net1_updated_dict)
            self.net1.load_state_dict(net1_dict)

    def load_old_model_params(self, path, mode):
        checkpoint = torch.load(path)
        if mode == 'RL_TDA':
            if self.net1 is not None:
                self.net1.load_state_dict(checkpoint['net1_state_dict'])
            if self.net2 is not None:
                self.net2.load_state_dict(checkpoint['net2_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def build_params(self, training_stage_freeze=None):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, self.net1.parameters()),
                "lr": float(FLAGS.lr) * FLAGS.lr_pose,
            }
        )
        return params_lr_list

    def RL_TDA_train_step(self, db, only_TDA=False, gt_pred_flag=False):
        PC = db['pcl_in'].to(self.device)
        obj_id = db['cat_id'].to(self.device)

        if not only_TDA:
            Aug_PC = db['aug_pcl_in'].to(self.device)
            results = self.net1(PC, obj_id)
            with torch.no_grad():
                results_2 = self.net2(Aug_PC, obj_id)
        else:
            results = self.net1(PC, obj_id)

        # loss
        loss_dict = {}
        if not only_TDA:
            # RL loss
            enc_feat_1 = results['feat_global']
            enc_feat_2 = results_2['feat_global']
            gt_R = db['rotation'].to(self.device)
            gt_t = db['translation'].to(self.device)
            sym = db['sym_info'].to(self.device)
            recon_1 = results['recon']
            recon_2 = results_2['recon']
            RL_loss = feat_consistency_loss(enc_feat_1, enc_feat_2)
            recon_1_loss = prop_sym_matching_loss(PC, recon_1, gt_R, gt_t, sym)
            recon_consistency_loss = prop_sym_matching_loss(recon_1, recon_2, gt_R, gt_t, sym)
            loss_dict['RL_loss'] = RL_loss
            loss_dict['recon_1_loss'] = recon_1_loss
            loss_dict['recon_consistency_loss'] = 0.2 * recon_consistency_loss
        else:
            loss_dict['RL_loss'] = torch.FloatTensor([0.0]).to(self.device)

        # TDA loss
        pred_TDA_list = {
            'Rot1': results['p_green_R'],
            'Rot1_f': results['f_green_R'],
            'Rot2': results['p_red_R'],
            'Rot2_f': results['f_red_R'],
            'Recon': results['recon'],
            'Tran': results['Pred_T'],
            'Size': results['Pred_s'],
            'TDA_h1': results['h1'],
            'TDA_h2': results['h2'],
        }

        gt_R = db['rotation'].to(self.device)
        gt_t = db['translation'].to(self.device)
        gt_s = db['fsnet_scale'].to(self.device)
        gt_h1 = db['pdh1'].to(self.device)
        gt_h2 = db['pdh2'].to(self.device)

        gt_green_v, gt_red_v = get_gt_v(gt_R)

        gt_TDA_list = {
            'Rot1': gt_green_v,
            'Rot2': gt_red_v,
            'Recon': PC,
            'Tran': gt_t,
            'Size': gt_s,
            'h1': gt_h1,
            'h2': gt_h2,
            'proto': None,
            'pdh1_category': db['pdh1_category'].to(self.device),
            'pdh2_category': db['pdh2_category'].to(self.device),
            'points_category': db['points_category'].to(self.device),
            'R': gt_R,
        }
        sym = db['sym_info'].to(self.device)
        loss_dict['TDA_loss'] = self.loss_tda_net(self.name_TDA_list, pred_TDA_list,
                                                  gt_TDA_list, sym, gt_pred_flag)

        output_dict = {}

        if not only_TDA:
            output_dict['enc_feat_2'] = enc_feat_2
        output_dict['enc_feat_1'] = results['feat_global']

        output_dict['PC'] = PC
        output_dict['obj_id'] = obj_id
        output_dict['gt_R'] = gt_R
        output_dict['gt_t'] = gt_t
        output_dict['gt_s'] = gt_s
        output_dict['gt_h1'] = gt_h1
        output_dict['gt_h2'] = gt_h2
        output_dict['sem_pro'] = None

        pred_keys = ['recon', 'p_green_R', 'p_red_R', 'f_green_R', 'f_red_R', 'Pred_T', 'Pred_s', 'h1', 'h2']
        for key in pred_keys:
            output_dict[key] = results[key]

        return output_dict, loss_dict

    def RL_TDA_train(self, train_dataloader, total_epoch):
        for e in range(total_epoch):
            epoch_s_time = time.time()
            for i, data in enumerate(train_dataloader, 1):
                iter_s_time = time.time()

                output_dict, loss_dict = self.RL_TDA_train_step(data)
                Con_loss = loss_dict['RL_loss']
                recon_1_loss = loss_dict['recon_1_loss']
                recon_consistency_loss = loss_dict['recon_consistency_loss']
                TDA_loss = loss_dict['TDA_loss']
                TDA_loss_sum = sum(TDA_loss.values())
                total_loss = 0.1 * Con_loss + 0.1 * recon_1_loss + 0.1 * recon_consistency_loss + 0.9 * TDA_loss_sum


                if math.isnan(total_loss):
                    print('Found nan in total loss')
                    i += 1
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net1.parameters(), 5)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if i % FLAGS.log_every == 0:
                    self.logger.info(
                        'Stage {} Epoch {} Batch {} L:{:.4f}, con_l:{:.4f},recon_1:{:.4f},recon_consist:{:.4f}, TDA_l:{:.4f}, rot_l:{:.4f}, size_l:{:.4f}, '
                        'trans_l:{:.4f}, h1_l:{:.4f}, h2_l:{:.4f}, h1_l_cate:{:.4f}, h2_l_cate:{:.4f},'
                        'R_DCD_cate_pred:{:.4f},Prop_sym:{:.4f}'.format(2, e, i,
                                                                        total_loss.item(),
                                                                        Con_loss.item(),
                                                                        recon_1_loss.item(),
                                                                        recon_consistency_loss.item(),
                                                                        TDA_loss_sum.item(),
                                                                        (TDA_loss['Rot1'] + TDA_loss[
                                                                            'Rot2']).item(),
                                                                        TDA_loss['Size'].item(),
                                                                        TDA_loss['Tran'].item(),
                                                                        TDA_loss['TDA_h1'].item(),
                                                                        TDA_loss['TDA_h2'].item(),
                                                                        TDA_loss['TDA_h1_cate'].item(),
                                                                        TDA_loss['TDA_h2_cate'].item(),
                                                                        TDA_loss['R_DCD_cate_pred'].item(),
                                                                        TDA_loss['Prop_sym'].item()
                                                                        ))
                    self.logger.info('The average running time of every {} is {:.4f} sec'.format(FLAGS.log_every,
                                                                                                 (
                                                                                                         time.time() - iter_s_time)
                                                                                                 ))
            self.logger.info('>>>>>>>>----------Epoch {:02d} train finish,'
                             'time is {:02f} sec---------<<<<<<<<'.format(e, time.time() - epoch_s_time))

            # save model
            if (e + 1) % FLAGS.save_every == 0 or (e + 1) == total_epoch:
                torch.save({'epoch': e,
                            'net1_state_dict': self.net1.state_dict(),
                            'net2_state_dict': self.net2.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict()},
                           '{0}/rl_tda_model_{1:02d}.pth'.format(FLAGS.model_save, e))

    def train(self, train_dataloader, mode):
        if mode == 'RL_TDA':
            self.RL_TDA_train(train_dataloader, 150)
