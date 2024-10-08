import torch
import time
import math
import absl.flags as flags
import numpy as np
from network.fs_net_repo.PoseNet9D import PoseNet9D
from tools.geom_utils import generate_RT
from evaluation.eval_utils_v1 import compute_degree_cm_mAP
import pickle

import absl.flags as flags
from tqdm import tqdm

FLAGS = flags.FLAGS


def creat_network(mode=None):
    net1 = PoseNet9D()
    return net1


class myEvaluater(object):
    def __init__(self, logger):
        self.logger = logger
        # self.device = torch.device(FLAGS.device)
        self.device = FLAGS.device
        self.net1 = None

    def setup(self):
        self.init_network()

    def init_network(self):
        self.net1 = creat_network()
        if self.net1 is not None:
            self.net1 = self.net1.to(self.device)

    def load_resume_model(self, model_path):
        if self.net1 is not None:
            self.net1.load_state_dict(torch.load(model_path)['net1_state_dict'])
            # for name, param in self.net1.named_parameters():
                # print('name: ', name)

            self.logger.info('load model from {}'.format(model_path))
        else:
            self.logger.info('no model to load')

    def __inference(self, db):
        self.net1.eval()
        PC = db['pcl_in'].to(self.device)
        Obj_id = db['cat_id_0base'].to(self.device)
        with torch.no_grad():
            output_dict = self.net1(PC, Obj_id)
        return output_dict

    def run(self, dataset):
        self.logger.info('>>>>>>>>----------Start TDA evaluation---------<<<<<<<<')
        self.net1.eval()
        t_inference = 0.0
        pred_results = []

        # ++++++++++++++++++++++++++++++++++++
        PC_list = []
        # Recon_List = []
        img_count = 0
        # ++++++++++++++++++++++++++++++++++++
        for i, data in tqdm(enumerate(dataset, 1), dynamic_ncols=True):
            if data is None:
                continue
            data, detection_dict, gts = data
            mean_shape = data['mean_shape'].to(self.device)
            sym = data['sym_info'].to(self.device)
            if len(data['cat_id_0base']) == 0:
                detection_dict['pred_RTs'] = np.zeros((0, 4, 4))
                detection_dict['pred_scales'] = np.zeros((0, 4, 4))
                pred_results.append(detection_dict)
                continue
            t_start = time.time()
            output_dict = self.__inference(data)
            output_dict['PC'] = data['pcl_in']

            p_green_R_vec = output_dict['p_green_R'].detach()
            p_red_R_vec = output_dict['p_red_R'].detach()
            p_T = output_dict['Pred_T'].detach()
            p_s = output_dict['Pred_s'].detach()
            f_green_R = output_dict['f_green_R'].detach()
            f_red_R = output_dict['f_red_R'].detach()

            PC = output_dict['PC'].detach().cpu().numpy()  # [20,1024,3]
            PC_list.append(PC)
            # recon = output_dict['recon'].detach().cpu().numpy()
            # Recon_List.append(recon)

            pred_s = p_s + mean_shape
            pred_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym)

            if pred_RT is not None:
                pred_RT = pred_RT.detach().cpu().numpy()
                pred_s = pred_s.detach().cpu().numpy()
                detection_dict['pred_RTs'] = pred_RT
                detection_dict['pred_scales'] = pred_s
            else:
                assert NotImplementedError

            t_inference += time.time() - t_start
            img_count += 1
            pred_results.append(detection_dict)
        print('inference time:{},\t img_count:{}'.format(t_inference / img_count, img_count))

        return pred_results


def calc_pose_metric(logger, pred_results, output_path):
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]

    # iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, output_path, degree_thres_list, shift_thres_list,
    #                                                  iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)
    synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    if FLAGS.per_obj in synset_names:
        idx = synset_names.index(FLAGS.per_obj)
    else:
        idx = -1
    iou_aps, pose_aps = compute_degree_cm_mAP(pred_results, synset_names, output_path, degree_thres_list,
                                              shift_thres_list,
                                              iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True, )

    # # fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)

    messages = []

    if FLAGS.per_obj in synset_names:
        # messages.append('Evaluation Seed: {}'.format(seed))
        messages.append('mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
        messages.append('5 degree: {:.1f}'.format(pose_aps[idx, degree_05_idx, -1] * 100))
        messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
        messages.append('2cm: {:.1f}'.format(pose_aps[idx, -1, shift_02_idx] * 100))
        messages.append('5cm: {:.1f}'.format(pose_aps[idx, -1, shift_05_idx] * 100))
        messages.append('10cm: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))
        # messages.append("Inference time: {:06f}  Average: {:06f}/image".format(t_inference, t_inference / img_count))
    else:
        # messages.append('Evaluation Seed: {}'.format(seed))
        messages.append('average mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
        messages.append('5 degree: {:.1f}'.format(pose_aps[idx, degree_05_idx, -1] * 100))
        messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
        messages.append('2cm: {:.1f}'.format(pose_aps[idx, -1, shift_02_idx] * 100))
        messages.append('5cm: {:.1f}'.format(pose_aps[idx, -1, shift_05_idx] * 100))
        messages.append('10cm: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))
        # messages.append("Inference time: {:06f}  Average: {:06f}/image".format(t_inference, t_inference / img_count))

        for idx in range(1, len(synset_names)):
            messages.append('category {}'.format(synset_names[idx]))
            messages.append('mAP:')
            messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
            messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
            messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
            messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
            messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
            messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
            messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
            messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
            messages.append('5 degree: {:.1f}'.format(pose_aps[idx, degree_05_idx, -1] * 100))
            messages.append('10 degree: {:.1f}'.format(pose_aps[idx, degree_10_idx, -1] * 100))
            messages.append('2cm: {:.1f}'.format(pose_aps[idx, -1, shift_02_idx] * 100))
            messages.append('5cm: {:.1f}'.format(pose_aps[idx, -1, shift_05_idx] * 100))
            messages.append('10cm: {:.1f}'.format(pose_aps[idx, -1, shift_10_idx] * 100))

    for msg in messages:
        logger.info(msg)
