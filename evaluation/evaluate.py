import os
import random

import torch
from absl import app

from config.config import *

FLAGS = flags.FLAGS
# from datasets.load_data import PoseDataset
from evaluation.load_data_eval import PoseDataset
import time
import numpy as np

# from creating log
import tensorflow as tf
from tools.eval_utils import setup_logger
from core.utils.pc_vis import vis_pointcloud
from evaluater.RT_TDA_Evaluater import myEvaluater, calc_pose_metric

# torch.autograd.set_detect_anomaly(False)
# device = 'cuda'


def seed_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def evaluate(argv):
    if FLAGS.eval_seed == -1:
        seed = int(time.time())
    else:
        seed = FLAGS.eval_seed
    seed_init_fn(seed)
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)

    logger = setup_logger('eval_log', os.path.join(FLAGS.model_save, 'log_eval.txt'))

    FLAGS.train = False

    model_name = os.path.basename(FLAGS.resume_model).split('.')[0]

    val_dataset = PoseDataset(source=FLAGS.dataset, mode='test')
    output_path = os.path.join(FLAGS.model_save, f'eval_result_{model_name}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    import pickle

    pred_result_save_path = os.path.join(output_path, 'pred_result.pkl')
    print(pred_result_save_path)
    logger.info(f'pred_result_save_path: {pred_result_save_path}')

    if os.path.exists(pred_result_save_path):
        with open(pred_result_save_path, 'rb') as file:
            pred_results = pickle.load(file)
    else:
        # create evaluator
        eval = myEvaluater(logger=logger)
        eval.setup()
        eval.load_resume_model(FLAGS.resume_model)
        pred_results = eval.run(val_dataset)
        with open(pred_result_save_path, 'wb') as file:
            pickle.dump(pred_results, file)

    calc_pose_metric(logger, pred_results, output_path)


if __name__ == "__main__":
    app.run(evaluate)
