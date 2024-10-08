import os
import random

import torch
from absl import app

from config.config import *

FLAGS = flags.FLAGS
from datasets.load_data import PoseDataset
import time
import numpy as np

import tensorflow as tf
from tools.eval_utils import setup_logger
from trainer.RL_TDA import RT_TDA_Trainer

# torch.autograd.set_detect_anomaly(False)
device = 'cuda'


def train(argv):
    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume_model)
        if 'seed' in checkpoint:
            seed = checkpoint['seed']
        else:
            seed = int(time.time()) if FLAGS.seed == -1 else FLAGS.seed
    else:
        seed = 3135
    seed_init_fn(seed)

    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    tf.compat.v1.disable_eager_execution()
    logger = setup_logger('train_log', os.path.join(FLAGS.model_save, 'log.txt'))


    run_stage = FLAGS.run_stage
    logger.info('>>>>>>>>----------RUN_STAGE:{}---------<<<<<<<<'.format(run_stage))
    # build dataset and dataloader
    train_dataset = PoseDataset(source=FLAGS.dataset, mode='train',
                                run_stage=run_stage, data_dir=FLAGS.dataset_dir, per_obj=FLAGS.per_obj)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                   num_workers=FLAGS.num_workers, pin_memory=True,
                                                   prefetch_factor=4,
                                                   worker_init_fn=seed_worker,
                                                   shuffle=True)

    # create trainer
    trainer = RT_TDA_Trainer(logger=logger)

    trainer.setup(mode=run_stage)
    if run_stage == 'RL_TDA' and FLAGS.RL_model_path != '':
        trainer.init_RL_TDA_model(FLAGS.RL_model_path)
    if FLAGS.resume and FLAGS.resume_model != '':
        trainer.load_old_model_params(FLAGS.resume_model, mode=run_stage)
    torch.cuda.empty_cache()
    trainer.train(train_dataloader, mode=run_stage)



def seed_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    app.run(train)
