import random
import numpy as np
import torch
from dataset.nuscenes_dataset import NusceneseDataset
import os
import copy

import argparse
import collections
from parse_config import ConfigParser
import torch.distributed as dist
from models.structures import Instances
from models.utils import cal_similarity
import lap

import datetime
from logger import TensorboardWriter
from utils import MetricTracker
from fvcore.common.timer import Timer

import models as module_arch
from torch.utils.data import DataLoader, RandomSampler
from utils import mot_collate_fn

from trainer.single_trainer import SingleTrainer as BaseTrainer


# For single gpu train

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', default="config/default.json", type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'])
    parser.add_argument('--local_rank', type=int)
    return parser


def main(opts, config):
    local_gpu_id = 1
    train_dataset = NusceneseDataset(config['train_dataset']['args']['ann_file'])
    train_sampler = RandomSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, batch_size=1, drop_last=True)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  collate_fn=mot_collate_fn,
                                  num_workers=0,
                                  pin_memory=True)
    model = config.init_obj('arch', module_arch)
    
    model = model.cuda(local_gpu_id)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']
    writer = TensorboardWriter(config.log_dir, logger, cfg_trainer['tensorboard'])
    train_loss_metrics = []
    for i in range(0, 4):
        train_loss_metrics.append(f'frame_{i + 1}_loss_affinity')
        train_loss_metrics.append(f'frame_{i + 1}_loss_attention')
        train_loss_metrics.append(f'frame_{i + 1}_loss_dn_affinity')
        train_loss_metrics.append(f'frame_{i + 1}_loss_dn_attention')
        for j in range(0, 2):
            train_loss_metrics.append(f'frame_{i + 1}_d{j}.loss_affinity')
            train_loss_metrics.append(f'frame_{i + 1}_d{j}.loss_attention')
            train_loss_metrics.append(f'frame_{i + 1}_d{j}.loss_dn_affinity')
            train_loss_metrics.append(f'frame_{i + 1}_d{j}.loss_dn_attention')
    train_metrics = MetricTracker('loss/total', *train_loss_metrics, writer=writer)

    global_step_time = Timer()

    log_step = config['trainer']['log_step']
    checkpoint_dir = config.save_dir

    device = "cuda:{}".format(local_gpu_id)

    trainer = BaseTrainer(
        model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        sampler=train_sampler,
        metrics=train_metrics,
        timer=global_step_time,
        checkpoint_dir=checkpoint_dir,
        opts=opts,
        log_step=log_step,
        writer=writer,
        logger=logger,
        device=device)

    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    config = ConfigParser.from_args(parser, '')

    checkpoint_dir = config.save_dir
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    main(opts, config)
