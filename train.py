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

import datetime
from logger import TensorboardWriter
from utils import MetricTracker
from fvcore.common.timer import Timer

import models as module_arch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import DistributedSampler
from utils import mot_collate_fn

from trainer.trainer import BaseTrainer
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_for_distributed(rank, opts):

    # 1. setting for distributed training
    opts.rank = rank
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    dist.init_process_group(backend='nccl',
                            init_method='tcp://localhost:2222',
                            world_size=opts.world_size,
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print(opts)
    return local_gpu_id

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', default="config/default.json", type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3', '4', '5', '6', '7']),
    parser.add_argument('--local_rank', type=int)
    return parser


def main(opts, config):
    rank = opts.local_rank
    dist.init_process_group(backend='nccl',
                            rank=rank,
                            world_size=int(os.environ['WORLD_SIZE']),
                            init_method='env://')
    # global_rank = dist.get_rank()
    # rank, world_size = get_dist_info()

    local_gpu_id = int(opts.gpu_ids[opts.local_rank])

    train_dataset = NusceneseDataset(config['train_dataset']['args']['ann_file'])
    train_sampler = DistributedSampler(dataset=train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, batch_size=1, drop_last=True)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  collate_fn=mot_collate_fn,
                                  num_workers=0,
                                  pin_memory=True)

    model = config.init_obj('arch', module_arch)
    model = model.cuda(local_gpu_id)
    model = DDP(model,
                device_ids=[local_gpu_id],)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']
    writer = TensorboardWriter(config.log_dir, logger, cfg_trainer['tensorboard'])

    train_loss_metrics = []
    for i in range(0, 7):
        train_loss_metrics.append(f'frame_{i + 1}_loss_affinity')
        train_loss_metrics.append(f'frame_{i + 1}_loss_attention')
        for j in range(0, 2):
            train_loss_metrics.append(f'frame_{i + 1}_d{j}.loss_affinity')
            train_loss_metrics.append(f'frame_{i + 1}_d{j}.loss_attention')
    train_metrics = MetricTracker('loss/total', *train_loss_metrics, writer=writer)

    global_step_time = Timer()

    log_step = config['trainer']['log_step']
    checkpoint_dir = config.save_dir
    device = local_gpu_id

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
