import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import argparse
import collections
from parse_config import ConfigParser

# checkpoint
import os.path as osp
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import dataset.base as module_dataloader
import dataset as module_dataset
import models as module_arch
from trainer.predictor import BasePredictor


def main(config, args):
    val_dataset = config.init_obj('val_dataset', module_dataset)
    val_dataloader = config.init_obj('val_data_loader', module_dataloader, dataset=val_dataset)

    model = config.init_obj('arch', module_arch)
    device = args.device

    output_name = str(args.eval_output)
    trainer = BasePredictor(model,
                            device=device,
                            data_loader=val_dataloader,
                            outputs=output_name)
    trainer._resume_checkpoint(args.resume)
    r = trainer.test()
    print(r)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument(
        '-c', '--config', default="config/val.json", type=str,
        help='config file path (default: None)'
        )
    args.add_argument(
        '-r', '--resume', default=None, type=str,
        help='path to latest checkpoint (default: None)'
        )
    args.add_argument(
        '-d', '--device', default=None, type=str,
        help='indices of GPUs to enable (default: all)'
        )
    args.add_argument(
        '--eval_only', action='store_true',
        help='whether to run in eval only mode'
        )
    args.add_argument(
        '-o', '--eval_output', default=None, type=str,
        help='Output files of evaluation'
        )
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = ""
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    main(config, args)
