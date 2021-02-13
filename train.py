import argparse
import os
import shutil
import numpy as np
import datetime

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from model.config import cfg
from model.engine.trainer import do_train
from model.modeling.build_model import ModelWithLoss
from model.data.samplers import IterationBasedBatchSampler
from model.utils.sync_batchnorm import convert_model
from model.utils.misc import str2bool, fix_model_state_dict
from model.provided_toolkit.datasets.synthetic_burst_train_set import SyntheticBurst
from model.provided_toolkit.datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB


def train(args, cfg):
    device = torch.device('cuda')
    model = ModelWithLoss(cfg).to(device)
    print('------------Model Architecture-------------')
    print(model)

    print('Loading Datasets...')
    data_loader = {}

    # train_transforms = 
    train_dataset = SyntheticBurst(ZurichRAW2RGB(cfg.DATASET.TRAIN), crop_sz=cfg.SOLVER.PATCH_SIZE)
    sampler = RandomSampler(train_dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.SOLVER.MAX_ITER)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)

    data_loader['train'] = train_loader

    # if args.eval_step != 0:
    #     val_transforms = 
    #     val_dataset = 
    #     sampler = SequentialSampler(val_dataset)
    #     batch_sampler = BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
    #     val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)

    #     data_loader['val'] = val_loader

    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.SOLVER.LR)

    if args.resume_iter != 0:
        print('Resume from {}'.format(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter))))
        model.load_state_dict(fix_model_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter)))))
        optimizer.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(args.resume_iter))))

    if cfg.SOLVER.SYNC_BATCHNORM:
        model = convert_model(model).to(device)
    
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    if not args.debug:
        summary_writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    else:
        summary_writer = None

    do_train(args, cfg, model, optimizer, data_loader, device, summary_writer)


def main():
    parser = argparse.ArgumentParser(description='pytorch training code')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--output_dirname', type=str, default='', help='')
    parser.add_argument('--log_step', type=int, default=50, help='')
    parser.add_argument('--eval_step', type=int, default=0, help='')
    parser.add_argument('--save_step', type=int, default=50000, help='')
    parser.add_argument('--num_gpus', type=int, default=1, help='')
    parser.add_argument('--num_workers', type=int, default=16, help='')
    parser.add_argument('--resume_iter', type=int, default=0, help='')

    args = parser.parse_args()

    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)
    
    if len(args.output_dirname) == 0:
        dt_now = datetime.datetime.now()
        output_dirname = os.path.join('output', str(dt_now.date()) + '_' + str(dt_now.time()))
    else:
        output_dirname = args.output_dirname
    cfg.OUTPUT_DIR = output_dirname
    cfg.freeze()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(cfg.SEED)
    else:
        raise Exception('GPU not found.')

    if not args.debug:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        if not len(args.config_file) == 0:
            shutil.copy2(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))

    train(args, cfg)


if __name__ == '__main__':
    main()
