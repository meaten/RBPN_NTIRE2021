import argparse
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler

from model.config import cfg


def test(args, cfg):
    device = torch.device('cuda')
    model = 
    print('------------Model Architecture-------------')
    print(model)

    print('Loading Datasets...')
    test_transforms = 
    test_dataset = 
    sampler = 
    batch_sampler = BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)

    model.load_state_dict(torch.load(args.trained_model))

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    do_test(args, cfg, model, test_loader, device)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='')
    parser.add_argument('--output_dirname', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--trained_model', type=str, default='')

    args = parser.parse_args()

    if len(args.config_file) > 0:
        print('Configuration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    if len(args.output_dirname) == 0:
        dt_now = datetime.datetime.now()
        output_dirname = os.path.join('output', str(dt_now.date()) + '_' + str(dt_now.time()))
    else:
        output_dirname = args.output_dirname
    cfg.OUTPUT_DIRNAME = output_dirname
    cfg.freeze()

    print('OUTPUT DIRNAME: {}'.format(cfg.OUTPUT_DIR))

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(cfg.SEED)
    else:
        raise Exception('GPU not found')

    test(args, cfg)

if __name__ == '__main__':
    main()