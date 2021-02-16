import argparse
import os
import numpy as np
import datetime
import cv2
from tqdm import tqdm

import torch

from model.config import cfg
from model.engine.trainer import do_train
from model.modeling.build_model import ModelWithLoss
from model.utils.misc import fix_model_state_dict
from model.provided_toolkit.datasets.synthetic_burst_val_set import SyntheticBurstVal


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='')
    parser.add_argument('--output_dirname', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--trained_model', type=str, default='')

    return parser.parse_args()    


def test(args, cfg):
    device = torch.device('cuda')
    model = ModelWithLoss(cfg).to(device)
    print('------------Model Architecture-------------')
    print(model)

    print('Loading Datasets...')
    # test_transforms = 
    test_dataset = SyntheticBurstVal(cfg.DATASET.TEST_SYNTHETIC)
    model.load_state_dict(fix_model_state_dict(torch.load(args.trained_model)))
    model.cuda()
    
    do_test(args, cfg, model, test_dataset, device)
    
    
def do_test(args, cfg, model, test_dataset, device):
    submit_dir = os.path.join(cfg.OUTPUT_DIRNAME, "submit")
    vis_dir = os.path.join(cfg.OUTPUT_DIRNAME, "visualize")
    os.makedirs(submit_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    for idx in tqdm(range(len(test_dataset))):
        burst, burst_name = test_dataset[idx]
        burst.to(device)
        shape = burst.shape
        
        burst_size = 14
        n_frames = cfg.MODEL.BURST_SIZE
        n_ensemble = burst_size - n_frames + 1
        ensemble_idx = np.array([np.arange(idx, idx+n_frames) for idx in range(n_ensemble)])
        ensemble_idx[:, 0] = 0
        ensemble_burst = torch.zeros([n_ensemble, n_frames, *shape[1:]]).to(device)
        for i, idx in enumerate(ensemble_idx):
            ensemble_burst[i] = burst[idx]
        ensemble_burst.to(device)
        
        with torch.no_grad():
            net_pred = model.model(ensemble_burst)
            net_pred = torch.mean(net_pred, axis=0)
        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_submit = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 14)).cpu().numpy().astype(np.uint16)
        net_pred_visualize = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 8)).cpu().numpy().astype(np.uint8)
        
        # Save predictions as png
        cv2.imwrite(os.path.join(submit_dir, '{}.png'.format(burst_name)), net_pred_submit)
        cv2.imwrite(os.path.join(vis_dir, '{}.png'.format(burst_name)), net_pred_visualize)
        
    import shutil
    shutil.make_archive(os.path.join(cfg.OUTPUT_DIRNAME, "submit"), 'zip', root_dir=submit_dir)


def main():
    args = parse_args()

    if len(args.config_file) > 0:
        print('Configuration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    if len(args.output_dirname) == 0:
        dt_now = datetime.datetime.now()
        output_dirname = os.path.join('output', "images", os.path.splitext(args.trained_model)[0])
    else:
        output_dirname = args.output_dirname
    os.makedirs(output_dirname, exist_ok=True)
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