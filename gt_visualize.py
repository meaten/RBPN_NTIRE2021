import argparse
import os
from train import train
import numpy as np
import datetime
import cv2
from tqdm import tqdm

import torch

from model.config import cfg
from model.modeling.build_model import ModelWithLoss
from model.utils.misc import fix_model_state_dict
from model.provided_toolkit.datasets.synthetic_burst_train_set import SyntheticBurst
from model.provided_toolkit.datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB


def gt_visualize(cfg):
    print('Loading Datasets...')
    train_dataset = SyntheticBurst(ZurichRAW2RGB(cfg.DATASET.TRAIN_SYNTHETIC), crop_sz=384, burst_size=1)
    
    submit_dir = os.path.join("output", "images", "gt", "submit")
    vis_dir = os.path.join("output", "images", "gt", "visualize")
    os.makedirs(submit_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    num_im = len(train_dataset)
    for i, (_, gt_frame, _, _) in tqdm(enumerate(train_dataset), total=num_im):
        gt_frame_submit = (gt_frame.permute(1, 2, 0) * (2 ** 14)).cpu().numpy().astype(np.uint16)
        gt_frame_visualize = (gt_frame.permute(1, 2, 0) * (2 ** 8)).cpu().numpy().astype(np.uint8)
        name = '{}.png'.format(str(i).zfill(len(str(num_im))))
        cv2.imwrite(os.path.join(submit_dir, name), gt_frame_submit)
        cv2.imwrite(os.path.join(vis_dir, name), gt_frame_visualize)
        
    
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


def main():
    gt_visualize(cfg)

if __name__ == '__main__':
    main()