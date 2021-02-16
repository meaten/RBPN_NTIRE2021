import os
import numpy as np
import cv2
from tqdm import tqdm

from model.config import cfg
from model.provided_toolkit.datasets.synthetic_burst_train_set import SyntheticBurst
from model.provided_toolkit.datasets.burstsr_dataset import BurstSRDataset
from model.provided_toolkit.datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB


def gt_visualize(cfg, track='synthetic'):
    print('Loading Datasets...')
    if track == 'synthetic':
        train_dataset = SyntheticBurst(ZurichRAW2RGB(cfg.DATASET.TRAIN_SYNTHETIC), crop_sz=384, burst_size=1)
    elif track == 'real':
        train_dataset = BurstSRDataset(cfg.DATASET.TRAIN_REAL, crop_sz=80, burst_size=1)
    
    submit_dir = os.path.join("output", "images", "gt", track, "submit")
    vis_dir = os.path.join("output", "images", "gt", track, "visualize")
    os.makedirs(submit_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    num_im = len(train_dataset)
    for i, (_, gt_frame, _, _) in tqdm(enumerate(train_dataset), total=num_im):
        gt_frame_submit = (gt_frame.permute(1, 2, 0) * (2 ** 14)).cpu().numpy().astype(np.uint16)
        gt_frame_visualize = (gt_frame.permute(1, 2, 0) * (2 ** 8)).cpu().numpy().astype(np.uint8)
        name = '{}.png'.format(str(i).zfill(len(str(num_im))))
        cv2.imwrite(os.path.join(submit_dir, name), gt_frame_submit)
        cv2.imwrite(os.path.join(vis_dir, name), gt_frame_visualize)


def main():
    gt_visualize(cfg)
    gt_visualize(cfg, track='real')

if __name__ == '__main__':
    main()