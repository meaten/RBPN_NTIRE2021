import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

import torch

from model.config import cfg
from model.modeling.build_model import ModelWithLoss
from model.utils.misc import fix_model_state_dict
from model.provided_toolkit.datasets.synthetic_burst_val_set import SyntheticBurstVal
from model.provided_toolkit.datasets.burstsr_dataset import BurstSRDataset

from model.provided_toolkit.utils.metrics import AlignedPSNR
from model.provided_toolkit.pwcnet.pwcnet import PWCNet


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
    model.load_state_dict(fix_model_state_dict(torch.load(args.trained_model)))
    model.cuda()

    print('Loading Datasets...')
    # test_transforms =
    if cfg.DATASET.TRACK == 'synthetic':
        test_dataset = SyntheticBurstVal(cfg.DATASET.TEST_SYNTHETIC)
        do_test_synthetic(args, cfg, model, test_dataset, device)
    elif cfg.DATASET.TRACK == 'real':
        test_dataset = BurstSRDataset(cfg.DATASET.REAL, split='val', crop_sz=80, burst_size=14, random_flip=False)
        do_test_real(args, cfg, model, test_dataset, device)
    
def do_test_synthetic(args, cfg, model, test_dataset, device):
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
        ensemble_burst = torch.zeros([n_ensemble, n_frames, *shape[1:]]).to(device)
        for i, idx in enumerate(get_ensemble_idx(burst_size=burst_size, num_frame_used=n_frames)):
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
    
    
def do_test_real(args, cfg, model, test_dataset, device):
    alignment_net = PWCNet(load_pretrained=True,
                           weights_path=cfg.PWCNET_WEIGHT)
    alignment_net = alignment_net.to(device)
    aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)
    
    scores_all = []
    for idx in tqdm(range(len(test_dataset)), total=len(test_dataset)):
        burst, frame_gt, meta_info_burst, meta_info_gt = test_dataset[idx]
        
        shape = burst.shape
        burst_size = 14
        n_frames = cfg.MODEL.BURST_SIZE
        n_ensemble = burst_size - n_frames + 1
        ensemble_burst = torch.zeros([n_ensemble, n_frames, *shape[1:]]).to(device)
        for i, idx in enumerate(get_ensemble_idx(burst_size=burst_size, num_frame_used=n_frames)):
            ensemble_burst[i] = burst[idx]
        ensemble_burst.to(device)
        
        with torch.no_grad():
            net_pred = model.model(ensemble_burst)
            net_pred = torch.mean(net_pred, axis=0)
            
        net_pred = net_pred.unsqueeze(0)
        burst = burst.unsqueeze(0).to(device)
        frame_gt = frame_gt.unsqueeze(0).to(device)
        
        # Calculate Aligned PSNR
        score = aligned_psnr_fn(net_pred, frame_gt, burst)

        scores_all.append(score)
        
    mean_psnr = sum(scores_all) / len(scores_all)

    print('Mean PSNR is {:0.3f}'.format(mean_psnr.item()))
    

def get_ensemble_idx(burst_size=14, num_frame_used=8):
    n_ensemble = burst_size - num_frame_used + 1
    ensemble_idx = np.array([np.arange(idx, idx + num_frame_used) for idx in range(n_ensemble)])
    ensemble_idx[:, 0] = 0
    
    return ensemble_idx


def main():
    args = parse_args()

    if len(args.config_file) > 0:
        print('Configuration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    if len(args.output_dirname) == 0:
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
