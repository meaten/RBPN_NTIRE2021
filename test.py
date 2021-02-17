import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

import torch

from model.config import cfg
from model.modeling.build_model import ModelWithLoss
from model.utils.misc import fix_model_state_dict
from model.provided_toolkit.datasets.synthetic_burst_train_set import SyntheticBurst
from model.provided_toolkit.datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from model.provided_toolkit.datasets.synthetic_burst_val_set import SyntheticBurstVal
from model.provided_toolkit.datasets.burstsr_dataset import BurstSRDataset

from model.provided_toolkit.utils.metrics import PSNR, AlignedPSNR
from model.provided_toolkit.pwcnet.pwcnet import PWCNet


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='')
    parser.add_argument('--trained_model', type=str, default='')

    return parser.parse_args()


def test(args, cfg):
    device = torch.device('cuda')
    model = ModelWithLoss(cfg).to(device)
    print('------------Model Architecture-------------')
    print(model)
    # model.load_state_dict(fix_model_state_dict(torch.load(args.trained_model)))
    model.load_state_dict(torch.load(args.trained_model))
    model.cuda()

    print('Loading Datasets...')
    # test_transforms =
    if cfg.DATASET.TRACK == 'synthetic':
        do_test_synthetic(args, cfg, model, device)
    elif cfg.DATASET.TRACK == 'real':
        do_test_real(args, cfg, model, device)
    
    
def do_test_synthetic(args, cfg, model, device):
    test_dataset = SyntheticBurst(ZurichRAW2RGB(cfg.DATASET.TRAIN_SYNTHETIC, split='test'), crop_sz=384, burst_size=14)
    
    vis_dir = os.path.join(cfg.OUTPUT_DIRNAME, "visualize")
    os.makedirs(vis_dir, exist_ok=True)
    
    psnr_fn = PSNR()
    
    print(f"visualize heatmaps on the validation data at {cfg.DATASET.TRAIN_SYNTHETIC}/test...")
    for idx in tqdm(range(len(test_dataset))):
        burst, frame_gt, gt_flow, meta_info = test_dataset[idx]
        
        net_pred = pred_ensemble(model, burst, cfg.MODEL.BURST_SIZE, device)
        frame_gt = frame_gt.to(device)
        psnr = psnr_fn(net_pred.unsqueeze(0), frame_gt.unsqueeze(0)).cpu().numpy()
        
        output_image = create_output_image(frame_gt, net_pred, psnr)
        
        # Save predictions as png
        name = name = str(idx).zfill(len(str(len(test_dataset))))
        cv2.imwrite(os.path.join(vis_dir, '{}.png'.format(name)), output_image)
    
    test_dataset = SyntheticBurstVal(cfg.DATASET.VAL_SYNTHETIC)
    
    submit_dir = os.path.join(cfg.OUTPUT_DIRNAME, "submit")
    os.makedirs(submit_dir, exist_ok=True)
    
    print(f"test on validation data at {cfg.DATASET.VAL_SYNTHETIC}...")
    for idx in tqdm(range(len(test_dataset))):
        burst, burst_name = test_dataset[idx]
        
        net_pred = pred_ensemble(model, burst, cfg.MODEL.BURST_SIZE, device)
        
        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_submit = (net_pred.permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 14)).cpu().numpy().astype(np.uint16)
        
        # Save predictions as png
        cv2.imwrite(os.path.join(submit_dir, '{}.png'.format(burst_name)), net_pred_submit)
        
    import shutil
    shutil.make_archive(os.path.join(cfg.OUTPUT_DIRNAME, "submit"), 'zip', root_dir=submit_dir)
    
    
def do_test_real(args, cfg, model, device):
    test_dataset = BurstSRDataset(cfg.DATASET.REAL, split='val', crop_sz=80, burst_size=14, random_flip=False)
    
    alignment_net = PWCNet(load_pretrained=True,
                           weights_path=cfg.PWCNET_WEIGHT)
    alignment_net = alignment_net.to(device)
    aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)
    
    # we don't submit images in this real track
    
    vis_dir = os.path.join(cfg.OUTPUT_DIRNAME, "visualize")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"testing and visualizing on validation data at {cfg.DATASET.REAL}/val...")
    scores_all = []
    for idx in tqdm(range(len(test_dataset)), total=len(test_dataset)):
        burst, frame_gt, meta_info_burst, meta_info_gt = test_dataset[idx]
        
        net_pred = pred_ensemble(model, burst, cfg.MODEL.BURST_SIZE, device)
        frame_gt = frame_gt.to(device)
        
        # Calculate Aligned PSNR
        score = aligned_psnr_fn(net_pred.unsqueeze(0), frame_gt.unsqueeze(0), burst.unsqueeze(0))
        scores_all.append(score)
        
        output_image = create_output_image(frame_gt, net_pred, score)
            
        # Save predictions as png
        name = str(idx).zfill(len(str(len(test_dataset))))
        cv2.imwrite(os.path.join(vis_dir, '{}.png'.format(name)), output_image)
        
    mean_psnr = sum(scores_all) / len(scores_all)

    print('Mean PSNR is {:0.3f}'.format(mean_psnr.item()))
    
    
def create_output_image(frame_gt, net_pred, psnr):
    max_heatmap = 0.3
    diff = torch.norm(frame_gt - net_pred, dim=0)
    diff = (diff.clamp(0.0, max_heatmap) / max_heatmap * (2 ** 8)).cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    
    net_pred = (net_pred.permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 8)).cpu().numpy().astype(np.uint8)
    frame_gt = (frame_gt.permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 8)).cpu().numpy().astype(np.uint8)
    
    alpha = 0.3
    heatmap_blended = cv2.addWeighted(net_pred, alpha, heatmap, 1 - alpha, 0)
    
    output_image = np.concatenate([frame_gt, net_pred, heatmap_blended], axis=1)
    cv2.putText(output_image, "GT", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
    cv2.putText(output_image, f"PSNR: {psnr:.3f}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
    cv2.putText(output_image, "SR", (10 + net_pred.shape[0], 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
    
    return output_image
    
    
def pred_ensemble(model, burst, num_frame, device):
    shape = burst.shape
    burst_size = shape[0]
    n_ensemble = burst_size - num_frame + 1
    ensemble_burst = torch.zeros([n_ensemble, num_frame, *shape[1:]]).to(device)
    for i, ens_idx in enumerate(get_ensemble_idx(burst_size=burst_size, num_frame=num_frame)):
        ensemble_burst[i] = burst[ens_idx]
    ensemble_burst.to(device)
    
    with torch.no_grad():
        net_pred = model.model(ensemble_burst)
        net_pred = torch.mean(net_pred, axis=0)
        
    return net_pred
    

def get_ensemble_idx(burst_size=14, num_frame=8):
    n_ensemble = burst_size - num_frame + 1
    ensemble_idx = np.array([np.arange(idx, idx + num_frame) for idx in range(n_ensemble)])
    ensemble_idx[:, 0] = 0
    
    return ensemble_idx


def main():
    args = parse_args()

    if len(args.config_file) > 0:
        print('Configuration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    output_dirname = os.path.join('output', "images", os.path.splitext(args.trained_model)[0])
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
