import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

import torch

from model.config import cfg
from model.modeling.build_model import ModelWithLoss
from model.provided_toolkit.datasets.synthetic_burst_test_set import SyntheticBurstVal
from model.provided_toolkit.datasets.burstsr_test_dataset import BurstSRDataset


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='')
    parser.add_argument('--trained_model', type=str, default='')
    return parser.parse_args()


def test(args, cfg):
    device = torch.device('cuda')
    model = ModelWithLoss(cfg).to(device)
    
    model.model.load_state_dict(torch.load(args.trained_model))
    if model.flow_refine:
        FR_model_path = os.path.dirname(args.trained_model)[:-5] + "FR_model/" + os.path.basename(args.trained_model)
        model.FR_model.load_state_dict(torch.load(FR_model_path))
    if model.denoise_burst:
        denoise_model_path = os.path.dirname(args.trained_model)[:-5] + "denoise_model/" + os.path.basename(args.trained_model)
        model.denoise_model.load_state_dict(torch.load(denoise_model_path))
    model.cuda()

    print('Loading Datasets...')
    # test_transforms =
    if cfg.DATASET.TRACK == 'synthetic':
        do_test_synthetic(args, cfg, model, device)
    elif cfg.DATASET.TRACK == 'real':
        do_test_real(args, cfg, model, device)
    
    
def do_test_synthetic(args, cfg, model, device):
    test_dataset = SyntheticBurstVal(cfg.DATASET.TEST_SYNTHETIC)
    
    submit_dir = os.path.join(cfg.OUTPUT_DIRNAME, "test_submit")
    os.makedirs(submit_dir, exist_ok=True)

    print(f"testing on validation data at {cfg.DATASET.TEST_SYNTHETIC}...")
    for idx in tqdm(range(len(test_dataset))):
        data_dict = test_dataset[idx]
        burst_name = data_dict['burst_name']
        
        net_pred = pred_ensemble(model, data_dict, cfg.MODEL.BURST_SIZE, device)
        
        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_submit = (net_pred.permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 14)).cpu().numpy().astype(np.uint16)
        
        # Save predictions as png
        cv2.imwrite(os.path.join(submit_dir, '{}.png'.format(burst_name)), net_pred_submit)
        
    import shutil
    shutil.make_archive(os.path.join(cfg.OUTPUT_DIRNAME, "test_submit"), 'zip', root_dir=submit_dir)
    return
        
    
def do_test_real(args, cfg, model, device):
    test_dataset = BurstSRDataset(cfg.DATASET.REAL, split='test', crop_sz=80, burst_size=14, random_flip=False)
    
    submit_dir = os.path.join(cfg.OUTPUT_DIRNAME, "test_submit")
    os.makedirs(submit_dir, exist_ok=True)
    
    print(f"testing and visualizing on validation data at {cfg.DATASET.REAL}/test...")
    for idx in tqdm(range(len(test_dataset)), total=len(test_dataset)):
        data_dict = test_dataset[idx]
        
        net_pred = pred_ensemble(model, data_dict, cfg.MODEL.BURST_SIZE, device)
        
        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_submit = (net_pred.permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 14)).cpu().numpy().astype(np.uint16)
        
        # Save predictions as png
        name = name = str(idx).zfill(len(str(len(test_dataset))))
        cv2.imwrite(os.path.join(submit_dir, '{}.png'.format(name)), net_pred_submit)
        
    import shutil
    shutil.make_archive(os.path.join(cfg.OUTPUT_DIRNAME, "test_submit"), 'zip', root_dir=submit_dir)
    return
    
    
def pred_ensemble(model, data_dict, num_frame, device):
    burst = data_dict['burst']
    shape = burst.shape
    burst_size = shape[0]
    n_ensemble = burst_size - num_frame + 1
    ensemble_burst = torch.zeros([n_ensemble, num_frame, *shape[1:]]).to(device)
    for i, ens_idx in enumerate(get_ensemble_idx(burst_size=burst_size, num_frame=num_frame)):
        ensemble_burst[i] = burst[ens_idx]
    ensemble_burst.to(device)
    data_dict['burst'] = ensemble_burst

    with torch.no_grad():
        ret_dict = model.pred(data_dict)
        net_pred = torch.mean(ret_dict['pred'], axis=0)
    
    data_dict['burst'] = burst
    
    return net_pred.clamp(0.0, 1.0)
    

def get_ensemble_idx(burst_size=14, num_frame=8):
    n_ensemble = burst_size - num_frame + 1
    ensemble_idx = np.array([np.arange(idx, idx + num_frame) for idx in range(n_ensemble)])
    ensemble_idx[:, 0] = 0
    
    return ensemble_idx


def imtext(img, msg, r, size, thickness, col, bgcol):
    cv2.putText(img, msg, r,
                cv2.FONT_HERSHEY_PLAIN, size,
                bgcol, int(4 * thickness), 1)
    cv2.putText(img, msg, r,
                cv2.FONT_HERSHEY_PLAIN, size,
                col, thickness, 1)


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
