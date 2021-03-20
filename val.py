import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import cfg
from model.modeling.build_model import ModelWithLoss
# from model.utils.misc import fix_model_state_dict
from model.provided_toolkit.datasets.synthetic_burst_train_set import SyntheticBurst
from model.provided_toolkit.datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from model.provided_toolkit.datasets.synthetic_burst_val_set import SyntheticBurstVal
from model.provided_toolkit.datasets.burstsr_dataset import BurstSRDataset

from model.provided_toolkit.utils.metrics import AlignedPSNR_custom, PSNR
from model.provided_toolkit.pwcnet.pwcnet import PWCNet

from model.utils.misc import SaveTorchImage, flow2color


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='')
    parser.add_argument('--trained_model', type=str, default='')
    parser.add_argument('--submit', action='store_true')
    return parser.parse_args()


def val(args, cfg):
    device = torch.device('cuda')
    model = ModelWithLoss(cfg).to(device)
    
    # model.load_state_dict(fix_model_state_dict(torch.load(args.trained_model)))
    model.model.load_state_dict(torch.load(args.trained_model))
    if model.flow_refine:
        FR_model_path = os.path.dirname(args.trained_model)[:-5] + "FR_model/" + os.path.basename(args.trained_model)
        model.FR_model.load_state_dict(torch.load(FR_model_path))
    if model.denoise_burst:
        denoise_model_path = os.path.dirname(args.trained_model)[:-5] + "denoise_model/" + os.path.basename(args.trained_model)
        model.denoise_model.load_state_dict(torch.load(denoise_model_path))
    model.cuda()

    model.eval()
    print('Loading Datasets...')
    if cfg.DATASET.TRACK == 'synthetic':
        do_val_synthetic(args, cfg, model, device)
    elif cfg.DATASET.TRACK == 'real':
        do_val_real(args, cfg, model, device)
    
    
def do_val_synthetic(args, cfg, model, device):
    val_dataset = SyntheticBurst(ZurichRAW2RGB(cfg.DATASET.TRAIN_SYNTHETIC, split='test'), crop_sz=384, burst_size=14)
    
    vis_dir = os.path.join(cfg.OUTPUT_DIRNAME, "visualize")
    os.makedirs(vis_dir, exist_ok=True)
    
    psnr_fn = PSNR()
    
    if args.submit:
        val_dataset = SyntheticBurstVal(cfg.DATASET.VAL_SYNTHETIC)
    
        submit_dir = os.path.join(cfg.OUTPUT_DIRNAME, "submit")
        os.makedirs(submit_dir, exist_ok=True)
    
        print(f"val on validation data at {cfg.DATASET.VAL_SYNTHETIC}...")
        for idx in tqdm(range(len(val_dataset))):
            data_dict = val_dataset[idx]
            burst_name = data_dict['burst_name']
            
            net_pred = self_ensemble(model, data_dict, cfg.MODEL.BURST_SIZE, device)
            
            # Normalize to 0  2^14 range and convert to numpy array
            net_pred_submit = (net_pred.permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 14)).cpu().numpy().astype(np.uint16)
            
            # Save predictions as png
            cv2.imwrite(os.path.join(submit_dir, '{}.png'.format(burst_name)), net_pred_submit)
            
        import shutil
        shutil.make_archive(os.path.join(cfg.OUTPUT_DIRNAME, "submit"), 'zip', root_dir=submit_dir)
        return
    
    print(f"visualize heatmaps on the validation data at {cfg.DATASET.TRAIN_SYNTHETIC}/test...")
    scores_all = []
    for idx in tqdm(range(len(val_dataset))):
        data_dict = val_dataset[idx]
        
        net_pred = self_ensemble(model, data_dict, cfg.MODEL.BURST_SIZE, device)
        # net_pred = burst_ensemble(model, data_dict, cfg.MODEL.BURST_SIZE, device)
        gt_frame = data_dict['gt_frame'].to(device)
        psnr = psnr_fn(net_pred.unsqueeze(0), gt_frame.unsqueeze(0)).cpu().numpy()
        scores_all.append(psnr)
        
        output_image = create_output_image(gt_frame, net_pred, psnr, data_dict['burst'])
        
        # Save predictions as png
        name = name = str(idx).zfill(len(str(len(val_dataset))))
        cv2.imwrite(os.path.join(vis_dir, '{}.png'.format(name)), output_image)
        print(psnr)
    mean_psnr = sum(scores_all) / len(scores_all)

    with open(os.path.join(cfg.OUTPUT_DIRNAME, 'result_psnr.txt'), 'w') as f:
        string = 'Mean PSNR is {:0.3f}\n'.format(mean_psnr.item())
        print(string)
        f.write(string)
        
    
def do_val_real(args, cfg, model, device):
    val_dataset = BurstSRDataset(cfg.DATASET.REAL, split='val', crop_sz=80, burst_size=14, random_flip=False)
    
    alignment_net = PWCNet(load_pretrained=True,
                           weights_path=cfg.PWCNET_WEIGHTS)
    alignment_net = alignment_net.to(device)
    aligned_psnr_fn = AlignedPSNR_custom(alignment_net=alignment_net, boundary_ignore=40)
    
    # we don't submit images in this real track
    
    vis_dir = os.path.join(cfg.OUTPUT_DIRNAME, "visualize")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"validating and visualizing on validation data at {cfg.DATASET.REAL}/val...")
    scores_all = []
    for idx in tqdm(range(len(val_dataset)), total=len(val_dataset)):
        data_dict = val_dataset[idx]
        
        # net_pred = self_ensemble(model, data_dict, cfg.MODEL.BURST_SIZE, device)
        net_pred = burst_ensemble(model, data_dict, cfg.MODEL.BURST_SIZE, device)
        
        # Calculate Aligned PSNR
        score, pred_warped_m, gt = aligned_psnr_fn(net_pred.unsqueeze(0),
                                                   data_dict['gt_frame'].unsqueeze(0).to(device),
                                                   data_dict['burst'].unsqueeze(0).to(device))
        scores_all.append(score)
        output_image = create_output_image(gt.squeeze(0), pred_warped_m.squeeze(0), score, data_dict['burst'])
            
        # Save predictions as png
        name = str(idx).zfill(len(str(len(val_dataset))))
        cv2.imwrite(os.path.join(vis_dir, '{}.png'.format(name)), output_image)
        print(score)
    mean_psnr = sum(scores_all) / len(scores_all)

    with open(os.path.join(cfg.OUTPUT_DIRNAME, 'result_psnr.txt'), 'w') as f:
        string = 'Mean PSNR is {:0.3f}'.format(mean_psnr.item())
        print(string)
        f.write(string)
    
    
def create_output_image(frame_gt, net_pred, psnr, burst):
    max_heatmap = 0.3
    diff = torch.norm(frame_gt - net_pred, dim=0)
    diff = (diff.clamp(0.0, max_heatmap) / max_heatmap * (2 ** 8 - 1)).cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    
    net_pred = (net_pred.permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 8 - 1)).cpu().numpy().astype(np.uint8)
    frame_gt = (frame_gt.permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 8 - 1)).cpu().numpy().astype(np.uint8)  # (384,384,3)

    alpha = 0.3
    heatmap_blended = cv2.addWeighted(frame_gt, alpha, heatmap, 1 - alpha, 0)
    
    # make rgb image from burst image
    # burst (14, 4, 48, 48), (num frame, channel, Height, Width)
    rgb = burst[:, [0, 1, 3], :, :]  # (14, 3, 48, 48)
    base_rgb_image = rgb[:1, :, :, :]  # (1, 3, 48, 48)

    # resize rgb image
    base_rgb_image_8x = torch.nn.functional.interpolate(base_rgb_image, scale_factor=8., mode='bilinear', align_corners=False)
    base_rgb_image_8x_np = (base_rgb_image_8x[0, :, :, :].permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 8 - 1)).cpu().numpy().astype(np.uint8)
    base_rgb_image_8x_np = center_crop(base_rgb_image_8x_np, frame_gt.shape)

    output_image = np.concatenate([frame_gt, net_pred, heatmap_blended, base_rgb_image_8x_np], axis=1)
    imtext(output_image, "GT", (10, 30), 1.5, 1, (0, 0, 0), (255, 255, 255))
    imtext(output_image, f"PSNR: {psnr:.3f}", (10, 60), 1.5, 1, (0, 0, 0), (255, 255, 255))
    imtext(output_image, "SR", (10 + net_pred.shape[0], 30), 1.5, 1, (0, 0, 0), (255, 255, 255))
    imtext(output_image, "Bilinear", (10 + 3 * net_pred.shape[0], 30), 1.5, 1, (0, 0, 0), (255, 255, 255))

    return output_image


def center_crop(img, shape):
    #  assume the shape of img is (h, w, c)
    img_shape = img.shape
    assert img_shape[2] == shape[2]
    if img_shape == shape:
        return img
    else:
        h_diff = img_shape[0] - shape[0]
        w_diff = img_shape[1] - shape[1]
        assert h_diff > 0 and w_diff > 0
        h_ignr = int(h_diff / 2)
        w_ignr = int(w_diff / 2)
        img = img[h_ignr:-h_ignr, w_ignr:-w_ignr]
        assert img.shape == shape
        return img
    
    
def self_ensemble(model, data_dict, num_frame, device):
    from itertools import product
    from copy import deepcopy
    # list_angle = list(range(4))
    # list_angle = [0] * 4
    list_angle = [0]
    # list_flip = [False, True]
    # list_flip = [True]
    list_flip = [False]

    burst = data_dict['burst']
    n_ensemble = len(list_angle) * len(list_flip)
    ensemble_burst = [deepcopy(data_dict) for _ in range(n_ensemble)]
    for i, (angle, flip) in enumerate(product(list_angle, list_flip)):
        # temp = rotate_image(flip_image(burst, flip=flip), angle=angle)
        # for j in range(temp.shape[0]):
        #     temp2 = (temp[j].permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 8 - 1)).cpu().numpy().astype(np.uint8)
        #     cv2.imwrite(os.path.join('temp', '{}_{}.png'.format(i, j)), temp2[:, :, [0, 1, 3]])
        ensemble_burst[i]['burst'] = rotate_image(flip_image(deepcopy(burst), flip=flip), angle=angle)

        for j in range(ensemble_burst[i]['burst'].shape[0]):
            temp = ensemble_burst[i]['burst'][j]
            temp = (temp.permute(1, 2, 0).clamp(0.0, 1.0) * (2 ** 8 - 1)).cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join('temp', '{}_{}.png'.format(i, j)), temp[:, :, [0, 1, 3]])

        # aug = rotate_image(flip_image(deepcopy(burst), flip=flip), angle=angle)
        # assert (burst == flip_image(rotate_image(deepcopy(aug), angle=-angle), flip=flip)).all()

    net_pred = [burst_ensemble(model, dic, num_frame, device) for dic in ensemble_burst]
    # net_pred = [model.pred(dic)['pred'] for dic in ensemble_burst]
    
    for i, (angle, flip) in enumerate(product(list_angle, list_flip)):
        net_pred[i] = flip_image(rotate_image(net_pred[i], angle=-angle), flip=flip)
    
    # net_pred = torch.mean(torch.stack(net_pred), axis=0)
    
    return net_pred[0].clamp(0.0, 1.0)


def flip_image(image, flip=True):
    if flip:
        if len(image.shape) == 3:
            channel, height, width = image.shape
            image = image[:, :, torch.arange(width - 1, -1, -1)]
    
        elif len(image.shape) == 4:
            for i in range(len(image)):
                channel, height, width = image[i].shape
                image[i] = image[i][:, :, torch.arange(width - 1, -1, -1)]
        else:
            raise ValueError
    return image
            
    
def rotate_image(image, angle=1):
    if angle:
        if len(image.shape) == 3:
            image = torch.rot90(image, angle, [1, 2])
        
        elif len(image.shape) == 4:
            for i in range(len(image)):
                image[i] = torch.rot90(image[i], angle, [1, 2])
        else:
            raise ValueError
    return image
    
    
def burst_ensemble(model, data_dict, num_frame, device):
    burst = data_dict['burst']
    shape = burst.shape
    burst_size = shape[0]
    n_ensemble = burst_size - num_frame + 1
    ensemble_burst = torch.zeros([n_ensemble, num_frame, *shape[1:]]).to(device)
    for i, ens_idx in enumerate(get_burst_ensemble_idx(burst_size=burst_size, num_frame=num_frame)):
        ensemble_burst[i] = burst[ens_idx]
    ensemble_burst.to(device)
    data_dict['burst'] = ensemble_burst

    up = nn.Upsample(scale_factor=4, mode='bilinear')

    with torch.no_grad():
        ret_dict = model.pred(data_dict)
        net_pred = torch.mean(ret_dict['pred'], axis=0)

    for i in range(ret_dict['refined_flow'].shape[0]):
        for j in range(ret_dict['refined_flow'].shape[1]):
            img = ret_dict['pred'][i].cpu().detach().clamp(0.0, 1.0).permute(1, 2, 0).numpy()*(2**8 - 1)
            img = img.astype(np.uint8)
            flow = up(ret_dict['refined_flow'][i, j].unsqueeze(0)).squeeze(0)
            # flow = F.interpolate(ret_dict['refined_flow'][i, j], size=384, mode='linear')
            
            flow = flow.cpu().detach().permute(1, 2, 0).numpy()
            flow = flow2color(flow)

            temp = cv2.addWeighted(img, 0.4, flow, 0.6, 0)

            cv2.imwrite(os.path.join('temp', 'f_{}_{}.png'.format(i, j)), temp)
            # flow = cv2.addWeighted(frame_gt, alpha, heatmap, 1 - alpha, 0)

                
    data_dict['burst'] = burst
    
    return net_pred.clamp(0.0, 1.0)
    

def get_burst_ensemble_idx(burst_size=14, num_frame=8):
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

    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(cfg.SEED)
    else:
        raise Exception('GPU not found')

    val(args, cfg)


if __name__ == '__main__':
    main()
