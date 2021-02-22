import time
import os
from tqdm import tqdm
import datetime

import torch

from model.utils.misc import SaveTorchImage


def do_train(args, cfg, model, optimizer, scheduler, data_loader, device, summary_writer):
    max_iter = len(data_loader['train']) + args.resume_iter
    trained_time = 0
    tic = time.time()
    end = time.time()

    logging_loss = 0

    print('Training Starts!!!')
    model.train()
    for iteration, (burst, gt_frame, gt_flow, meta_info) in enumerate(data_loader['train'], args.resume_iter+1):
        optimizer.zero_grad()
        loss = model(burst, gt_frame)
        loss = loss.mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        logging_loss += loss.item()

        trained_time += time.time() - end
        end = time.time()

        if iteration % args.log_step == 0:
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            logging_loss /= args.log_step
            print('===> Iter: {:07d}, LR: {:.06f}, Cost: {:2f}s, Eta: {}, Loss: {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), logging_loss))

            if summary_writer:
                summary_writer.add_scalar('train/loss', logging_loss, global_step=iteration)
                summary_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step=iteration)

            logging_loss = 0

            tic = time.time()

        if iteration % args.save_step == 0 and not args.debug:
            model_path = os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(iteration))
            optimizer_path = os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(iteration))

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)

            if args.num_gpus > 1:
                torch.save(model.model.module.state_dict(), model_path)
            else:
                torch.save(model.model.state_dict(), model_path)

            torch.save(optimizer.state_dict(), optimizer_path)

            print('=====> Save Checkpoint to {}'.format(model_path))

        if 'val' in data_loader.keys() and iteration % args.eval_step == 0:
            print('Validating...')
            model.eval()
            eval_loss = 0
            for _ in tqdm(data_loader['val']):
                loss = model()
                eval_loss += loss.item()

            val_loss /= len(data_loader['val'])

            validation_time = time.time() - end
            trained_time += validation_time
            end = time.time()
            tic = time.time()
            print('======> Cost: {:2f}s, Loss: {:.06f}'.format(validation_time, eval_loss))

            if summary_writer:
                summary_writer.add_scalar('val/loss', val_loss, global_step=iteration)

            model.train()
