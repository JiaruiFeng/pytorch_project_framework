"""Train a model on your project.

Author:
    Chris Chute (chute@stanford.edu)
Edior:
    Jiarui Feng

"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util
from args import get_train_args
from collections import OrderedDict
from json import dumps
from model import demo

from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load

import math

def get_model(log,args):
    if args.model_name=="demo":
        model = demo(input_size=args.input_size)
    else:
        raise ValueError("Model name doesn't exist.")
    model = nn.DataParallel(model, args.gpu_ids)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0

    return model,step


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, type="train")
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get your model
    log.info('Building model...')
    model, step=get_model(log,args)
    model = model.to(device)
    model.train()

    #Exponential moving average
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(),lr=1,betas=[0.8,0.999],eps=1e-7,weight_decay=args.l2_wd)
    scheduler=sched.LambdaLR(optimizer,lambda step: args.l2_wd)

    #get loss computer
    cri=nn.CrossEntropyLoss()

    # Get data loader
    log.info('Building dataset...')

    train_dataset = util.load_dataset(args.train_file)
    dev_dataset = util.load_dataset(args.dev_file)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=util.collate_fn)

    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=util.collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
                for batch_x,batch_y in train_loader:
                    # Setup for forward
                    batch_x=batch_x.to(device)
                    batch_y=batch_y.to(device)
                    batch_size=batch_x.size(0)
                    optimizer.zero_grad()

                    # Forward
                    output=model(batch_x)
                    loss = cri(output,batch_y)
                    loss_val = loss.item()

                    # Backward
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    ema(model, step // batch_size)

                    # Log info
                    step += batch_size
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(epoch=epoch,
                                             NLL=loss_val)
                    tbx.add_scalar('train/Loss', loss_val, step)
                    tbx.add_scalar('train/LR',
                                   optimizer.param_groups[0]['lr'],
                                   step)

                    steps_till_eval -= batch_size
                    if steps_till_eval <= 0:
                        steps_till_eval = args.eval_steps

                        # Evaluate and save checkpoint
                        log.info(f'Evaluating at step {step}...')
                        ema.assign(model)
                        results, pred_dict = evaluate(model, dev_loader, device,cri,
                                                      args.dev_eval_file)
                        saver.save(step, model, results[args.metric_name], device)
                        ema.resume(model)

                        # Log to console
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                        log.info(f'Dev {results_str}')


def evaluate(model, data_loader, device,cri, eval_file):
    loss_meter = util.AverageMeter()
    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for batch_x,batch_y in data_loader:
            # Setup for forward
            # Setup for forward
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_x.size(0)

            # Forward
            output = model(batch_x)
            loss = cri(output, batch_y)

            loss_val = loss.item()
            loss_meter.update(loss_val, batch_size)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=loss_meter.avg)

    model.train()

    results = util.eval_dicts(gold_dict, output)
    results_list = [('Loss', loss_meter.avg),
                    ('Accuracy', results['Accuracy'])]
    results = OrderedDict(results_list)
    return results, pred_dict





if __name__ == '__main__':
    main(get_train_args())
