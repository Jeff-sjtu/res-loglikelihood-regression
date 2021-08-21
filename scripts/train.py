import logging
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data
from rlepose.models import builder
from rlepose.opt import cfg, logger, opt
from rlepose.trainer import train, validate, validate_gt, validate_gt_3d
from rlepose.utils.env import init_dist
from rlepose.utils.metrics import NullWriter
from rlepose.utils.transforms import get_coord

num_gpu = torch.cuda.device_count()


def _init_fn(worker_id):
    np.random.seed(opt.seed)
    random.seed(opt.seed)


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    if opt.seed is not None:
        setup_seed(opt.seed)

    if opt.launcher == 'slurm':
        main_worker(None, opt, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        opt.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt, cfg))


def main_worker(gpu, opt, cfg):
    if opt.seed is not None:
        setup_seed(opt.seed)

    if gpu is not None:
        opt.gpu = gpu

    init_dist(opt)

    if opt.log:
        cfg_file_name = os.path.basename(opt.cfg)
        filehandler = logging.FileHandler(
            './exp/{}-{}/training.log'.format(opt.exp_id, cfg_file_name))
        streamhandler = logging.StreamHandler()

        logger.setLevel(logging.INFO)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)
    else:
        null_writer = NullWriter()
        sys.stdout = null_writer

    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    opt.nThreads = int(opt.nThreads / num_gpu)

    # Model Initialize
    m = preset_model(cfg)

    m.cuda(opt.gpu)
    m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[opt.gpu])

    criterion = builder.build_loss(cfg.LOSS).cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(m.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0001)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    train_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True, heatmap2coord=cfg.TEST.HEATMAP2COORD)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=opt.world_size, rank=opt.rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), num_workers=opt.nThreads, sampler=train_sampler, worker_init_fn=_init_fn)

    output_3d = cfg.DATA_PRESET.get('OUT_3D', False)

    heatmap_to_coord = get_coord(cfg, cfg.DATA_PRESET.HEATMAP_SIZE, output_3d)

    opt.trainIters = 0
    best_err = 999

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        train_sampler.set_epoch(i)
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loss, acc = train(opt, cfg, train_loader, m, criterion, optimizer)
        logger.epochInfo('Train', opt.epoch, loss, acc)

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            # Save checkpoint
            if opt.log:
                torch.save(m.module.state_dict(), './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
            # Prediction Test
            with torch.no_grad():
                if output_3d:
                    err = validate_gt_3d(m, opt, cfg, heatmap_to_coord)
                    if opt.log and err <= best_err:
                        best_err = err
                        torch.save(m.module.state_dict(), './exp/{}-{}/best_model.pth'.format(opt.exp_id, cfg.FILE_NAME))

                    logger.info(f'##### Epoch {opt.epoch} | gt results: {err}/{best_err} #####')

                else:
                    gt_AP = validate_gt(m, opt, cfg, heatmap_to_coord)
                    det_AP = validate(m, opt, cfg, heatmap_to_coord)
                    logger.info(f'##### Epoch {opt.epoch} | gt mAP: {gt_AP} | det mAP: {det_AP} #####')

        torch.distributed.barrier()  # Sync

    torch.save(m.module.state_dict(), './exp/{}-{}/final.pth'.format(opt.exp_id, cfg.FILE_NAME))


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model


if __name__ == "__main__":

    if opt.world_size > num_gpu:
        print(f'Wrong world size. Changing it from {opt.world_size} to {num_gpu}.')
        opt.world_size = num_gpu
    main()
