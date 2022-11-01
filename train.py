import logging
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import random
import cv2
import yaml

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor

from utils import setup_dataset, LINZ_mapper, handle_custom_args
from models import PDwRN

import pdb
import time

logger = logging.getLogger("detectron2")

    
def do_train(cfg, model, resume=False):
    # Set the model to training mode
    model.train()
    
    # Create the optimizer and the LR-scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    
    # TO DO: CHECK THE CONFIGURATION BELOW -- NOT SURE WHAT ALL OF THEM MEAN
    checkpointer = DetectionCheckpointer(
        model, os.path.join(cfg.OUTPUT_DIR, "training"), optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    writers = default_writers(os.path.join(cfg.OUTPUT_DIR, "training"), max_iter) if comm.is_main_process() else []
    
    # Create a dataloader for the training dataset
    data_loader = build_detection_train_loader(cfg, mapper=LINZ_mapper)     # create the dataloader
    
    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            
            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

def setup_custom(args, config):
    vars(args)['config_file'] = config['config_file']
    vars(args)['custom_config'] = config
            
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(
        cfg, args
    )
    return cfg

def main(args):
    # Configurations setup
    cfg = setup(args)
    custom_config = args.custom_config

    # Build the model
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=os.path.join(cfg.OUTPUT_DIR, "training")).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume,
            find_unused_parameters=True
        )
        return do_test(cfg, model)
    
    # Distribute among GPU-s
    # NOTE: find_unused_parameters=True is used to ignore the p3-p7 level features. ADRESS THIS ISSUE IN THE FUTURE!
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True
        )
    
    # Register the LINZ-Real dataset
    data_path = custom_config['data_path']
    debug_on = custom_config['debug_on']
    setup_dataset(data_path=data_path, debug_on=debug_on)

    # Train
    do_train(cfg, model, resume=args.resume)
    logger.info("Training finished.")

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    
    # Load custom configurations
    with open("configs/custom_config.yaml", 'r') as f:
        custom_config = yaml.unsafe_load(f)
    setup_custom(args, custom_config)
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )