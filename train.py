import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

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

from utils import register_dataset, get_category_names

logger = logging.getLogger("detectron2")

def do_test(cfg, model):
    print("Doing testing")
    
def do_train(cfg, model, resume=False):
    model.train()
    print("Doing training")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def main(args):
    cfg = setup(args)
    
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)
    
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    
    # Register the COCO (LINZ-Real in the future) dataset
    data_path = "/home/myeghiaz/detectron2/datasets/coco"
    debug_on = True
    for mode in ["train", "val"]:
        # Register the dataset
        DatasetCatalog.register("coco_custom_" + mode, lambda d=mode: register_dataset(data_path, d, debug_on))
        
        # Update the metadata
        category_names = get_category_names(os.path.join(data_path, f"annotations/instances_{mode}2017.json"))
        MetadataCatalog.get("coco_custom_" + mode).set(thing_classes=category_names)    

    # TO DO: need to replace the model with another model
    
    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )