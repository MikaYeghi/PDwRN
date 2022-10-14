import logging
from collections import OrderedDict
import os
import cv2
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, launch, default_setup, default_writers
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.engine import DefaultPredictor

from models import PDwRN
from utils import setup_dataset
from evaluator import PDwRNEvaluator

import pdb

logger = logging.getLogger("detectron2")


def do_test_new(cfg, predictor, test_data_paths, reduce_datasets=True):
    results = OrderedDict()
    for dataset_name, test_data_path in zip(cfg.DATASETS.TEST, test_data_paths):
        logger.info(f"Running evaluation on {dataset_name}")
        dataset_dicts = DatasetCatalog.get(dataset_name)
        if reduce_datasets:
            old_length = len(dataset_dicts)
            dataset_dicts = [dataset_dict for dataset_dict in dataset_dicts if len(dataset_dict['annotations']) > 0]
            new_length = len(dataset_dicts)
            logger.info(f"Successfully removed {old_length - new_length} images from the dataset.")
            logger.info(f"Continuing with {new_length} images.")
        evaluator = PDwRNEvaluator(cfg, test_data_path)

        for input_ in tqdm(dataset_dicts):
            image = cv2.imread(input_['file_name'])
            output_ = predictor(image)
            evaluator.process([input_], [output_])
            
        results_i = evaluator.evaluate()
        if results_i is None:
            results_i = {}
        
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_test(cfg, predictor, test_data_paths, reduce_datasets=True):
    results = OrderedDict()
    for dataset_name, test_data_path in zip(cfg.DATASETS.TEST, test_data_paths):
        logger.info(f"Running evaluation on {dataset_name}")
        dataset_dicts = DatasetCatalog.get(dataset_name)
        if reduce_datasets:
            old_length = len(dataset_dicts)
            dataset_dicts = [dataset_dict for dataset_dict in dataset_dicts if len(dataset_dict['annotations']) > 0]
            new_length = len(dataset_dicts)
            logger.info(f"Successfully removed {old_length - new_length} images from the dataset.")
            logger.info(f"Continuing with {new_length} images.")
        evaluator = PDwRNEvaluator(cfg, test_data_path)

        for input_ in tqdm(dataset_dicts):
            image = cv2.imread(input_['file_name'])
            output_ = predictor(image)
            evaluator.process([input_], [output_])
            
        results_i = evaluator.evaluate()
        if results_i is None:
            results_i = {}
        
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

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
    cfg = setup(args)

    # model = build_model(cfg)
    predictor = DefaultPredictor(cfg)
    logger.info("Predictor model:\n{}".format(predictor.model))

    distributed = comm.get_world_size() > 1
    if distributed:
        predictor = DistributedDataParallel(
            predictor, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    # Register the LINZ-Real dataset
    data_path = "/home/myeghiaz/Project/PDRN/data/"
    debug_on = False
    setup_dataset(data_path=data_path, debug_on=debug_on)
    
    # Record the test data sets paths
    test_data_paths = [os.path.join(data_path, "test")]
        
    # Run testing
    do_test(cfg, predictor, test_data_paths, reduce_datasets=False)

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