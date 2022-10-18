import logging
from collections import OrderedDict
import os
import cv2
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import random
import pandas as pd

import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, launch, default_setup, default_writers
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.engine import DefaultPredictor

from models import PDwRN
from utils import setup_dataset, get_image_statistics
from evaluator import PDwRNEvaluator

import pdb

logger = logging.getLogger("detectron2")


def do_test(cfg, predictor, test_data_paths, reduce_datasets=False, fast_AP=False):
    results = OrderedDict()
    for dataset_name, test_data_path in zip(cfg.DATASETS.TEST, test_data_paths):
        logger.info(f"Running evaluation on {dataset_name} with confidence threshold {cfg.MODEL.RETINANET.SCORE_THRESH_TEST}")
        dataset_dicts = DatasetCatalog.get(dataset_name)
        if reduce_datasets:
            old_length = len(dataset_dicts)
            dataset_dicts = [dataset_dict for dataset_dict in dataset_dicts if len(dataset_dict['annotations']) > 0]
            new_length = len(dataset_dicts)
            logger.info(f"Successfully removed {old_length - new_length} images from the dataset. Continuing with {new_length} images.")
        if fast_AP:
            logger.info("Running fast AP computation using 3000 images.")
            dataset_dicts = random.sample(dataset_dicts, 3000)
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

def compute_dataset_AP(cfg, dataset_name, conf_thresh):    
    logger.info("Loading the dataset.")
    
    # Initialize the parallel lists where the predictions info will be stored
    file_names = []        # a list of file names for every single detection (NOT image, but detection!)
    confidences = []       # a list of confidences per detection
    true_positives = []    # a list of boolean values indicating whether the prediction is TP (True) or FP (False)
    
    # Load the dataset
    test_dataset = DatasetCatalog.get(dataset_name)
    total_gt_count = 0
    
    # Build the predictor
    cfg.merge_from_list(["MODEL.RETINANET.SCORE_THRESH_TEST", conf_thresh])
    predictor = DefaultPredictor(cfg)
    distributed = comm.get_world_size() > 1
    if distributed:
        predictor = DistributedDataParallel(
            predictor, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
        
    logger.info("Generating the precision-recall curve.")
    # Generate predictions
    for input_ in tqdm(test_dataset):
        # Generate the predictions
        file_name = input_['file_name']
        image = cv2.imread(file_name)
        output_ = predictor(image)
        
        # Record the predictions
        file_names_, confidences_, true_positives_ = get_image_statistics(input_, output_)
        
        # Merge the statistics into the global lists
        file_names = file_names + file_names_
        confidences = confidences + confidences_
        true_positives = true_positives + true_positives_
        
        # Ground-truth counter
        total_gt_count += len(input_['annotations'])
    
    # Construct the evaluation info dataframe
    evaluation_info = pd.DataFrame(
        list(zip(file_names, confidences, true_positives)), 
        columns=["File names", "Confidence", "TP"]
    )
    
    # Sort the dataframe in descending confidence order
    evaluation_info = evaluation_info.sort_values(by=['Confidence'], ascending=False)
    
    # Create the FP column
    evaluation_info["FP"] = evaluation_info.apply(lambda row : 1 - row['TP'], axis=1)
    
    # Create the accumulated TP and FP columns
    evaluation_info["Acc_TP"] = evaluation_info['TP'].cumsum()
    evaluation_info["Acc_FP"] = evaluation_info['FP'].cumsum()
    
    # Create the "Precision" and "Recall" columns
    evaluation_info["Precision"] = evaluation_info.apply(lambda row : row["Acc_TP"] / (row["Acc_TP"] + row["Acc_FP"]), axis=1)
    evaluation_info["Recall"] = evaluation_info.apply(lambda row : row["Acc_TP"] / total_gt_count, axis=1)
    
    # Extract the lists
    recall_list = evaluation_info["Recall"].to_numpy()
    precision_list = evaluation_info["Precision"].to_numpy()
    recall_list = np.concatenate(([0.], recall_list, [1.]))
    precision_list = np.concatenate(([1.], precision_list, [0.]))
    
    # Compute and the average precision
    for i in range(precision_list.size - 1, 0, -1):
        precision_list[i - 1] = np.maximum(precision_list[i - 1], precision_list[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recall_list[1:] != recall_list[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recall_list[i + 1] - recall_list[i]) * precision_list[i + 1])
    logger.info(f"Average precision: {round(100 * ap, 2)}%.")
    
    # Save the plot
    fig = plt.figure()
    plt.plot(recall_list, precision_list, 'b')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("Precision-recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    fig.savefig("metrics/PR_curve.jpg", dpi=fig.dpi)
    
    # Save the dataframe to 'metrics/results.csv' and AP to results.txt
    logger.info("Saving the results to the 'metrics' folder...")
    evaluation_info.to_csv("metrics/results.csv")
    text = f"Average Precision (AP)={round(100 * ap, 2)}%"
    with open("metrics/results.txt", 'w') as f:
        f.write(text)
    
    logger.info("Evaluation finished.")

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
    data_path = "/home/myeghiaz/Project/PDwRN/data/"
    debug_on = False     # If set to true, only a small random portion of the dataset will be loaded
    compute_AP = True    # If True, average precision is computed
    fast_AP = True       # If True, a subset consisting of 3000 images is used for computing AP
    conf_thresh = 0.000001   # Confidence threshold used for computing the Precision-Recall curve

    setup_dataset(data_path=data_path, debug_on=debug_on)
    reduce_dataset = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS
    
    # Record the test data sets paths
    test_data_paths = [os.path.join(data_path, "test")]
        
    # Run testing
    if compute_AP:
        compute_dataset_AP(cfg, "LINZ_test", conf_thresh)
    else:
        do_test(cfg, predictor, test_data_paths)

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