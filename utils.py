import os
import json
from tqdm import tqdm
import copy
import cv2
import torch
import torch.nn as nn
import pickle
from random import randrange
import argparse

from detectron2.structures import BoxMode, Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils

import time
import pdb

def register_LINZ(data_path, mode, debug_on=False):
    """
    TO DO: DESCRIPTION OF THIS FUNCTION
    """
    data_path = copy.deepcopy(data_path)
    data_path = os.path.join(data_path, mode)
    annotations_dir = os.path.join(data_path, "annotations")    # full path to the annotations directory
    images_dir = os.path.join(data_path, "images")              # full path to the images directory
    
    annotations_list = os.listdir(annotations_dir)              # list of image filenames
    images_list = os.listdir(images_dir)                        # list of annotations filenames
    
    # Initialize the return list where the dicts will be stored
    annotations_list = annotations_list[:10] if debug_on else annotations_list
    dataset_dicts = [None for _ in range(len(annotations_list))]
    
    # Loop through the images
    for idx, annotation_file in enumerate(tqdm(annotations_list)):
        st = time.time()
        record = {}
        
        # Record preliminary information about the image
        # NOTE: reading image height and width slows down data set loading A LOT. Temporary fix -- supply (384, 384)
        file_name = annotation_file.split('.')[0] + '.jpg'
        image_id = idx
        # height, width = cv2.imread(os.path.join(images_dir, file_name)).shape[:2]
        height, width = 384, 384
        
        record["file_name"] = os.path.join(images_dir, file_name)
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width
        
        # Record detections
        vehicles = []
        with open(os.path.join(annotations_dir, annotation_file), 'rb') as f:
            annotations = pickle.load(f)
        del annotations['unknown'] # delete the unknown vehicles from the dataset
        for vehicle_type in annotations.keys():
            for vehicle_coordinate in annotations[vehicle_type]:
                vehicle = {
                    "gt_point": vehicle_coordinate,
                    "category_id": 0
                }
                vehicles.append(vehicle) # appends a numpy array consisting of 2 values in (x, y) format
        record["annotations"] = vehicles
        
        # dataset_dicts.append(record)
        dataset_dicts[idx] = record
        et = time.time()
    
    return dataset_dicts

def setup_dataset(data_path, debug_on):
    for mode in ["train", "test"]:
        # Register the dataset
        DatasetCatalog.register("LINZ_" + mode, lambda mode_=mode : register_LINZ(data_path, mode_, debug_on))
        
        # Update the metadata
        MetadataCatalog.get("LINZ_" + mode).set(thing_classes=["vehicle"])

def XYWH2XYXY(bbox):
    """
    This function takes a list of 4 values which represent a bounding box in XYWH format, and converts it into the XYXY format.
    
    Input:
        - bbox: list (or other iterable consisting of 4 values)
    """
    assert len(bbox) == 4, f"Bounding box has length {len(bbox)}. Expected length 4."
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]
    return bbox

def handle_custom_args(args):
    assert isinstance(args, argparse.Namespace), "The args variable supplied is not a valid argparse.Namespace object."
    args_dict = vars(args)
    custom_options = args_dict['opts']
    custom_args = {"debug_on": False}
    for custom_option in custom_options:
        if '=' in custom_option:
            k = custom_option.split('=')[0]
            v = custom_option.split('=')[1]
            custom_args[k] = v
        elif custom_option == "debug":
            custom_args['debug_on'] = True
        else:
            raise NotImplementedError
    vars(args)['opts'] = []
    return custom_args

def LINZ_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    out_data = {}
    
    # Record the basic info, except annotations
    for k, v in dataset_dict.items():
        if k != "annotations":
            out_data[k] = v
    
    # Record instances with random bounding boxes (only box centres matter)
    annos = Instances(image_size=(dataset_dict['height'], dataset_dict['width']))
    bboxes = Boxes(torch.tensor(
        [XYWH2XYXY([obj['gt_point'][0] - 25, obj['gt_point'][1] - 25, 50, 50]) for obj in dataset_dict['annotations']], 
        dtype=torch.float
    ))
    obj_classes = torch.tensor(
        [veh['category_id'] for veh in dataset_dict['annotations']]
    )
    annos.set("gt_boxes", bboxes)
    annos.set("gt_classes", obj_classes)
    out_data["instances"] = annos
    
    # Read the image
    file_name = dataset_dict['file_name']
    image = cv2.imread(file_name)                                    # read the image
    image = torch.tensor(image[..., ::-1].copy())                    # BGR -> RGB
    image = image.permute(2, 0, 1)                                   # (H, W, C) -> (C, H, W)
    
    # Record the image tensor
    out_data["image"] = image
    out_data["num_instances"] = len(dataset_dict["annotations"])

    return out_data   

def register_dataset(data_path, mode, debug_on=False):
    """
    This function registers a new dataset. In order to do this, it takes as inputs some parameters which describe the dataset,
    and transforms them into the dataset dictionary format expected by the model.
    """
    anns_dir = "annotations"
    imgs_dir = "train2017" if mode == "train" else "val2017"
    annotations_file = os.path.join(data_path, anns_dir, "instances_train2017.json")
    
    # Initialize the return list where the dicts will be stored
    dataset_dicts = []
    
    # Load the annotations file
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Create the category mapping from COCO to ordered id-s
    category_map = category_mapping(annotations['categories'])
    
    # Loop through the images
    full_annotations = annotations['images'][:1] if debug_on else annotations['images']
    for idx, image in enumerate(tqdm(full_annotations)):
        record = {}
        
        # Record preliminary information about the image
        record['file_name'] = os.path.join(data_path, imgs_dir, image['file_name'])
        record['image_id'] = idx
        record['height'] = image['height']
        record['width'] = image['width']
        
        # Record the detections (into objs)
        objs = []
        detections = [ann for ann in annotations['annotations'] if ann['image_id']==image['id']]
        for detection in detections:
            obj = {
                "bbox": detection['bbox'],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": category_map[detection['category_id']]
            }
            objs.append(obj)
        
        # Record the detections
        record['annotations'] = objs
        
        # Add the image description to the dataset
        dataset_dicts.append(record)
    
    return dataset_dicts