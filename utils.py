import os
import json
from tqdm import tqdm
import copy
import cv2
import torch
import pickle
from random import randrange

from detectron2.structures import BoxMode, Instances, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils

import pdb

def register_LINZ(data_path, mode, debug_on=False):
    annotations_dir = os.path.join(data_path, "annotations")    # full path to the annotations directory
    images_dir = os.path.join(data_path, "images")              # full path to the images directory
    
    annotations_list = os.listdir(annotations_dir)              # list of image filenames
    images_list = os.listdir(images_dir)                        # list of annotations filenames
    
    # Initialize the return list where the dicts will be stored
    dataset_dicts = []
    
    # Loop through the images
    annotations_list = annotations_list[:1000] if debug_on else annotations_list
    for idx, annotation_file in enumerate(tqdm(annotations_list)):
        record = {}
        
        # Record preliminary information about the image
        # The line below causes a decrease in the speed of the loop
#         assert annotation.split('.')[0] + '.jpg' in images_list, f"{annotation.split('.')[0] + '.jpg'} not found in {images_dir}" 
        file_name = annotation_file.split('.')[0] + '.jpg'
        image_id = idx
        height, width = cv2.imread(os.path.join(images_dir, file_name)).shape[:2]
        
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
        
        dataset_dicts.append(record)
    
    return dataset_dicts

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

def setup_dataset(data_path, debug_on):
    # TO DO: UPDATE THIS FUNCTION TO HANDLE 2 DATASETS -- COCO AND LINZ
    
#     for mode in ["train", "val"]:
#         # Register the dataset
#         DatasetCatalog.register("COCO_CUSTOM_" + mode, lambda d=mode: register_dataset(data_path, d, debug_on))
        
#         # Update the metadata
#         category_names = get_category_names(os.path.join(data_path, f"annotations/instances_{mode}2017.json"))
#         MetadataCatalog.get("COCO_CUSTOM_" + mode).set(thing_classes=category_names)
    
    for mode in ["train", "val"]:
        # Register the dataset
        DatasetCatalog.register("LINZ_" + mode, lambda mode_=mode: register_LINZ(data_path, mode_, debug_on))
        
        # Update the metadata
#         category_names = get_category_names(os.path.join(data_path, f"annotations/instances_{mode}2017.json"))
        MetadataCatalog.get("LINZ_" + mode).set(thing_classes=["vehicle"])

def category_mapping(categories):
    """
    This function takes COCO format category descriptions and maps them to new, ordered categories.
    The reason for implementing this function is that COCO misses some category id-s. For example, it doesn't have
    a category id 12.
    """
    k = 0
    category_map = {}
    for category in categories:
        category_map[category['id']] = k
        k += 1
    return category_map

def get_category_names(anns_file_path):
    # Load the annotations file
    with open(anns_file_path, 'r') as f:
        annotations = json.load(f)
    
    category_names = []
    for category in annotations['categories']:
        category_names.append(category['name'])
    
    return category_names

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

def LINZ_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    out_data = {}
    
    # Record the basic info
    for k, v in dataset_dict.items():
        out_data[k] = v
    
    # Read the image
    file_name = dataset_dict['file_name']
    image = cv2.imread(file_name)                                    # read the image
    image = torch.tensor(image[..., ::-1].copy())                    # BGR -> RGB
    image = image.permute(2, 0, 1)                                   # (H, W, C) -> (C, H, W)
    
    # Record the image tensor
    out_data["image"] = image

    return out_data

def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    out_data = {}
    
    # Read the image info
    file_name = dataset_dict['file_name']
    image_id = dataset_dict['image_id']
    
    # Read the image
    image = cv2.imread(file_name)                                    # read the image
    image = torch.tensor(image[..., ::-1].copy())                    # BGR -> RGB
    image = image.permute(2, 0, 1)                                   # (H, W, C) -> (C, H, W)
    H, W = image.shape[1:]                                           # read height and width
    
    # Read the annotations
    annos = Instances(image_size=(H, W))
    bboxes = Boxes(torch.tensor(
        [XYWH2XYXY(obj['bbox']) for obj in dataset_dict['annotations']], 
        dtype=torch.float
    ))
    obj_classes = torch.tensor(
        [obj['category_id'] for obj in dataset_dict['annotations']],
        dtype=torch.long
    )
    annos.set("gt_boxes", bboxes)
    annos.set("gt_classes", obj_classes)
    
    # Fill the output data
    out_data["file_name"] = file_name
    out_data["image_id"] = image_id
    out_data["image"] = image
    out_data["height"] = H
    out_data["width"] = W
    out_data["instances"] = annos
    
    return out_data
    